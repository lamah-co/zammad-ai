"""Zammad knowledge-base API compatibility tests."""

import os

os.environ.setdefault("ZAMMAD_AI_ZAMMAD", '{"type":"api","base_url":"https://zammad.example.com","knowledge_base_id":2,"auth_token":"test-token"}')

from job.settings.zammad import ZammadAPISettings
from job.zammad.api import ZammadAPIClient


def test_get_kb_answer_uses_translation_content_id(monkeypatch) -> None:
    """Current Zammad keys rich-text content by content ID, not answer ID."""
    client = ZammadAPIClient(
        ZammadAPISettings(
            base_url="https://zammad.example.com",
            knowledge_base_id=2,
            auth_token="test-token",
        )
    )
    first_response = {
        "id": 42,
        "assets": {
            "KnowledgeBaseAnswerTranslation": {
                "42": {"title": "سؤال", "content_id": 99},
            },
        },
    }
    content_response = {
        "id": 42,
        "assets": {
            "KnowledgeBaseAnswer": {
                "42": {
                    "attachments": [],
                    "created_at": "2026-08-13T00:00:00Z",
                    "updated_at": "2026-08-13T00:00:00Z",
                },
            },
            "KnowledgeBaseAnswerTranslationContent": {
                "99": {"body": "<p>إجابة</p>"},
            },
        },
    }
    calls: list[str] = []

    def fake_request(method: str, path: str, **_kwargs):
        calls.append(path)
        return content_response if "include_contents=99" in path else first_response

    monkeypatch.setattr(client, "_request", fake_request)

    answer = client.get_kb_answer_by_id(42)

    assert answer is not None
    assert answer.answerTitle == "سؤال"
    assert answer.answerBody.strip() == "إجابة"
    assert calls == [
        "/api/v1/knowledge_bases/2/answers/42",
        "/api/v1/knowledge_bases/2/answers/42?include_contents=99",
    ]
    client.close()
