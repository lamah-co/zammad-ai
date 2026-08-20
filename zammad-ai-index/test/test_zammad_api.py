"""Tests for Zammad API response parsing."""

import os
from unittest import TestCase
from unittest.mock import Mock

os.environ.setdefault("ZAMMAD_AI_DISABLE_YAML", "true")
os.environ.setdefault("ZAMMAD_AI_ZAMMAD__TYPE", "api")
os.environ.setdefault("ZAMMAD_AI_ZAMMAD__BASE_URL", "https://zammad.example.test")
os.environ.setdefault("ZAMMAD_AI_ZAMMAD__KNOWLEDGE_BASE_ID", "2")
os.environ.setdefault("ZAMMAD_AI_ZAMMAD__AUTH_TOKEN", "token")

from pydantic import SecretStr

from job.settings.zammad import ZammadAPISettings
from job.zammad.api import ZammadAPIClient


class ZammadAPIClientTest(TestCase):
    """Tests for the token-based Zammad API client."""

    def make_client(self) -> ZammadAPIClient:
        """Create a test client with network calls mocked by each test."""
        return ZammadAPIClient(
            ZammadAPISettings(
                base_url="https://zammad.example.test",
                knowledge_base_id=2,
                auth_token=SecretStr("token"),
            )
        )

    def test_fetches_answer_body_by_translation_content_id(self) -> None:
        """Fetch answer content through the translation content_id."""
        client = self.make_client()
        client._request = Mock(  # type: ignore[method-assign]
            side_effect=[
                {
                    "id": 1,
                    "assets": {
                        "KnowledgeBaseAnswer": {
                            "1": {
                                "attachments": [],
                                "created_at": "2026-08-13T01:25:01Z",
                                "updated_at": "2026-08-13T01:25:03Z",
                            }
                        },
                        "KnowledgeBaseAnswerTranslation": {
                            "1": {
                                "answer_id": 1,
                                "content_id": 2,
                                "title": "Arabic FAQ title",
                            }
                        },
                    },
                },
                {
                    "id": 1,
                    "assets": {
                        "KnowledgeBaseAnswerTranslationContent": {
                            "2": {
                                "body": "<p>Arabic FAQ body</p>",
                            }
                        }
                    },
                },
            ]
        )

        answer = client.get_kb_answer_by_id(1)

        assert answer is not None
        self.assertEqual(answer.answerTitle, "Arabic FAQ title")
        self.assertEqual(answer.answerBody.strip(), "Arabic FAQ body")
        client._request.assert_any_call("GET", "/api/v1/knowledge_bases/2/answers/1")
        client._request.assert_any_call("GET", "/api/v1/knowledge_bases/2/answers/1?include_contents=2")

    def test_accepts_legacy_content_keyed_by_answer_id(self) -> None:
        """Keep compatibility with responses that key content by answer ID."""
        client = self.make_client()
        client._request = Mock(  # type: ignore[method-assign]
            return_value={
                "id": 1,
                "assets": {
                    "KnowledgeBaseAnswer": {
                        "1": {
                            "attachments": [],
                            "created_at": "2026-08-13T01:25:01Z",
                            "updated_at": "2026-08-13T01:25:03Z",
                        }
                    },
                    "KnowledgeBaseAnswerTranslation": {
                        "1": {
                            "answer_id": 1,
                            "title": "Legacy FAQ title",
                        }
                    },
                    "KnowledgeBaseAnswerTranslationContent": {
                        "1": {
                            "body": "<p>Legacy FAQ body</p>",
                        }
                    },
                },
            }
        )

        answer = client.get_kb_answer_by_id(1)

        assert answer is not None
        self.assertEqual(answer.answerTitle, "Legacy FAQ title")
        self.assertEqual(answer.answerBody.strip(), "Legacy FAQ body")
        client._request.assert_called_once_with("GET", "/api/v1/knowledge_bases/2/answers/1")
