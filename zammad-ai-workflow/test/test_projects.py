"""Tests for answering each project from its own collection, chosen by the event's request type."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from faststream.exceptions import AckMessage
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.answer.knowledgebase import QdrantKBClient
from app.kafka.broker import build_router
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadArticle, ZammadTicket
from app.projects import load_project_use_cases, project_settings
from app.settings import ZammadAISettings
from app.settings.triage import Action, ActionTypes, Category

# The shared project file from zammad-infra; the workflow reads only key and qdrant.
ERSAAL = """\
key: ersaal
zammad:
  group: ERSAAL Support
  adopt_groups: [ERSAAL]
kb:
  root_category: ERSAAL
qdrant:
  collection_name: ersaal_kb
docs:
  repo: lamah-co/ersaal-docs
  site_url: https://docs.getersaal.com
  sections: [{path: content/docs, kind: guide}]
"""


def test_shared_project_file_loads_and_scopes_the_answer_settings(
    tmp_path: Path, settings_factory: Callable[..., ZammadAISettings]
) -> None:
    """The project's collection and key replace the default ones; everything else stays shared."""
    (tmp_path / "ersaal.yaml").write_text(ERSAAL, encoding="utf-8")
    base = settings_factory()

    project = load_project_use_cases(tmp_path)["ersaal"]
    scoped = project_settings(base, project)

    assert scoped.answer.qdrant.collection_name == "ersaal_kb"
    assert scoped.answer.qdrant.project == "ersaal"
    assert base.answer.qdrant.project is None
    assert scoped.triage == base.triage


def test_projects_cannot_share_a_collection(tmp_path: Path) -> None:
    """Two projects in one collection would see each other's answers."""
    (tmp_path / "ersaal.yaml").write_text(ERSAAL, encoding="utf-8")
    (tmp_path / "demo.yaml").write_text(ERSAAL.replace("key: ersaal", "key: demo"), encoding="utf-8")

    with pytest.raises(ValueError, match="share a Qdrant collection"):
        load_project_use_cases(tmp_path)


@pytest.mark.parametrize(
    ("project", "extra", "expected"),
    [
        (None, None, None),
        ("ersaal", None, Filter(must=[FieldCondition(key="metadata.project", match=MatchValue(value="ersaal"))])),
    ],
)
def test_retrieval_is_restricted_to_the_project(
    project: str | None, extra: Filter | None, expected: Filter | None
) -> None:
    """A project's searches only see vectors tagged with its key."""
    client = QdrantKBClient.__new__(QdrantKBClient)
    client.qdrant_settings = SimpleNamespace(project=project)

    assert client._scoped(extra) == expected


def test_project_filter_keeps_an_existing_filter() -> None:
    """Law lookups and similar filters still apply inside the project."""
    client = QdrantKBClient.__new__(QdrantKBClient)
    client.qdrant_settings = SimpleNamespace(project="ersaal")
    law = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value="law"))])

    scoped = client._scoped(law)

    assert scoped is not None and scoped.must is not None
    assert law in scoped.must and len(scoped.must) == 2


@pytest.mark.asyncio
async def test_events_are_answered_by_their_projects_services(
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: Callable[..., ZammadAISettings],
    kafka_message_factory: Callable[..., dict[str, str]],
) -> None:
    """An ersaal event uses ERSAAL's action service; a support event keeps the shared one."""
    triage = MagicMock()
    triage.zammad_client = MagicMock()
    triage.zammad_client.get_ticket = AsyncMock(
        return_value=ZammadTicket(
            id=3720, articles=[ZammadArticle(id=1, ticket_id=3720, text="msg", sender="Customer")]
        )
    )
    triage.perform_triage = AsyncMock(
        return_value=TriageResult(
            user_text="",
            category=Category(name="Test"),
            action=Action(name="Keine_Aktion", description="No action", type=ActionTypes.NoAction),
            reasoning="test",
            confidence=0.9,
        )
    )
    shared, ersaal = MagicMock(), MagicMock()
    for action in (shared, ersaal):
        action.execute_action = AsyncMock()
        action.hand_off_failed_ticket = AsyncMock()
    monkeypatch.setattr("app.kafka.broker.get_triage_service", lambda *a, **k: triage)
    monkeypatch.setattr("app.kafka.broker.get_answer_service", lambda *a, **k: MagicMock())
    monkeypatch.setattr("app.kafka.broker.get_action_service", lambda *a, **k: shared)
    monkeypatch.setattr("app.kafka.broker.build_project_actions", lambda settings: {"ersaal": ersaal})
    settings = settings_factory(valid_request_types=["support", "ersaal"])
    _, handler = build_router(settings=settings)

    for request_type in ("ersaal", "support"):
        with pytest.raises(AckMessage):
            await handler(event=Event.model_validate(kafka_message_factory(anliegenart=request_type)))

    ersaal.execute_action.assert_awaited_once()
    shared.execute_action.assert_awaited_once()
