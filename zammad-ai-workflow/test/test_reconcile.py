"""Tests for answering each unanswered customer turn once (conversation mode)."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from faststream.exceptions import AckMessage
from pydantic import HttpUrl, SecretStr

from app.action.service import ActionService
from app.errors import TriageError, ZammadMissingTicketAttributeError
from app.kafka.broker import build_router
from app.models.answer import StaticAnswer
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings import ZammadAISettings
from app.settings.conversation import ConversationSettings
from app.settings.localization import LocalizationSettings
from app.settings.triage import Action, ActionTypes, Category
from app.settings.zammad import ZammadAPISettings
from app.zammad.api import ZammadAPIClient

AI_USER = 5
HUMAN_AGENT = 7


def customer(article_id: int, text: str = "msg") -> ZammadArticle:
    """Return a public WhatsApp message from the customer."""
    return ZammadArticle(id=article_id, ticket_id=3720, text=text, sender="Customer", type="whatsapp message")


def reply(article_id: int, *, by: int) -> ZammadArticle:
    """Return a public agent reply written by the given user."""
    return ZammadArticle(
        id=article_id, ticket_id=3720, text="reply", sender="Agent", type="whatsapp message", created_by_id=by
    )


def triage_result() -> TriageResult:
    """Return a minimal triage result."""
    return TriageResult(
        user_text="",
        category=Category(name="Test"),
        action=Action(name="Keine_Aktion", description="No action", type=ActionTypes.NoAction),
        reasoning="test",
        confidence=0.95,
    )


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch the router's services with mocks; set `ticket` to control what Zammad returns."""
    triage = MagicMock()
    triage.zammad_client = MagicMock()
    triage.zammad_client.get_ticket_with_context = AsyncMock()
    triage.zammad_client.get_own_user_id = AsyncMock(return_value=AI_USER)
    triage.zammad_client.remove_tag_from_ticket = AsyncMock()
    triage.perform_triage = AsyncMock(return_value=triage_result())
    action = MagicMock()
    action.execute_action = AsyncMock()
    action.hand_off_failed_ticket = AsyncMock(return_value=True)
    monkeypatch.setattr("app.kafka.broker.get_triage_service", lambda *args, **kwargs: triage)
    monkeypatch.setattr("app.kafka.broker.get_action_service", lambda *args, **kwargs: action)
    monkeypatch.setattr("app.kafka.broker.get_answer_service", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.kafka.broker.asyncio.sleep", AsyncMock())
    ns = SimpleNamespace(triage=triage, action=action)

    def set_ticket(ticket: ZammadTicket) -> None:
        triage.zammad_client.get_ticket_with_context.return_value = ticket

    ns.set_ticket = set_ticket
    return ns


async def handle(
    settings_factory: Callable[..., ZammadAISettings], kafka_message_factory: Callable[..., dict[str, str]]
) -> ZammadAISettings:
    """Send one ticket event through the handler in conversation mode and return the settings used."""
    settings = settings_factory(
        valid_request_types=["technischer Bürgersupport"], conversation=ConversationSettings(enabled=True)
    )
    _, event_handler = build_router(settings=settings)
    with pytest.raises(AckMessage):
        await event_handler(event=Event.model_validate(kafka_message_factory()))
    return settings


@pytest.mark.asyncio
async def test_burst_is_answered_once_with_the_turn_marker(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """All customer messages after the marker go to triage together, and the answer records the newest one."""
    services.set_ticket(
        ZammadTicket(
            id=3720,
            ai_answered_through_article_id=1,
            articles=[customer(1), reply(2, by=AI_USER), customer(3, "Test"), customer(4, "sender ID?")],
        )
    )

    await handle(settings_factory, kafka_message_factory)

    turn = services.triage.perform_triage.call_args.kwargs["turn"]
    assert [a.id for a in turn] == [3, 4]
    assert services.action.execute_action.call_args.kwargs["answered_through"] == 4


@pytest.mark.asyncio
async def test_duplicate_event_after_the_answer_does_nothing(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """Events for an already answered turn (bursts, redeliveries, our own reply) change nothing."""
    services.set_ticket(
        ZammadTicket(
            id=3720, ai_answered_through_article_id=4, articles=[customer(3), customer(4), reply(5, by=AI_USER)]
        )
    )

    await handle(settings_factory, kafka_message_factory)

    services.triage.perform_triage.assert_not_called()
    services.action.execute_action.assert_not_called()


@pytest.mark.asyncio
async def test_ticket_with_a_human_reply_is_left_to_the_person(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """Once a human agent replied, new customer messages are not answered by the AI."""
    services.set_ticket(ZammadTicket(id=3720, articles=[customer(1), reply(2, by=HUMAN_AGENT), customer(3)]))

    await handle(settings_factory, kafka_message_factory)

    services.triage.perform_triage.assert_not_called()
    services.action.execute_action.assert_not_called()


@pytest.mark.asyncio
async def test_resume_tag_is_consumed_after_answering(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """The resume tag lets the AI answer once and is then removed, so the marker carries the conversation on."""
    services.set_ticket(
        ZammadTicket(
            id=3720, tags=["ai_failed", "ai_resume"], articles=[customer(1), reply(2, by=HUMAN_AGENT), customer(3)]
        )
    )

    await handle(settings_factory, kafka_message_factory)

    assert services.action.execute_action.call_args.kwargs["answered_through"] == 3
    services.triage.zammad_client.remove_tag_from_ticket.assert_awaited_once_with(ticket_id=3720, tag="ai_resume")


@pytest.mark.asyncio
async def test_hand_off_after_retries_records_the_turn(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """A failed turn is handed off with the whole turn's text, and marked as handled."""
    services.set_ticket(ZammadTicket(id=3720, articles=[customer(1, "Test"), customer(2, "sender ID?")]))
    services.triage.perform_triage.side_effect = TriageError("retryable triage", retryable=True)

    await handle(settings_factory, kafka_message_factory)

    kwargs = services.action.hand_off_failed_ticket.call_args.kwargs
    assert kwargs["answered_through"] == 2
    assert "1. Test" in kwargs["customer_message"] and "2. sender ID?" in kwargs["customer_message"]


def _action_service() -> tuple[ActionService, SimpleNamespace]:
    service = object.__new__(ActionService)
    service.settings = SimpleNamespace(
        triage=SimpleNamespace(
            public_reply_subject="Answer",
            human_review_public_fallback="We will get back to you.",
            no_answer_public_fallback="We will get back to you.",
            human_review_internal_note="Review",
            no_answer_internal_note="Review",
            human_review_tag="ai_human_review",
            no_answer_tag="ai_no_answer",
            handoff_ticket_state=None,
            no_category_name="Unknown",
        ),
        zammad=SimpleNamespace(pending_close_after_days=None),
        localization=LocalizationSettings(),
    )
    client = SimpleNamespace(
        post_answer=AsyncMock(),
        post_shared_draft=AsyncMock(),
        set_answered_through=AsyncMock(),
        add_tag_to_ticket=AsyncMock(),
        set_ticket_state=AsyncMock(),
    )
    service.zammad_client = client  # type: ignore[assignment]
    service.logger = MagicMock()
    return service, client


def _static_triage(*, auto_publish: bool) -> TriageResult:
    return TriageResult(
        user_text="Help",
        category=Category(name="FAQ", auto_publish=auto_publish),
        action=Action(name="static", description="Static", type=ActionTypes.StaticAnswer, answer="Answer text"),
        reasoning="test",
        confidence=1.0,
    )


@pytest.mark.asyncio
async def test_published_answer_writes_the_marker_in_the_same_request() -> None:
    """The public answer carries the marker, so no separate write can be lost."""
    service, client = _action_service()
    service.get_answer = AsyncMock(return_value=StaticAnswer(response="Answer text"))

    await service.execute_action(ticket_id=42, triage=_static_triage(auto_publish=True), answered_through=9)

    assert client.post_answer.call_args.kwargs["answered_through"] == 9
    client.set_answered_through.assert_not_called()


@pytest.mark.asyncio
async def test_shared_draft_marks_the_turn_as_handled() -> None:
    """A draft for an agent still records the turn, so the next event does not draft it again."""
    service, client = _action_service()
    service.get_answer = AsyncMock(return_value=StaticAnswer(response="Answer text"))

    await service.execute_action(ticket_id=42, triage=_static_triage(auto_publish=False), answered_through=9)

    client.post_shared_draft.assert_awaited_once()
    client.set_answered_through.assert_awaited_once_with(ticket_id=42, article_id=9)


@pytest.mark.asyncio
async def test_single_pass_mode_writes_no_marker() -> None:
    """Without a turn the action behaves as before."""
    service, client = _action_service()
    service.get_answer = AsyncMock(return_value=StaticAnswer(response="Answer text"))

    await service.execute_action(ticket_id=42, triage=_static_triage(auto_publish=True))

    assert "answered_through" not in client.post_answer.call_args.kwargs
    client.set_answered_through.assert_not_called()


def _api_client(handler: Callable[[httpx.Request], httpx.Response]) -> ZammadAPIClient:
    client = ZammadAPIClient(
        ZammadAPISettings(base_url=HttpUrl("https://zammad.example.test"), auth_token=SecretStr("t"))
    )
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test", transport=httpx.MockTransport(handler), headers=client.client.headers
    )
    return client


@pytest.mark.asyncio
async def test_api_client_posts_reply_and_marker_in_one_ticket_update() -> None:
    """With a marker, the reply article and the attribute go in a single PUT /tickets/{id}."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 17,
                        "ticket_id": 42,
                        "body": "hi",
                        "internal": False,
                        "sender": "Customer",
                        "type": "whatsapp message",
                        "from": "+218900000000",
                    }
                ],
            )
        return httpx.Response(200, json={"id": 42})

    client = _api_client(handler)
    await client.post_answer(ticket_id=42, text="Hello", subject="Answer", answered_through=17)

    writes = [r for r in requests if r.method != "GET"]
    assert len(writes) == 1
    assert (writes[0].method, writes[0].url.path) == ("PUT", "/api/v1/tickets/42")
    body = json.loads(writes[0].content)
    assert body["ai_answered_through_article_id"] == 17
    assert body["article"]["body"] == "Hello" and body["article"]["type"] == "whatsapp message"
    assert "ticket_id" not in body["article"]


@pytest.mark.asyncio
async def test_api_client_marker_and_tag_removal_payloads() -> None:
    """The standalone marker write and the tag removal use Zammad's ticket and tag endpoints."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    client = _api_client(handler)
    await client.set_answered_through(ticket_id=42, article_id=17)
    await client.remove_tag_from_ticket(ticket_id=42, tag="ai_resume")

    assert (requests[0].method, requests[0].url.path) == ("PUT", "/api/v1/tickets/42")
    assert json.loads(requests[0].content) == {"ai_answered_through_article_id": 17}
    assert (requests[1].method, requests[1].url.path) == ("POST", "/api/v1/tags/remove")
    assert json.loads(requests[1].content) == {"object": "Ticket", "o_id": 42, "item": "ai_resume"}


@pytest.mark.asyncio
async def test_api_client_refuses_conversation_mode_without_the_ticket_attribute() -> None:
    """Without the attribute the marker would be dropped and every event would answer again, so loading fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ticket_articles/by_ticket/42"):
            return httpx.Response(200, json=[{"id": 1, "ticket_id": 42, "body": "hi", "sender": "Customer"}])
        return httpx.Response(200, json={"id": 42, "group_id": 3, "owner_id": 1})

    client = _api_client(handler)

    with pytest.raises(ZammadMissingTicketAttributeError):
        await client.get_ticket_with_context(42)


@pytest.mark.asyncio
async def test_missing_ticket_attribute_posts_nothing(
    services: SimpleNamespace, settings_factory: Callable, kafka_message_factory: Callable
) -> None:
    """The event is acknowledged without triage, an answer, or a hand-off reply, and without retries."""
    services.triage.zammad_client.get_ticket_with_context.side_effect = ZammadMissingTicketAttributeError("missing")

    await handle(settings_factory, kafka_message_factory)

    services.triage.perform_triage.assert_not_called()
    services.action.execute_action.assert_not_called()
    services.action.hand_off_failed_ticket.assert_not_called()
    assert services.triage.zammad_client.get_ticket_with_context.await_count <= 2
