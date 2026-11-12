"""Tests for the Kafka retry budget, backoff and failure hand-off."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from faststream.exceptions import AckMessage, NackMessage

from app.action.service import ActionService
from app.errors import (
    AckDecision,
    AnswerServiceError,
    GenAIBillingError,
    GenAIQuotaError,
    TriageError,
    classify_exception,
    classify_provider_error,
)
from app.kafka.broker import build_router
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings import ZammadAISettings
from app.settings.kafka import KafkaRetrySettings, KafkaSettings
from app.settings.localization import LocalizationSettings
from app.settings.triage import Action, ActionTypes, Category
from app.zammad.base import TicketNotFoundError, ZammadConnectionError

DEPLETED_CREDITS = (
    "402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits are depleted.'}}"
)


@pytest.fixture
def event() -> Event:
    """Return a valid ticket event for ticket 3720."""
    return Event.model_validate(
        {
            "action": "created",
            "ticket": "3720",
            "status": "new",
            "statusId": "1",
            "anliegenart": "support",
            "lhmExtId": "",
        }
    )


@pytest.fixture
def triage() -> MagicMock:
    """Return a triage service whose ticket holds one English customer message."""
    service = MagicMock()
    service.zammad_client = MagicMock()
    service.zammad_client.get_ticket = AsyncMock(
        return_value=ZammadTicket(
            id=3720,
            articles=[ZammadArticle(id=1, ticket_id=3720, text="How do I get an API key?", sender="Customer")],
        )
    )
    service.perform_triage = AsyncMock(
        return_value=TriageResult(
            user_text="How do I get an API key?",
            category=Category(name="General"),
            action=Action(name="No Action", description="No action", type=ActionTypes.NoAction),
            reasoning="ok",
            confidence=0.9,
        )
    )
    return service


@pytest.fixture
def action() -> MagicMock:
    """Return an action service with async execution and hand-off."""
    service = MagicMock()
    service.execute_action = AsyncMock()
    service.hand_off_failed_ticket = AsyncMock(return_value=True)
    return service


@pytest.fixture
def sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Record backoff waits instead of sleeping."""
    mock = AsyncMock()
    monkeypatch.setattr("app.kafka.broker.asyncio.sleep", mock)
    return mock


@pytest.fixture
def handler(
    monkeypatch: pytest.MonkeyPatch,
    settings_factory: Callable[..., ZammadAISettings],
    triage: MagicMock,
    action: MagicMock,
) -> Callable[..., object]:
    """Build the event handler with three attempts and fake services."""
    monkeypatch.setattr("app.kafka.broker.get_triage_service", lambda *args, **kwargs: triage)
    monkeypatch.setattr("app.kafka.broker.get_answer_service", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.kafka.broker.get_action_service", lambda *args, **kwargs: action)
    settings = settings_factory(valid_request_types=["support"])
    settings.kafka.retry = KafkaRetrySettings(max_attempts=3, initial_backoff_seconds=2, max_backoff_seconds=60)
    _router, event_handler = build_router(settings=settings)
    return event_handler


@pytest.mark.asyncio
async def test_temporary_error_recovers_within_the_budget(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """A temporary error that clears on the next attempt is processed normally."""
    triage.perform_triage.side_effect = [TriageError("flaky", retryable=True), triage.perform_triage.return_value]

    with pytest.raises(AckMessage):
        await handler(event=event)

    assert triage.perform_triage.await_count == 2
    action.execute_action.assert_awaited_once()
    action.hand_off_failed_ticket.assert_not_awaited()
    assert [call.args[0] for call in sleep.await_args_list] == [2]


@pytest.mark.asyncio
async def test_exhausted_budget_hands_off_and_acknowledges(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """The event is tried max_attempts times with doubling waits, then handed off instead of looping."""
    triage.perform_triage.side_effect = TriageError("still failing", retryable=True)

    with pytest.raises(AckMessage):
        await handler(event=event)

    assert triage.perform_triage.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [2, 4]
    action.hand_off_failed_ticket.assert_awaited_once_with(
        ticket_id=3720,
        reason="TriageError after 3 attempt(s)",
        customer_message="How do I get an API key?",
    )


@pytest.mark.asyncio
async def test_billing_error_hands_off_on_the_first_failure(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """Depleted credits cannot be fixed by retrying, so the ticket goes to a person at once."""
    triage.perform_triage.side_effect = classify_provider_error(Exception(DEPLETED_CREDITS))

    with pytest.raises(AckMessage):
        await handler(event=event)

    assert triage.perform_triage.await_count == 1
    sleep.assert_not_awaited()
    action.hand_off_failed_ticket.assert_awaited_once()


@pytest.mark.asyncio
async def test_billing_error_wrapped_in_a_retryable_error_still_hands_off(
    handler, event: Event, triage: MagicMock, action: MagicMock
) -> None:
    """A billing error raised underneath a generic retryable error is found through the cause chain."""

    async def fail(**_kwargs: object) -> None:
        try:
            raise GenAIBillingError("no credits")
        except GenAIBillingError as e:
            raise AnswerServiceError("Answer generation failed", retryable=True) from e

    triage.perform_triage.side_effect = fail

    with pytest.raises(AckMessage):
        await handler(event=event)

    assert triage.perform_triage.await_count == 1
    action.hand_off_failed_ticket.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_is_kept_when_zammad_cannot_take_the_hand_off(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """If the hand-off cannot be written, the event is not acknowledged, and the wait is the maximum backoff."""
    triage.perform_triage.side_effect = TriageError("still failing", retryable=True)
    action.hand_off_failed_ticket.side_effect = ZammadConnectionError("zammad down")

    with pytest.raises(NackMessage):
        await handler(event=event)

    assert sleep.await_args_list[-1].args[0] == 60


@pytest.mark.asyncio
async def test_permanent_error_on_a_real_ticket_is_handed_off(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """A permanent error never leaves the customer silent: it is handed off at once, without retries."""
    triage.perform_triage.side_effect = TriageError("invalid", retryable=False)

    with pytest.raises(AckMessage):
        await handler(event=event)

    assert triage.perform_triage.await_count == 1
    sleep.assert_not_awaited()
    action.hand_off_failed_ticket.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_without_a_usable_ticket_is_dropped(
    handler, event: Event, triage: MagicMock, action: MagicMock
) -> None:
    """A ticket that no longer exists cannot be handed off, so the event is dropped."""
    triage.zammad_client.get_ticket.side_effect = TicketNotFoundError("gone")

    with pytest.raises(AckMessage):
        await handler(event=event)

    action.hand_off_failed_ticket.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_hand_off_error_does_not_block_the_queue(
    handler, event: Event, triage: MagicMock, action: MagicMock, sleep: AsyncMock
) -> None:
    """A bug in the hand-off is acknowledged and counted instead of redelivering the event forever."""
    triage.perform_triage.side_effect = TriageError("invalid", retryable=False)
    action.hand_off_failed_ticket.side_effect = AttributeError("bug in hand-off")

    with pytest.raises(AckMessage):
        await handler(event=event)

    sleep.assert_not_awaited()


def test_backoff_doubles_and_is_capped() -> None:
    """Waits grow 2, 4, 8 ... and never exceed max_backoff_seconds."""
    retry = KafkaRetrySettings(initial_backoff_seconds=2, max_backoff_seconds=10)

    assert [retry.backoff_seconds(attempt) for attempt in range(1, 6)] == [2, 4, 8, 10, 10]


def test_retry_waits_must_fit_well_inside_the_poll_interval() -> None:
    """A budget whose waits exceed half the poll interval is rejected, so an event is never processed twice."""
    KafkaSettings()  # defaults: 30 s of waits plus 60 s, well inside 900 s

    with pytest.raises(ValueError, match="max_poll_interval_seconds"):
        KafkaSettings(
            retry=KafkaRetrySettings(max_attempts=10, initial_backoff_seconds=30, max_backoff_seconds=120),
            max_poll_interval_seconds=600,
        )


def test_depleted_credits_are_a_billing_error_not_a_model_error() -> None:
    """The production 402 message must map to the billing class, and quota errors keep their class."""
    assert isinstance(classify_provider_error(Exception(DEPLETED_CREDITS)), GenAIBillingError)
    status_error = Exception("payment required")
    status_error.code = 402  # type: ignore
    assert isinstance(classify_provider_error(status_error), GenAIBillingError)
    assert not isinstance(
        classify_provider_error(Exception("timeout for request id 402 at /billing")), GenAIBillingError
    )
    assert isinstance(classify_provider_error(Exception("429 quota exceeded")), GenAIQuotaError)
    decision = classify_exception(GenAIBillingError("no credits"), category_wrong_retry_confidence_threshold=0.5)
    assert decision.decision == AckDecision.HANDOFF


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "language"), [("How do I get an API key?", "en"), ("كيف أحصل على مفتاح؟", "ar")])
async def test_failure_hand_off_posts_note_tag_and_localized_acknowledgement(text: str, language: str) -> None:
    """The failure hand-off reuses the human-review path with the failure note and tag."""
    service = ActionService.__new__(ActionService)
    service.logger = MagicMock()
    service.zammad_client = MagicMock(
        post_answer=AsyncMock(),
        add_tag_to_ticket=AsyncMock(),
        set_ticket_state=AsyncMock(),
        get_ticket_tags=AsyncMock(return_value=[]),
    )
    service.settings = SimpleNamespace(
        localization=LocalizationSettings(),
        triage=SimpleNamespace(
            human_review_public_fallback={"ar": "تم استلام طلبك", "en": "We received your request"},
            no_answer_public_fallback=None,
            human_review_internal_note=None,
            no_answer_internal_note=None,
            no_action_internal_note=None,
            human_review_tag="ai_human_review",
            no_answer_tag=None,
            failed_internal_note="Failed: {reason}\n{customer_message}",
            failed_category_name="Processing failed",
            failed_tag="ai_failed",
            handoff_ticket_state="open",
            public_reply_subject={"ar": "الرد على طلبك", "en": "Answer"},
        ),
    )

    await service.hand_off_failed_ticket(ticket_id=7, reason="TriageError after 3 attempt(s)", customer_message=text)

    note, acknowledgement = (call.kwargs for call in service.zammad_client.post_answer.await_args_list)
    assert note["internal"] is True and "TriageError after 3 attempt(s)" in note["text"]
    service.zammad_client.add_tag_to_ticket.assert_awaited_once_with(ticket_id=7, tag="ai_failed")
    service.zammad_client.set_ticket_state.assert_awaited_once_with(ticket_id=7, state="open")
    expected = "We received your request" if language == "en" else "تم استلام طلبك"
    assert acknowledgement["internal"] is False and acknowledgement["text"] == expected


@pytest.mark.asyncio
async def test_failure_hand_off_is_not_repeated_on_an_already_failed_ticket() -> None:
    """If the failure tag is already there, a redelivered hand-off posts nothing."""
    service = ActionService.__new__(ActionService)
    service.logger = MagicMock()
    service.zammad_client = MagicMock(post_answer=AsyncMock(), get_ticket_tags=AsyncMock(return_value=["ai_failed"]))
    service.settings = SimpleNamespace(triage=SimpleNamespace(failed_tag="ai_failed"))

    handed_off = await service.hand_off_failed_ticket(ticket_id=7, reason="r", customer_message="hi")

    assert handed_off is False
    service.zammad_client.post_answer.assert_not_awaited()
