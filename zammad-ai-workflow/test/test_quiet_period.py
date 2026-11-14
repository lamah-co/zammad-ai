"""Tests for waiting until the customer has finished writing (quiet period and cap), with a fake clock."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from faststream.exceptions import AckMessage

from app.conversation.recheck import RecheckScheduler
from app.conversation.turn import Readiness, message_gaps, seconds_after_ai_reply, turn_readiness
from app.kafka.broker import build_router
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings import ZammadAISettings
from app.settings.conversation import ConversationSettings
from app.settings.triage import Action, ActionTypes, Category

T0 = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)
AI_USER = 5


def at(seconds: float) -> datetime:
    """Return T0 plus the given seconds."""
    return T0 + timedelta(seconds=seconds)


def customer(article_id: int, sent: float) -> ZammadArticle:
    """Return a customer WhatsApp message sent `sent` seconds after T0."""
    return ZammadArticle(
        id=article_id,
        ticket_id=1,
        text=f"msg {article_id}",
        sender="Customer",
        type="whatsapp message",
        created_at=at(sent),
    )


def readiness(sent: list[float], now: float) -> Readiness:
    """Apply the default 10 s quiet period and 60 s cap."""
    return turn_readiness(
        [customer(i, t) for i, t in enumerate(sent, start=1)], now=at(now), quiet_seconds=10, max_wait_seconds=60
    )


def test_single_message_waits_for_the_quiet_period() -> None:
    """A message is answered once it is 10 s old."""
    assert readiness([0], now=1).wait_seconds == pytest.approx(9)
    assert readiness([0], now=10).ready


def test_burst_waits_for_its_newest_message() -> None:
    """Messages 3 s apart keep the turn open until 10 s after the last one."""
    assert readiness([0, 3, 6], now=7).wait_seconds == pytest.approx(9)
    assert readiness([0, 3, 6], now=16).ready


def test_continuous_messages_are_answered_by_the_cap() -> None:
    """A customer writing every 5 s is answered 60 s after the first message."""
    sent = [float(t) for t in range(0, 60, 5)]
    assert readiness(sent, now=56).wait_seconds == pytest.approx(4)
    capped = readiness([*sent, 60.0], now=60)
    assert capped.ready and capped.after_cap


def test_zero_quiet_period_and_missing_timestamps_answer_at_once() -> None:
    """Quiet period off, or a message without a timestamp, never delays the answer."""
    articles = [customer(1, 0)]
    assert turn_readiness(articles, now=at(0), quiet_seconds=0, max_wait_seconds=60).ready
    untimed = ZammadArticle(id=2, ticket_id=1, text="x", sender="Customer")
    assert turn_readiness([*articles, untimed], now=at(0), quiet_seconds=10, max_wait_seconds=60).ready


def test_cap_shorter_than_quiet_period_is_rejected() -> None:
    """The cap must leave room for the quiet period."""
    with pytest.raises(ValueError, match="max_wait_seconds"):
        ConversationSettings(quiet_seconds=30, max_wait_seconds=10)


def test_gaps_and_time_since_the_ai_reply() -> None:
    """The metrics see the gaps inside a turn and how soon it followed the AI's reply."""
    reply = ZammadArticle(
        id=2,
        ticket_id=1,
        text="answer",
        sender="Agent",
        type="whatsapp message",
        created_by_id=AI_USER,
        created_at=at(20),
    )
    turn = [customer(3, 25), customer(4, 28)]
    ticket = ZammadTicket(id=1, articles=[customer(1, 0), reply, *turn])

    assert message_gaps(turn) == [3.0]
    assert seconds_after_ai_reply(ticket, turn, ai_user_id=AI_USER) == 5.0
    assert seconds_after_ai_reply(ZammadTicket(id=1, articles=turn), turn, ai_user_id=AI_USER) is None


class ManualSleep:
    """A sleep that waits until the test releases it."""

    def __init__(self) -> None:
        """Start with no recorded sleeps."""
        self.calls: list[float] = []
        self.release = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        """Record the delay and wait for the release."""
        self.calls.append(seconds)
        await self.release.wait()


async def settle() -> None:
    """Let scheduled tasks run."""
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_new_event_replaces_the_pending_recheck() -> None:
    """Only one re-check per ticket is pending; the latest delay wins."""
    sleep = ManualSleep()
    scheduler = RecheckScheduler(sleep=sleep)
    runs: list[int] = []

    async def recheck(ticket_id: int) -> None:
        runs.append(ticket_id)

    scheduler.schedule(7, 9, recheck)
    await settle()
    scheduler.schedule(7, 10, recheck)
    await settle()
    sleep.release.set()
    await settle()

    assert runs == [7]
    assert not scheduler.pending(7)


@pytest.mark.asyncio
async def test_ticket_lock_serialises_work_on_one_ticket() -> None:
    """An event and a re-check of the same ticket never run at once; other tickets are not held up."""
    scheduler = RecheckScheduler()
    order: list[str] = []
    gate = asyncio.Event()

    async def slow(name: str, ticket_id: int) -> None:
        async with scheduler.ticket_lock(ticket_id):
            order.append(f"{name} start")
            await gate.wait()
            order.append(f"{name} end")

    first = asyncio.create_task(slow("a", 1))
    second = asyncio.create_task(slow("b", 1))
    other = asyncio.create_task(slow("c", 2))
    await settle()
    assert order == ["a start", "c start"]
    gate.set()
    await asyncio.gather(first, second, other)
    assert order.index("a end") < order.index("b start")


@pytest.fixture
def flow(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch the router's services and clock; the re-check sleep advances the fake clock."""
    clock = SimpleNamespace(now=at(0))
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += timedelta(seconds=seconds)

    triage = MagicMock()
    triage.zammad_client = MagicMock()
    triage.zammad_client.get_ticket_with_context = AsyncMock()
    triage.zammad_client.get_own_user_id = AsyncMock(return_value=AI_USER)
    triage.zammad_client.remove_tag_from_ticket = AsyncMock()
    triage.perform_triage = AsyncMock(
        return_value=TriageResult(
            user_text="",
            category=Category(name="Test"),
            action=Action(name="Keine_Aktion", description="No action", type=ActionTypes.NoAction),
            reasoning="test",
            confidence=0.95,
        )
    )
    action = MagicMock()
    action.execute_action = AsyncMock()
    action.hand_off_failed_ticket = AsyncMock(return_value=True)
    monkeypatch.setattr("app.kafka.broker.get_triage_service", lambda *args, **kwargs: triage)
    monkeypatch.setattr("app.kafka.broker.get_action_service", lambda *args, **kwargs: action)
    monkeypatch.setattr("app.kafka.broker.get_answer_service", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("app.kafka.broker.utc_now", lambda: clock.now)
    monkeypatch.setattr("app.kafka.broker.RecheckScheduler", lambda: RecheckScheduler(sleep=fake_sleep))
    return SimpleNamespace(clock=clock, sleeps=sleeps, triage=triage, action=action)


async def send_event(handler: Callable, kafka_message_factory: Callable) -> None:
    """Deliver one ticket event; the handler must acknowledge it without waiting."""
    with pytest.raises(AckMessage):
        await handler(event=Event.model_validate(kafka_message_factory()))


@pytest.mark.asyncio
async def test_burst_three_seconds_apart_gets_one_reply_after_the_quiet_period(
    flow: SimpleNamespace, settings_factory: Callable[..., ZammadAISettings], kafka_message_factory: Callable
) -> None:
    """Each event is acknowledged at once; the turn is answered once, 10 s after the last message."""
    settings = settings_factory(
        valid_request_types=["technischer Bürgersupport"], conversation=ConversationSettings(enabled=True)
    )
    _, handler = build_router(settings=settings)
    client = flow.triage.zammad_client

    for sent in (0.0, 3.0, 6.0):
        flow.clock.now = at(sent)
        count = int(sent // 3) + 1
        client.get_ticket_with_context.return_value = ZammadTicket(
            id=3720, articles=[customer(i, 3.0 * (i - 1)) for i in range(1, count + 1)]
        )
        await send_event(handler, kafka_message_factory)
        flow.triage.perform_triage.assert_not_called()

    await settle()

    assert flow.triage.perform_triage.await_count == 1
    assert [a.id for a in flow.triage.perform_triage.call_args.kwargs["turn"]] == [1, 2, 3]
    assert flow.action.execute_action.call_args.kwargs["answered_through"] == 3
    assert flow.clock.now >= at(16)


@pytest.mark.asyncio
async def test_single_message_is_answered_about_ten_seconds_later(
    flow: SimpleNamespace, settings_factory: Callable[..., ZammadAISettings], kafka_message_factory: Callable
) -> None:
    """The first event schedules a re-check for when the message is 10 s old, which then answers it."""
    settings = settings_factory(
        valid_request_types=["technischer Bürgersupport"], conversation=ConversationSettings(enabled=True)
    )
    _, handler = build_router(settings=settings)
    flow.triage.zammad_client.get_ticket_with_context.return_value = ZammadTicket(id=3720, articles=[customer(1, 0)])
    flow.clock.now = at(0.2)

    await send_event(handler, kafka_message_factory)
    await settle()

    assert flow.sleeps == [pytest.approx(10.3)]
    assert flow.triage.perform_triage.await_count == 1
