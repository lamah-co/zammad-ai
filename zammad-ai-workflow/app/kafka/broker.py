"""Kafka router setup for Zammad AI ticket events."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from logging import Logger

from faststream import AckPolicy
from faststream.exceptions import AckMessage, NackMessage
from faststream.kafka.fastapi import KafkaRouter
from faststream.kafka.prometheus import KafkaPrometheusMiddleware
from faststream.security import BaseSecurity
from prometheus_client import REGISTRY, Counter, Histogram
from pydantic import ValidationError

from app.action.service import ActionService, get_action_service
from app.answer.service import AnswerService, get_answer_service
from app.conversation.recheck import RecheckScheduler
from app.conversation.turn import (
    TurnDecision,
    decide_turn,
    message_gaps,
    seconds_after_ai_reply,
    turn_readiness,
    turn_text,
    unanswered_turn,
)
from app.errors import AckDecision, ExceptionDecision, KafkaPayloadError, classify_exception
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadTicket
from app.settings import ZammadAISettings
from app.settings.conversation import ConversationSettings
from app.settings.kafka import KafkaRetrySettings
from app.triage.triage import TriageService, get_triage_service
from app.utils.logging import getLogger
from app.utils.status import track_activity
from app.zammad.base import BaseZammadClient, TicketNotFoundError, ZammadConnectionError, ZammadRetryableError

from .security import setup_security

logger: Logger = getLogger(name="zammad-ai.kafka.broker")

EVENT_RETRIES_TOTAL = Counter(
    name="zammad_ai_event_retries_total",
    documentation="Ticket event retries after a retryable error, by error class.",
    labelnames=("reason",),
)
EVENT_HANDOFFS_TOTAL = Counter(
    name="zammad_ai_event_handoffs_total",
    documentation="Ticket events handed to a human after the retry budget ran out or a person was required.",
    labelnames=("reason",),
)
TURNS_SKIPPED_TOTAL = Counter(
    name="zammad_ai_turns_skipped_total",
    documentation="Ticket events that needed no answer: nothing new since the last answer, or a person owns the conversation.",
    labelnames=("reason",),
)
TURNS_DEFERRED_TOTAL = Counter(
    name="zammad_ai_turns_deferred_total",
    documentation="Ticket events whose turn was not quiet yet, so a re-check was scheduled.",
)
TURNS_ANSWERED_AFTER_CAP_TOTAL = Counter(
    name="zammad_ai_turns_answered_after_cap_total",
    documentation="Turns answered because the maximum wait ran out while the customer was still writing.",
)
TURNS_SOON_AFTER_REPLY_TOTAL = Counter(
    name="zammad_ai_turns_soon_after_reply_total",
    documentation="Turns that started within the maximum wait after an AI reply; many suggest the quiet period is too short.",
)
TURN_MESSAGE_GAP_SECONDS = Histogram(
    name="zammad_ai_turn_message_gap_seconds",
    documentation="Seconds between consecutive customer messages within an answered turn.",
    buckets=(1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120),
)


# Re-check slightly after the quiet period ends, so rounding never makes the turn look one moment too young.
RECHECK_MARGIN_SECONDS = 0.5


def utc_now() -> datetime:
    """Return the current time; tests replace it with a fake clock."""
    return datetime.now(UTC)


EVENT_HANDOFF_FAILURES_TOTAL = Counter(
    name="zammad_ai_event_handoff_failures_total",
    documentation="Hand-offs that failed for a reason other than Zammad being unreachable; the event was acknowledged.",
    labelnames=("reason",),
)


def build_router(settings: ZammadAISettings) -> tuple[KafkaRouter, Callable]:
    """Create and configure a KafkaRouter and its subscriber event handler for ticket triage.

    Parameters:
        settings (ZammadAISettings): Application settings containing Kafka configuration and the set of valid request types.

    Returns:
        tuple[KafkaRouter, Callable]: The configured KafkaRouter and its subscriber event handler.
    """
    logger.info("Building Kafka router")

    # Security setup
    security: BaseSecurity = setup_security(kafka_settings=settings.kafka)

    # Kafka Router
    router = KafkaRouter(
        bootstrap_servers=settings.kafka.broker_url,
        client_id=settings.kafka.client_id,
        security=security,
        middlewares=(
            KafkaPrometheusMiddleware(
                registry=REGISTRY,
                app_name="zammad-ai",
                metrics_prefix="zammad_ai_kafka",
            ),
        ),
    )

    triage_service: TriageService = get_triage_service(settings=settings)
    answer_service: AnswerService = get_answer_service(settings=settings)
    action_service: ActionService = get_action_service(settings=settings, answer_service=answer_service)

    retry: KafkaRetrySettings = settings.kafka.retry
    conversation: ConversationSettings = settings.conversation
    rechecks: RecheckScheduler = RecheckScheduler()
    handoff_tags: tuple[str, ...] = tuple(
        tag
        for tag in (settings.triage.human_review_tag, settings.triage.no_answer_tag, settings.triage.failed_tag)
        if tag
    )

    async def process_ticket(ticket_id: int) -> None:
        """Triage one ticket and execute its action. Errors propagate to the retry loop."""
        # Get ticket details from Zammad
        try:
            zammad_client: BaseZammadClient = triage_service.zammad_client
            ticket: ZammadTicket = (
                await zammad_client.get_ticket_with_context(id=ticket_id)
                if conversation.enabled
                else await zammad_client.get_ticket(id=ticket_id)
            )
        except TicketNotFoundError as e:
            logger.info(f"Ticket {ticket_id} does not exist anymore.", exc_info=True)
            raise KafkaPayloadError("Ticket was not found", retryable=False) from e
        except (ZammadRetryableError, ZammadConnectionError) as e:
            logger.error("Error connecting to Zammad", exc_info=True)
            raise KafkaPayloadError("Failed due to Zammad connection error", retryable=True) from e

        # Answer only the unanswered turn, and only while no person owns the conversation.
        turn: TurnDecision | None = None
        if conversation.enabled:
            turn = decide_turn(
                ticket,
                ai_user_id=await zammad_client.get_own_user_id(),
                handoff_tags=handoff_tags,
                resume_tag=conversation.resume_tag,
            )
            if not turn.articles:
                TURNS_SKIPPED_TOTAL.labels(reason="nothing_new").inc()
                logger.info(f"Ticket {ticket_id} has no unanswered customer message; skipping the event.")
                return
            if turn.blocked_by is not None:
                TURNS_SKIPPED_TOTAL.labels(reason="person_involved").inc()
                logger.info(f"Ticket {ticket_id} is with a person ({turn.blocked_by}); not answering.")
                return
            # Wait for the customer to finish writing, without blocking the partition: re-check later instead.
            readiness = turn_readiness(
                turn.articles,
                now=utc_now(),
                quiet_seconds=conversation.quiet_seconds,
                max_wait_seconds=conversation.max_wait_seconds,
            )
            if not readiness.ready:
                TURNS_DEFERRED_TOTAL.inc()
                logger.info(
                    f"Ticket {ticket_id}: waiting {readiness.wait_seconds:.1f}s for the customer to finish the turn."
                )
                rechecks.schedule(ticket_id, readiness.wait_seconds + RECHECK_MARGIN_SECONDS, recheck)
                return
            if readiness.after_cap:
                TURNS_ANSWERED_AFTER_CAP_TOTAL.inc()
            for gap in message_gaps(turn.articles):
                TURN_MESSAGE_GAP_SECONDS.observe(gap)
            since_reply = seconds_after_ai_reply(
                ticket, turn.articles, ai_user_id=await zammad_client.get_own_user_id()
            )
            if since_reply is not None and since_reply < conversation.max_wait_seconds:
                TURNS_SOON_AFTER_REPLY_TOTAL.inc()

        # Store the original group ID
        original_group_id: int | None = ticket.group_id

        # Move ticket to AI group if configured and not already in that group
        if (
            settings.zammad.type == "eai"
            and settings.zammad.ai_ticket_group_id is not None
            and original_group_id is not None
            and original_group_id != settings.zammad.ai_ticket_group_id
        ):
            try:
                await zammad_client.update_ticket_group(
                    ticket_id=ticket_id, group_id=settings.zammad.ai_ticket_group_id
                )
                logger.info(f"Moved ticket {ticket_id} to AI group {settings.zammad.ai_ticket_group_id}")
            except (ZammadRetryableError, ZammadConnectionError) as e:
                logger.error("Error connecting to Zammad while moving ticket to AI group", exc_info=True)
                raise KafkaPayloadError(
                    "Failed due to Zammad connection error while moving ticket to AI group",
                    retryable=True,
                ) from e
        else:
            original_group_id = None  # No need to move back cause it stays in the same group

        # Perform triage and execute corresponding actions
        try:
            if turn is None:
                result: TriageResult = await triage_service.perform_triage(ticket=ticket)
                await action_service.execute_action(ticket_id=ticket_id, triage=result)
            else:
                result = await triage_service.perform_triage(ticket=ticket, turn=turn.articles)
                await action_service.execute_action(
                    ticket_id=ticket_id, triage=result, answered_through=turn.newest_article_id
                )
                if conversation.resume_tag in ticket.tags:
                    await zammad_client.remove_tag_from_ticket(ticket_id=ticket_id, tag=conversation.resume_tag)
            logger.debug(
                f"Triage result for ticket {ticket_id}: category: {result.category.name}, action: {result.action.name}"
            )
        # Move ticket back to original group if it was moved to AI group
        finally:
            if original_group_id is not None:
                try:
                    await zammad_client.update_ticket_group(ticket_id=ticket_id, group_id=original_group_id)
                    logger.info(f"Moved ticket {ticket_id} back to original group {original_group_id}")
                except (ZammadRetryableError, ZammadConnectionError) as e:
                    logger.error("Error connecting to Zammad while moving ticket back to original group", exc_info=True)
                    raise KafkaPayloadError(
                        "Failed due to Zammad connection error while moving ticket back to original group",
                        retryable=True,
                    ) from e

    async def hand_off(ticket_id: int, decision: ExceptionDecision, attempts: int) -> None:
        """Hand the ticket to a human. Keep the event only while Zammad is unreachable."""
        try:
            reason = f"{decision.error_class} after {attempts} attempt(s)"
            if conversation.enabled:
                ticket: ZammadTicket = await triage_service.zammad_client.get_ticket_with_context(id=ticket_id)
                articles = unanswered_turn(ticket)
                handed_off: bool = await action_service.hand_off_failed_ticket(
                    ticket_id=ticket_id,
                    reason=reason,
                    customer_message=turn_text(articles) if articles else "",
                    answered_through=articles[-1].id if articles else None,
                )
            else:
                ticket = await triage_service.zammad_client.get_ticket(id=ticket_id)
                article = ticket.latest_customer_article()
                handed_off = await action_service.hand_off_failed_ticket(
                    ticket_id=ticket_id,
                    reason=reason,
                    customer_message=article.text if article else "",
                )
            if handed_off:
                logger.warning(f"Handed off ticket {ticket_id} after {attempts} attempt(s): {decision.error_class}")
        except TicketNotFoundError as e:
            logger.info(f"Ticket {ticket_id} disappeared before the hand-off.", exc_info=True)
            raise AckMessage() from e
        except (ZammadRetryableError, ZammadConnectionError) as e:
            logger.error(f"Zammad is unreachable; keeping the event for ticket {ticket_id}.", exc_info=True)
            await asyncio.sleep(retry.max_backoff_seconds)
            raise NackMessage() from e
        except Exception as e:
            # Never block the queue on a hand-off bug: count it, alert on it, and move on.
            EVENT_HANDOFF_FAILURES_TOTAL.labels(reason=type(e).__name__).inc()
            logger.error(f"Hand-off failed for ticket {ticket_id}; acknowledging the event.", exc_info=True)
            raise AckMessage() from e

    async def run_ticket(ticket_id: int) -> None:
        """Process the ticket with retries and a final hand-off. Always ends in AckMessage or NackMessage."""
        async with rechecks.ticket_lock(ticket_id):
            attempt = 0
            while True:
                attempt += 1
                try:
                    await process_ticket(ticket_id)
                except AckMessage, NackMessage:
                    raise
                except Exception as e:
                    decision: ExceptionDecision = classify_exception(
                        e,
                        category_wrong_retry_confidence_threshold=settings.triage.category_wrong_retry_confidence_threshold,
                    )
                    # Only an event without a usable ticket is dropped; every other failure reaches a person.
                    drop: bool = decision.decision == AckDecision.ACK_DROP and isinstance(e, KafkaPayloadError)
                    retry_next: bool = decision.decision == AckDecision.NACK_RETRY and attempt < retry.max_attempts
                    next_step = (
                        f"retry in {retry.backoff_seconds(attempt):.0f}s"
                        if retry_next
                        else ("drop" if drop else "hand off")
                    )
                    logger.error(
                        f"Kafka event decision={decision.decision} reason={decision.reason} error_class={decision.error_class} ticket={ticket_id} attempt={attempt}/{retry.max_attempts} next={next_step}",
                        exc_info=True,
                    )
                    if drop:
                        raise AckMessage() from e
                    if retry_next:
                        EVENT_RETRIES_TOTAL.labels(reason=decision.error_class).inc()
                        await asyncio.sleep(retry.backoff_seconds(attempt))
                        continue
                    EVENT_HANDOFFS_TOTAL.labels(reason=decision.error_class).inc()
                    await hand_off(ticket_id, decision, attempt)
                    raise AckMessage() from e
                raise AckMessage()

    async def recheck(ticket_id: int) -> None:
        """Process a ticket whose turn was not quiet yet, outside any Kafka event."""
        async with track_activity():
            logger.info(f"Re-checking ticket {ticket_id} after its quiet period.")
            try:
                await run_ticket(ticket_id)
            except AckMessage:
                pass
            except NackMessage:
                # Zammad was unreachable; there is no event to redeliver, so try again later.
                rechecks.schedule(ticket_id, retry.max_backoff_seconds, recheck)

    @router.subscriber(
        settings.kafka.topic,
        group_id=settings.kafka.group_id,
        ack_policy=AckPolicy.NACK_ON_ERROR,
        max_poll_interval_ms=settings.kafka.max_poll_interval_seconds * 1000,
    )
    async def event_handler(
        event: dict[str, object],
    ) -> None:
        """Process a Kafka event by performing ticket triage and acknowledging or negatively acknowledging the message.

        Raises:
            AckMessage: If the event is successfully processed or intentionally skipped due to unsupported request type.
            NackMessage: If processing fails.
        """
        async with track_activity():
            logger.debug(f"Received event: {event}")

            # Parse and validate the incoming event
            try:
                parsed_event: Event = Event.model_validate(event)
            except ValidationError as e:
                logger.error("Failed to parse Kafka event payload", exc_info=True)
                raise AckMessage() from e

            # Skip events with unsupported request types
            if parsed_event.request_type not in settings.kafka.event_processing.valid_request_types:
                logger.info(f"Skipping event with request type: {parsed_event.request_type}")
                raise AckMessage()

            # Skip events with wrong action types
            if parsed_event.action not in settings.kafka.event_processing.valid_action_types:
                logger.info(f"Skipping event with action type: {parsed_event.action}")
                raise AckMessage()

            # Extract ticket ID
            try:
                ticket_id: int = int(parsed_event.ticket)
            except (TypeError, ValueError) as e:
                logger.error(f"Kafka event has an invalid ticket id: {parsed_event.ticket!r}", exc_info=True)
                raise AckMessage() from e

            await run_ticket(ticket_id)

    return router, event_handler
