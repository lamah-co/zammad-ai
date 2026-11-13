"""Kafka router setup for Zammad AI ticket events."""

import asyncio
from collections.abc import Callable
from logging import Logger

from faststream import AckPolicy
from faststream.exceptions import AckMessage, NackMessage
from faststream.kafka.fastapi import KafkaRouter
from faststream.kafka.prometheus import KafkaPrometheusMiddleware
from faststream.security import BaseSecurity
from prometheus_client import REGISTRY, Counter
from pydantic import ValidationError

from app.action.service import ActionService, get_action_service
from app.answer.service import AnswerService, get_answer_service
from app.errors import AckDecision, ExceptionDecision, KafkaPayloadError, classify_exception
from app.models.kafka import Event
from app.models.triage import TriageResult
from app.models.zammad import ZammadTicket
from app.settings import ZammadAISettings
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

    async def process_ticket(ticket_id: int) -> None:
        """Triage one ticket and execute its action. Errors propagate to the retry loop."""
        # Get ticket details from Zammad
        try:
            zammad_client: BaseZammadClient = triage_service.zammad_client
            ticket: ZammadTicket = await zammad_client.get_ticket(id=ticket_id)
        except TicketNotFoundError as e:
            logger.info(f"Ticket {ticket_id} does not exist anymore.", exc_info=True)
            raise KafkaPayloadError("Ticket was not found", retryable=False) from e
        except (ZammadRetryableError, ZammadConnectionError) as e:
            logger.error("Error connecting to Zammad", exc_info=True)
            raise KafkaPayloadError("Failed due to Zammad connection error", retryable=True) from e

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
            result: TriageResult = await triage_service.perform_triage(ticket=ticket)
            logger.debug(
                f"Triage result for ticket {ticket_id}: category: {result.category.name}, action: {result.action.name}"
            )
            await action_service.execute_action(ticket_id=ticket_id, triage=result)
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
            ticket: ZammadTicket = await triage_service.zammad_client.get_ticket(id=ticket_id)
            article = ticket.latest_customer_article()
            handed_off: bool = await action_service.hand_off_failed_ticket(
                ticket_id=ticket_id,
                reason=f"{decision.error_class} after {attempts} attempt(s)",
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

    return router, event_handler
