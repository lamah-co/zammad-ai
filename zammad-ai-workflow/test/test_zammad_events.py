"""Tests for standard Zammad webhook event ingress."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.v1.events import process_zammad_event
from app.models.api_v1 import ZammadTicketEventInput
from app.models.zammad import ZammadArticle, ZammadTicket


@pytest.mark.asyncio
async def test_customer_article_event_runs_triage_and_action() -> None:
    ticket = ZammadTicket(
        id=42,
        articles=[
            ZammadArticle(
                id=7,
                ticket_id=42,
                text="Help",
                sender="Customer",
                type="email",
            )
        ],
    )
    triage_result = SimpleNamespace()
    triage_service = SimpleNamespace(
        zammad_client=SimpleNamespace(get_ticket=AsyncMock(return_value=ticket)),
        perform_triage=AsyncMock(return_value=triage_result),
    )
    action_service = SimpleNamespace(execute_action=AsyncMock())

    result = await process_zammad_event(
        input=ZammadTicketEventInput(ticket_id=42, article_id=7),
        triage_service=triage_service,
        action_service=action_service,
        credentials=None,
    )

    assert result.status == "processed"
    triage_service.perform_triage.assert_awaited_once_with(ticket=ticket)
    action_service.execute_action.assert_awaited_once_with(ticket_id=42, triage=triage_result)


@pytest.mark.asyncio
async def test_agent_article_event_is_ignored() -> None:
    ticket = ZammadTicket(
        id=42,
        articles=[
            ZammadArticle(
                id=8,
                ticket_id=42,
                text="Answer",
                sender="Agent",
                type="email",
            )
        ],
    )
    triage_service = SimpleNamespace(
        zammad_client=SimpleNamespace(get_ticket=AsyncMock(return_value=ticket)),
        perform_triage=AsyncMock(),
    )
    action_service = SimpleNamespace(execute_action=AsyncMock())

    result = await process_zammad_event(
        input=ZammadTicketEventInput(ticket_id=42, article_id=8),
        triage_service=triage_service,
        action_service=action_service,
        credentials=None,
    )

    assert result.status == "ignored"
    triage_service.perform_triage.assert_not_awaited()
    action_service.execute_action.assert_not_awaited()
