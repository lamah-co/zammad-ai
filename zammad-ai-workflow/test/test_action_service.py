"""Tests for customer-delivery fallbacks in the action service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.action.service import ActionService
from app.errors import UnsupportedReplyChannelError
from app.models.answer import StaticAnswer
from app.models.triage import TriageResult
from app.settings.triage import Action, ActionTypes, Category


@pytest.mark.asyncio
async def test_unsupported_reply_channel_becomes_shared_draft() -> None:
    """Unsupported customer channels must fall back to a shared draft."""
    service = object.__new__(ActionService)
    service.settings = SimpleNamespace(zammad=SimpleNamespace(pending_close_after_days=None))
    service.get_answer = AsyncMock(return_value=StaticAnswer(response="A reviewed answer"))
    service.zammad_client = SimpleNamespace(
        post_answer=AsyncMock(side_effect=UnsupportedReplyChannelError("unsupported web channel")),
        post_shared_draft=AsyncMock(),
    )
    triage = TriageResult(
        user_text="Help",
        category=Category(name="FAQ", auto_publish=True),
        action=Action(
            name="static",
            description="Static",
            type=ActionTypes.StaticAnswer,
            answer="A reviewed answer",
        ),
        reasoning="test",
        confidence=1.0,
    )

    await service.execute_action(ticket_id=42, triage=triage)

    service.zammad_client.post_shared_draft.assert_awaited_once_with(ticket_id=42, text="A reviewed answer")
