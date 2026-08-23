"""Tests for customer-delivery fallbacks in the action service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.action.service import ActionService
from app.errors import UnsupportedReplyChannelError
from app.models.answer import NoAnswerPossible, StaticAnswer
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


@pytest.mark.asyncio
async def test_no_answer_possible_creates_agent_handoff_and_customer_acknowledgement() -> None:
    """No-answer outcomes should be visible to agents and acknowledged to customers."""
    service = object.__new__(ActionService)
    service.settings = SimpleNamespace(
        triage=SimpleNamespace(
            no_category_name="Human review required",
            no_action_internal_note=None,
            no_answer_public_fallback="تم استلام سؤالك، سنقوم بالرد عليك في أقرب وقت ممكن.",
            no_answer_internal_note="Category: {category}\nMessage: {customer_message}\nReason: {reason}",
            no_answer_tag="ai_no_answer",
            human_review_public_fallback=None,
            human_review_internal_note=None,
            human_review_tag=None,
            handoff_ticket_state="open",
        )
    )
    service.get_answer = AsyncMock(
        return_value=NoAnswerPossible(
            reasoning=(
                "The knowledge base did not contain enough grounded details for this answer, so the workflow must "
                "hand the ticket to a human instead of inventing unsupported information."
            )
        )
    )
    service.zammad_client = SimpleNamespace(
        post_answer=AsyncMock(),
        post_shared_draft=AsyncMock(),
        add_tag_to_ticket=AsyncMock(),
        set_ticket_state=AsyncMock(),
    )
    action = Action(name="ai_answer", description="AI answer", type=ActionTypes.AIAnswer)
    triage = TriageResult(
        user_text="شن نوع Authorization المستخدم؟",
        category=Category(name="FAQ answerable", auto_publish=True),
        action=action,
        reasoning="FAQ-like request.",
        confidence=0.9,
    )

    await service.execute_action(ticket_id=42, triage=triage)

    service.zammad_client.add_tag_to_ticket.assert_awaited_once_with(ticket_id=42, tag="ai_no_answer")
    service.zammad_client.set_ticket_state.assert_awaited_once_with(ticket_id=42, state="open")
    service.zammad_client.post_answer.assert_any_await(
        ticket_id=42,
        text="تم استلام سؤالك، سنقوم بالرد عليك في أقرب وقت ممكن.",
        subject="Answer",
        internal=False,
    )
    assert service.zammad_client.post_answer.await_args_list[0].kwargs["internal"] is True
    assert service.zammad_client.post_answer.await_args_list[0].kwargs["subject"] == "Handoff"


@pytest.mark.asyncio
async def test_human_review_no_action_creates_handoff_acknowledgement() -> None:
    """Human-review outcomes should not be silent."""
    human_review_message = "تم استلام استفسارك ، ويحتاج إلى مراجعة من الفريق المختص . سنقوم بالرد عليك في أقرب وقت ممكن"
    service = object.__new__(ActionService)
    service.settings = SimpleNamespace(
        triage=SimpleNamespace(
            no_category_name="Human review required",
            no_action_internal_note=None,
            no_answer_public_fallback=None,
            no_answer_internal_note=None,
            no_answer_tag=None,
            human_review_public_fallback=human_review_message,
            human_review_internal_note="Human review required for: {customer_message}",
            human_review_tag="ai_human_review",
            handoff_ticket_state="open",
        )
    )
    service.get_answer = AsyncMock(
        return_value=NoAnswerPossible(
            reasoning=(
                "No answer was generated because the selected action is configured as NoAction and the customer "
                "request needs account-specific human review before a final support response can be provided."
            )
        )
    )
    service.zammad_client = SimpleNamespace(
        post_answer=AsyncMock(),
        post_shared_draft=AsyncMock(),
        add_tag_to_ticket=AsyncMock(),
        set_ticket_state=AsyncMock(),
    )
    action = Action(name="no_action", description="Human review", type=ActionTypes.NoAction)
    triage = TriageResult(
        user_text="عندي رقم مرسل كيف يتم تفعيله؟",
        category=Category(name="Human review required", auto_publish=False),
        action=action,
        reasoning="Account-specific activation request.",
        confidence=0.9,
    )

    await service.execute_action(ticket_id=43, triage=triage)

    service.zammad_client.add_tag_to_ticket.assert_awaited_once_with(ticket_id=43, tag="ai_human_review")
    service.zammad_client.set_ticket_state.assert_awaited_once_with(ticket_id=43, state="open")
    service.zammad_client.post_answer.assert_any_await(
        ticket_id=43,
        text=human_review_message,
        subject="Answer",
        internal=False,
    )
