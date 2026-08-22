"""Tests for moderation-driven action execution outcomes."""

import pytest

from app.action.service import ActionService
from app.guardrails import GuardrailService
from app.models.answer import AnswerCandidate
from app.models.moderation import ModerationResult
from app.models.triage import TriageResult
from app.settings import ZammadAISettings
from app.settings.triage import Action, ActionTypes, Category


class RecordingZammadClient:
    """Record outbound writes made by ActionService."""

    def __init__(self) -> None:
        """Initialize empty write records."""
        self.answers: list[dict[str, object]] = []
        self.shared_drafts: list[dict[str, object]] = []

    async def post_answer(
        self,
        ticket_id: int,
        text: str,
        subject: str | None = None,
        internal: bool = False,
    ) -> None:
        """Record a public or internal answer post."""
        self.answers.append(
            {
                "ticket_id": ticket_id,
                "text": text,
                "subject": subject,
                "internal": internal,
            }
        )

    async def post_shared_draft(self, ticket_id: int, text: str) -> None:
        """Record a shared draft post."""
        self.shared_drafts.append({"ticket_id": ticket_id, "text": text})

    async def close(self) -> None:
        """No-op close for the fake client."""
        return None


class FakeAnswerService:
    """Return a configured AI answer candidate."""

    def __init__(self) -> None:
        """Initialize call counters."""
        self.generated_answer_calls = 0
        self.generated_conversational_calls = 0

    async def generate_answer(self, **_kwargs) -> AnswerCandidate:
        """Return a regular KB-grounded answer candidate."""
        self.generated_answer_calls += 1
        return AnswerCandidate(
            response=(
                "This is a generated support reply with enough content to satisfy the schema minimum length. "
                "It contains detailed guidance for the customer and would normally be eligible for automatic "
                "publication when the category allows auto publish and the answer candidate permits it."
            ),
            documents=[],
            auto_publish=True,
        )

    async def generate_conversational_answer(self, **_kwargs) -> AnswerCandidate:
        """Return a conversational answer candidate."""
        self.generated_conversational_calls += 1
        return AnswerCandidate(
            response=(
                "مرحبا، شكرا لتواصلك معنا. يسعدنا مساعدتك في نطاق الدعم المتاح. "
                "فضلا أرسل تفاصيل الخدمة أو المشروع أو المشكلة التي تحتاج إلى مساعدة بشأنها، "
                "وسنوجهك للخطوة المناسبة أو للفريق المختص حسب حالتك. "
                "يمكنك كتابة سؤالك بشكل مختصر أو إرسال تفاصيل إضافية، وسنتعامل مع رسالتك ضمن نطاق الدعم المتاح."
            ),
            documents=[],
            auto_publish=True,
        )


class FakeModerationService:
    """Return a configured moderation decision for generated responses."""

    async def moderate_response(self, **_kwargs) -> ModerationResult:
        """Return a high-risk moderation decision."""
        return ModerationResult(
            decision="unsafe",
            language="ar",
            risk_level="high",
            harm_categories=["prompt_injection"],
            reason="Generated response requires review.",
            customer_response_type="human_review",
        )


@pytest.mark.asyncio
async def test_out_of_scope_static_fallback_is_published(settings_factory) -> None:
    """Out-of-scope synthetic triage should send the configured reusable static fallback."""
    fallback = (
        "Thanks for reaching out. I may not be the right assistant for that request, "
        "but your message has been received. Please send a few details about the service, "
        "project, or support issue you need help with, and our team will guide you to the right next step."
    )
    out_of_scope_action = Action(
        name="out_of_scope_static_response",
        description="Static out-of-scope fallback",
        type=ActionTypes.StaticAnswer,
        answer=fallback,
    )
    settings: ZammadAISettings = settings_factory()
    settings.triage.actions.append(out_of_scope_action)

    service = ActionService.__new__(ActionService)
    service.settings = settings
    service.answer_service = object()
    service.guardrail_service = GuardrailService(settings=settings.guardrails)
    service.max_user_text_length = settings.max_user_text_length
    service.zammad_client = RecordingZammadClient()

    triage = TriageResult(
        user_text="write me a poem",
        category=Category(name="Out of scope", auto_publish=True),
        action=out_of_scope_action,
        reasoning="Message asks for unrelated content.",
        confidence=0.85,
        extracted_values=None,
    )

    await service.execute_action(ticket_id=123, triage=triage)

    assert service.zammad_client.answers == [
        {
            "ticket_id": 123,
            "text": fallback,
            "subject": "Answer",
            "internal": False,
        }
    ]
    assert service.zammad_client.shared_drafts == []


@pytest.mark.asyncio
async def test_gemini_moderation_blocks_auto_publish_for_ai_answer(settings_factory) -> None:
    """Unsafe generated AI replies should be stored as shared drafts for human review."""
    ai_action = Action(
        name="conversational_ai_answer",
        description="Generate conversational answer",
        type=ActionTypes.AIAnswer,
    )
    settings: ZammadAISettings = settings_factory()
    settings.moderation.enabled = True
    settings.triage.actions.append(ai_action)

    service = ActionService.__new__(ActionService)
    service.settings = settings
    service.answer_service = FakeAnswerService()
    service.guardrail_service = GuardrailService(settings=settings.guardrails)
    service.moderation_service = FakeModerationService()
    service.max_user_text_length = settings.max_user_text_length
    service.zammad_client = RecordingZammadClient()

    triage = TriageResult(
        user_text="مرحبا",
        category=Category(name="Conversational support", auto_publish=True),
        action=ai_action,
        reasoning="Arabic greeting.",
        confidence=1.0,
        extracted_values=None,
    )

    await service.execute_action(ticket_id=123, triage=triage)

    assert service.zammad_client.answers == []
    assert len(service.zammad_client.shared_drafts) == 1
    assert service.zammad_client.shared_drafts[0]["ticket_id"] == 123


@pytest.mark.asyncio
async def test_conversational_support_uses_conversational_generator(settings_factory) -> None:
    """Small-talk actions should not use the KB-grounded answer path."""
    ai_action = Action(
        name="conversational_ai_answer",
        description="Generate conversational answer",
        type=ActionTypes.AIAnswer,
    )
    settings: ZammadAISettings = settings_factory()
    settings.moderation.small_talk_category_name = "Conversational support"
    settings.triage.actions.append(ai_action)

    answer_service = FakeAnswerService()
    service = ActionService.__new__(ActionService)
    service.settings = settings
    service.answer_service = answer_service
    service.guardrail_service = GuardrailService(settings=settings.guardrails)
    service.moderation_service = FakeModerationService()
    service.max_user_text_length = settings.max_user_text_length
    service.zammad_client = RecordingZammadClient()

    triage = TriageResult(
        user_text="السلام عليكم",
        category=Category(name="Conversational support", auto_publish=True),
        action=ai_action,
        reasoning="Arabic greeting.",
        confidence=1.0,
        extracted_values=None,
    )

    await service.execute_action(ticket_id=123, triage=triage)

    assert answer_service.generated_answer_calls == 0
    assert answer_service.generated_conversational_calls == 1
    assert len(service.zammad_client.answers) == 1
    assert service.zammad_client.answers[0]["internal"] is False
