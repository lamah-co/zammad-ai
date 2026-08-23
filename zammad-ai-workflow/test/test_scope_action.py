"""Tests for moderation-driven action execution outcomes."""

import pytest

from app.action.service import ActionService
from app.guardrails import GuardrailService
from app.models.answer import AnswerCandidate, NoAnswerPossible
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
        self.tags: list[dict[str, object]] = []
        self.states: list[dict[str, object]] = []

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

    async def add_tag_to_ticket(self, ticket_id: int, tag: str) -> None:
        """Record a tag assignment."""
        self.tags.append({"ticket_id": ticket_id, "tag": tag})

    async def set_ticket_state(self, ticket_id: int, state: str) -> None:
        """Record a ticket state update."""
        self.states.append({"ticket_id": ticket_id, "state": state})

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


class NoAnswerService(FakeAnswerService):
    """Return a no-answer outcome from the KB-grounded answer path."""

    async def generate_answer(self, **_kwargs) -> NoAnswerPossible:
        """Return a no-answer outcome with schema-compliant reasoning."""
        self.generated_answer_calls += 1
        return NoAnswerPossible(
            reasoning=(
                "The retrieved knowledge-base passages did not contain enough grounded information to answer the "
                "customer question confidently, so the workflow must avoid inventing unsupported operational details."
            )
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


@pytest.mark.asyncio
async def test_no_answer_possible_sends_customer_fallback_and_agent_handoff(settings_factory) -> None:
    """FAQ requests that cannot be grounded should acknowledge the customer and alert agents."""
    ai_action = Action(
        name="ai_answer",
        description="Generate grounded answer",
        type=ActionTypes.AIAnswer,
    )
    settings: ZammadAISettings = settings_factory()
    settings.triage.actions.append(ai_action)
    settings.triage.no_answer_public_fallback = "تم استلام سؤالك، سنقوم بالرد عليك في أقرب وقت ممكن."
    settings.triage.no_answer_internal_note = (
        "AI could not generate a grounded answer.\n"
        "Category: {category}\n"
        "Action: {action}\n"
        "Customer message: {customer_message}\n"
        "Reason: {reason}"
    )
    settings.triage.no_answer_tag = "ai_no_answer"
    settings.triage.handoff_ticket_state = "open"

    service = ActionService.__new__(ActionService)
    service.settings = settings
    service.answer_service = NoAnswerService()
    service.guardrail_service = GuardrailService(settings=settings.guardrails)
    service.moderation_service = FakeModerationService()
    service.max_user_text_length = settings.max_user_text_length
    service.zammad_client = RecordingZammadClient()

    triage = TriageResult(
        user_text="شن نوع Authorization المستخدم؟",
        category=Category(name="FAQ answerable", auto_publish=True),
        action=ai_action,
        reasoning="Question appears answerable from the FAQ scope.",
        confidence=0.92,
        extracted_values=None,
    )

    await service.execute_action(ticket_id=123, triage=triage)

    public_answers = [answer for answer in service.zammad_client.answers if not answer["internal"]]
    internal_answers = [answer for answer in service.zammad_client.answers if answer["internal"]]

    assert public_answers == [
        {
            "ticket_id": 123,
            "text": "تم استلام سؤالك، سنقوم بالرد عليك في أقرب وقت ممكن.",
            "subject": "Answer",
            "internal": False,
        }
    ]
    assert len(internal_answers) == 1
    assert internal_answers[0]["subject"] == "Handoff"
    assert "شن نوع Authorization المستخدم؟" in str(internal_answers[0]["text"])
    assert service.zammad_client.tags == [{"ticket_id": 123, "tag": "ai_no_answer"}]
    assert service.zammad_client.states == [{"ticket_id": 123, "state": "open"}]
    assert service.zammad_client.shared_drafts == []


@pytest.mark.asyncio
async def test_human_review_required_sends_acknowledgement_and_handoff(settings_factory) -> None:
    """Human review outcomes should not remain silent."""
    no_action = Action(
        name="no_action",
        description="Escalate to a human",
        type=ActionTypes.NoAction,
    )
    human_review_message = "تم استلام استفسارك ، ويحتاج إلى مراجعة من الفريق المختص . سنقوم بالرد عليك في أقرب وقت ممكن"
    settings: ZammadAISettings = settings_factory()
    settings.triage.actions.append(no_action)
    settings.triage.no_category_name = "Human review required"
    settings.triage.human_review_public_fallback = human_review_message
    settings.triage.human_review_internal_note = (
        "AI routed this ticket to human review.\n"
        "Category: {category}\n"
        "Action: {action}\n"
        "Customer message: {customer_message}\n"
        "Reason: {reason}"
    )
    settings.triage.human_review_tag = "ai_human_review"
    settings.triage.handoff_ticket_state = "open"

    service = ActionService.__new__(ActionService)
    service.settings = settings
    service.answer_service = FakeAnswerService()
    service.guardrail_service = GuardrailService(settings=settings.guardrails)
    service.moderation_service = FakeModerationService()
    service.max_user_text_length = settings.max_user_text_length
    service.zammad_client = RecordingZammadClient()

    triage = TriageResult(
        user_text="عندي رقم مرسل كيف يتم تفعيله؟",
        category=Category(name="Human review required", auto_publish=False),
        action=no_action,
        reasoning="Customer asks for account-specific sender activation.",
        confidence=0.88,
        extracted_values=None,
    )

    await service.execute_action(ticket_id=456, triage=triage)

    public_answers = [answer for answer in service.zammad_client.answers if not answer["internal"]]
    internal_answers = [answer for answer in service.zammad_client.answers if answer["internal"]]

    assert public_answers == [
        {
            "ticket_id": 456,
            "text": human_review_message,
            "subject": "Answer",
            "internal": False,
        }
    ]
    assert len(internal_answers) == 1
    assert internal_answers[0]["subject"] == "Handoff"
    assert "عندي رقم مرسل كيف يتم تفعيله؟" in str(internal_answers[0]["text"])
    assert service.zammad_client.tags == [{"ticket_id": 456, "tag": "ai_human_review"}]
    assert service.zammad_client.states == [{"ticket_id": 456, "state": "open"}]
    assert service.zammad_client.shared_drafts == []
