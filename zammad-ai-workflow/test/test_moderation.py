"""Tests for Gemini-backed moderation routing."""

import pytest

from app.models.moderation import ModerationResult
from app.moderation.service import GeminiModerationService


class FakeModerationAgent:
    """Return a configured structured moderation result."""

    def __init__(self, result: ModerationResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def ainvoke(self, *, input: dict, config: object) -> dict:
        self.calls.append({"input": input, "config": config})
        return {"structured_response": self.result}


@pytest.mark.asyncio
async def test_gemini_moderation_disabled_returns_safe_support(settings_factory) -> None:
    """Disabled moderation should fail open for existing local/test deployments."""
    service = GeminiModerationService(settings_factory())

    result = await service.moderate_prompt("write me a poem")

    assert result.decision == "safe_support"
    assert result.customer_response_type == "continue_triage"
    assert result.language == "und"


@pytest.mark.asyncio
async def test_gemini_moderation_empty_text_returns_human_review(settings_factory) -> None:
    """Enabled moderation should send empty text to human review without a model call."""
    service = GeminiModerationService.__new__(GeminiModerationService)
    service.settings = settings_factory().moderation
    service.settings.enabled = True
    service._enabled = True
    service._agent = FakeModerationAgent(
        ModerationResult(
            decision="safe_support",
            language="ar",
            risk_level="low",
            harm_categories=[],
            reason="Should not be used.",
            customer_response_type="continue_triage",
        )
    )

    result = await service.moderate_prompt("   ")

    assert result.decision == "uncertain"
    assert result.customer_response_type == "human_review"
    assert result.language == "und"
    assert service._agent.calls == []


@pytest.mark.asyncio
async def test_gemini_moderation_uses_structured_arabic_decision(settings_factory) -> None:
    """Gemini structured moderation output should preserve Arabic language metadata."""
    expected = ModerationResult(
        decision="small_talk",
        language="ar",
        risk_level="low",
        harm_categories=[],
        reason="Arabic greeting.",
        customer_response_type="conversational_ai",
    )
    service = GeminiModerationService.__new__(GeminiModerationService)
    service.settings = settings_factory().moderation
    service.settings.enabled = True
    service._enabled = True
    service._agent = FakeModerationAgent(expected)

    result = await service.moderate_prompt("السلام عليكم")

    assert result == expected
    assert service._agent.calls


@pytest.mark.asyncio
async def test_gemini_moderation_checks_generated_response(settings_factory) -> None:
    """Generated answers should be moderated with both the prompt and response text."""
    expected = ModerationResult(
        decision="unsafe",
        language="ar",
        risk_level="high",
        harm_categories=["credential_extraction"],
        reason="Asks for private credentials.",
        customer_response_type="human_review",
    )
    service = GeminiModerationService.__new__(GeminiModerationService)
    service.settings = settings_factory().moderation
    service.settings.enabled = True
    service._enabled = True
    service._agent = FakeModerationAgent(expected)

    result = await service.moderate_response(
        prompt="كيف أغير كلمة المرور؟",
        response="Send your password to support.",
        session_id="session-id",
    )

    message = service._agent.calls[0]["input"]["messages"][0].content
    assert result == expected
    assert "Customer prompt:" in message
    assert "Generated response:" in message
    assert "session-id" in message
