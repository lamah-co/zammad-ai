"""Models for Gemini moderation decisions."""

from typing import Literal

from pydantic import BaseModel, Field


ModerationDecision = Literal["safe_support", "small_talk", "out_of_scope", "unsafe", "uncertain"]
ModerationRiskLevel = Literal["low", "medium", "high"]
ModerationRoute = Literal["continue_triage", "conversational_ai", "static_fallback", "human_review"]


class ModerationResult(BaseModel):
    """Structured moderation result used before triage and before publishing generated responses."""

    decision: ModerationDecision = Field(description="Moderation decision for the evaluated text")
    language: str = Field(description="Detected BCP-47 language code, such as ar, en, ar-LY, or und")
    risk_level: ModerationRiskLevel = Field(description="Overall risk level")
    harm_categories: list[str] = Field(default_factory=list, description="Detected harm or policy categories")
    reason: str = Field(description="Short internal reason for the decision")
    customer_response_type: ModerationRoute = Field(description="Workflow route recommended by moderation")
