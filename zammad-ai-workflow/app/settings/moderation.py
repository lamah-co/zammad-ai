"""Settings for Gemini-backed moderation."""

from typing import Literal

from pydantic import BaseModel, Field


class ModerationSettings(BaseModel):
    """Configuration for Gemini moderation routing and response checks."""

    enabled: bool = Field(default=False, description="Enable Gemini moderation checks.")
    provider: Literal["gemini"] = Field(default="gemini", description="Moderation provider identifier.")
    model: str | None = Field(
        default=None,
        description="Gemini model for moderation. Falls back to genai.judge_model or genai.chat_model.",
    )
    block_on_unsafe: bool = Field(
        default=True,
        description="Route unsafe generated responses away from publication.",
    )
    uncertain_action: str = Field(
        default="human_review",
        description="Route to use when moderation is uncertain.",
    )
    small_talk_category_name: str = Field(
        default="Conversational support",
        description="Category name used for small-talk moderation decisions.",
    )
    out_of_scope_category_name: str = Field(
        default="Out of scope",
        description="Synthetic category name used for out-of-scope moderation decisions.",
    )
    out_of_scope_action_name: str = Field(
        default="out_of_scope_static_response",
        description="Configured StaticAnswer action used for out-of-scope fallback responses.",
    )
