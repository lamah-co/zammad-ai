"""Settings for answering ticket conversations turn by turn."""

from typing import Self

from pydantic import BaseModel, Field, NonNegativeFloat, model_validator


class ConversationSettings(BaseModel):
    """How the AI decides whether it may answer a ticket's unanswered customer messages."""

    enabled: bool = Field(
        default=True,
        description="Answer each unanswered customer turn once. Off: one pass on the newest customer message per event.",
    )
    resume_tag: str = Field(
        default="ai_resume",
        description="Tag an agent adds to hand a ticket back to the AI after a person was involved.",
    )
    quiet_seconds: NonNegativeFloat = Field(
        default=10,
        description="Answer a turn only once its newest customer message is this old, so a burst gets one reply. 0 answers at once.",
    )
    max_wait_seconds: NonNegativeFloat = Field(
        default=60,
        description="Answer a turn at the latest this long after its first message, even while the customer keeps writing.",
    )

    @model_validator(mode="after")
    def validate_wait(self) -> Self:
        """The cap must not be shorter than the quiet period."""
        if self.max_wait_seconds < self.quiet_seconds:
            raise ValueError("conversation.max_wait_seconds must be at least conversation.quiet_seconds")
        return self
