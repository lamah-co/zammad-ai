"""Settings for answering ticket conversations turn by turn."""

from pydantic import BaseModel, Field


class ConversationSettings(BaseModel):
    """How the AI decides whether it may answer a ticket's unanswered customer messages."""

    resume_tag: str = Field(
        default="ai_resume",
        description="Tag an agent adds to hand a ticket back to the AI after a person was involved.",
    )
