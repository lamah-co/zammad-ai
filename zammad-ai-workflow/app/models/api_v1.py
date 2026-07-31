"""Request and response models for the public v1 API."""

from pydantic import BaseModel, Field

from .answer import DocumentDict
from .triage import TriageResult


class TriageInput(BaseModel):
    """Payload for a triage request."""

    text: str
    session_id: str | None = None


class TriageOutput(BaseModel):
    """Response payload for a triage request."""

    triage: TriageResult
    session_id: str


class AnswerInput(BaseModel):
    """Payload for an answer request."""

    text: str
    session_id: str | None = None
    ticket_id: int | None = None
    category: str
    action: str


class AnswerOutput(BaseModel):
    """Response payload for an answer request."""

    response: str = Field(description="The final answer to the user's question.")
    documents: list[DocumentDict] = Field(description="List of documents supporting the answer.")
    auto_publish: bool = Field(
        description="Indicates whether the generated answer should be automatically published based on the judge's opinion.",
        default=False,
    )


class ZammadTicketEventInput(BaseModel):
    """A Zammad trigger event scoped to one newly created article."""

    ticket_id: int = Field(description="Ticket containing the new customer article")
    article_id: int = Field(description="New article that caused the Zammad trigger")


class ZammadTicketEventOutput(BaseModel):
    """Processing result returned to the Zammad webhook."""

    status: str = Field(description="processed or ignored")
    reason: str | None = Field(default=None, description="Reason an event was ignored")


class HealthCheckResponse(BaseModel):
    """Health check response returned by the backend."""

    status: str = Field(
        description="Health status of the API. Expected value is 'healthy' when the API is operational.",
        default="healthy",
    )
