"""Select a ticket's unanswered customer turn and decide whether the AI may answer it."""

from collections.abc import Collection
from dataclasses import dataclass

from app.models.zammad import ZammadArticle, ZammadTicket

# Zammad's "-" user: a ticket owned by it is unassigned.
UNASSIGNED_OWNER_ID = 1


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """The unanswered customer messages of a ticket, and why the AI must not answer them, if it must not."""

    articles: tuple[ZammadArticle, ...]
    blocked_by: str | None

    @property
    def answerable(self) -> bool:
        """Return whether there is a turn and the AI may answer it."""
        return bool(self.articles) and self.blocked_by is None

    @property
    def newest_article_id(self) -> int | None:
        """Return the ID of the newest message in the turn."""
        return self.articles[-1].id if self.articles else None


def is_customer_message(article: ZammadArticle) -> bool:
    """Return whether the article is a public message from the customer, excluding web-form submissions."""
    return (
        not article.internal
        and (article.sender or "").casefold() == "customer"
        and (article.type or "").casefold() != "web"
    )


def unanswered_turn(ticket: ZammadTicket) -> tuple[ZammadArticle, ...]:
    """Return the customer messages after the newest one the AI answered, oldest first."""
    answered_through: int = ticket.ai_answered_through_article_id or 0
    return tuple(
        sorted(
            (article for article in ticket.articles if article.id > answered_through and is_customer_message(article)),
            key=lambda article: article.id,
        )
    )


def takeover_reason(
    ticket: ZammadTicket, *, ai_user_id: int, handoff_tags: Collection[str], resume_tag: str
) -> str | None:
    """Return why a person now owns the conversation, or None when the AI may answer.

    A person owns it after a public reply from a human agent since the AI's last answer, when a human agent
    owns the ticket, or when it carries a hand-off tag. The resume tag hands it back to the AI.
    """
    if resume_tag in ticket.tags:
        return None
    for tag in handoff_tags:
        if tag in ticket.tags:
            return f"hand-off tag {tag}"
    if ticket.owner_id not in (None, UNASSIGNED_OWNER_ID, ai_user_id):
        return "human owner"
    answered_through: int = ticket.ai_answered_through_article_id or 0
    for article in ticket.articles:
        if (
            article.id > answered_through
            and not article.internal
            and (article.sender or "").casefold() == "agent"
            and article.created_by_id not in (None, ai_user_id)
        ):
            return "human reply"
    return None


def decide_turn(
    ticket: ZammadTicket, *, ai_user_id: int, handoff_tags: Collection[str], resume_tag: str
) -> TurnDecision:
    """Combine the unanswered turn with the takeover rule."""
    return TurnDecision(
        articles=unanswered_turn(ticket),
        blocked_by=takeover_reason(ticket, ai_user_id=ai_user_id, handoff_tags=handoff_tags, resume_tag=resume_tag),
    )
