"""Select a ticket's unanswered customer turn and decide whether the AI may answer it."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

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


# Tells triage and the answer model that a burst is one request, so a greeting next to a question isn't small talk.
BURST_PREAMBLE = (
    "The customer sent these messages in a row, oldest first. Read them together as one request. "
    "Greetings, thanks and test messages are not the request when another message asks for something."
)


def turn_text(articles: Sequence[ZammadArticle]) -> str:
    """Return the text the AI answers: a single message as is, several as a numbered list, oldest first."""
    if len(articles) == 1:
        return articles[0].text
    lines = "\n".join(f"{number}. {article.text}" for number, article in enumerate(articles, start=1))
    return f"{BURST_PREAMBLE}\n{lines}"


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


@dataclass(frozen=True, slots=True)
class Readiness:
    """Whether a turn can be answered now, or how long to wait for the customer to finish writing."""

    wait_seconds: float
    after_cap: bool = False

    @property
    def ready(self) -> bool:
        """Return whether the turn can be answered now."""
        return self.wait_seconds <= 0


def turn_readiness(
    articles: Sequence[ZammadArticle], *, now: datetime, quiet_seconds: float, max_wait_seconds: float
) -> Readiness:
    """Return whether the turn is ready: its newest message is quiet_seconds old, or its first is max_wait_seconds old.

    Messages without a timestamp never delay the answer.
    """
    times = [article.created_at for article in articles if article.created_at is not None]
    if quiet_seconds <= 0 or not times or len(times) != len(articles):
        return Readiness(wait_seconds=0)
    quiet_left = quiet_seconds - (now - max(times)).total_seconds()
    if quiet_left <= 0:
        return Readiness(wait_seconds=0)
    cap_left = max_wait_seconds - (now - min(times)).total_seconds()
    if cap_left <= 0:
        return Readiness(wait_seconds=0, after_cap=True)
    return Readiness(wait_seconds=min(quiet_left, cap_left))


def message_gaps(articles: Sequence[ZammadArticle]) -> list[float]:
    """Return the seconds between consecutive messages of a turn."""
    times = sorted(article.created_at for article in articles if article.created_at is not None)
    return [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:], strict=False)]


def seconds_after_ai_reply(ticket: ZammadTicket, articles: Sequence[ZammadArticle], *, ai_user_id: int) -> float | None:
    """Return how long after the AI's previous public reply the turn started, or None without such a reply."""
    if not articles or articles[0].created_at is None:
        return None
    start = articles[0].created_at
    replies = [
        article.created_at
        for article in ticket.articles
        if article.created_at is not None
        and article.created_at <= start
        and not article.internal
        and (article.sender or "").casefold() == "agent"
        and article.created_by_id == ai_user_id
    ]
    return (start - max(replies)).total_seconds() if replies else None
