"""Tests for selecting the unanswered turn and applying the human takeover rule."""

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from app.conversation.turn import BURST_PREAMBLE, decide_turn, takeover_reason, turn_text, unanswered_turn
from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings.zammad import ZammadAPISettings
from app.zammad.api import ZammadAPIClient

AI_USER = 5
HUMAN_AGENT = 7
CUSTOMER = 9
HANDOFF_TAGS = ("ai_human_review", "ai_no_answer", "ai_failed")


def customer(article_id: int, text: str = "msg", type_: str = "whatsapp message") -> ZammadArticle:
    """Return a public customer article."""
    return ZammadArticle(id=article_id, ticket_id=1, text=text, sender="Customer", type=type_, created_by_id=CUSTOMER)


def agent(article_id: int, *, by: int, internal: bool = False) -> ZammadArticle:
    """Return an agent article written by the given user."""
    return ZammadArticle(
        id=article_id,
        ticket_id=1,
        text="reply",
        sender="Agent",
        type="note" if internal else "whatsapp message",
        internal=internal,
        created_by_id=by,
    )


def decide(ticket: ZammadTicket):
    """Apply the rule with the production tag names."""
    return decide_turn(ticket, ai_user_id=AI_USER, handoff_tags=HANDOFF_TAGS, resume_tag="ai_resume")


def test_burst_after_the_last_ai_answer_is_one_turn() -> None:
    """Customer messages after the marker form the turn, oldest first."""
    ticket = ZammadTicket(
        id=1,
        ai_answered_through_article_id=11,
        articles=[customer(10), customer(11), agent(12, by=AI_USER), customer(15), customer(13), customer(14)],
    )

    decision = decide(ticket)

    assert [a.id for a in decision.articles] == [13, 14, 15]
    assert decision.answerable
    assert decision.newest_article_id == 15


def test_internal_agent_system_and_web_articles_are_not_part_of_a_turn() -> None:
    """Only public customer messages count, and web-form submissions are excluded."""
    ticket = ZammadTicket(
        id=1,
        articles=[
            customer(1, type_="web"),
            agent(2, by=HUMAN_AGENT, internal=True),
            ZammadArticle(id=3, ticket_id=1, text="auto reply", sender="System", type="email"),
            ZammadArticle(id=4, ticket_id=1, text="note", sender="Customer", type="note", internal=True),
            customer(5),
        ],
    )

    assert [a.id for a in unanswered_turn(ticket)] == [5]


def test_nothing_new_since_the_marker_is_not_answerable() -> None:
    """A redelivered or duplicate event finds an empty turn."""
    ticket = ZammadTicket(
        id=1, ai_answered_through_article_id=3, articles=[customer(1), customer(3), agent(4, by=AI_USER)]
    )

    decision = decide(ticket)

    assert decision.articles == ()
    assert not decision.answerable


def test_ai_replies_and_agent_internal_notes_do_not_block() -> None:
    """The AI's own replies and private notes by agents leave the conversation with the AI."""
    ticket = ZammadTicket(
        id=1,
        ai_answered_through_article_id=1,
        articles=[customer(1), agent(2, by=AI_USER), agent(3, by=HUMAN_AGENT, internal=True), customer(4)],
    )

    assert decide(ticket).answerable


def test_public_human_reply_blocks_the_ai() -> None:
    """Once a human agent replies publicly, the AI stays out."""
    ticket = ZammadTicket(
        id=1, ai_answered_through_article_id=1, articles=[customer(1), agent(2, by=HUMAN_AGENT), customer(3)]
    )

    assert decide(ticket).blocked_by == "human reply"


@pytest.mark.parametrize(("owner", "blocked"), [(None, False), (1, False), (AI_USER, False), (HUMAN_AGENT, True)])
def test_only_a_human_owner_blocks_the_ai(owner: int | None, blocked: bool) -> None:
    """Unassigned (owner 1) or AI-owned tickets stay with the AI; a human owner takes over."""
    ticket = ZammadTicket(id=1, owner_id=owner, articles=[customer(1)])

    assert (decide(ticket).blocked_by == "human owner") is blocked


@pytest.mark.parametrize("tag", HANDOFF_TAGS)
def test_hand_off_tags_block_the_ai(tag: str) -> None:
    """After the AI handed a ticket off, a person owns it."""
    ticket = ZammadTicket(id=1, tags=[tag], articles=[customer(1)])

    assert decide(ticket).blocked_by == f"hand-off tag {tag}"


def test_resume_tag_hands_the_ticket_back_to_the_ai() -> None:
    """An agent can hand the conversation back, even after a human reply, ownership and a hand-off."""
    ticket = ZammadTicket(
        id=1,
        owner_id=HUMAN_AGENT,
        tags=["ai_human_review", "ai_resume"],
        articles=[customer(1), agent(2, by=HUMAN_AGENT), customer(3)],
    )

    assert decide(ticket).answerable


def test_human_reply_before_the_last_ai_answer_no_longer_blocks() -> None:
    """After a resume, the AI answered past the human reply, so that reply no longer counts."""
    ticket = ZammadTicket(
        id=1,
        ai_answered_through_article_id=3,
        articles=[customer(1), agent(2, by=HUMAN_AGENT), customer(3), agent(4, by=AI_USER), customer(5)],
    )

    assert takeover_reason(ticket, ai_user_id=AI_USER, handoff_tags=HANDOFF_TAGS, resume_tag="ai_resume") is None


def test_ticket_5_from_production_is_left_to_the_human_who_replied() -> None:
    """Ticket 5 on 2026-11-05..07: a human answered "Test", so the AI must not answer the later messages."""
    ticket = ZammadTicket(
        id=5,
        owner_id=1,
        articles=[
            customer(160, "Test"),
            agent(161, by=HUMAN_AGENT),
            customer(162, "اهلا ، نبي نطلب معرف مرسل"),
            ZammadArticle(id=163, ticket_id=5, text="Reminder", sender="System", type="whatsapp message"),
            customer(164, "اهلا"),
        ],
    )

    decision = decide(ticket)

    assert [a.id for a in decision.articles] == [160, 162, 164]
    assert decision.blocked_by == "human reply"
    assert not decision.answerable


def test_turn_text_keeps_a_single_message_and_numbers_a_burst() -> None:
    """One message is answered as is; a burst is one numbered request, so a greeting doesn't hide the question."""
    assert turn_text([customer(1, "اهلا")]) == "اهلا"
    assert turn_text([customer(1, "Test"), customer(2, "اهلا ، نبي نطلب معرف مرسل"), customer(3, "اهلا")]) == (
        f"{BURST_PREAMBLE}\n1. Test\n2. اهلا ، نبي نطلب معرف مرسل\n3. اهلا"
    )


@pytest.mark.asyncio
async def test_api_client_loads_ticket_context_and_caches_its_own_user() -> None:
    """The REST client fills owner, tags and the marker, and asks for its own user only once."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("/ticket_articles/by_ticket/42"):
            return httpx.Response(
                200, json=[{"id": 1, "ticket_id": 42, "body": "hi", "sender": "Customer", "created_by_id": CUSTOMER}]
            )
        if request.url.path.endswith("/tickets/42"):
            return httpx.Response(
                200, json={"id": 42, "group_id": 3, "owner_id": 1, "ai_answered_through_article_id": 0}
            )
        if request.url.path.endswith("/tags"):
            return httpx.Response(200, json={"tags": ["ai_resume"]})
        return httpx.Response(200, json={"id": AI_USER})

    client = ZammadAPIClient(
        ZammadAPISettings(base_url=HttpUrl("https://zammad.example.test"), auth_token=SecretStr("t"))
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test", transport=httpx.MockTransport(handler), headers=client.client.headers
    )

    ticket = await client.get_ticket_with_context(42)
    first = await client.get_own_user_id()
    second = await client.get_own_user_id()

    assert (ticket.group_id, ticket.owner_id, ticket.ai_answered_through_article_id, ticket.tags) == (
        3,
        1,
        0,
        ["ai_resume"],
    )
    assert ticket.articles[0].created_by_id == CUSTOMER
    assert first == second == AI_USER
    assert requests.count("/api/v1/users/me") == 1
