"""Contract tests for channel-aware Zammad article creation."""

import json

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from app.errors import UnsupportedReplyChannelError
from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings.zammad import ZammadAPISettings
from app.zammad.api import ZammadAPIClient


def _settings() -> ZammadAPISettings:
    return ZammadAPISettings(
        base_url=HttpUrl("https://zammad.example.test"),
        auth_token=SecretStr("access-token"),
    )


def _json(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def test_ticket_selects_latest_public_customer_article() -> None:
    ticket = ZammadTicket(
        id=42,
        articles=[
            ZammadArticle(id=1, ticket_id=42, text="old", sender="Customer", type="email"),
            ZammadArticle(id=3, ticket_id=42, text="agent", sender="Agent", type="email"),
            ZammadArticle(id=2, ticket_id=42, text="new", sender="Customer", type="whatsapp message"),
        ],
    )

    assert ticket.latest_customer_article() is not None
    assert ticket.latest_customer_article().id == 2  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_post_answer_replies_through_email_channel() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 17,
                        "ticket_id": 42,
                        "body": "Where is my order?",
                        "internal": False,
                        "sender": "Customer",
                        "type": "email",
                        "from": "Customer <customer@example.test>",
                        "to": "Support <support@example.test>",
                        "subject": "Order 1001",
                    }
                ],
            )
        return httpx.Response(201, json={"id": 18})

    client = ZammadAPIClient(_settings())
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test",
        transport=httpx.MockTransport(handler),
        headers=client.client.headers,
    )

    article_id = await client.post_answer(ticket_id=42, text="Your order is on its way.", subject="Order 1001")

    assert article_id == 18
    assert requests[0].headers["Authorization"] == "Token token=access-token"
    assert _json(requests[1]) == {
        "ticket_id": 42,
        "body": "Your order is on its way.",
        "internal": False,
        "subject": "Order 1001",
        "content_type": "text/html",
        "type": "email",
        "sender": "Agent",
        "from": "Support <support@example.test>",
        "to": "Customer <customer@example.test>",
    }
    await client.close()


@pytest.mark.asyncio
async def test_post_answer_replies_through_whatsapp_channel() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 21,
                        "ticket_id": 43,
                        "body": "اين طلبي؟",
                        "internal": False,
                        "sender": "Customer",
                        "type": "whatsapp message",
                    }
                ],
            )
        return httpx.Response(201, json={"id": 22})

    client = ZammadAPIClient(_settings())
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test",
        transport=httpx.MockTransport(handler),
        headers=client.client.headers,
    )

    article_id = await client.post_answer(ticket_id=43, text="طلبك في الطريق")

    assert article_id == 22
    assert _json(requests[1]) == {
        "ticket_id": 43,
        "body": "طلبك في الطريق",
        "internal": False,
        "content_type": "text/plain",
        "type": "whatsapp message",
        "sender": "Agent",
    }
    await client.close()


@pytest.mark.asyncio
async def test_internal_answer_is_an_explicit_note_without_channel_lookup() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"id": 31})

    client = ZammadAPIClient(_settings())
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test",
        transport=httpx.MockTransport(handler),
        headers=client.client.headers,
    )

    await client.post_answer(ticket_id=44, text="Escalated", internal=True)

    assert len(requests) == 1
    assert _json(requests[0])["type"] == "note"
    assert _json(requests[0])["sender"] == "Agent"
    await client.close()


@pytest.mark.asyncio
async def test_public_answer_rejects_unsupported_customer_channel() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": 51,
                    "ticket_id": 45,
                    "body": "Hello",
                    "internal": False,
                    "sender": "Customer",
                    "type": "web",
                }
            ],
        )

    client = ZammadAPIClient(_settings())
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://zammad.example.test",
        transport=httpx.MockTransport(handler),
        headers=client.client.headers,
    )

    with pytest.raises(UnsupportedReplyChannelError, match="web"):
        await client.post_answer(ticket_id=45, text="Hello back")
    await client.close()
