"""Regression tests for channel-aware replies and ticket article selection."""

from datetime import datetime, timezone

import pytest

from app.models.zammad import ZammadArticle, ZammadTicket
from app.settings.zammad import ZammadAPISettings
from app.zammad.api import ZammadAPIClient


@pytest.mark.asyncio
async def test_post_answer_uses_latest_customer_article_channel(settings_factory) -> None:
    """Public replies should use the latest customer article channel instead of creating notes."""
    settings = settings_factory().zammad
    assert isinstance(settings, ZammadAPISettings)
    client = ZammadAPIClient(settings=settings)
    requests: list[dict[str, object]] = []

    async def fake_get_ticket(_ticket_id: int) -> ZammadTicket:
        return ZammadTicket(
            id=5,
            articles=[
                ZammadArticle(
                    id=1,
                    ticket_id=5,
                    text="old",
                    type="email",
                    sender="Customer",
                    from_="old@example.test",
                    to="support@example.test",
                    created_at=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
                ),
                ZammadArticle(
                    id=2,
                    ticket_id=5,
                    text="latest",
                    type="whatsapp message",
                    sender="Customer",
                    from_="Asma (+218910055964)",
                    to="Lamah Sanad (+218 93-4363964)",
                    created_at=datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
                ),
            ],
        )

    async def fake_request(method: str, url: str, **kwargs) -> None:
        requests.append({"method": method, "url": url, "json": kwargs["json"]})

    client.get_ticket = fake_get_ticket  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    await client.post_answer(ticket_id=5, text="hello", subject="Answer", internal=False)

    payload = requests[0]["json"]
    assert payload["type"] == "whatsapp message"
    assert payload["sender"] == "Agent"
    assert payload["internal"] is False
    assert payload["from"] == "Lamah Sanad (+218 93-4363964)"
    assert payload["to"] == "Asma (+218910055964)"


def test_latest_customer_article_ignores_old_and_system_articles() -> None:
    """Follow-up tickets should select the newest public customer article."""
    ticket = ZammadTicket(
        id=5,
        articles=[
            ZammadArticle(
                id=1,
                ticket_id=5,
                text="السلام عليكم",
                sender="Customer",
                type="whatsapp message",
                created_at=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
            ),
            ZammadArticle(
                id=2,
                ticket_id=5,
                text="System reminder",
                sender="System",
                type="whatsapp message",
                created_at=datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc),
            ),
            ZammadArticle(
                id=3,
                ticket_id=5,
                text="كيف يمكن طلب اسم المرسل",
                sender="Customer",
                type="whatsapp message",
                created_at=datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
            ),
        ],
    )

    article = ticket.latest_customer_article()

    assert article is not None
    assert article.id == 3
    assert article.text == "كيف يمكن طلب اسم المرسل"
