"""Tests for the REST API surface exposed next to the Kafka consumer."""

import pytest
from fastapi.testclient import TestClient

import app.api.v1.utils as api_utils


def test_direct_zammad_event_route_is_not_registered(test_client: TestClient) -> None:
    """Kafka is the only ticket-event ingress, so the direct webhook route must not exist."""
    response = test_client.post("/api/v1/events/zammad", json={"ticket_id": 1, "article_id": 1})

    assert response.status_code == 404


def test_health_check_does_not_require_api_key(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Container health probes must keep working when an API key is configured."""
    monkeypatch.setattr(api_utils.settings.api, "api_key", "expected-key")

    response = test_client.get("/api/v1/health")

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/triage", {"text": "Hello"}),
        ("/api/v1/answer", {"text": "Hello", "category": "General", "action": "ai_answer"}),
    ],
)
def test_rest_endpoints_reject_missing_or_wrong_api_key(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str, payload: dict[str, str]
) -> None:
    """Triage and answer endpoints must reject requests without the configured bearer key."""
    monkeypatch.setattr(api_utils.settings.api, "api_key", "expected-key")

    missing = test_client.post(path, json=payload)
    wrong = test_client.post(path, json=payload, headers={"Authorization": "Bearer wrong-key"})

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_triage_endpoint_accepts_configured_api_key(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request carrying the configured bearer key passes authentication."""
    monkeypatch.setattr(api_utils.settings.api, "api_key", "expected-key")

    response = test_client.post(
        "/api/v1/triage", json={"text": "Hello"}, headers={"Authorization": "Bearer expected-key"}
    )

    assert response.status_code != 401
