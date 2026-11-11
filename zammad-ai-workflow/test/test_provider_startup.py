"""Tests for starting the workflow while the GenAI embedding provider is unavailable."""

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.answer.knowledgebase as knowledgebase
from app.answer.knowledgebase import QdrantKBClient, QdrantKBError, watch_provider
from app.settings import QdrantSettings
from app.settings.genai import GenAIGeminiSettings
from app.utils.status import provider_status, set_provider_ready

DEPLETED_CREDITS = (
    "402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits are depleted.'}}"
)


class FakeEmbeddings:
    """Embeddings that fail with a provider error or answer with a fixed vector size."""

    def __init__(self, *, error: Exception | None = None, dimension: int = 3) -> None:
        """Store the behaviour for embed_query."""
        self.error = error
        self.dimension = dimension

    def embed_query(self, _text: str) -> list[float]:
        """Raise the configured error or return a vector of the configured size."""
        if self.error is not None:
            raise self.error
        return [0.0] * self.dimension


@pytest.fixture(autouse=True)
def reset_provider_status() -> Generator[None, None, None]:
    """Keep provider readiness from leaking between tests."""
    set_provider_ready(True)
    yield
    set_provider_ready(True)


@pytest.fixture
def vectorstore(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace Qdrant clients and the vector store so no network is used."""
    qdrant = MagicMock()
    qdrant.collection_exists.return_value = True
    qdrant.get_collection.return_value = SimpleNamespace(points_count=15)
    monkeypatch.setattr(knowledgebase, "QdrantClient", lambda **_kwargs: qdrant)
    monkeypatch.setattr(knowledgebase, "AsyncQdrantClient", lambda **_kwargs: MagicMock())
    store = MagicMock()
    monkeypatch.setattr(knowledgebase, "QdrantVectorStore", store)
    return store


def build_client(monkeypatch: pytest.MonkeyPatch, embeddings: FakeEmbeddings) -> QdrantKBClient:
    """Construct the client with the given fake embeddings."""
    monkeypatch.setattr(knowledgebase, "get_embedding_model", lambda *_args: embeddings)
    return QdrantKBClient(qdrant_settings=QdrantSettings(vector_dimension=3), genai_settings=GenAIGeminiSettings())


def test_provider_outage_does_not_abort_startup(monkeypatch: pytest.MonkeyPatch, vectorstore: MagicMock) -> None:
    """Depleted credits at startup leave the client usable and the provider marked unavailable."""
    build_client(monkeypatch, FakeEmbeddings(error=Exception(DEPLETED_CREDITS)))

    assert provider_status() == (False, "GenAIBillingError")
    # The vector store's own collection check embeds a test text, so it is skipped while the provider is down.
    assert vectorstore.call_args.kwargs["validate_collection_config"] is False


def test_working_provider_keeps_full_startup_validation(
    monkeypatch: pytest.MonkeyPatch, vectorstore: MagicMock
) -> None:
    """When the provider answers, startup behaves exactly as before."""
    build_client(monkeypatch, FakeEmbeddings(dimension=3))

    assert provider_status() == (True, None)
    assert vectorstore.call_args.kwargs["validate_collection_config"] is True


def test_dimension_mismatch_still_stops_startup(monkeypatch: pytest.MonkeyPatch, vectorstore: MagicMock) -> None:
    """A wrong vector size is a configuration error, not an outage, so it still fails fast."""
    with pytest.raises(QdrantKBError, match="dimension mismatch"):
        build_client(monkeypatch, FakeEmbeddings(dimension=768))


@pytest.mark.asyncio
async def test_watch_provider_marks_ready_once_the_provider_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The background check retries with growing waits and stops once the provider is back."""
    set_provider_ready(False, "GenAIBillingError")
    sleep = AsyncMock()
    monkeypatch.setattr(knowledgebase.asyncio, "sleep", sleep)
    results = iter([False, False, True])

    def verify() -> bool:
        ready = next(results)
        if ready:
            set_provider_ready(True)
        return ready

    await watch_provider(SimpleNamespace(verify_embeddings=verify))  # type: ignore

    assert provider_status() == (True, None)
    assert [call.args[0] for call in sleep.await_args_list] == [5.0, 10.0, 20.0]


def test_health_reports_degraded_while_the_provider_is_unavailable(test_client: TestClient) -> None:
    """Health returns 503 with the reason during an outage, and 200 again afterwards."""
    set_provider_ready(False, "GenAIBillingError")
    degraded = test_client.get("/api/v1/health")
    set_provider_ready(True)
    healthy = test_client.get("/api/v1/health")

    assert degraded.status_code == 503
    assert degraded.json() == {"status": "degraded", "reason": "GenAIBillingError"}
    assert healthy.status_code == 200
    assert healthy.json()["status"] == "healthy"


def test_invalid_gemini_key_is_a_credential_error() -> None:
    """Gemini's invalid-key response hands off at once instead of using up the retry budget."""
    from app.errors import GenAIAuthError, classify_provider_error

    error = Exception("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key. reason: API_KEY_INVALID")

    assert isinstance(classify_provider_error(error), GenAIAuthError)
