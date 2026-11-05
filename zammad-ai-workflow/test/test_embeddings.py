"""Tests for knowledge-base embedding provider selection."""

import sys
from types import ModuleType

from app.settings.genai import GenAIGeminiSettings
from app.utils.genai_provider import get_embedding_model


class FakeGoogleGenerativeAIEmbeddings:
    """Capture Gemini embedding constructor arguments."""

    def __init__(self, **kwargs: object) -> None:
        """Store the constructor arguments for assertions."""
        self.kwargs = kwargs


def test_get_embedding_model_supports_gemini(monkeypatch) -> None:
    """Gemini GenAI settings should build Gemini embeddings for Qdrant."""
    fake_module = ModuleType("langchain_google_genai")
    fake_module.GoogleGenerativeAIEmbeddings = FakeGoogleGenerativeAIEmbeddings  # type: ignore
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    embeddings = get_embedding_model(
        genai_settings=GenAIGeminiSettings(embedding_model="gemini-embedding-001"),
        vector_dimension=3072,
    )

    assert isinstance(embeddings, FakeGoogleGenerativeAIEmbeddings)
    assert embeddings.kwargs == {
        "model": "gemini-embedding-001",
        "output_dimensionality": 3072,
        "vertexai": False,
    }
