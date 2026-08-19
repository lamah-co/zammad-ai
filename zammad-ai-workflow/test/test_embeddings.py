"""Tests for knowledge-base embedding provider selection."""

import sys
from types import ModuleType

from app.answer.knowledgebase import build_embeddings
from app.settings.answer import QdrantSettings
from app.settings.genai import GenAIGeminiSettings


class FakeGoogleGenerativeAIEmbeddings:
    """Capture Gemini embedding constructor arguments."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_build_embeddings_supports_gemini(monkeypatch) -> None:
    """Gemini GenAI settings should build Gemini embeddings for Qdrant."""
    fake_module = ModuleType("langchain_google_genai")
    fake_module.GoogleGenerativeAIEmbeddings = FakeGoogleGenerativeAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    embeddings = build_embeddings(
        genai_settings=GenAIGeminiSettings(embedding_model="gemini-embedding-001"),
        qdrant_settings=QdrantSettings(vector_dimension=3072),
    )

    assert isinstance(embeddings, FakeGoogleGenerativeAIEmbeddings)
    assert embeddings.kwargs == {
        "model": "gemini-embedding-001",
        "output_dimensionality": 3072,
    }
