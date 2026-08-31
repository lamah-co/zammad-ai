"""Provider contract tests for index-job embeddings."""

import sys
import types

from job.settings.genai import GenAIGeminiSettings, GenAIOpenAISettings


def _make_fake_module(name: str, class_name: str):
    module = types.ModuleType(name)

    class FakeEmbeddings:
        def __init__(self, *args, **kwargs):
            self._init_args = args
            self._init_kwargs = kwargs

    setattr(module, class_name, FakeEmbeddings)
    return module


def _make_forbidden_module(name: str, class_name: str):
    module = types.ModuleType(name)

    class ForbiddenProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError(f"{class_name} must not be used for this provider")

    setattr(module, class_name, ForbiddenProvider)
    return module


def test_get_embedding_model_gemini(monkeypatch) -> None:
    """The index job must use Gemini's native embedding adapter."""
    fake_mod = _make_fake_module("langchain_google_genai", "GoogleGenerativeAIEmbeddings")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _make_forbidden_module("langchain_openai", "OpenAIEmbeddings"),
    )

    from job.utils.genai_provider import get_embedding_model

    settings = GenAIGeminiSettings(embedding_model="gemini-embedding-001")
    model = get_embedding_model(settings, vector_dimension=768)

    assert getattr(model, "_init_kwargs") == {
        "model": "gemini-embedding-001",
        "output_dimensionality": 768,
        "vertexai": False,
    }


def test_gemini_settings_use_native_model_defaults() -> None:
    """Selecting Gemini must not inherit OpenAI model defaults."""
    settings = GenAIGeminiSettings()

    assert settings.chat_model == "gemini-2.5-flash"
    assert settings.embedding_model == "gemini-embedding-001"


def test_get_embedding_model_openai(monkeypatch) -> None:
    """The index job must preserve native OpenAI embedding configuration."""
    fake_mod = _make_fake_module("langchain_openai", "OpenAIEmbeddings")
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    from job.utils.genai_provider import get_embedding_model

    settings = GenAIOpenAISettings(embedding_model="text-embedding-test")
    model = get_embedding_model(settings, vector_dimension=1536)

    assert getattr(model, "_init_kwargs") == {
        "model": "text-embedding-test",
        "dimensions": 1536,
        "max_retries": 3,
    }
