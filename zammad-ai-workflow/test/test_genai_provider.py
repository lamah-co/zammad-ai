"""Tests for the genai provider helpers.

These tests create fake langchain provider modules and ensure the
helper `get_chat_model` constructs the expected chat model objects
and passes through configuration like reasoning effort.
"""

import sys
import types

import pytest

from app.settings.genai import GenAIAnthropicSettings, GenAIGeminiSettings, GenAIOpenAISettings


def _make_fake_module(name: str, class_name: str):
    m = types.ModuleType(name)

    class FakeChat:
        def __init__(self, *args, **kwargs):
            # record construction args for assertions
            self._init_args = args
            self._init_kwargs = kwargs

    setattr(m, class_name, FakeChat)
    return m


def _make_forbidden_module(name: str, class_name: str):
    """Create a provider module whose constructor fails if it is selected."""
    module = types.ModuleType(name)

    class ForbiddenProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError(f"{class_name} must not be used for this provider")

    setattr(module, class_name, ForbiddenProvider)
    return module


def test_get_chat_model_openai(monkeypatch):
    """Ensure an OpenAI chat model is constructed with expected kwargs."""
    # Provide fake langchain_openai module with ChatOpenAI
    fake_mod = _make_fake_module("langchain_openai", "ChatOpenAI")
    # Use monkeypatch to ensure the fake module is removed after the test
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    settings = GenAIOpenAISettings(chat_model="gpt-test")

    from app.utils.genai_provider import get_chat_model

    model = get_chat_model(settings, "answer")
    assert hasattr(model, "_init_kwargs")


def test_get_chat_model_anthropic(monkeypatch):
    """Ensure an Anthropic chat model is constructed and reasoning maps to kwargs."""
    # Provide fake langchain_anthropic module with ChatAnthropic
    fake_mod = _make_fake_module("langchain_anthropic", "ChatAnthropic")
    # Use monkeypatch to ensure the fake module is removed after the test
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_mod)

    # (leave out the unused base settings variable)

    from app.utils.genai_provider import get_chat_model

    # Also test that reasoning mapping results in kwargs when configured
    settings_with_reasoning = GenAIAnthropicSettings(chat_model="claude-test")
    model = get_chat_model(settings_with_reasoning, "answer")
    # Our fake ChatAnthropic stores init kwargs on the instance
    assert hasattr(model, "_init_kwargs")
    # Ensure model was constructed with thinking/effort when reasoning provided
    assert ("thinking" in model._init_kwargs) or ("effort" in model._init_kwargs)


def test_get_chat_model_gemini_preserves_provider_configuration(monkeypatch):
    """Gemini chat must use the native provider and role-specific settings."""
    fake_mod = _make_fake_module("langchain_google_genai", "ChatGoogleGenerativeAI")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _make_forbidden_module("langchain_openai", "ChatOpenAI"),
    )

    from app.utils.genai_provider import get_chat_model

    settings = GenAIGeminiSettings(
        chat_model="gemini-fallback",
        answer_model="gemini-answer",
        answer_thinking_budget=1024,
        include_thoughts=False,
        vertexai=False,
    )

    model = get_chat_model(settings, "answer")

    assert getattr(model, "_init_kwargs") == {
        "model": "gemini-answer",
        "temperature": 0.0,
        "max_retries": 3,
        "include_thoughts": False,
        "vertexai": False,
        "thinking_budget": 1024,
    }


def test_get_chat_model_gemini_uses_thinking_level_for_gemini_3(monkeypatch):
    """Gemini 3 thinking levels must be passed without an incompatible budget."""
    fake_mod = _make_fake_module("langchain_google_genai", "ChatGoogleGenerativeAI")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)

    from app.utils.genai_provider import get_chat_model

    settings = GenAIGeminiSettings(
        chat_model="gemini-3-flash",
        judge_thinking_level="low",
    )

    model = get_chat_model(settings, "judge")

    assert getattr(model, "_init_kwargs")["thinking_level"] == "low"
    assert "thinking_budget" not in getattr(model, "_init_kwargs")


def test_gemini_settings_reject_conflicting_thinking_controls() -> None:
    """A role cannot configure Gemini 2.5 and Gemini 3 thinking simultaneously."""
    with pytest.raises(ValueError, match="answer.*thinking"):
        GenAIGeminiSettings(
            answer_thinking_budget=1024,
            answer_thinking_level="low",
        )


def test_gemini_settings_use_native_model_defaults() -> None:
    """Selecting Gemini must not inherit OpenAI model defaults."""
    settings = GenAIGeminiSettings()

    assert settings.chat_model == "gemini-2.5-flash"
    assert settings.embedding_model == "gemini-embedding-001"


def test_get_embedding_model_gemini_uses_native_dimensions(monkeypatch):
    """Gemini embeddings must use string-native provider calls and fixed output size."""
    fake_mod = _make_fake_module("langchain_google_genai", "GoogleGenerativeAIEmbeddings")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_mod)
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _make_forbidden_module("langchain_openai", "OpenAIEmbeddings"),
    )

    from app.utils.genai_provider import get_embedding_model

    settings = GenAIGeminiSettings(embedding_model="gemini-embedding-001")
    model = get_embedding_model(settings, vector_dimension=768)

    assert getattr(model, "_init_kwargs") == {
        "model": "gemini-embedding-001",
        "output_dimensionality": 768,
    }


def test_get_embedding_model_openai_keeps_openai_dimensions(monkeypatch):
    """Existing OpenAI embedding behavior must remain unchanged."""
    fake_mod = _make_fake_module("langchain_openai", "OpenAIEmbeddings")
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    from app.utils.genai_provider import get_embedding_model

    settings = GenAIOpenAISettings(embedding_model="text-embedding-test")
    model = get_embedding_model(settings, vector_dimension=1536)

    assert getattr(model, "_init_kwargs") == {
        "model": "text-embedding-test",
        "dimensions": 1536,
        "max_retries": 3,
    }
