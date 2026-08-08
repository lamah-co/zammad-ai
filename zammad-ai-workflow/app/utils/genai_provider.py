"""Provider factory helpers for creating LangChain chat model instances."""

from __future__ import annotations

from logging import Logger, getLogger
from typing import Literal

from langchain_core.embeddings import Embeddings

from app.settings.genai import GenAIAnthropicSettings, GenAIGeminiSettings, GenAIOpenAISettings, GenAIProviderSettings

logger: Logger = getLogger("zammad-ai.genai_provider")


def _openai_role_config(genai_settings: GenAIOpenAISettings, role: Literal["triage", "answer", "judge"]):
    """Return openai model name, temperature and reasoning config for a specific role."""
    if role == "triage":
        model = genai_settings.triage_model or genai_settings.chat_model
        temp = genai_settings.triage_temperature
        reasoning = genai_settings.triage_reasoning_config
    elif role == "answer":
        model = genai_settings.answer_model or genai_settings.chat_model
        temp = genai_settings.answer_temperature
        reasoning = genai_settings.answer_reasoning_config
    elif role == "judge":
        model = genai_settings.judge_model or genai_settings.chat_model
        temp = genai_settings.judge_temperature
        reasoning = genai_settings.judge_reasoning_config
    else:
        raise ValueError(f"Unknown role: {role}")
    return model, temp, genai_settings.max_retries, reasoning


def _anthropic_role_config(genai_settings: GenAIAnthropicSettings, role: Literal["triage", "answer", "judge"]):
    """Return anthropic model name, temperature and reasoning config for a specific role."""
    if role == "triage":
        model = genai_settings.triage_model or genai_settings.chat_model
        temp = genai_settings.triage_temperature
        thinking = genai_settings.triage_thinking
        effort = genai_settings.triage_effort
    elif role == "answer":
        model = genai_settings.answer_model or genai_settings.chat_model
        temp = genai_settings.answer_temperature
        thinking = genai_settings.answer_thinking
        effort = genai_settings.answer_effort
    elif role == "judge":
        model = genai_settings.judge_model or genai_settings.chat_model
        temp = genai_settings.judge_temperature
        thinking = genai_settings.judge_thinking
        effort = genai_settings.judge_effort
    else:
        raise ValueError(f"Unknown role: {role}")
    return model, temp, genai_settings.max_retries, thinking, effort


def _gemini_role_config(genai_settings: GenAIGeminiSettings, role: Literal["triage", "answer", "judge"]):
    """Return model and native Gemini thinking controls for one workflow role."""
    if role == "triage":
        model = genai_settings.triage_model or genai_settings.chat_model
        temp = genai_settings.triage_temperature
    elif role == "answer":
        model = genai_settings.answer_model or genai_settings.chat_model
        temp = genai_settings.answer_temperature
    elif role == "judge":
        model = genai_settings.judge_model or genai_settings.chat_model
        temp = genai_settings.judge_temperature
    else:
        raise ValueError(f"Unknown role: {role}")

    return (
        model,
        temp,
        getattr(genai_settings, f"{role}_thinking_level"),
        getattr(genai_settings, f"{role}_thinking_budget"),
    )


def get_chat_model(genai_settings: GenAIProviderSettings, role: Literal["triage", "answer", "judge"]):
    """Construct a LangChain chat model instance for the configured SDK."""
    match genai_settings.sdk:
        case "openai":
            model_name, temperature, max_retries, reasoning = _openai_role_config(genai_settings, role)
            try:
                from langchain_openai import ChatOpenAI

                chat = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_retries=max_retries,
                    reasoning=reasoning,
                )
                return chat
            except ImportError:
                logger.error("langchain_openai is required for sdk 'openai'", exc_info=True)
                raise
        case "anthropic":
            model_name, temperature, max_retries, thinking, effort = _anthropic_role_config(genai_settings, role)
            try:
                from langchain_anthropic import ChatAnthropic

                chat = ChatAnthropic(
                    model_name=model_name,
                    temperature=temperature,
                    max_retries=max_retries,
                    thinking=thinking.model_dump() if thinking is not None else None,
                    effort=effort,
                )
                return chat
            except ImportError:
                logger.error("langchain_anthropic and anthropic SDK are required for sdk 'anthropic'", exc_info=True)
                raise
        case "gemini":
            model_name, temperature, thinking_level, thinking_budget = _gemini_role_config(genai_settings, role)
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                kwargs = {
                    "model": model_name,
                    "temperature": temperature,
                    "max_retries": genai_settings.max_retries,
                    "include_thoughts": genai_settings.include_thoughts,
                    "vertexai": False,
                }
                if genai_settings.base_url is not None:
                    kwargs["base_url"] = genai_settings.base_url
                if thinking_level is not None:
                    kwargs["thinking_level"] = thinking_level
                if thinking_budget is not None:
                    kwargs["thinking_budget"] = thinking_budget
                return ChatGoogleGenerativeAI(**kwargs)
            except ImportError:
                logger.error("langchain_google_genai is required for sdk 'gemini'", exc_info=True)
                raise
        case _:
            raise ValueError(f"Unsupported GenAI SDK: {genai_settings.sdk}")


def get_embedding_model(genai_settings: GenAIProviderSettings, vector_dimension: int) -> Embeddings:
    """Construct a provider-native embedding model with an explicit output size."""
    match genai_settings.sdk:
        case "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            if genai_settings.base_url is not None:
                return GoogleGenerativeAIEmbeddings(
                    model=genai_settings.embedding_model,
                    output_dimensionality=vector_dimension,
                    base_url=genai_settings.base_url,
                    vertexai=False,
                )
            return GoogleGenerativeAIEmbeddings(
                model=genai_settings.embedding_model,
                output_dimensionality=vector_dimension,
                vertexai=False,
            )
        case "openai" | "anthropic":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=genai_settings.embedding_model,
                dimensions=vector_dimension,
                max_retries=genai_settings.max_retries,
            )
        case _:
            raise ValueError(f"Unsupported GenAI SDK for embeddings: {genai_settings.sdk}")
