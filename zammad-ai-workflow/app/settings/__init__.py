"""Settings models and configuration loading for Zammad AI."""

from .answer import AnswerSettings, JudgeSettings, JudgeThresholds, LawToolSettings, QdrantSettings
from .frontend import FrontendSettings
from .genai import GenAIAnthropicSettings, GenAIGeminiSettings, GenAIOpenAISettings, GenAIProviderSettings
from .guardrails import GuardrailSettings
from .kafka import KafkaSettings
from .moderation import ModerationSettings
from .preparser import PreparserSettings
from .settings import ZammadAISettings, get_settings
from .triage import TriageSettings
from .usecase import UseCaseSettings
from .zammad import BaseZammadSettings, ZammadAPISettings, ZammadEAISettings

__all__: list[str] = [
    "AnswerSettings",
    "BaseZammadSettings",
    "FrontendSettings",
    "GenAIProviderSettings",
    "GenAIGeminiSettings",
    "GenAIOpenAISettings",
    "GenAIAnthropicSettings",
    "GenAIGeminiSettings",
    "get_settings",
    "KafkaSettings",
    "ModerationSettings",
    "JudgeSettings",
    "JudgeThresholds",
    "LawToolSettings",
    "QdrantSettings",
    "TriageSettings",
    "UseCaseSettings",
    "ZammadAISettings",
    "ZammadAPISettings",
    "ZammadEAISettings",
    "GuardrailSettings",
    "PreparserSettings",
]
