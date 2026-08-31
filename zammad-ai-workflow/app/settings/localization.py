"""Localization settings for customer-facing workflow messages."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SupportedLanguage = Literal["ar", "en"]
UnsupportedLanguageAction = Literal["human_review", "default_language"]
LocalizedText = str | dict[str, str]


class LocalizationSettings(BaseModel):
    """Supported interaction languages and localization fallback behavior."""

    default_language: SupportedLanguage = Field(
        default="ar",
        description="Language used when a supported language cannot be confidently detected.",
    )
    supported_languages: list[SupportedLanguage] = Field(
        default_factory=lambda: ["ar", "en"],
        description="Customer languages supported by the automated workflow.",
    )
    unsupported_language_action: UnsupportedLanguageAction = Field(
        default="human_review",
        description="How to handle clearly unsupported customer languages.",
    )

    @field_validator("supported_languages")
    @classmethod
    def validate_supported_languages(cls, value: list[SupportedLanguage]) -> list[SupportedLanguage]:
        """Require at least one supported language and remove duplicate entries."""
        unique_languages = list(dict.fromkeys(value))
        if not unique_languages:
            raise ValueError("At least one supported language is required")
        return unique_languages
