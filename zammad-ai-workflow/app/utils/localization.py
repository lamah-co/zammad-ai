"""Helpers for resolving customer-facing localized text."""

from app.settings.localization import LocalizationSettings, LocalizedText

REPLY_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ar": (
        "Reply language: Arabic. Write the subject and the final answer in clear Modern Standard Arabic. "
        "If the sources are in another language, translate the facts into Arabic before answering."
    ),
    "en": (
        "Reply language: English. The sources may be in Arabic. Translate the facts into English before providing "
        "the final answer, and write the subject and the final answer in English only. Keep URLs, numbers, and "
        "product names unchanged."
    ),
}


def normalize_supported_language(language: str | None, settings: LocalizationSettings) -> str | None:
    """Normalize a BCP-47 language code to a configured base language."""
    if language is None or not language.strip():
        return settings.default_language

    code = language.strip().lower()
    if code == "und":
        return settings.default_language

    base_language = code.replace("_", "-").split("-", maxsplit=1)[0]
    if base_language in settings.supported_languages:
        return base_language

    return None


def resolve_localized_text(
    value: LocalizedText | None,
    *,
    language: str | None,
    settings: LocalizationSettings,
) -> str | None:
    """Resolve a localized config value for a detected customer language."""
    if value is None:
        return None
    if isinstance(value, str):
        return value

    language_code = normalize_supported_language(language, settings) or settings.default_language
    if text := _non_empty(value.get(language_code)):
        return text
    if text := _non_empty(value.get(settings.default_language)):
        return text

    for text_value in value.values():
        if text := _non_empty(text_value):
            return text

    return None


def reply_language_instruction(language: str | None, settings: LocalizationSettings) -> str:
    """Return the generation instruction for the customer's language, or an empty string when unknown."""
    if language is None:
        return ""
    language_code = normalize_supported_language(language, settings) or settings.default_language
    return REPLY_LANGUAGE_INSTRUCTIONS.get(language_code, "")


def guess_language(text: str, settings: LocalizationSettings) -> str:
    """Guess the customer language from the script of the text, for when moderation never ran."""
    if any("\u0600" <= char <= "\u06ff" for char in text):
        language = "ar"
    elif text.strip():
        language = "en"
    else:
        return settings.default_language
    return language if language in settings.supported_languages else settings.default_language


def is_blank_localized_text(value: LocalizedText | None) -> bool:
    """Return true when a plain or localized text setting has no usable text."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return not any(_non_empty(text) for text in value.values())


def _non_empty(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
