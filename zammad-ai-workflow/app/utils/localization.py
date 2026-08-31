"""Helpers for resolving customer-facing localized text."""


from app.settings.localization import LocalizationSettings, LocalizedText


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
