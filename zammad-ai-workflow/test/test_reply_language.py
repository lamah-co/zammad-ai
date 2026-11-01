"""Tests for matching customer replies to the detected language."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.action.service import ActionService
from app.settings.localization import LocalizationSettings, missing_languages, static_reply_errors
from app.utils.localization import reply_language_instruction


def test_english_instruction_requires_translation_before_the_final_answer() -> None:
    """English customers get facts translated from the Arabic knowledge base."""
    instruction = reply_language_instruction("en", LocalizationSettings())

    assert "Translate the facts into English before providing the final answer" in instruction


def test_regional_language_codes_use_the_base_language() -> None:
    """A regional code such as ar-LY resolves to the Arabic instruction."""
    assert "Reply language: Arabic" in reply_language_instruction("ar-LY", LocalizationSettings())


def test_unsupported_language_falls_back_to_the_default_language() -> None:
    """An unsupported language uses the configured default."""
    settings = LocalizationSettings(default_language="en")

    assert "Reply language: English" in reply_language_instruction("fr", settings)


def test_unknown_language_adds_no_instruction() -> None:
    """Callers without a detected language keep the prompt unchanged."""
    assert reply_language_instruction(None, LocalizationSettings()) == ""


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"ar": "مرحبا", "en": "Hello"}, []),
        ({"ar": "مرحبا"}, ["en"]),
        ({"ar": "مرحبا", "en": "  "}, ["en"]),
        ("Hello", ["ar", "en"]),
    ],
)
def test_missing_languages(reply: str | dict[str, str], expected: list[str]) -> None:
    """Static replies must have usable text for every supported language."""
    assert missing_languages(reply, LocalizationSettings()) == expected


def test_judge_prompt_excludes_language_from_groundedness() -> None:
    """A correct translation must not reduce the groundedness score."""
    prompt = (Path(__file__).parents[1] / "prompts" / "judge" / "judge.prompt.md").read_text(encoding="utf-8")

    assert "Language is not a criterion." in prompt
    assert "a correct translation of a document counts as supported" in prompt


@pytest.mark.parametrize(
    ("language", "subject"), [("ar", "الرد على طلبك"), ("en", "Answer"), ("ar-LY", "الرد على طلبك")]
)
def test_public_subject_follows_the_customer_language(language: str, subject: str) -> None:
    """Static replies use the subject of the detected language."""
    service = ActionService.__new__(ActionService)
    service.settings = SimpleNamespace(
        triage=SimpleNamespace(public_reply_subject={"ar": "الرد على طلبك", "en": "Answer"}),
        localization=LocalizationSettings(),
    )

    assert service._public_subject(language) == subject


def test_static_reply_errors_name_each_incomplete_reply() -> None:
    """Incomplete static replies are reported so the service refuses to start."""
    errors = static_reply_errors(
        {"complete": {"ar": "مرحبا", "en": "Hello"}, "partial": {"ar": "مرحبا"}, "unset": None},
        LocalizationSettings(),
    )

    assert errors == ["partial is missing: en"]
