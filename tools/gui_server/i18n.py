"""Per-request translation of server messages.

Server code describes user-facing text as an English template plus arguments
(``Message``). Rendering happens at the HTTP/SSE boundary in the caller's
language, so a single job or status can be shown in Chinese and English to two
browsers at the same time. Templates are the keys of
``tools/i18n/locale/easy_gui_zh_CN.json`` (English source text, ``{0}`` style).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

from tools.easy_gui_i18n import GuiI18n, UI_LANGUAGES, read_gui_settings

from . import paths

LANGUAGES = tuple(code for _, code in UI_LANGUAGES)
DEFAULT_LANGUAGE = "zh_CN"
_TRANSLATORS = {code: GuiI18n(paths.SETTINGS_FILE, language=code) for code in LANGUAGES}
_CURRENT = contextvars.ContextVar("indextts_ui_language", default=None)


def normalize(language: str | None) -> str:
    language = (language or "").replace("-", "_")
    if language in _TRANSLATORS:
        return language
    if language.lower().startswith("en"):
        return "en_US"
    if language.lower().startswith("zh"):
        return "zh_CN"
    return saved_language()


def saved_language() -> str:
    value = read_gui_settings(paths.SETTINGS_FILE).get("ui_language", DEFAULT_LANGUAGE)
    return value if value in _TRANSLATORS else DEFAULT_LANGUAGE


def current_language() -> str:
    return _CURRENT.get() or saved_language()


def set_current_language(language: str | None):
    return _CURRENT.set(normalize(language) if language else None)


def reset_current_language(token) -> None:
    _CURRENT.reset(token)


def translator(language: str | None = None) -> GuiI18n:
    return _TRANSLATORS[normalize(language) if language else current_language()]


def t(source: str, *args, language: str | None = None) -> str:
    """Translate an English template now, in the current request language."""
    return translator(language).t(source, *args)


def backend(message: Any, language: str | None = None) -> str:
    """Translate a complete message produced by indextts_training modules."""
    return translator(language).backend_message(message)


@dataclass(frozen=True)
class Message:
    """A deferred, translatable message: English template plus format args."""

    template: str
    args: tuple = field(default_factory=tuple)
    raw: bool = False  # raw=True: a complete backend string, translated via templates

    def render(self, language: str | None = None) -> str:
        if self.raw:
            return backend(self.template, language)
        args = tuple(a.render(language) if isinstance(a, Message) else a for a in self.args)
        try:
            return translator(language).t(self.template, *args)
        except (IndexError, KeyError, ValueError):
            return self.template

    def __str__(self) -> str:  # English, for logs
        try:
            return self.template.format(*self.args) if self.args and not self.raw else self.template
        except (IndexError, KeyError, ValueError):
            return self.template


def msg(template: str, *args) -> Message:
    return Message(template, tuple(args))


def raw(text: Any) -> Message:
    return Message(str(text or ""), (), raw=True)


def render(value: Any, language: str | None = None) -> Any:
    """Recursively render Message objects inside JSON-like data."""
    if isinstance(value, Message):
        return value.render(language)
    if isinstance(value, dict):
        return {k: render(v, language) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [render(v, language) for v in value]
    return value
