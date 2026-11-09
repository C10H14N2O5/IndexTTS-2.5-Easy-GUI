"""Small, application-only translations for Easy GUI.

Stored values, user content and the console are deliberately left untouched.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from string import Formatter


_SETTINGS_LOCK = threading.RLock()
_FORMATTER = Formatter()
UI_LANGUAGES = [("简体中文", "zh_CN"), ("English", "en_US")]


def read_gui_settings(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def update_gui_settings(path: Path, updates: dict) -> dict:
    """Merge both language and engine updates without discarding other keys."""
    with _SETTINGS_LOCK:
        data = read_gui_settings(path)
        data.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)
        return data


class GuiI18n:
    def __init__(self, settings_file: Path, language: str | None = None):
        selected = language or read_gui_settings(settings_file).get("ui_language", "zh_CN")
        self.language = selected if selected in {"zh_CN", "en_US"} else "zh_CN"
        locale_dir = Path(__file__).resolve().parent / "i18n" / "locale"
        self.translations = json.loads((locale_dir / "easy_gui_zh_CN.json").read_text(encoding="utf-8"))
        backend_templates = json.loads((locale_dir / "easy_gui_backend_templates.json").read_text(encoding="utf-8"))
        self._backend_patterns = []
        for template in backend_templates:
            pattern = []
            fields = set()
            for literal, field, spec, conversion in _FORMATTER.parse(template):
                pattern.append(re.escape(literal))
                if field is not None:
                    # These are explicit, numbered message templates only.
                    if not field.isdecimal():
                        raise ValueError(f"Unsupported backend message placeholder: {field}")
                    if field in fields:
                        pattern.append(f"(?P=p{field})")
                    else:
                        pattern.append(f"(?P<p{field}>.*?)")
                        fields.add(field)
            if fields:
                self._backend_patterns.append((re.compile("".join(pattern), re.DOTALL), template))

    def t(self, source: str, *args, **kwargs) -> str:
        translated = self.translations.get(source, source) if self.language == "zh_CN" else source
        return translated.format(*args, **kwargs) if args or kwargs else translated

    def choices(self, values):
        """Translate enum labels; keep original values sent to Gradio callbacks."""
        return [(self.t(item[0]), item[1]) if isinstance(item, (tuple, list))
                else (self.t(item), item) if isinstance(item, str) else item for item in values]

    def backend_message(self, message) -> str:
        """Translate only known complete backend messages at the display boundary."""
        text = str(message or "")
        if self.language == "en_US":
            return text
        if text in self.translations:
            return self.translations[text]
        for pattern, source in self._backend_patterns:
            match = pattern.fullmatch(text)
            if match:
                # Captured fields are already formatted: preserve their exact bytes,
                # including paths, exception details, and numeric precision.
                result = []
                for literal, field, spec, conversion in _FORMATTER.parse(self.translations.get(source, source)):
                    result.append(literal)
                    if field is not None:
                        value = match.group(f"p{field}")
                        if source == "Training failed: {0}: {1}" and field == "1":
                            value = self.backend_message(value)
                        result.append(value)
                return "".join(result)
        return text
