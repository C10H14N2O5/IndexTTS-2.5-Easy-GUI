"""Offline checks for locale selection, settings, and stable enum values.

Run with: .venv\Scripts\python.exe -B -m unittest discover -s tests -p test_easy_gui_i18n.py
"""
import json
import tempfile
import unittest
from pathlib import Path
from string import Formatter

from tools.easy_gui_i18n import GuiI18n, read_gui_settings, update_gui_settings


class GuiLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Path(self.temp.name) / "gui_settings.json"
        self.zh = GuiI18n(self.settings)
        self.en = GuiI18n(self.settings, language="en_US")

    def test_locale_defaults_and_fallback(self):
        self.assertEqual(self.zh.language, "zh_CN")
        self.assertEqual(self.zh.t("Status"), "状态")
        self.assertEqual(self.en.t("Status"), "Status")
        self.assertEqual(self.zh.t("Unknown {0}", "测试"), "Unknown 测试")
        update_gui_settings(self.settings, {"ui_language": "en_US"})
        self.assertEqual(GuiI18n(self.settings).language, "en_US")
        self.assertEqual(GuiI18n(self.settings, language="invalid").language, "zh_CN")

    def test_settings_updates_preserve_other_fields(self):
        original = {"use_bf16": True, "use_flash_accel": False, "custom": {"preserve": 1}}
        update_gui_settings(self.settings, original)
        update_gui_settings(self.settings, {"ui_language": "en_US"})
        update_gui_settings(self.settings, {"use_bf16": False})
        stored = read_gui_settings(self.settings)
        self.assertEqual(stored["ui_language"], "en_US")
        self.assertEqual(stored["custom"], original["custom"])
        self.assertIs(stored["use_bf16"], False)
        self.assertIs(stored["use_flash_accel"], False)

    def test_enum_values_and_paths_are_unchanged(self):
        values = ["Chinese", "Japanese", "Auto-detect", "Steps", "Epochs", "Emotion vector", "Paragraph/Sentence Auto"]
        for locale in (self.zh, self.en):
            self.assertEqual([v for _, v in locale.choices(values)], values)
            path = r"D:\角色\Chinese\checkpoint-step-100"
            self.assertEqual(locale.choices([("None", "__NONE__"), ("User 名称", path)])[1], ("User 名称", path))
        self.assertEqual(self.zh.choices(["Chinese"]), [("中文", "Chinese")])

    def test_formatted_messages_preserve_numeric_specs_and_details(self):
        self.assertEqual(self.zh.t("{0}h {1:02d}m", 1, 2), "1 小时 02 分钟")
        message = "Training step 12/1,000 | epoch 2 | lr 4.00e-05"
        self.assertEqual(self.zh.backend_message(message), "训练步数 12/1,000 | 轮次 2 | 学习率 4.00e-05")
        self.assertEqual(self.en.backend_message(message), message)
        checkpoint = "Saved checkpoint: 检查点 | Eval: preview.wav"
        self.assertEqual(self.zh.backend_message(checkpoint), "已保存检查点：检查点 | 试听：preview.wav")
        detail = r"ValueError: D:\目录\中文.wav {raw} <tag>"
        self.assertTrue(self.zh.backend_message("Training failed: RuntimeError: " + detail).endswith(detail))
        self.assertEqual(self.zh.backend_message(detail), detail)

    def test_translation_placeholders_are_compatible(self):
        formatter = Formatter()
        for source, translated in self.zh.translations.items():
            original_fields = {name: (spec, conversion) for _, name, spec, conversion in formatter.parse(source) if name is not None}
            actual_fields = {name: (spec, conversion) for _, name, spec, conversion in formatter.parse(translated) if name is not None}
            # The English plural suffix has no equivalent in Chinese.
            omitted = {"6"} if source.startswith("Generation complete.\nLanguage:") else set()
            self.assertEqual(actual_fields, {key: value for key, value in original_fields.items() if key not in omitted}, source)


if __name__ == "__main__":
    unittest.main()
