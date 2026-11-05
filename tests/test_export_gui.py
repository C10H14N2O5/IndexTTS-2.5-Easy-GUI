"""Exercise GUI export contracts without importing the model-heavy Gradio app."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
HELPERS = {
    "checkpoint_export_adapter_info", "select_checkpoint_export_adapter",
    "checkpoint_export_mode_visibility", "browse_checkpoint_export_destination",
    "checkpoint_export_values", "checkpoint_export_snapshot",
    "start_checkpoint_export", "cancel_checkpoint_export",
}


class ExportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "easy_gui.py").read_text(encoding="utf-8"))
        cls.module = ast.Module(body=[node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in HELPERS], type_ignores=[])
        cls.code = compile(cls.module, "export_gui_helpers", "exec")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="export_gui_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "current-model"
        self.model_dir.mkdir()
        self.details = Mock(return_value={"valid": True, "filename": "voice_step100.pth",
                                         "training_base": "", "error": ""})
        self.picker = Mock(return_value=("", ""))
        self.manager = SimpleNamespace(
            start=Mock(return_value={"status": "running", "phase": "starting"}),
            snapshot=Mock(return_value={"status": "idle"}),
            cancel=Mock(return_value={"status": "cancelling", "phase": "saving"}),
        )
        self.scope = {
            "Path": Path, "MODEL_DIR": self.model_dir, "OUTPUTS_DIR": self.root / "outputs",
            "NONE_SENTINEL": "__NONE__", "EXPORT_MANAGER": self.manager,
            "adapter_details": self.details, "choose_export_path": self.picker,
            "t": lambda text, *args: text.format(*args),
            "gr": SimpleNamespace(update=lambda **kwargs: kwargs, Error=ValueError),
        }
        exec(self.code, self.scope)

    def call(self, name, *args):
        return self.scope[name](*args)

    def test_recorded_training_directory_is_metadata_not_a_base_checkpoint(self):
        historical = self.root / "historical-model-directory"
        historical.mkdir()
        (historical / "gpt.pth").write_bytes(b"historical placeholder")
        self.details.return_value["training_base"] = str(historical)
        filename, base, notice = self.call("checkpoint_export_adapter_info", "adapter")
        self.assertEqual(filename, "voice_step100.pth")
        self.assertEqual(base, str(self.model_dir / "gpt.pth"))
        self.assertIn(str(historical), notice)
        self.assertIn("does not verify", notice)
        self.assertNotIn("unavailable", notice)
        # A moved/deleted historical directory is still metadata, not an error
        # or permission to silently substitute a different model.
        self.details.return_value["training_base"] = str(self.root / "missing-old-model")
        _, missing_base, missing_notice = self.call("checkpoint_export_adapter_info", "adapter")
        self.assertEqual(missing_base, base)
        self.assertIn("does not verify", missing_notice)

    def test_adapter_selection_and_no_selection(self):
        selected = self.call("select_checkpoint_export_adapter", "chosen-adapter")
        self.assertEqual(selected[:2], ("chosen-adapter", "voice_step100.pth"))
        self.assertEqual(self.call("select_checkpoint_export_adapter", "__NONE__")[0], "")
        self.details.return_value.update(valid=False, error="Missing adapter weights")
        self.assertEqual(self.call("checkpoint_export_adapter_info", "broken")[2], "Missing adapter weights")

    def test_save_dialog_cancellation_and_failure_do_not_change_path_or_start_export(self):
        current = str(self.root / "chosen.pth")
        value, notice = self.call("browse_checkpoint_export_destination", current, "unused.pth")
        self.assertEqual(value, {})
        self.assertIn("cancelled", notice)
        self.manager.start.assert_not_called()
        self.picker.return_value = (current, "Desktop dialog unavailable")
        value, notice = self.call("browse_checkpoint_export_destination", current, "unused.pth")
        self.assertEqual(value, {})
        self.assertEqual(notice, "Desktop dialog unavailable")
        self.manager.start.assert_not_called()

    def test_start_routes_browser_filename_and_explicit_server_path(self):
        self.call("start_checkpoint_export", "adapter", "browser", "voice.pth", "ignored.pth", "base", "config", "")
        self.manager.start.assert_called_with("adapter", "browser", "voice.pth", "base", "config", "")
        destination = str(self.root / "chosen.pth")
        self.call("start_checkpoint_export", "adapter", "path", "ignored.pth", destination, "base", "config", "deployment-base")
        self.manager.start.assert_called_with("adapter", "path", destination, "base", "config", "deployment-base")
        for args in (("", "browser", "voice.pth", "", "", "", ""),
                     ("adapter", "browser", None, "", "", "", ""),
                     ("adapter", "path", "voice.pth", "", "", "", "")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.call("start_checkpoint_export", *args)

    def test_failed_cancelled_and_running_tasks_never_show_download_links(self):
        for status in ("idle", "running", "cancelling", "failed", "cancelled"):
            with self.subTest(status=status):
                rendered = self.call("checkpoint_export_values", {
                    "status": status, "phase": "saving", "task_id": "some-task", "token": "secret",
                    "output": "model.pth", "error": "Full diagnostic text" if status == "failed" else "",
                })
                self.assertEqual(rendered[3], "")
                self.assertEqual(rendered[4]["interactive"], status not in ("running", "cancelling"))
                self.assertEqual(rendered[5]["interactive"], status == "running")
                if status == "failed":
                    self.assertIn("Full diagnostic text", rendered[0])

    def test_success_links_are_scoped_encoded_and_escape_filename(self):
        rendered = self.call("checkpoint_export_values", {
            "status": "succeeded", "phase": "completed", "task_id": "id/with space",
            "token": 'token"<>', "output": 'model"><script>.pth',
        })
        links = rendered[3]
        self.assertIn("/exports/id%2Fwith%20space/model?token=token%22%3C%3E", links)
        self.assertIn("/exports/id%2Fwith%20space/report?token=token%22%3C%3E", links)
        self.assertNotIn("<script>", links)
        self.assertIn("&quot;&gt;&lt;script&gt;", links)
        self.assertEqual(links.count("<a "), 2)

    def test_success_summary_uses_report_evidence_and_does_not_invent_verification(self):
        report = {
            "tensor_count": 456, "lora_target_count": 96, "changed_tensor_count": 96,
            "base_sha256": "a" * 64, "output_sha256": "b" * 64,
            "cpu_forward_max_abs_difference": 2e-6, "source_files_unchanged": True,
            "round_trip_tensors_identical": True,
        }
        rendered = self.call("checkpoint_export_values", {"status": "succeeded", "phase": "completed", "report": report})
        self.assertIn("Validated tensors: 456", rendered[0])
        self.assertIn("LoRA targets: 96", rendered[0])
        self.assertIn("a" * 64, rendered[0])
        self.assertIn("b" * 64, rendered[0])
        self.assertIn("Source files unchanged: Yes", rendered[0])
        self.assertIn("Saved tensors identical: Yes", rendered[0])
        del report["source_files_unchanged"]
        del report["round_trip_tensors_identical"]
        partial = self.call("checkpoint_export_values", {"status": "succeeded", "report": report})
        self.assertIn("Source files unchanged: Not verified", partial[0])
        self.assertIn("Saved tensors identical: Not verified", partial[0])

    def test_stage_labels_and_chinese_translations_are_present(self):
        translations = json.loads((ROOT / "tools/i18n/locale/easy_gui_zh_CN.json").read_text(encoding="utf-8"))
        for node in ast.walk(self.module):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                self.assertIn(node.args[0].value, translations)
        for phase in ("idle", "starting", "hashing", "building", "checking", "saving", "verifying", "publishing", "completed"):
            with self.subTest(phase=phase):
                rendered = self.call("checkpoint_export_values", {"status": "running", "phase": phase, "elapsed": 1.25})
                self.assertIn("Elapsed 1.2s", rendered[0])
                self.assertNotIn(f" · {phase} · ", rendered[0])


if __name__ == "__main__":
    unittest.main()
