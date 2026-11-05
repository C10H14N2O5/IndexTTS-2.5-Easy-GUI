"""Lightweight transactional export tests; no models, CUDA or ML imports required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import merge_lora_to_pth as exporter


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="export-transaction-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "merged.pth"
        self.report = self.output.with_suffix(".merge.json")

    @staticmethod
    def write_pair(model, report):
        model.write_bytes(b"complete verified checkpoint")
        report.write_text(json.dumps({"output_sha256": exporter.digest(model)}), encoding="utf-8")

    def test_publish_complete_pair_and_remove_only_staging(self):
        with exporter.export_staging(self.output) as (model, report):
            folder = model.parent
            self.write_pair(model, report)
            identities = exporter._file_identity(model), exporter._file_identity(report)
            exporter.publish_export(model, report, self.output)
            self.assertEqual(exporter._file_identity(self.output), identities[0])
            self.assertEqual(exporter._file_identity(self.report), identities[1])
        self.assertFalse(folder.exists())
        self.assertEqual(self.output.read_bytes(), b"complete verified checkpoint")
        self.assertEqual(json.loads(self.report.read_text())["output_sha256"], exporter.digest(self.output))

    def test_existing_model_is_never_overwritten(self):
        self.output.write_bytes(b"other user's model")
        with self.assertRaises(FileExistsError), exporter.export_staging(self.output):
            self.fail("Should not yield a staging area")
        self.assertEqual(self.output.read_bytes(), b"other user's model")
        self.assertFalse(self.report.exists())

    def test_existing_report_is_never_overwritten(self):
        self.report.write_bytes(b"other user's report")
        with self.assertRaises(FileExistsError), exporter.export_staging(self.output):
            self.fail("Should not yield a staging area")
        self.assertEqual(self.report.read_bytes(), b"other user's report")
        self.assertFalse(self.output.exists())

    def test_concurrent_model_creation_rolls_back_own_report_only(self):
        original_link = exporter._publish_no_replace
        def link(source, destination):
            if Path(destination) == self.output:
                self.output.write_bytes(b"concurrent export")
            return original_link(source, destination)
        with exporter.export_staging(self.output) as (model, report):
            self.write_pair(model, report)
            with patch.object(exporter, "_publish_no_replace", side_effect=link), self.assertRaises(FileExistsError):
                exporter.publish_export(model, report, self.output)
        self.assertEqual(self.output.read_bytes(), b"concurrent export")
        self.assertFalse(self.report.exists())

    def test_concurrent_report_creation_preserves_both_foreign_and_staging_data(self):
        original_link = exporter._publish_no_replace
        def link(source, destination):
            if Path(destination) == self.report:
                self.report.write_bytes(b"concurrent report")
            return original_link(source, destination)
        with exporter.export_staging(self.output) as (model, report):
            self.write_pair(model, report)
            with patch.object(exporter, "_publish_no_replace", side_effect=link), self.assertRaises(FileExistsError):
                exporter.publish_export(model, report, self.output)
            self.assertEqual(model.read_bytes(), b"complete verified checkpoint")
            self.assertTrue(report.is_file())
        self.assertEqual(self.report.read_bytes(), b"concurrent report")
        self.assertFalse(self.output.exists())

    def test_failed_publication_does_not_remove_replacement_report(self):
        original_link = exporter._publish_no_replace
        def link(source, destination):
            if Path(destination) == self.output:
                self.report.unlink()
                self.report.write_bytes(b"foreign replacement")
                raise OSError("simulated model publication failure")
            return original_link(source, destination)
        with exporter.export_staging(self.output) as (model, report):
            self.write_pair(model, report)
            with patch.object(exporter, "_publish_no_replace", side_effect=link), self.assertRaisesRegex(OSError, "publication failure"):
                exporter.publish_export(model, report, self.output)
        self.assertEqual(self.report.read_bytes(), b"foreign replacement")
        self.assertFalse(self.output.exists())

    def test_report_write_failure_leaves_no_export_or_foreign_partial_damage(self):
        old_partial = self.output.with_suffix(".pth.partial")
        old_partial.write_bytes(b"another task's old partial")
        with self.assertRaisesRegex(OSError, "report disk failure"):
            with exporter.export_staging(self.output) as (model, report):
                folder = model.parent
                model.write_bytes(b"complete model")
                report.write_bytes(b"incomplete json")
                raise OSError("report disk failure")
        self.assertFalse(folder.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual(old_partial.read_bytes(), b"another task's old partial")

    def test_unique_staging_for_simultaneous_attempts(self):
        with exporter.export_staging(self.output) as first:
            with exporter.export_staging(self.output) as second:
                self.assertNotEqual(first[0].parent, second[0].parent)
                self.assertEqual(first[0].parent.parent, self.output.parent)
                self.assertEqual(second[0].parent.parent, self.output.parent)
                self.write_pair(*first)
                self.write_pair(*second)
                exporter.publish_export(*first, self.output)
                with self.assertRaises(FileExistsError):
                    exporter.publish_export(*second, self.output)
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.report.is_file())

    def test_empty_model_or_invalid_report_cannot_publish(self):
        for model_bytes, report_text in ((b"", "{}"), (b"model", ""), (b"model", "{broken"), (b"model", "[]")):
            with self.subTest(model=model_bytes, report=report_text), exporter.export_staging(self.output) as (model, report):
                model.write_bytes(model_bytes)
                report.write_text(report_text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    exporter.publish_export(model, report, self.output)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.report.exists())

    def test_broken_destination_link_is_reserved(self):
        try:
            self.output.symlink_to(self.root / "missing.pth")
        except OSError:
            self.skipTest("Creating symlinks requires Windows developer mode or privilege")
        with self.assertRaises(FileExistsError), exporter.export_staging(self.output):
            self.fail("Broken symlink must not be replaced")
        self.assertTrue(self.output.is_symlink())

    def test_config_and_expected_base_mutation_are_detected(self):
        sources = [self.root / name for name in ("base.pth", "config.yaml", "adapter_model.safetensors",
                                                "adapter_config.json", "expected_base.pth")]
        for source in sources:
            source.write_bytes(b"original")
        hashes = {source: exporter.digest(source) for source in sources}
        exporter.verify_source_hashes(hashes)
        for source in sources:
            with self.subTest(source=source.name):
                source.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "source file changed"):
                    exporter.verify_source_hashes(hashes)
                source.write_bytes(b"original")
        sources[1].unlink()
        with self.assertRaisesRegex(RuntimeError, "config.yaml"):
            exporter.verify_source_hashes(hashes)

    def test_import_and_help_do_not_load_ml_runtime(self):
        project_root = Path(__file__).resolve().parents[1]
        code = ("import sys;from tools import merge_lora_to_pth;"
                "assert not any(name in sys.modules for name in ('torch','peft','transformers','safetensors'));"
                "merge_lora_to_pth.main(['--help'])")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=project_root,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"--expected-base", result.stdout)


if __name__ == "__main__":
    unittest.main()
