"""Stored validation results (``validation_posthoc.json``) shared by trainer and GUI."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from indextts_training import validation_results as vr

FP_A, FP_B = "a" * 64, "b" * 64


def entry(name, fingerprint, loss, step=None, **extra):
    return dict(name=name, step=step, loss=loss, validation_fingerprint=fingerprint, samples=1,
                computed_at=1.0, **extra)


class ValidationResultsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "adapter"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def names(self):
        return [(item["name"], item["validation_fingerprint"], item["loss"]) for item in vr.load(self.root)]

    def test_constants(self):
        self.assertEqual((vr.RESULTS_FILE, vr.BASELINE), ("validation_posthoc.json", "base_model"))

    def test_load_missing_or_corrupt_file_is_empty(self):
        self.assertEqual(vr.load(self.root), [])
        self.assertEqual(vr.load(self.root / "does-not-exist"), [])
        path = self.root / vr.RESULTS_FILE
        for content in ("{not json", "", "[1, 2]", '{"entries": null}', '"text"', '{"entries": "abc"}'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                self.assertEqual(vr.load(self.root), [])
        path.write_bytes(b"\xff\xfe\x00garbage")
        self.assertEqual(vr.load(self.root), [])
        self.assertFalse(vr.has_baseline(self.root, FP_A))

    def test_load_non_list_entries_is_empty(self):
        (self.root / vr.RESULTS_FILE).write_text('{"entries": 5}', encoding="utf-8")
        self.assertEqual(vr.load(self.root), [])

    def test_load_skips_malformed_entries(self):
        good = entry("checkpoint-step-000002", FP_A, 1.5, 2)
        (self.root / vr.RESULTS_FILE).write_text(json.dumps({"version": 1, "entries": [
            good, {"name": "no-loss"}, {"loss": 1.0}, "string", None, {"name": "text-loss", "loss": "1.0"},
            {"name": "bool-loss", "loss": True}, {"name": 7, "loss": 1.0}]}), encoding="utf-8")
        self.assertEqual(vr.load(self.root), [good])

    def test_upsert_replaces_same_name_and_fingerprint_only(self):
        vr.upsert(self.root, entry("checkpoint-step-000002", FP_A, 1.5, 2))
        vr.upsert(self.root, entry("checkpoint-step-000002", FP_B, 1.4, 2))
        vr.upsert(self.root, entry("checkpoint-step-000004", FP_A, 1.2, 4))
        vr.upsert(self.root, entry("checkpoint-step-000002", FP_A, 1.1, 2))
        self.assertEqual(self.names(), [
            ("checkpoint-step-000002", FP_A, 1.1), ("checkpoint-step-000004", FP_A, 1.2),
            ("checkpoint-step-000002", FP_B, 1.4)])
        data = json.loads((self.root / vr.RESULTS_FILE).read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)

    def test_save_orders_baseline_first_within_each_fingerprint(self):
        vr.save(self.root, [
            entry("final_adapter", FP_B, 1.0, None),
            entry("checkpoint-step-000010", FP_B, 1.1, 10),
            entry("checkpoint-step-000002", FP_B, 1.3, 2),
            entry(vr.BASELINE, FP_B, 2.9, 0),
            entry("best_adapter", FP_A, 0.9, 4),
            entry("checkpoint-step-000004", FP_A, 0.9, 4),
            entry(vr.BASELINE, FP_A, 3.0, 0),
        ])
        self.assertEqual([(item["validation_fingerprint"], item["name"]) for item in vr.load(self.root)], [
            (FP_A, vr.BASELINE), (FP_A, "best_adapter"), (FP_A, "checkpoint-step-000004"),
            (FP_B, vr.BASELINE), (FP_B, "checkpoint-step-000002"), (FP_B, "checkpoint-step-000010"),
            (FP_B, "final_adapter")])

    def test_has_baseline_matches_fingerprint(self):
        self.assertFalse(vr.has_baseline(self.root, FP_A))
        vr.upsert(self.root, entry("checkpoint-step-000002", FP_A, 1.5, 2))
        self.assertFalse(vr.has_baseline(self.root, FP_A))
        vr.upsert(self.root, entry(vr.BASELINE, FP_A, 3.0, 0))
        self.assertTrue(vr.has_baseline(self.root, FP_A))
        self.assertFalse(vr.has_baseline(self.root, FP_B))
        self.assertFalse(vr.has_baseline(self.root, None))
        self.assertFalse(vr.has_baseline(self.root, ""))

    def test_record_baseline_writes_training_entry(self):
        root = self.root / "new" / "adapter"  # created on demand
        before = time.time()
        vr.record_baseline(root, loss=3, fingerprint=FP_A, samples=2, model_dir=Path("models/x"))
        [saved] = vr.load(root)
        self.assertEqual({key: value for key, value in saved.items() if key != "computed_at"}, dict(
            name="base_model", step=0, loss=3.0, validation_fingerprint=FP_A, samples=2,
            model_dir=str(Path("models/x")), same_as=None, weights=None, source="training"))
        self.assertIsInstance(saved["loss"], float)
        self.assertGreaterEqual(saved["computed_at"], before)
        self.assertTrue(vr.has_baseline(root, FP_A))
        # Recording again for the same fingerprint replaces it; another fingerprint is kept alongside.
        vr.upsert(root, entry("checkpoint-step-000002", FP_A, 1.5, 2))
        vr.record_baseline(root, loss=2.5, fingerprint=FP_A, samples=2, model_dir="m")
        vr.record_baseline(root, loss=2.0, fingerprint=FP_B, samples=3, model_dir="m")
        self.assertEqual([(item["name"], item["validation_fingerprint"], item["loss"]) for item in vr.load(root)], [
            (vr.BASELINE, FP_A, 2.5), ("checkpoint-step-000002", FP_A, 1.5), (vr.BASELINE, FP_B, 2.0)])

    def test_concurrent_upserts_are_serialized(self):
        names = [f"checkpoint-step-{step:06d}" for step in range(1, 21)]
        threads = [threading.Thread(target=vr.upsert, args=(self.root, entry(name, FP_A, 1.0, index + 1)))
                   for index, name in enumerate(names)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(sorted(item["name"] for item in vr.load(self.root)), names)


if __name__ == "__main__":
    unittest.main()
