"""Post-hoc checkpoint validation through the real API with the fake engine."""
from __future__ import annotations

import json
import math
import shutil
import threading

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths

CKPT2, CKPT4, CKPT6 = "checkpoint-step-000002", "checkpoint-step-000004", "checkpoint-step-000006"


def fake_loss(step):
    return 0.9 + 1.6 * math.exp(-step / 300.0)


class PosthocApiTests(GuiApiTestCase):
    def setUp(self):
        super().setUp()
        audio = self.audio("corpus/a.wav", .6)
        audio.with_suffix(".txt").write_text("training text", encoding="utf-8")
        self.assertEqual(self.client.post("/api/projects", json={"name": "post", "source_folder": str(audio.parent)}).status_code, 201)
        response = self.client.put("/api/projects/post/dataset", json={"prepare": True})
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.put("/api/projects/post/training", json={
            "adapter_name": "post", "training_steps": 6, "save_steps": 2, "eval_enabled": False})
        self.assertEqual(response.status_code, 200, response.text)
        self.validation_dir = paths.VALIDATION_DIR / "post"
        response = self.client.put("/api/projects/post/validation", json={"source_folder": str(self.validation_dir)})
        self.assertEqual(response.status_code, 200, response.text)
        self.add_validation_clip("v1", .9)
        self.prepare_validation()
        self.out = self.root / "training/outputs/post"
        self.calls = []
        fake = self.state.extras["posthoc_fn"]

        def recording(model_dir, targets, rows, **kwargs):
            self.calls.append([(target["name"], target.get("step")) for target in targets])
            return fake(model_dir, targets, rows, **kwargs)
        self.state.extras["posthoc_fn"] = recording

    # ------------------------------------------------------------------ helpers
    def add_validation_clip(self, stem, seconds):
        clip = self.audio(f"validation/post/{stem}.wav", seconds)
        clip.with_suffix(".txt").write_text(f"validation text {stem}", encoding="utf-8")
        return clip

    def prepare_validation(self, **values):
        response = self.client.put("/api/projects/post/validation", json={"prepare": True, "enabled": True, **values})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def train(self, **overrides):
        response = self.client.post("/api/training/start", json={"project": "post", **overrides})
        self.assertEqual(response.status_code, 202, response.text)
        self.wait_until(lambda: not self.state.training.snapshot()["running"], timeout=30)
        self.state.training.thread.join(10)
        status = self.client.get("/api/training/status").json()
        self.assertEqual(status["status"], "succeeded", status)
        return status

    def results(self, adapter=None):
        params = {"adapter": adapter} if adapter else {}
        response = self.client.get("/api/projects/post/validation-results", params=params)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def items(self):
        return {item["name"]: item for item in self.results()["items"]}

    def run_job(self, expected="succeeded", **body):
        response = self.client.post("/api/projects/post/validation-runs", json=body)
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()["job"]
        self.assertEqual(job["kind"], "validation_posthoc")
        self.assertTrue(job["gpu"])
        final = self.wait_job(job["id"], timeout=30)
        self.assertEqual(final["state"], expected, final)
        return job, final

    def rejected(self, status, **body):
        response = self.client.post("/api/projects/post/validation-runs", json=body)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()["error"]["message"]

    def saved_entries(self):
        return json.loads((self.out / "validation_posthoc.json").read_text(encoding="utf-8"))["entries"]

    def posthoc_series(self):
        metrics = self.client.get("/api/projects/post/metrics").json()
        return next((series for series in metrics["series"] if series["label"] == "validation_posthoc/loss"), None)

    # -------------------------------------------------------------------- tests
    def test_adapters_listing(self):
        self.assertEqual(self.client.get("/api/projects/post/adapters").json(), [])
        self.train()
        (self.root / "training/outputs/empty_folder").mkdir(parents=True)
        other = self.root / "training/outputs/another/checkpoint-step-000001"
        other.mkdir(parents=True)
        (other / "adapter_config.json").write_text("{}", encoding="utf-8")
        stray = self.root / "training/outputs/another/notes"
        stray.mkdir()
        (stray / "adapter_config.json").write_text("{}", encoding="utf-8")
        response = self.client.get("/api/projects/post/adapters")
        self.assertEqual(response.status_code, 200, response.text)
        listing = {item["name"]: item for item in response.json()}
        self.assertEqual(set(listing), {"another", "post"})
        self.assertEqual((listing["post"]["checkpoints"], listing["post"]["has_final"],
                          listing["post"]["is_project_adapter"]), (3, True, True))
        self.assertEqual((listing["another"]["checkpoints"], listing["another"]["has_final"],
                          listing["another"]["is_project_adapter"]), (1, False, False))
        self.assertEqual(self.client.get("/api/projects/missing/adapters").status_code, 404)

    def test_run_results_metrics_tensorboard_and_recompute(self):
        before = self.results()
        self.assertEqual(before["items"], [])
        self.assertTrue(before["validation"]["ready"], before)
        self.assertEqual(before["validation"]["samples"], 1)
        self.assertIsNone(before["training_fingerprint"])
        self.train()
        fingerprint = self.results()["validation"]["fingerprint"]
        self.assertEqual(self.results()["training_fingerprint"], fingerprint)
        self.assertIsNone(self.posthoc_series())

        job, final = self.run_job(checkpoints=[CKPT4, CKPT2], include_baseline=True)
        self.assertEqual(job["meta"]["targets"], ["base_model", CKPT2, CKPT4])
        self.assertEqual(final["result"]["computed"], ["base_model", CKPT2, CKPT4])
        self.assertFalse(final["result"]["stopped"])
        self.assertEqual(self.calls, [[("base_model", 0), (CKPT2, 2), (CKPT4, 4)]])
        items = self.items()
        self.assertEqual(set(items), {"base_model", CKPT2, CKPT4})
        self.assertEqual(items["base_model"]["kind"], "baseline")
        self.assertEqual(items[CKPT2]["kind"], "checkpoint")
        self.assertAlmostEqual(items["base_model"]["loss"], 2.9)
        self.assertAlmostEqual(items[CKPT2]["loss"], fake_loss(2))
        self.assertAlmostEqual(items[CKPT4]["loss"], fake_loss(4))
        self.assertTrue(all(item["status"] == "current" and item["samples"] == 1
                            and item["validation_fingerprint"] == fingerprint for item in items.values()))
        entries = self.saved_entries()
        self.assertEqual([entry["name"] for entry in entries], ["base_model", CKPT2, CKPT4])
        self.assertIsNone(entries[0]["weights"])
        self.assertEqual(len(entries[1]["weights"]), 2)

        series = self.posthoc_series()
        self.assertIsNotNone(series)
        self.assertEqual([point[0] for point in series["points"]], [0, 2, 4])
        self.assertAlmostEqual(series["points"][0][1], 2.9)
        labels = {s["label"]: s for s in self.client.get("/api/projects/post/metrics").json()["series"]}
        self.assertIn("train/loss_ema", labels)
        # The training-time validation curve is not mixed with the post-hoc run.
        self.assertEqual([point[0] for point in labels["validation/loss"]["points"]], [2, 4, 6])
        events = list((self.out / "tensorboard/posthoc").glob("events.out.tfevents.*"))
        self.assertEqual(len(events), 1)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        log = EventAccumulator(str(self.out / "tensorboard/posthoc")).Reload()
        self.assertEqual([point.step for point in log.Scalars("validation/loss")], [0, 2, 4])

        # Everything selected is current: nothing to do without Recompute.
        message = self.rejected(400, checkpoints=[CKPT2, CKPT4], include_baseline=True)
        self.assertIn("Recompute", message)
        computed_at = items[CKPT2]["computed_at"]
        _, final = self.run_job(checkpoints=[CKPT2], recompute=True)
        self.assertEqual(final["result"]["computed"], [CKPT2])
        self.assertGreater(self.items()[CKPT2]["computed_at"], computed_at)
        self.assertEqual(len(self.saved_entries()), 3)
        # A partially current selection only evaluates the missing checkpoint.
        job, _ = self.run_job(checkpoints=[CKPT2, CKPT6])
        self.assertEqual(job["meta"]["targets"], [CKPT6])
        self.assertEqual(self.calls[-1], [(CKPT6, 6)])
        # Only one event file survives each rewrite.
        self.assertEqual(len(list((self.out / "tensorboard/posthoc").glob("events.out.tfevents.*"))), 1)
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [0, 2, 4, 6])

    def test_baseline_only_and_unknown_adapter_results(self):
        self.train()
        _, final = self.run_job(include_baseline=True)
        self.assertEqual(final["result"]["computed"], ["base_model"])
        self.assertEqual(self.rejected(400, include_baseline=True).count("Recompute"), 1)
        empty = self.results(adapter="unknown_adapter")
        self.assertEqual((empty["adapter"], empty["items"]), ("unknown_adapter", []))
        self.assertEqual(self.client.get("/api/projects/post/validation-results",
                                         params={"adapter": "../escape"}).status_code, 400)

    def test_final_adapter_reuses_identical_checkpoint(self):
        self.train()
        self.assertEqual((self.out / "final_adapter/adapter_model.safetensors").read_bytes(),
                         (self.out / CKPT6 / "adapter_model.safetensors").read_bytes())
        _, final = self.run_job(checkpoints=["final_adapter", CKPT6])
        self.assertEqual(self.calls, [[(CKPT6, 6)]])
        self.assertEqual(sorted(final["result"]["computed"]), sorted([CKPT6, "final_adapter"]))
        items = self.items()
        self.assertEqual(items["final_adapter"]["kind"], "final")
        self.assertEqual(items["final_adapter"]["same_as"], CKPT6)
        self.assertEqual(items["final_adapter"]["step"], 6)
        self.assertEqual(items["final_adapter"]["loss"], items[CKPT6]["loss"])
        self.assertEqual(items["final_adapter"]["status"], "current")
        self.assertIsNone(items[CKPT6]["same_as"])
        # One point per step: the checkpoint wins over the identical final adapter.
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [6])
        # Recompute of the final adapter alone still reuses the current twin result.
        self.calls.clear()
        _, final = self.run_job(checkpoints=["final_adapter"], recompute=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(final["result"]["computed"], ["final_adapter"])
        self.assertEqual(self.items()["final_adapter"]["same_as"], CKPT6)

    def test_final_adapter_without_current_twin_is_evaluated_at_its_step(self):
        self.train()
        _, final = self.run_job(checkpoints=["final_adapter"])
        self.assertEqual(self.calls, [[("final_adapter", 6)]])
        item = self.items()["final_adapter"]
        self.assertEqual((item["step"], item["same_as"]), (6, None))
        self.assertAlmostEqual(item["loss"], fake_loss(6))

    def test_distinct_final_adapter_uses_sidecar_step(self):
        self.train()
        (self.out / "final_adapter/adapter_model.safetensors").write_bytes(b"DIFFERENT")
        self.run_job(checkpoints=["final_adapter", CKPT6])
        self.assertEqual(self.calls, [[(CKPT6, 6), ("final_adapter", 6)]])
        self.assertIsNone(self.items()["final_adapter"]["same_as"])

    def test_best_adapter_reuses_identical_checkpoint_and_evaluates_when_distinct(self):
        status = self.train(training={"auto_stop_enabled": True, "auto_stop_grace_steps": 0})
        self.assertFalse(status["auto_stopped"])
        listing = {item["name"]: item for item in self.client.get("/api/projects/post/checkpoints").json()}
        best = listing["best_adapter"]
        step = best["step"]
        self.assertEqual((best["kind"], step), ("best", status["auto_stop"]["best_step"]))
        twin = f"checkpoint-step-{step:06d}"
        self.assertEqual((self.out / "best_adapter/adapter_model.safetensors").read_bytes(),
                         (self.out / twin / "adapter_model.safetensors").read_bytes())
        _, final = self.run_job(checkpoints=["best_adapter", twin])
        self.assertEqual(self.calls, [[(twin, step)]])
        self.assertEqual(sorted(final["result"]["computed"]), sorted([twin, "best_adapter"]))
        items = self.items()
        self.assertEqual((items["best_adapter"]["kind"], items["best_adapter"]["step"],
                          items["best_adapter"]["same_as"], items["best_adapter"]["status"]), ("best", step, twin, "current"))
        self.assertEqual(items["best_adapter"]["loss"], items[twin]["loss"])
        # One point per step: the checkpoint wins over the identical best adapter.
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [step])
        # Best adapter alone with Recompute still reuses the current twin.
        self.calls.clear()
        _, final = self.run_job(checkpoints=["best_adapter"], recompute=True)
        self.assertEqual((self.calls, final["result"]["computed"]), ([], ["best_adapter"]))
        # Distinct weights are evaluated directly at the sidecar step.
        (self.out / "best_adapter/adapter_model.safetensors").write_bytes(b"DIFFERENT BEST")
        self.assertEqual(self.items()["best_adapter"]["status"], "outdated")
        _, final = self.run_job(checkpoints=["best_adapter"])
        self.assertEqual(self.calls, [[("best_adapter", step)]])
        item = self.items()["best_adapter"]
        self.assertEqual((item["kind"], item["step"], item["same_as"], item["status"]), ("best", step, None, "current"))
        self.assertAlmostEqual(item["loss"], fake_loss(step))

    def test_best_adapter_without_twin_is_evaluated(self):
        self.train(training={"auto_stop_enabled": True, "auto_stop_grace_steps": 0})
        step = self.client.get("/api/projects/post/checkpoints").json()
        step = next(item["step"] for item in step if item["name"] == "best_adapter")
        shutil.rmtree(self.out / f"checkpoint-step-{step:06d}")
        job, _ = self.run_job(checkpoints=["best_adapter"], include_baseline=True)
        self.assertEqual(job["meta"]["targets"], ["base_model", "best_adapter"])
        self.assertEqual(self.calls, [[("base_model", 0), ("best_adapter", step)]])
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [0, step])

    def test_rebuilt_validation_manifest_marks_results_outdated(self):
        self.train()
        self.run_job(checkpoints=[CKPT2, CKPT4], include_baseline=True)
        old = self.results()["validation"]["fingerprint"]
        self.add_validation_clip("v2", 1.1)
        # Stale (folder changed, manifest not rebuilt): refused, results remain readable.
        self.assertIn("needs updating", self.rejected(400, checkpoints=[CKPT2]))
        stale = self.results()
        self.assertFalse(stale["validation"]["ready"])
        self.assertTrue(stale["validation"]["message"])
        self.prepare_validation()
        info = self.results()
        self.assertNotEqual(info["validation"]["fingerprint"], old)
        self.assertEqual(info["validation"]["samples"], 2)
        self.assertTrue(all(item["status"] == "outdated" for item in info["items"]))
        self.assertIsNone(self.posthoc_series())
        # The new manifest may be evaluated without Recompute; old results are kept.
        _, final = self.run_job(checkpoints=[CKPT2])
        self.assertEqual(final["result"]["computed"], [CKPT2])
        statuses = sorted((item["name"], item["status"]) for item in self.results()["items"])
        self.assertEqual(statuses, sorted([("base_model", "outdated"), (CKPT2, "outdated"),
                                           (CKPT4, "outdated"), (CKPT2, "current")]))
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [2])
        self.assertEqual(len(list((self.out / "tensorboard/posthoc").glob("events.out.tfevents.*"))), 1)

    def test_changed_adapter_weights_mark_result_outdated(self):
        self.train()
        self.run_job(checkpoints=[CKPT2, CKPT4])
        (self.out / CKPT2 / "adapter_model.safetensors").write_bytes(b"retrained weights of another size")
        items = self.items()
        self.assertEqual((items[CKPT2]["status"], items[CKPT4]["status"]), ("outdated", "current"))
        job, _ = self.run_job(checkpoints=[CKPT2, CKPT4])
        self.assertEqual(job["meta"]["targets"], [CKPT2])
        self.assertEqual(self.items()[CKPT2]["status"], "current")

    def test_preflight_errors(self):
        # Adapter folder does not exist yet.
        self.assertEqual(self.client.post("/api/projects/post/validation-runs",
                                          json={"include_baseline": True}).status_code, 404)
        self.assertEqual(self.client.post("/api/projects/missing/validation-runs",
                                          json={"include_baseline": True}).status_code, 404)
        self.train()
        self.assertEqual(self.client.post("/api/projects/post/validation-runs",
                                          json={"include_baseline": True, "adapter": "nope"}).status_code, 404)
        self.assertIn("checkpoint-step-999999", self.rejected(404, checkpoints=["checkpoint-step-999999"]))
        self.assertIn("Select at least one", self.rejected(400))
        self.assertIn("Select at least one", self.rejected(400, checkpoints=[], recompute=True))
        # Validation folder unset.
        response = self.client.put("/api/projects/post/validation", json={"source_folder": "", "enabled": False})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("validation folder", self.rejected(400, checkpoints=[CKPT2]))
        self.assertFalse(self.results()["validation"]["ready"])
        self.assertEqual(self.state.jobs.list(), [])

    def test_validation_manifest_never_built(self):
        self.train()
        other = paths.VALIDATION_DIR / "fresh"
        other.mkdir(parents=True)
        clip = self.audio("validation/fresh/x.wav", 1.3)
        clip.with_suffix(".txt").write_text("fresh", encoding="utf-8")
        response = self.client.put("/api/projects/post/validation", json={"source_folder": str(other)})
        self.assertEqual(response.status_code, 200, response.text)
        message = self.rejected(400, checkpoints=[CKPT2])
        self.assertTrue(message)
        self.assertEqual(self.state.jobs.list(), [])

    def test_training_overlap_is_rejected(self):
        self.train()
        response = self.client.put("/api/projects/post/validation", json={
            "source_folder": str(self.root / "corpus"), "prepare": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("same audio", self.rejected(400, checkpoints=[CKPT2]))
        # Results stay readable without the overlap check.
        self.assertTrue(self.results()["validation"]["ready"])

    def test_busy_training_and_gpu_job(self):
        self.train()
        response = self.client.post("/api/training/start", json={
            "project": "post", "training": {"adapter_name": "post_long", "training_steps": 400}})
        self.assertEqual(response.status_code, 202, response.text)
        try:
            self.assertIn("Stop training", self.rejected(409, checkpoints=[CKPT2]))
        finally:
            self.client.post("/api/training/stop")
            self.wait_until(lambda: not self.state.training.snapshot()["running"], timeout=20)
            self.state.training.thread.join(10)
        release = threading.Event()
        blocker = self.state.jobs.submit("synthesis", "Blocking GPU job", lambda ctx: release.wait(10), gpu=True)
        try:
            self.assertIn("GPU is busy", self.rejected(409, checkpoints=[CKPT2]))
        finally:
            release.set()
            self.wait_job(blocker.id)
        self.run_job(checkpoints=[CKPT2])

    def test_cancel_keeps_partial_results_and_blocks_training(self):
        self.train()
        first = threading.Event()
        release = threading.Event()
        fake = self.state.extras["posthoc_fn"]

        def gated(model_dir, targets, rows, *, on_result, **kwargs):
            def record(target, loss):
                on_result(target, loss)
                first.set()
                release.wait(10)
            return fake(model_dir, targets, rows, on_result=record, **kwargs)
        self.state.extras["posthoc_fn"] = gated
        response = self.client.post("/api/projects/post/validation-runs",
                                    json={"checkpoints": [CKPT2, CKPT4, CKPT6], "include_baseline": True})
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()["job"]
        try:
            self.assertTrue(first.wait(10))
            # The GPU lease is held while the job runs.
            self.assertEqual(self.client.post("/api/training/start", json={"project": "post"}).status_code, 409)
            cancelled = self.client.post(f"/api/jobs/{job['id']}/cancel")
            self.assertEqual(cancelled.status_code, 200, cancelled.text)
        finally:
            release.set()
        final = self.wait_job(job["id"], timeout=20)
        self.assertEqual(final["state"], "cancelled", final)
        self.assertEqual(final["result"]["computed"], ["base_model"])
        self.assertTrue(final["result"]["stopped"])
        self.assertEqual(list(self.items()), ["base_model"])
        self.assertEqual([point[0] for point in self.posthoc_series()["points"]], [0])
        self.assertEqual(len(list((self.out / "tensorboard/posthoc").glob("events.out.tfevents.*"))), 1)

    def test_cancel_skips_reused_final_adapter(self):
        self.train()
        self.run_job(checkpoints=[CKPT6])
        first = threading.Event()
        release = threading.Event()
        fake = self.state.extras["posthoc_fn"]

        def gated(model_dir, targets, rows, *, on_result, **kwargs):
            def record(target, loss):
                on_result(target, loss)
                first.set()
                release.wait(10)
            return fake(model_dir, targets, rows, on_result=record, **kwargs)
        self.state.extras["posthoc_fn"] = gated
        response = self.client.post("/api/projects/post/validation-runs",
                                    json={"checkpoints": [CKPT2, CKPT4, "final_adapter"]})
        job = response.json()["job"]
        self.assertTrue(first.wait(10))
        self.client.post(f"/api/jobs/{job['id']}/cancel")
        release.set()
        final = self.wait_job(job["id"], timeout=20)
        self.assertEqual(final["state"], "cancelled", final)
        self.assertEqual(final["result"]["computed"], [CKPT2])
        self.assertNotIn("final_adapter", self.items())

    def test_evaluation_failure_keeps_completed_results(self):
        self.train()
        broken = self.out / "checkpoint-step-fail"
        shutil.copytree(self.out / CKPT4, broken)
        (broken / "adapter_model.safetensors").write_bytes(b"FAKE-SAFETENSORS-failing")
        from indextts_training.schedule import write_adapter_state
        write_adapter_state(broken, 99, 6, 6, False)
        names = {item["name"]: item for item in self.client.get("/api/projects/post/checkpoints").json()}
        self.assertEqual(names["checkpoint-step-fail"]["step"], 99)
        _, final = self.run_job(expected="failed", checkpoints=["checkpoint-step-fail", CKPT2])
        self.assertIn("Fake validation failure", final["error"]["message"])
        self.assertEqual(list(self.items()), [CKPT2])
        # The TensorBoard run reflects the result kept from before the failure.
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        log = EventAccumulator(str(self.out / "tensorboard/posthoc")).Reload()
        self.assertEqual([point.step for point in log.Scalars("validation/loss")], [2])
        self.run_job(checkpoints=[CKPT4])
