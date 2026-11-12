from __future__ import annotations
import json
from unittest.mock import patch
from gui_api_support import GuiApiTestCase


class TrainingApiTests(GuiApiTestCase):
    def prepare(self, name="trainer", **training):
        audio = self.audio("corpus/a.wav", .6)
        audio.with_suffix(".txt").write_text("training text", encoding="utf-8")
        self.assertEqual(self.client.post("/api/projects", json={"name": name, "source_folder": str(audio.parent)}).status_code, 201)
        response = self.client.put(f"/api/projects/{name}/dataset", json={"prepare": True})
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.put(f"/api/projects/{name}/training", json={"training_steps": 3, "save_steps": 1, "eval_enabled": False, **training})
        self.assertEqual(response.status_code, 200, response.text)
        return name

    def finish(self):
        self.wait_until(lambda: not self.state.training.snapshot()["running"], timeout=20)
        self.state.training.thread.join(10)
        return self.client.get("/api/training/status").json()

    def test_training_happy_seed_checkpoints_metrics_and_sse(self):
        name = self.prepare(seed=777)
        with patch.object(self.state, "publish", wraps=self.state.publish) as publish:
            response = self.client.post("/api/training/start", json={"project": name})
            self.assertEqual(response.status_code, 202, response.text)
            terminal = self.finish()
            self.assertEqual(terminal["status"], "succeeded")
            self.assertEqual(terminal["step"], 3)
            self.assertTrue(any(call.args[0] == "training" for call in publish.call_args_list))
        self.assertIn("training", self.state.providers)
        persisted = json.loads((self.root / "training/outputs/trainer/training_config.json").read_text())
        self.assertEqual(persisted["seed"], 777)
        self.assertEqual(persisted["lr"], 4e-5)
        self.assertEqual(persisted["grad_clip"], .5)
        self.assertFalse(persisted["differentiable_asr_enabled"])
        checkpoints = self.client.get(f"/api/projects/{name}/checkpoints").json()
        self.assertEqual(len(checkpoints), 4)
        self.assertIn("final", [row["kind"] for row in checkpoints])
        metrics = self.client.get(f"/api/projects/{name}/metrics", params={"max_points": 2}).json()
        self.assertTrue(metrics["available"])
        self.assertTrue(all(len(series["points"]) <= 2 for series in metrics["series"]))
        self.assertIn("train/loss_ema", [series["label"] for series in metrics["series"]])
        activity = self.client.get("/api/activity").json()
        self.assertTrue(any("Training complete" in item["message"] for item in activity["status"]))

    def test_early_stop_finishes_at_stop_step_with_planned_schedule(self):
        name = self.prepare(training_steps=10, save_steps=4, early_stop_enabled=True, early_stop_step=5)
        project = self.client.get(f"/api/projects/{name}").json()
        self.assertTrue(project["training"]["early_stop_enabled"])
        self.assertEqual(project["training"]["early_stop_step"], 5)
        self.assertEqual(self.client.post("/api/training/start", json={"project": name}).status_code, 202)
        terminal = self.finish()
        self.assertEqual(terminal["status"], "succeeded", terminal)
        self.assertEqual((terminal["step"], terminal["total"], terminal["schedule_total"]), (5, 5, 10))
        self.assertIn("finished early at step 5", terminal["message"])
        self.assertIn("LR schedule 10", terminal["message"])
        persisted = json.loads((self.root / "training/outputs/trainer/training_config.json").read_text())
        self.assertEqual((persisted["end_step"], persisted["steps"]), (5, 10))
        checkpoints = {row["name"]: row for row in self.client.get(f"/api/projects/{name}/checkpoints").json()}
        self.assertEqual(set(checkpoints), {"checkpoint-step-000004", "checkpoint-step-000005", "final_adapter"})
        self.assertTrue(checkpoints["checkpoint-step-000005"]["early_stopped"])
        self.assertFalse(checkpoints["checkpoint-step-000004"]["early_stopped"])
        self.assertTrue(checkpoints["final_adapter"]["early_stopped"])
        self.assertEqual(checkpoints["final_adapter"]["step"], 5)
        activity = self.client.get("/api/activity").json()
        self.assertTrue(any("finished early" in str(item) for item in activity["status"]))
        # Resuming the early-stopped run without early stop continues the original plan.
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "early_stop_enabled": False, "resume_checkpoint": checkpoints["checkpoint-step-000005"]["path"]}})
        self.assertEqual(response.status_code, 202, response.text)
        terminal = self.finish()
        self.assertEqual((terminal["status"], terminal["step"], terminal["total"], terminal["schedule_total"]),
                         ("succeeded", 10, 10, 10))
        self.assertIn("Training complete", terminal["message"])
        final = {row["name"]: row for row in self.client.get(f"/api/projects/{name}/checkpoints").json()}["final_adapter"]
        self.assertEqual((final["step"], final["early_stopped"]), (10, False))

    def test_early_stop_preflight_rejects_invalid_steps(self):
        name = self.prepare(training_steps=10, save_steps=4)
        for stop in (10, 11):
            with self.subTest(stop=stop):
                response = self.client.post("/api/training/start", json={"project": name, "training": {
                    "early_stop_enabled": True, "early_stop_step": stop}})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("planned 10", response.json()["error"]["message"])
        # Epoch mode plans epochs x updates (one clip -> one update per epoch).
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "mode": "Epochs", "training_epochs": 3, "early_stop_enabled": True, "early_stop_step": 3}})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("planned 3", response.json()["error"]["message"])
        # A disabled early stop ignores its step entirely.
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "training_steps": 4, "early_stop_enabled": False, "early_stop_step": 1}})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(self.finish()["step"], 4)
        checkpoint = self.root / "training/outputs/trainer/checkpoint-step-000004"
        self.assertTrue(checkpoint.is_dir())
        for stop in (3, 4):
            with self.subTest(resume_stop=stop):
                response = self.client.post("/api/training/start", json={"project": name, "training": {
                    "early_stop_enabled": True, "early_stop_step": stop, "resume_checkpoint": str(checkpoint)}})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("already at step 4", response.json()["error"]["message"])
        self.assertFalse(self.state.training.snapshot()["running"])
        # The sidecar wins over the folder name for the resume step.
        renamed = checkpoint.with_name("checkpoint-renamed")
        checkpoint.rename(renamed)
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "early_stop_enabled": True, "early_stop_step": 4, "resume_checkpoint": str(renamed)}})
        self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "early_stop_enabled": True, "early_stop_step": 6, "resume_checkpoint": str(checkpoint.with_name("checkpoint-step-000004"))}})
        self.assertEqual(response.status_code, 400, response.text)  # resume folder no longer exists

    def test_training_failure_and_stop_are_terminal_and_release_gpu(self):
        self.prepare(adapter_name="fail_training", training_steps=7)
        self.assertEqual(self.client.post("/api/training/start", json={"project": "trainer"}).status_code, 202)
        self.assertEqual(self.finish()["status"], "failed")
        self.client.put("/api/projects/trainer/training", json={"adapter_name": "stopped_training", "training_steps": 100})
        self.assertEqual(self.client.post("/api/training/start", json={"project": "trainer"}).status_code, 202)
        stop = self.client.post("/api/training/stop")
        self.assertEqual(stop.status_code, 200)
        self.assertIn(stop.json()["status"], {"stopping", "stopped"})
        terminal = self.finish()
        self.assertEqual(terminal["status"], "stopped")
        self.assertTrue(terminal["stopped"])
        from indextts_training.activity import acquire, release
        lease = acquire("test")
        release(lease)

    def test_training_conflicts_and_invalid_preflight_do_not_unload(self):
        self.prepare()
        from indextts_training.activity import acquire, release
        token = acquire("inference")
        try:
            with patch.object(self.state.training, "before_start") as unload:
                response = self.client.post("/api/training/start", json={"project": "trainer"})
                self.assertEqual(response.status_code, 409, response.text)
                unload.assert_not_called()
        finally:
            release(token)
        (self.root / "corpus/a.txt").write_text("stale source", encoding="utf-8")
        self.assertEqual(self.client.post("/api/training/start", json={"project": "trainer"}).status_code, 400)
        self.client.put("/api/projects/trainer/dataset", json={"prepare": True})
        self.assertEqual(self.client.post("/api/training/start", json={"project": "trainer", "training": {"training_steps": 100}}).status_code, 202)
        self.assertEqual(self.client.post("/api/training/start", json={"project": "trainer"}).status_code, 409)
        self.assertEqual(self.client.put("/api/projects/trainer/training", json={"seed": 2}).status_code, 409)
        job = self.state.jobs
        from tools.gui_server.errors import Busy
        with self.assertRaises(Busy):
            job.submit("synthesis", "Synthesis", lambda ctx: {}, gpu=True)
        self.client.post("/api/training/stop")
        self.finish()

    def test_validation_overlap_is_rejected(self):
        self.prepare()
        response = self.client.put("/api/projects/trainer/validation", json={"source_folder": str(self.root / "corpus"), "prepare": True, "enabled": True})
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.post("/api/training/start", json={"project": "trainer"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("same audio", response.json()["error"]["message"])

    # ---------------------------------------------------------------- auto stop
    def enable_validation(self, name="trainer"):
        from tools.gui_server import paths
        folder = paths.VALIDATION_DIR / name
        response = self.client.put(f"/api/projects/{name}/validation", json={"source_folder": str(folder)})
        self.assertEqual(response.status_code, 200, response.text)
        clip = self.audio(f"validation/{name}/v1.wav", .9)
        clip.with_suffix(".txt").write_text("validation text v1", encoding="utf-8")
        response = self.client.put(f"/api/projects/{name}/validation", json={"prepare": True, "enabled": True})
        self.assertEqual(response.status_code, 200, response.text)

    AUTO_STOP = dict(adapter_name="overfit_run", training_steps=20, save_steps=2, auto_stop_enabled=True,
                     auto_stop_patience_steps=3, auto_stop_min_delta=0.01, auto_stop_watch_interval_steps=3)

    def test_auto_stop_ends_overfitting_run_and_keeps_best_adapter(self):
        name = self.prepare(**self.AUTO_STOP)
        self.enable_validation(name)
        project = self.client.get(f"/api/projects/{name}").json()["training"]
        self.assertEqual((project["auto_stop_enabled"], project["auto_stop_patience_steps"],
                          project["auto_stop_min_delta"], project["auto_stop_watch_interval_steps"]), (True, 3, 0.01, 3))
        self.assertEqual(self.client.post("/api/training/start", json={"project": name}).status_code, 202)
        terminal = self.finish()
        # The fake U-shaped curve bottoms out at step 8 (40 % of 20); the watch starting at
        # step 10 ends at its deadline 13, before the planned 20 steps.
        self.assertEqual(terminal["status"], "succeeded", terminal)
        self.assertEqual((terminal["step"], terminal["auto_stopped"]), (13, True))
        self.assertEqual(terminal["auto_stop"]["phase"], "stopped")
        self.assertEqual((terminal["auto_stop"]["best_step"], terminal["auto_stop"]["valley_step"],
                          terminal["auto_stop"]["watch_start_step"], terminal["auto_stop"]["deadline_step"]), (8, 8, 10, 13))
        self.assertIn("stopped automatically at step 13", terminal["message"])
        self.assertIn("within 3 steps", terminal["message"])
        self.assertIn("at step 8", terminal["message"])
        persisted = json.loads((self.root / "training/outputs/overfit_run/training_config.json").read_text())
        self.assertEqual(persisted["auto_stop"], {"patience_steps": 3, "min_delta": 0.01, "watch_interval_steps": 3})
        checkpoints = {row["name"]: row for row in self.client.get(f"/api/projects/{name}/checkpoints").json()}
        self.assertEqual(set(checkpoints), {f"checkpoint-step-{step:06d}" for step in (2, 4, 6, 8, 10, 12, 13)}
                         | {"best_adapter", "final_adapter"})
        best = checkpoints["best_adapter"]
        self.assertEqual((best["kind"], best["step"], best["auto_stopped"], best["evaluation"]), ("best", 8, False, None))
        self.assertAlmostEqual(best["validation_loss"], terminal["auto_stop"]["best_loss"])
        self.assertTrue(checkpoints["checkpoint-step-000013"]["auto_stopped"])
        self.assertFalse(checkpoints["checkpoint-step-000012"]["auto_stopped"])
        self.assertIsNone(checkpoints["checkpoint-step-000012"]["validation_loss"])
        self.assertEqual((checkpoints["final_adapter"]["step"], checkpoints["final_adapter"]["auto_stopped"],
                          checkpoints["final_adapter"]["early_stopped"]), (13, True, False))
        folders = {row["name"]: row for row in self.client.get(f"/api/projects/{name}/adapters").json()}
        self.assertEqual((folders["overfit_run"]["checkpoints"], folders["overfit_run"]["has_final"],
                          folders["overfit_run"]["has_best"]), (7, True, True))
        adapters = [item for item in self.client.get("/api/adapters").json() if item["project"] == "overfit_run"]
        best_items = [item for item in adapters if item["kind"] == "best"]
        self.assertEqual(len(best_items), 1, adapters)
        self.assertEqual(best_items[0]["checkpoint"], "best_adapter")
        self.assertEqual(best_items[0]["label"], "overfit_run · best")
        self.assertEqual([item["kind"] for item in adapters][:2], ["final", "best"])
        metrics = self.client.get(f"/api/projects/{name}/metrics", params={"adapter": "overfit_run"}).json()
        labels = {series["label"]: series for series in metrics["series"]}
        self.assertEqual([point[0] for point in labels["validation/loss"]["points"]], [2, 4, 6, 8, 10, 12, 13])

        # Resuming from the auto-stop checkpoint restores the saved comparison (best at step 8).
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "resume_checkpoint": checkpoints["checkpoint-step-000013"]["path"]}})
        self.assertEqual(response.status_code, 202, response.text)
        terminal = self.finish()
        self.assertEqual(terminal["status"], "succeeded", terminal)
        self.assertTrue(terminal["auto_stopped"])
        self.assertGreater(terminal["step"], 13)
        self.assertEqual(terminal["auto_stop"]["best_step"], 8)  # restored, not reset to the resumed run
        self.assertEqual(json.loads((self.root / "training/outputs/overfit_run/best_adapter/adapter_state.json")
                                    .read_text())["step"], 8)

    def test_auto_stop_without_validation_or_with_long_watch_interval_is_rejected(self):
        name = self.prepare(auto_stop_enabled=True)
        response = self.client.post("/api/training/start", json={"project": name})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Auto stop needs validation loss", response.json()["error"]["message"])
        self.enable_validation(name)
        response = self.client.post("/api/training/start", json={"project": name, "training": {
            "auto_stop_patience_steps": 10, "auto_stop_watch_interval_steps": 11}})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("must not exceed the patience", response.json()["error"]["message"])
        self.assertFalse(self.state.training.snapshot()["running"])
        self.assertFalse((self.root / "training/outputs/trainer/training_config.json").exists())
        # Interval equal to patience is accepted; a disabled auto stop ignores its values.
        for training in ({"auto_stop_patience_steps": 10, "auto_stop_watch_interval_steps": 10},
                         {"auto_stop_enabled": False, "auto_stop_patience_steps": 1, "auto_stop_watch_interval_steps": 5}):
            with self.subTest(training=training):
                response = self.client.post("/api/training/start", json={"project": name, "training": training})
                self.assertEqual(response.status_code, 202, response.text)
                terminal = self.finish()
                self.assertEqual((terminal["status"], terminal["step"], terminal["auto_stopped"]), ("succeeded", 3, False))
        self.assertIsNone(terminal["auto_stop"])
        self.assertEqual(self.client.put(f"/api/projects/{name}/training", json={
            "auto_stop_patience_steps": 0}).status_code, 400)

    def test_tensorboard_never_opens_browser(self):
        self.prepare()
        with patch.object(self.state.training, "start_tensorboard", return_value="TensorBoard started.") as start:
            response = self.client.post("/api/training/tensorboard", json={"project": "trainer", "reload": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(start.call_args.kwargs["open_browser"])
        self.assertTrue(start.call_args.kwargs["force_restart"])
        self.assertTrue(response.json()["local_only"])
        self.assertEqual(response.json()["url"], "http://127.0.0.1:6006")

    def test_metrics_bound_and_empty(self):
        self.prepare()
        self.assertFalse(self.client.get("/api/projects/trainer/metrics").json()["available"])
        self.assertEqual(self.client.get("/api/projects/trainer/metrics?max_points=2001").status_code, 400)
        from torch.utils.tensorboard import SummaryWriter
        logdir = self.root / "training/outputs/trainer/tensorboard"
        with SummaryWriter(str(logdir)) as writer:
            for step in range(5000):
                writer.add_scalar("train/loss_ema", 1 / (step + 1), step)
        result = self.client.get("/api/projects/trainer/metrics").json()["series"][0]
        self.assertLessEqual(len(result["points"]), 2000)
        self.assertEqual(result["points"][-1][0], 4999)
