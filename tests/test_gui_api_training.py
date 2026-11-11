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
