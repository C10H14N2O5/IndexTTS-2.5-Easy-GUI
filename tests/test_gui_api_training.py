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
