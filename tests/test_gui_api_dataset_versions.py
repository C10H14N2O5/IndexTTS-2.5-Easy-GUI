from __future__ import annotations
import json
from pathlib import Path
from gui_api_support import GuiApiTestCase


class DatasetVersionApiTests(GuiApiTestCase):
    name = "voice"

    def url(self, role="training", suffix=""):
        return f"/api/projects/{self.name}/dataset-versions/{role}{suffix}"

    def setup_project(self, validation=True):
        for index, seconds in enumerate((.6, .65)):
            audio = self.audio(f"corpus/c{index}.wav", seconds)
            audio.with_suffix(".txt").write_text(f"training text {index}", encoding="utf-8")
        self.corpus = audio.parent
        response = self.client.post("/api/projects", json={"name": self.name, "source_folder": str(self.corpus)})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(self.client.put(f"/api/projects/{self.name}/dataset", json={"prepare": True}).status_code, 200)
        if validation:
            audio = self.audio("validation/voice/v.wav", .9)
            audio.with_suffix(".txt").write_text("validation text", encoding="utf-8")
            self.validation_audio = audio
            response = self.client.put(f"/api/projects/{self.name}/validation",
                                       json={"source_folder": str(audio.parent), "enabled": True, "prepare": True})
            self.assertEqual(response.status_code, 200, response.text)
        response = self.client.put(f"/api/projects/{self.name}/training",
                                   json={"training_steps": 2, "save_steps": 1, "eval_enabled": False})
        self.assertEqual(response.status_code, 200, response.text)

    def finish(self):
        self.wait_until(lambda: not self.state.training.snapshot()["running"], timeout=20)
        self.state.training.thread.join(10)
        return self.client.get("/api/training/status").json()

    def test_freeze_list_rename_select_and_delete(self):
        self.setup_project(validation=False)
        listing = self.client.get(self.url()).json()
        self.assertEqual((listing["active"], listing["versions"], listing["next_name"]), (None, [], "v1"))
        self.assertIsNone(listing["freeze_reason"])
        frozen = self.client.post(self.url(), json={"note": "first"})
        self.assertEqual(frozen.status_code, 201, frozen.text)
        info = frozen.json()
        self.assertEqual((info["id"], info["name"], info["active"], info["integrity"]["status"]),
                         ("train-001", "v1", True, "ready"))
        self.assertEqual(info["diff"], {"added": 0, "removed": 0, "text_changed": 0})
        self.assertEqual(self.client.post(self.url(), json={}).status_code, 409)
        listing = self.client.get(self.url()).json()
        self.assertIn("identical to version v1", listing["freeze_reason"])
        project = self.client.get(f"/api/projects/{self.name}").json()
        self.assertEqual(project["dataset_version"], "train-001")
        self.assertEqual((project["training_dataset"]["version_id"], project["training_dataset"]["stale"]),
                         ("train-001", False))
        renamed = self.client.patch(self.url(suffix="/train-001"), json={"name": "基线"})
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["name"], "基线")
        self.assertEqual(self.client.patch(self.url(suffix="/train-404"), json={"name": "x"}).status_code, 404)
        self.assertEqual(self.client.get(f"/api/projects/{self.name}/dataset-versions/bogus").status_code, 400)
        selected = self.client.put(self.url(suffix="/active"), json={"version_id": None})
        self.assertEqual(selected.status_code, 200, selected.text)
        self.assertIsNone(selected.json()["training_dataset"]["version_id"])
        self.assertEqual(self.client.put(self.url(suffix="/active"), json={"version_id": "train-009"}).status_code, 404)
        deleted = self.client.delete(self.url(suffix="/train-001"))
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(self.client.get(self.url()).json()["versions"], [])

    def test_training_uses_frozen_versions_and_records_them(self):
        self.setup_project()
        self.assertEqual(self.client.post(self.url(), json={"name": "t1"}).status_code, 201)
        self.assertEqual(self.client.post(self.url("validation"), json={"name": "val1"}).status_code, 201)
        # The working sets go stale but training still uses the intact frozen versions.
        self.audio("corpus/c9.wav", .7).with_suffix(".txt").write_text("new text", encoding="utf-8")
        self.validation_audio.with_suffix(".txt").write_text("edited", encoding="utf-8")
        project = self.client.get(f"/api/projects/{self.name}").json()
        self.assertTrue(project["dataset_status"]["stale"])
        self.assertFalse(project["training_dataset"]["stale"])
        self.assertFalse(project["validation_dataset"]["stale"])
        response = self.client.post("/api/training/start", json={"project": self.name})
        self.assertEqual(response.status_code, 202, response.text)
        busy = self.client.put(self.url(suffix="/active"), json={"version_id": None})
        self.assertEqual(busy.status_code, 409)
        self.assertEqual(self.finish()["status"], "succeeded")
        saved = json.loads((self.root / f"training/outputs/{self.name}/training_config.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["dataset"]["version_id"], saved["dataset"]["version_name"], saved["dataset"]["revision"]),
                         ("train-001", "t1", 1))
        self.assertEqual(saved["dataset"]["samples"], 2)
        self.assertTrue(saved["manifest"].endswith(str(Path("versions/training/train-001/manifest.jsonl"))))
        self.assertEqual((saved["validation"]["version_id"], saved["validation"]["version_name"]), ("val-001", "val1"))
        listing = self.client.get(self.url()).json()
        self.assertEqual(listing["versions"][0]["used_by"], [self.name])
        self.assertEqual(listing["versions"][0]["diff"]["added"], 0)  # the working manifest was not rebuilt
        results = self.client.get(f"/api/projects/{self.name}/validation-results").json()
        self.assertEqual(results["trained_with"]["training"]["version_id"], "train-001")
        self.assertEqual(results["trained_with"]["validation"]["name"], "val1")
        self.assertEqual(results["validation"]["version"], {"id": "val-001", "name": "val1", "revision": 1})
        job = self.submit(f"/api/projects/{self.name}/validation-runs", {"checkpoints": ["final_adapter"]})
        self.assertEqual(self.wait_job(job["id"])["state"], "succeeded")
        items = self.client.get(f"/api/projects/{self.name}/validation-results").json()["items"]
        final = next(item for item in items if item["name"] == "final_adapter")
        self.assertEqual((final["status"], final["validation_version"]["id"]), ("current", "val-001"))

    def test_damaged_version_blocks_training_until_repaired(self):
        self.setup_project(validation=False)
        self.assertEqual(self.client.post(self.url(), json={}).status_code, 201)
        moved = self.corpus / "sub" / "c0.wav"
        moved.parent.mkdir()
        (self.corpus / "c0.wav").rename(moved)
        project = self.client.get(f"/api/projects/{self.name}").json()
        self.assertEqual((project["training_dataset"]["status"], project["training_dataset"]["stale"]), ("damaged", True))
        refused = self.client.post("/api/training/start", json={"project": self.name})
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertIn("missing", refused.json()["error"]["message"])
        report = self.client.get(self.url(suffix="/train-001/integrity")).json()
        self.assertEqual((report["missing_count"], report["relocatable"]), (1, 1))
        repaired = self.client.post(self.url(suffix="/train-001/repair"), json={"changed": "accept"})
        self.assertEqual(repaired.status_code, 200, repaired.text)
        self.assertEqual(repaired.json()["versions"][0]["revision"], 2)
        project = self.client.get(f"/api/projects/{self.name}").json()
        self.assertEqual((project["training_dataset"]["status"], project["training_dataset"]["revision"]), ("ready", 2))
