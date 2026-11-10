from __future__ import annotations
import json
from pathlib import Path
from gui_api_support import GuiApiTestCase


class ProjectsApiTests(GuiApiTestCase):
    def create(self, name="voice_project"):
        response = self.client.post("/api/projects", json={"name": name})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_crud_defaults_clone_and_confirmed_delete_preserve_user_data(self):
        project = self.create("voice project")
        self.assertEqual(project["project_name"], "voice_project")
        cfg = project["training"]
        self.assertEqual((cfg["training_steps"], cfg["save_steps"], cfg["learning_rate"], cfg["seed"]), (1000, 100, 4e-5, 1234))
        self.assertEqual(self.client.post("/api/projects", json={"name": "voice project"}).status_code, 409)
        source = self.audio("corpus/a.wav", .6)
        source.with_suffix(".txt").write_text("a transcript", encoding="utf-8")
        response = self.client.put("/api/projects/voice_project/dataset", json={"source_folder": str(source.parent), "prepare": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["dataset_status"]["status"], "ready")
        project_path = self.root / "training/projects/voice_project/project.json"
        persisted = json.loads(project_path.read_text(encoding="utf-8"))
        persisted["legacy_extension"] = {"keep": True}
        project_path.write_text(json.dumps(persisted), encoding="utf-8")
        reference = self.audio(".runtime/temp/eval.wav")
        saved = self.client.put("/api/projects/voice_project/training", json={"seed": 42, "eval_reference": str(reference), "semantic_loss_weight": 2})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["training"]["seed"], 42)
        self.assertEqual(saved.json()["training"]["semantic_loss_weight"], 0)
        self.assertTrue(Path(saved.json()["training"]["eval_reference"]).is_relative_to(project_path.parent))
        self.assertEqual(saved.json()["legacy_extension"], {"keep": True})
        clone = self.client.post("/api/projects/voice_project/clone", json={"name": "my copy"})
        self.assertEqual(clone.status_code, 201, clone.text)
        self.assertEqual(clone.json()["project_name"], "my_copy")
        self.assertNotEqual(clone.json()["manifest"], response.json()["manifest"])
        self.assertEqual(clone.json()["training"]["adapter_name"], "my_copy")
        self.assertEqual(len(self.client.get("/api/projects").json()), 2)
        output = self.root / "training/outputs/voice_project/keep.bin"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"keep")
        refused = self.client.request("DELETE", "/api/projects/voice_project", json={"confirm_name": "wrong"})
        self.assertEqual(refused.status_code, 400)
        deleted = self.client.request("DELETE", "/api/projects/voice_project", json={"confirm_name": "voice_project"})
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertTrue(source.is_file())
        self.assertTrue(output.is_file())
        self.assertFalse(project_path.exists())
        self.assertEqual(self.client.get("/api/projects/voice_project").status_code, 404)

    def test_validation_preparation_and_staleness(self):
        self.create()
        audio = self.audio("validation/set/a.wav", .7)
        audio.with_suffix(".txt").write_text("validation text", encoding="utf-8")
        response = self.client.put("/api/projects/voice_project/validation", json={"source_folder": str(audio.parent), "enabled": True, "interval_steps": 0, "prepare": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["validation"]["enabled"])
        self.assertIsNone(response.json()["validation"]["interval_steps"])
        self.assertEqual(response.json()["validation_status"]["status"], "ready")
        audio.with_suffix(".txt").write_text("changed text", encoding="utf-8")
        self.assertTrue(self.client.get("/api/projects/voice_project").json()["validation_status"]["stale"])
        self.assertEqual(self.client.put("/api/projects/voice_project/validation", json={"interval_steps": -1}).status_code, 400)

    def test_autotune_and_invalid_names_or_parameters(self):
        self.create()
        result = self.client.post("/api/projects/voice_project/autotune", json={"vram_preset": "32 GB+"})
        self.assertEqual(result.status_code, 200, result.text)
        values = result.json()["training"]
        self.assertEqual((values["lora_rank"], values["training_steps"], values["learning_rate"], values["grad_accum"]), (16, 900, 2e-4, 4))
        # Autotune is a preview; does not silently overwrite the saved controls.
        self.assertEqual(self.client.get("/api/projects/voice_project").json()["training"]["learning_rate"], 4e-5)
        self.assertEqual(self.client.put("/api/projects/voice_project/training", json={"seed": -1}).status_code, 400)
        self.assertEqual(self.client.get("/api/projects/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/projects/voice_project/checkpoints", params={"adapter": "../outside"}).status_code, 400)
        self.assertEqual(self.client.put("/api/projects/voice_project/dataset", json={"source_folder": str(self.root / "missing")}).status_code, 400)
        self.assertEqual(self.client.post("/api/projects", json={"name": "x", "language": "BAD"}).status_code, 400)
        self.assertEqual(self.client.post("/api/projects", json={"name": "CON"}).status_code, 400)
        self.assertEqual(self.client.put("/api/projects/voice_project/training", json={"adapter_name": "NUL"}).status_code, 400)
