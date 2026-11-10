from __future__ import annotations
import json
from pathlib import Path
from gui_api_support import GuiApiTestCase


class AbxApiTests(GuiApiTestCase):
    def create_session(self, texts="hello", repetitions=1):
        voice = self.audio()
        voice.with_suffix(".txt").write_text("reference transcript", encoding="utf-8")
        adapter = self.root / "training/outputs/test/final_adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        response = self.client.post("/api/abx/sessions", json={"voice": str(voice), "adapter_b": str(adapter),
                                    "texts": texts, "repetitions": repetitions, "settings": {"temperature": .45}})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_paired_generation_rating_reveal_and_no_assignment_leak(self):
        session = self.create_session()
        sid = session["id"]
        self.assertFalse(session["generation_complete"])
        self.assertIsNone(session["trial"])
        self.assertIsNone(session["results"])
        self.assertEqual(self.client.put(f"/api/abx/sessions/{sid}/rate", json={"index": 0, "score": 1}).status_code, 400)
        job = self.submit(f"/api/abx/sessions/{sid}/generate", {})
        self.assertEqual(self.wait_job(job["id"])["state"], "succeeded")
        detail = self.client.get(f"/api/abx/sessions/{sid}").json()
        self.assertTrue(detail["generation_complete"])
        self.assertEqual(detail["clips_done"], 2)
        self.assertNotIn("first", detail["trial"])
        self.assertNotIn("side", detail["trial"])
        self.assertNotIn("clips", detail)
        self.assertIsNone(detail["results"])
        self.assertTrue(Path(detail["trial"]["sample_1"]).is_file())
        # Stored format remains identical to 1.x, including paired seeds and frozen decoding.
        disk = json.loads((self.root / "outputs/abx" / sid / "session.json").read_text())
        self.assertEqual(disk["settings"]["temperature"], .45)
        self.assertEqual(disk["clips"][0]["seed"], disk["clips"][1]["seed"])
        response = self.client.put(f"/api/abx/sessions/{sid}/rate", json={"index": 0, "score": 2})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rated"], 1)
        revealed = self.client.post(f"/api/abx/sessions/{sid}/reveal")
        self.assertEqual(revealed.status_code, 200, revealed.text)
        self.assertTrue(revealed.json()["revealed"])
        self.assertIsNotNone(revealed.json()["results"])
        self.assertEqual(self.client.put(f"/api/abx/sessions/{sid}/rate", json={"index": 0, "score": 0}).status_code, 400)
        self.assertEqual(len(self.client.get("/api/abx/sessions").json()), 1)

    def test_failed_generation_and_cancel_resume(self):
        bad = self.create_session("[fail]")
        job = self.submit(f"/api/abx/sessions/{bad['id']}/generate", {})
        self.assertEqual(self.wait_job(job["id"])["state"], "failed")
        session = self.create_session("one\ntwo", 2)
        sid = session["id"]
        job = self.submit(f"/api/abx/sessions/{sid}/generate", {})
        self.assertEqual(self.client.post(f"/api/abx/sessions/{sid}/generate").status_code, 409)
        self.client.post(f"/api/jobs/{job['id']}/cancel")
        self.assertEqual(self.wait_job(job["id"])["state"], "cancelled")
        resumed = self.submit(f"/api/abx/sessions/{sid}/generate", {})
        self.assertEqual(self.wait_job(resumed["id"], timeout=20)["state"], "succeeded")
        self.assertTrue(self.client.get(f"/api/abx/sessions/{sid}").json()["generation_complete"])

    def test_invalid_inputs_missing_session_and_activity_conflict(self):
        voice = self.audio()
        response = self.client.post("/api/abx/sessions", json={"voice": str(voice), "texts": "hello"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/abx/sessions/20200101_000000").status_code, 404)
        session = self.create_session()
        sid = session["id"]
        self.assertEqual(self.client.post(f"/api/abx/sessions/{sid}/reveal").status_code, 400)
        self.assertEqual(self.client.put(f"/api/abx/sessions/{sid}/rate", json={"index": 0, "score": 3}).status_code, 400)
        from indextts_training.activity import acquire, release
        token = acquire("training")
        try:
            self.assertEqual(self.client.post(f"/api/abx/sessions/{sid}/generate").status_code, 409)
        finally:
            release(token)
