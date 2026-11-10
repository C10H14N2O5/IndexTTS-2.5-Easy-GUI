from __future__ import annotations
import json
from unittest.mock import patch
from gui_api_support import GuiApiTestCase


class ExportApiTests(GuiApiTestCase):
    def adapter(self):
        directory = self.root / "training/outputs/voice/final_adapter"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "bias": "none", "r": 16}))
        (directory / "adapter_model.safetensors").write_bytes(b"FAKE")
        return directory

    def finish(self):
        self.wait_until(lambda: self.state.export.snapshot()["status"] not in {"running", "cancelling"}, timeout=30)
        self.state.export.thread.join(10)
        return self.client.get("/api/export/status").json()

    def test_adapter_info_export_download_tokens_and_events(self):
        adapter = self.adapter()
        info = self.client.get("/api/export/adapter-info", params={"adapter": str(adapter)})
        self.assertEqual(info.status_code, 200, info.text)
        self.assertTrue(info.json()["valid"])
        self.assertIn("fake-engine", info.json()["base"])
        with patch.object(self.state, "publish", wraps=self.state.publish) as publish:
            response = self.client.post("/api/export/start", json={"adapter": str(adapter), "destination": "voice.pth"})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertFalse(response.json()["downloads"])
            self.assertEqual(self.client.post("/api/export/start", json={"adapter": str(adapter)}).status_code, 409)
            done = self.finish()
            self.assertEqual(done["status"], "succeeded", done)
            self.assertTrue(any(call.args[0] == "export" for call in publish.call_args_list))
        for kind in ("model", "report"):
            url = done["downloads"][kind]
            download = self.client.get(url)
            self.assertEqual(download.status_code, 200, download.text[:300])
            self.assertEqual(download.headers["cache-control"], "private, no-store")
            self.assertEqual(self.client.get(url.split("?")[0]).status_code, 404)
            self.assertEqual(self.client.get(url.split("?")[0] + "?token=wrong").status_code, 404)
        self.assertIn("export", self.state.providers)

    def test_export_failure_cancel_and_conflicts(self):
        adapter = self.adapter()
        response = self.client.post("/api/export/start", json={"adapter": str(adapter), "destination": "fail.pth"})
        self.assertEqual(response.status_code, 202, response.text)
        failed = self.finish()
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["downloads"])
        response = self.client.post("/api/export/start", json={"adapter": str(adapter), "destination": "cancel.pth"})
        self.assertEqual(response.status_code, 202, response.text)
        cancelled = self.client.post("/api/export/cancel")
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(self.finish()["status"], "cancelled")
        from indextts_training.activity import acquire, release
        token = acquire("training")
        try:
            self.assertEqual(self.client.post("/api/export/start", json={"adapter": str(adapter)}).status_code, 409)
        finally:
            release(token)

    def test_destination_validation_and_never_overwrite(self):
        adapter = self.adapter()
        for destination in ("../bad.pth", "NUL.pth", "invalid.txt"):
            response = self.client.post("/api/export/start", json={"adapter": str(adapter), "destination": destination})
            self.assertEqual(response.status_code, 400, response.text)
        output = self.root / "existing.pth"
        output.write_bytes(b"precious")
        response = self.client.post("/api/export/start", json={"adapter": str(adapter), "mode": "path", "destination": str(output)})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(output.read_bytes(), b"precious")
        response = self.client.get("/api/export/adapter-info", params={"adapter": str(self.root / "missing")})
        self.assertFalse(response.json()["valid"])
        self.assertEqual(self.client.get("/exports/unknown/model?token=bad").status_code, 404)
