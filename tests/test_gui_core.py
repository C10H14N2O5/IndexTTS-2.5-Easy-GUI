"""Core web GUI infrastructure: events, jobs, console, guards, SPA fallback."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from gui_api_support import GuiApiTestCase

from tools.gui_server import i18n, paths
from tools.gui_server.console import ConsoleBuffer, classify
from tools.gui_server.errors import Busy, ServiceError
from tools.gui_server.events import EventHub
from tools.gui_server.jobs import JobManager


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class EventHubTests(unittest.TestCase):
    def test_snapshot_first_then_events_rendered_per_language(self):
        async def scenario():
            hub = EventHub()
            hub.attach_loop(asyncio.get_running_loop())
            zh = hub.stream("zh_CN", lambda: {"detail": i18n.msg("not installed")})
            en = hub.stream("en_US", lambda: {"detail": i18n.msg("not installed")})
            self.assertEqual(await zh.__anext__(), "retry: 2000\n\n")
            self.assertEqual(await en.__anext__(), "retry: 2000\n\n")
            zh_snapshot = json.loads((await zh.__anext__()).split("data: ", 1)[1])
            en_snapshot = json.loads((await en.__anext__()).split("data: ", 1)[1])
            self.assertEqual(zh_snapshot["type"], "snapshot")
            self.assertEqual(en_snapshot["data"]["detail"], "not installed")
            self.assertNotEqual(zh_snapshot["data"]["detail"], "not installed")
            pending = asyncio.ensure_future(zh.__anext__())
            await asyncio.sleep(0)
            hub.publish("job", {"stage": i18n.msg("not installed")})
            event = json.loads((await asyncio.wait_for(pending, 2)).split("data: ", 1)[1])
            self.assertEqual(event["type"], "job")
            self.assertEqual(event["data"]["stage"], zh_snapshot["data"]["detail"])
            self.assertEqual(hub.subscriber_count, 2)
            hub.close()
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(zh.__anext__(), 2)
            await en.aclose()
            self.assertEqual(hub.subscriber_count, 0)

        asyncio.run(scenario())

    def test_publish_without_loop_is_a_no_op(self):
        EventHub().publish("job", {"x": 1})


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.jobs = JobManager(publish=lambda kind, data: self.events.append((kind, data)))

    def test_gpu_slot_busy_cancel_and_result(self):
        release = threading.Event()

        def work(ctx):
            ctx.progress(0.5, "Chunk {0}/{1}", 1, 2)
            while not release.wait(0.01):
                ctx.check_cancelled()
            return {"path": "x.wav"}

        first = self.jobs.submit("synthesis", "Synthesis", work, gpu=True)
        self.assertTrue(wait_for(lambda: first.progress == 0.5))
        with self.assertRaises(Busy) as busy:
            self.jobs.submit("dialogue", "Dialogue", work, gpu=True)
        self.assertEqual(busy.exception.extra["busy_job"]["id"], first.id)
        # Non-GPU work is not blocked by the GPU slot.
        cpu = self.jobs.submit("asr_reference", "ASR", lambda ctx: {"text": "hi"})
        self.assertTrue(wait_for(lambda: cpu.state == "succeeded"))
        self.assertEqual(cpu.result, {"text": "hi"})
        self.jobs.cancel(first.id)
        self.assertTrue(wait_for(lambda: first.state == "cancelled"))
        self.assertIsNotNone(first.finished_at)
        second = self.jobs.submit("synthesis", "Synthesis", lambda ctx: {"ok": True}, gpu=True)
        self.assertTrue(wait_for(lambda: second.state == "succeeded"))
        self.assertEqual(second.progress, 1.0)
        self.assertTrue(any(kind == "job" and data["id"] == first.id and data["state"] == "cancelled"
                            for kind, data in self.events))

    def test_training_and_export_share_the_gpu_lease(self):
        from indextts_training import activity
        lease = activity.acquire("training")
        try:
            with self.assertRaises(Busy):
                self.jobs.submit("synthesis", "Synthesis", lambda ctx: None, gpu=True)
        finally:
            activity.release(lease)
        finished = threading.Event()
        job = self.jobs.submit("synthesis", "Synthesis", lambda ctx: finished.wait(2), gpu=True)
        try:
            with self.assertRaises(RuntimeError):
                activity.acquire("export")
        finally:
            finished.set()
            self.assertTrue(wait_for(lambda: job.state != "running"))
        lease = activity.acquire("training")
        activity.release(lease)

    def test_failed_gpu_job_releases_lease_and_shutdown_rejects_work(self):
        from indextts_training import activity
        def fail(ctx):
            raise ServiceError("Invalid input")
        job = self.jobs.submit("synthesis", "Synthesis", fail, gpu=True)
        self.assertTrue(wait_for(lambda: job.state == "failed"))
        lease = activity.acquire("export")
        activity.release(lease)
        self.jobs.shutdown()
        with self.assertRaises(Busy):
            self.jobs.submit("asr_reference", "ASR", lambda ctx: None)

    def test_snapshots_and_events_do_not_share_mutable_metadata(self):
        release = threading.Event()
        job = self.jobs.submit("asr_reference", "ASR", lambda ctx: release.wait(2), meta={"nested": {"x": 1}})
        first_event = self.events[0][1]
        snapshot = job.to_dict()
        snapshot["meta"]["nested"]["x"] = 2
        job.meta["nested"]["x"] = 3
        release.set()
        self.assertTrue(wait_for(lambda: job.state != "running"))
        self.assertEqual(first_event["meta"]["nested"]["x"], 1)
        self.assertEqual(job.to_dict()["meta"]["nested"]["x"], 3)

    def test_failures_are_reported(self):
        def service_failure(ctx):
            raise ServiceError("Invalid theme.")

        def crash(ctx):
            raise ZeroDivisionError("boom")

        failed = self.jobs.submit("synthesis", "Synthesis", service_failure, gpu=True)
        self.assertTrue(wait_for(lambda: failed.state == "failed"))
        self.assertEqual(failed.error["code"], "invalid")
        self.assertEqual(failed.error["message"].render("en_US"), "Invalid theme.")
        with mock.patch("traceback.print_exc"):
            crashed = self.jobs.submit("synthesis", "Synthesis", crash, gpu=True)
            self.assertTrue(wait_for(lambda: crashed.state == "failed"))
        self.assertEqual(crashed.error["code"], "internal")
        self.assertIn("ZeroDivisionError", crashed.error["message"].render("en_US"))
        listed = [job["id"] for job in self.jobs.list()]
        self.assertEqual(listed[:2], [crashed.id, failed.id])


class ConsoleBufferTests(unittest.TestCase):
    def test_progress_lines_overwrite_in_place(self):
        console = ConsoleBuffer()
        console.write("Loading model\n")
        console.write(" 10%|#   | 1/10 [00:01<00:09, 1.00it/s]\r")
        first = console.drain()
        self.assertEqual([line["text"] for line in first["lines"]][0], "Loading model")
        console.write(" 50%|#####| 5/10 [00:05<00:05, 1.00it/s]\r")
        changed = console.drain()
        self.assertEqual(len(changed["lines"]), 1)
        self.assertIn("50%", changed["lines"][0]["text"])
        self.assertEqual(changed["lines"][0]["id"], first["lines"][-1]["id"])
        self.assertEqual(changed["lines"][0]["level"], "progress")
        self.assertIsNone(console.drain())
        console.write("\r100%|#####| 10/10 [00:10<00:00, 1.00it/s]\n")
        finished = console.drain()
        self.assertEqual(finished["lines"][0]["id"], first["lines"][-1]["id"])
        self.assertIn("100%", finished["lines"][0]["text"])
        console.write("partial")
        self.assertEqual(console.drain()["current"], "partial")
        snapshot = console.snapshot()
        self.assertEqual(len(snapshot["lines"]), 2)
        console.clear()
        self.assertEqual(console.snapshot()["lines"], [])
        self.assertEqual(console.drain(), {"lines": [], "current": "", "reset": True})
        console.write("brand new\n")
        self.assertFalse(console.drain()["reset"])

    def test_levels(self):
        self.assertEqual(classify("Traceback (most recent call last):"), "error")
        self.assertEqual(classify("UserWarning: fallback"), "warn")
        self.assertEqual(classify("Saved output"), "success")
        self.assertEqual(classify(">> gpt_gen_time: 1.2"), "metric")
        self.assertEqual(classify("plain text"), "plain")


class AppTests(GuiApiTestCase):

    def test_bootstrap_and_settings(self):
        data = self.client.get("/api/bootstrap").json()
        self.assertTrue(data["fake_engine"])
        self.assertEqual(data["settings"]["ui_language"], "zh_CN")
        self.assertIn("Neutral / balanced", data["catalog"]["emotion_presets"])
        updated = self.client.put("/api/settings", json={"theme": "dark", "ui_language": "en_US"})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["theme"], "dark")
        invalid = self.client.put("/api/settings", json={"theme": "pink"})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["error"]["code"], "invalid")
        self.assertNotIn("input", invalid.json()["error"]["detail"][0])
        self.assertEqual(self.client.put("/api/settings", json={"use_torch_compile": True}).status_code, 400)

    def test_same_origin_guard(self):
        self.assertEqual(self.client.put("/api/settings", json={"theme": "light"},
                                         headers={"Origin": "http://evil.example"}).status_code, 403)
        self.assertEqual(self.client.put("/api/settings", json={"theme": "light"},
                                         headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)
        self.assertEqual(self.client.put("/api/settings", json={"theme": "light"},
                                         headers={"Origin": "http://testserver"}).status_code, 200)
        # Reads are never blocked.
        self.assertEqual(self.client.get("/api/settings", headers={"Origin": "http://evil.example"}).status_code, 200)

    def test_errors_are_structured_and_localised(self):
        response = self.client.get("/api/jobs/missing", headers={"X-UI-Language": "en_US"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "not_found")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        missing = self.client.get("/api/does-not-exist")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "not_found")
        self.assertEqual(self.client.get("/exports/x").status_code, 404)

    def test_upload_request_limit_rejects_before_spooling_and_handles_chunked(self):
        from tools.gui_server import security
        body = (b'--sample\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                b'Content-Type: audio/wav\r\n\r\n' + b'x' * 2000 + b'\r\n--sample--\r\n')
        headers = {"Content-Type": "multipart/form-data; boundary=sample"}
        with mock.patch.object(security, "MAX_UPLOAD_BODY_BYTES", 256):
            direct = self.client.post("/api/uploads", content=body, headers=headers)
            streamed = self.client.post("/api/uploads", content=iter([body[:200], body[200:]]), headers=headers)
        for response in (direct, streamed):
            self.assertEqual(response.status_code, 413, response.text)
            self.assertEqual(response.json()["error"]["code"], "too_large")
        self.assertEqual(list(paths.UPLOADS_DIR.iterdir()), [])

    def test_request_state_is_not_the_last_constructed_app(self):
        from tools.gui_server.state import build, current, install
        other = build(fake=False)
        install(other)
        try:
            response = self.client.get("/api/bootstrap")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["fake_engine"])
            self.assertIs(current(), other)
        finally:
            install(self.state)

    def test_fresh_training_manager_does_not_inherit_fake_trainer(self):
        from tools.gui_server.app import create_app
        from tools.gui_server.state import install
        try:
            real_app = create_app(fake=False, background=False)
            self.assertIsNot(self.state.training, real_app.state.gui.training)
            self.assertIsNone(real_app.state.gui.training.train_fn)
        finally:
            install(self.state)

    def test_openapi_documents_uniform_validation_errors(self):
        schema = self.client.get("/api/openapi.json").json()
        responses = schema["paths"]["/api/settings"]["put"]["responses"]
        self.assertNotIn("422", responses)
        self.assertIn("400", responses)
        self.assertEqual(responses["400"]["content"]["application/json"]["schema"]["$ref"],
                         "#/components/schemas/ErrorResponse")

    def test_events_stream_starts_with_snapshot(self):
        self.app.state.gui.hub.close()  # let the stream end after the snapshot
        try:
            response = self.client.get("/api/events?lang=en_US")
        finally:
            self.app.state.gui.hub._closed = False
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(chunk.split("data: ", 1)[1]) for chunk in response.text.split("\n\n") if "data: " in chunk]
        self.assertEqual(events[0]["type"], "snapshot")
        for key in ("jobs", "status", "console", "settings", "system"):
            self.assertIn(key, events[0]["data"])

    def test_legacy_proofread_entry_redirects_without_old_html(self):
        for url in ("/proofread", "/proofread/"):
            response = self.client.get(url, follow_redirects=False)
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/projects")
            self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/proofread/api/session").status_code, 403)

    def test_spa_fallback(self):
        from tools.gui_server.app import web_assets_ok
        response = self.client.get("/projects/demo/training")
        if (paths.WEB_DIR / "index.html").is_file():
            self.assertEqual(response.status_code, 200)
            self.assertIn("default-src 'self'", response.headers["content-security-policy"])
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertTrue(web_assets_ok()[0], web_assets_ok()[1])
        else:
            self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/assets/missing.js").status_code, 404)


if __name__ == "__main__":
    unittest.main()
