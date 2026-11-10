"""Isolated fake GUI API fixture; no test writes into real user data.

Usage: subclass GuiApiTestCase, use self.client / self.root / self.state,
self.audio(relative='voices/reference.wav'), self.wait_job(id), self.wait_until(fn).
All directory constants and the legacy CREATED_DIRS tuple are patched before
create_app. The static WEB_DIR intentionally remains the read-only built app.
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from tools.gui_server import paths


class GuiApiTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._stack = contextlib.ExitStack()
        self.addCleanup(self._stack.close)
        self.root = Path(self._stack.enter_context(tempfile.TemporaryDirectory(prefix='gui-api-')))
        original_root = paths.ROOT
        redirected = {}
        for name, value in vars(paths).copy().items():
            if name == 'WEB_DIR' or not isinstance(value, Path):
                continue
            try:
                redirected[name] = self.root / value.relative_to(original_root)
            except ValueError:
                continue
        for name, value in redirected.items():
            self._stack.enter_context(patch.object(paths, name, value))
        self._stack.enter_context(patch.object(paths, 'CREATED_DIRS', tuple(
            self.root / p.relative_to(original_root) for p in paths.CREATED_DIRS)))
        for name, path in redirected.items():
            if name not in {'SETTINGS_FILE'}:
                path.mkdir(parents=True, exist_ok=True)
        from tools.gui_server.app import create_app
        from tools.gui_server.services import engine
        # First app must be fake before engine.unload, otherwise torch is imported unnecessarily.
        self.app = create_app(fake=True, background=False)
        self.state = self.app.state.gui
        engine.unload()
        self.client = self._stack.enter_context(TestClient(self.app, headers={'X-UI-Language': 'en_US'}))
        self.addCleanup(self._stop_workers)

    def _stop_workers(self):
        for job in self.state.jobs.list():
            if job['state'] == 'running' and job['cancellable']:
                self.state.jobs.cancel(job['id'])
        self.state.training.stop_event.set()
        self.state.export.cancel()
        if self.state.training.thread:
            self.state.training.thread.join(10)
        if self.state.export.thread:
            self.state.export.thread.join(10)
        self.wait_until(lambda: not any(j['state'] == 'running' for j in self.state.jobs.list()), timeout=10)
        self.state.training._stop_tensorboard_server()
        from tools.gui_server.services import engine
        engine.unload()

    def audio(self, relative='voices/reference.wav', seconds=0.08):
        from tools.gui_server.fake import write_tone
        return write_tone(self.root / relative, seconds)

    def audio_bytes(self, seconds=0.05):
        path = self.audio('.runtime/temp/test-upload.wav', seconds)
        return path.read_bytes()

    def upload_audio(self, name='reference.wav'):
        response = self.client.post('/api/uploads', files={'file': (name, io.BytesIO(self.audio_bytes()), 'audio/wav')})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def wait_until(self, predicate, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        self.fail('Timed out waiting for background operation')

    def wait_job(self, job_id, timeout=15.0):
        def done():
            response = self.client.get(f'/api/jobs/{job_id}')
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            return value if value['state'] != 'running' else None
        return self.wait_until(done, timeout)

    def submit(self, endpoint, body):
        response = self.client.post(endpoint, json=body)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['job']
