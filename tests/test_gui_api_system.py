"""Fake server runtime, filesystem boundary and media integration tests."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from tools.gui_server.services import engine
from indextts_training.activity import acquire, release


class SystemApiTests(GuiApiTestCase):
    def test_system_status_and_unload(self):
        response = self.client.get('/api/system/status')
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertTrue(data['fake_engine'])
        self.assertTrue(data['asr']['available'])
        self.assertIn(str(self.root), data['completion_sound_path'])
        engine.load_model(True, False, False, False, False)
        self.assertTrue(self.client.get('/api/system/status').json()['model']['loaded'])
        response = self.client.post('/api/model/unload')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(engine.is_loaded())

    def test_model_cache_context_does_not_cross_fake_real(self):
        fake = engine.load_model(True, False, False, False, False)
        self.assertIs(fake, engine.load_model(True, False, False, False, False))
        with patch.object(self.state, 'fake', False):
            self.assertFalse(engine.loaded_info()['loaded'])
            with self.assertRaisesRegex(Exception, 'model is missing'):
                engine.load_model(True, False, False, False, False)
        self.assertIsNot(fake, engine.load_model(True, False, False, False, False))

    def test_unload_and_cleanup_respect_training_lease(self):
        voice = self.audio()
        lease = acquire('training')
        try:
            for method, endpoint in [('post', '/api/model/unload'), ('delete', '/api/outputs'), ('delete', '/api/voices')]:
                response = getattr(self.client, method)(endpoint)
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(response.json()['error']['code'], 'busy')
            self.assertTrue(voice.exists())
        finally:
            release(lease)

    def test_fs_filters_and_read_only(self):
        self.audio()
        (paths.VOICES_DIR / 'notes.txt').write_text('not audio', encoding='utf-8')
        (paths.VOICES_DIR / 'model.pth').write_bytes(b'checkpoint')
        (paths.VOICES_DIR / 'folder').mkdir()
        roots = self.client.get('/api/fs/roots').json()
        self.assertTrue(any(item['path'] == str(self.root.resolve()) for item in roots))
        for kind, expected in [('audio', {'reference.wav', 'folder'}), ('pth', {'model.pth', 'folder'}), ('dir', {'folder'})]:
            response = self.client.get('/api/fs/list', params={'path': str(paths.VOICES_DIR), 'kind': kind})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual({row['name'] for row in response.json()['entries']}, expected)
        self.assertEqual(self.client.get('/api/fs/list', params={'path': str(self.root / 'missing')}).status_code, 404)
        self.assertEqual(self.client.post('/api/fs/list', json={'path': str(paths.VOICES_DIR)}).status_code, 405)

    def test_media_range_head_and_download(self):
        audio = self.audio()
        response = self.client.get('/api/media', params={'path': str(audio)}, headers={'Range': 'bytes=0-15'})
        self.assertEqual(response.status_code, 206, response.text)
        self.assertEqual(response.content, audio.read_bytes()[:16])
        self.assertEqual(response.headers['accept-ranges'], 'bytes')
        self.assertIn('bytes 0-15/', response.headers['content-range'])
        self.assertEqual(self.client.head('/api/media', params={'path': str(audio)}).content, b'')
        download = self.client.get('/api/media', params={'path': str(audio), 'download': 'true'})
        self.assertIn('attachment', download.headers['content-disposition'])

    def test_media_rejects_outside_and_nonmedia(self):
        secret = self.root / 'secret.txt'
        secret.write_text('secret', encoding='utf-8')
        outside = self.audio('outside.wav')
        for path in [secret, outside]:
            response = self.client.get('/api/media', params={'path': str(path)})
            self.assertEqual(response.status_code, 403, response.text)
        inside = paths.OUTPUTS_DIR / 'private.json'
        inside.write_text('{"secret":true}', encoding='utf-8')
        self.assertEqual(self.client.get('/api/media', params={'path': str(inside)}).status_code, 403)
        self.assertEqual(self.client.get('/api/media', params={'path': str(paths.VOICES_DIR / 'missing.wav')}).status_code, 404)
        sound = self.audio('assets/inference_training_done.wav')
        self.assertEqual(self.client.get('/api/media', params={'path': 'assets/inference_training_done.wav'}).content,
                         sound.read_bytes())

    def test_demo_seed_is_explicit_isolated_and_additive(self):
        from tools.gui_server.fake.seed_demo import main
        directory = self.root / 'isolated-demo'
        with contextlib.ExitStack() as stack:
            for name, value in vars(paths).copy().items():
                if isinstance(value, Path) or name == 'CREATED_DIRS':
                    stack.enter_context(patch.object(paths, name, value))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--data-dir', str(directory)]), 0)
            voice = directory / 'voices/Demo_Warm.txt'
            voice.write_text('User edited transcript', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--data-dir', str(directory)]), 0)
            self.assertEqual(voice.read_text(encoding='utf-8'), 'User edited transcript')
        project = json.loads((directory / 'training/projects/Demo_Narrator/project.json').read_text(encoding='utf-8'))
        self.assertEqual(project['stats']['valid'], 3)
        self.assertEqual(project['training']['seed'], 1234)
        self.assertEqual(len(list((directory / 'voices').glob('*.wav'))), 3)
        # Refuse an existing unmarked root and never change its files.
        existing = self.root / 'unmarked'
        existing.mkdir()
        sentinel = existing / 'keep.txt'
        sentinel.write_text('Keep me', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--data-dir', str(existing)])
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'Keep me')
        self.assertEqual(list(existing.iterdir()), [sentinel])

    def test_openapi_contains_all_creation_contracts(self):
        schema = self.app.openapi()
        for name in ['SynthesisRequest', 'DialogueRequest', 'Voice', 'AudioUpload', 'SystemStatus', 'HistoryItem', 'DirectoryListing']:
            self.assertIn(name, schema['components']['schemas'])
        self.assertIn('voice_path', schema['components']['schemas']['SynthesisRequest']['properties'])
        self.assertIn('emotion_source', schema['components']['schemas']['DialogueTurn']['properties'])
        self.assertIn('/api/asr/transcribe', schema['paths'])
