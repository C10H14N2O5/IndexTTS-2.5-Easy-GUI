"""Real multipart parsing, audio validation and flat-library persistence tests."""
from __future__ import annotations

import asyncio
import io
import os
import time
from unittest.mock import patch

from starlette.datastructures import UploadFile

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from tools.gui_server.services import uploads


class LibraryApiTests(GuiApiTestCase):
    def test_upload_voice_transcript_rename_and_delete(self):
        upload = self.upload_audio()
        self.assertTrue(upload['path'].startswith(str(paths.UPLOADS_DIR)))
        self.assertGreater(upload['duration'], 0)
        response = self.client.post('/api/voices', json={'name': 'Test Speaker', 'source_path': upload['path'], 'transcript': '  Hello voice  '})
        self.assertEqual(response.status_code, 201, response.text)
        voice = response.json()
        self.assertEqual(voice['name'], 'Test_Speaker.wav')
        self.assertEqual(voice['transcript'], 'Hello voice')
        self.assertEqual((paths.VOICES_DIR / 'Test_Speaker.txt').read_text(encoding='utf-8'), 'Hello voice\n')
        self.assertEqual(self.client.get('/api/voices').json()[0], voice)
        response = self.client.patch('/api/voices/Test_Speaker.wav', json={'new_name': 'Renamed', 'transcript': 'New text'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['transcript'], 'New text')
        self.assertFalse((paths.VOICES_DIR / 'Test_Speaker.wav').exists())
        self.assertFalse((paths.VOICES_DIR / 'Test_Speaker.txt').exists())
        self.assertTrue((paths.VOICES_DIR / 'Renamed.txt').is_file())
        self.assertEqual(self.client.delete('/api/voices/Renamed.wav').json(), {'deleted': 1})
        self.assertFalse((paths.VOICES_DIR / 'Renamed.txt').exists())

    def test_voice_overwrite_requires_confirmation_and_clears_transcript(self):
        source = self.audio('.runtime/temp/source.wav')
        body = {'name': 'Alice', 'source_path': str(source), 'transcript': 'Hello'}
        self.assertEqual(self.client.post('/api/voices', json=body).status_code, 201)
        duplicate = self.client.post('/api/voices', json=body)
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        self.assertEqual(duplicate.json()['error']['code'], 'conflict')
        response = self.client.post('/api/voices', json={**body, 'overwrite': True, 'transcript': ''})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()['transcript'], '')
        self.assertFalse((paths.VOICES_DIR / 'Alice.txt').exists())

    def test_upload_rejects_extension_empty_and_disguised_audio(self):
        for name, data in [('a.exe', self.audio_bytes()), ('empty.wav', b''), ('fake.wav', b'not audio')]:
            response = self.client.post('/api/uploads', files={'file': (name, data, 'audio/wav')})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(list(paths.UPLOADS_DIR.glob('*')), [])

    def test_upload_size_limit_both_announced_and_streamed(self):
        with patch.object(uploads, 'MAX_UPLOAD_BYTES', 32):
            response = self.client.post('/api/uploads', files={'file': ('too-big.wav', self.audio_bytes(), 'audio/wav')})
            self.assertEqual(response.status_code, 413, response.text)
            unknown_size = UploadFile(io.BytesIO(b'x' * 33), filename='unknown.wav', size=None)
            with self.assertRaisesRegex(Exception, '200 MB'):
                asyncio.run(uploads.save_upload(unknown_size))
        self.assertEqual(list(paths.UPLOADS_DIR.glob('*')), [])

    def test_expired_upload_purge_does_not_touch_fresh_or_library(self):
        old = self.audio('.runtime/temp/uploads/old.wav')
        fresh = self.audio('.runtime/temp/uploads/fresh.wav')
        voice = self.audio()
        before = time.time() - uploads.UPLOAD_TTL_SECONDS - 30
        os.utime(old, (before, before))
        os.utime(voice, (before, before))
        uploads.purge_expired()
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(voice.exists())

    def test_description_crud_and_collision(self):
        body = {'name': 'Warm narrator', 'text': 'Warm, calm delivery.'}
        saved = self.client.post('/api/emotion-descriptions', json=body)
        self.assertEqual(saved.status_code, 201, saved.text)
        self.assertEqual(saved.json()['name'], 'Warm_narrator.txt')
        self.assertEqual(self.client.get('/api/emotion-descriptions').json(), [saved.json()])
        self.assertEqual(self.client.post('/api/emotion-descriptions', json=body).status_code, 409)
        response = self.client.post('/api/emotion-descriptions', json={**body, 'text': 'New delivery.', 'overwrite': True})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(self.client.delete('/api/emotion-descriptions/Warm_narrator.txt').json(), {'deleted': 1})
        self.assertEqual(self.client.get('/api/emotion-descriptions').json(), [])

    def test_invalid_library_names_and_path_traversal(self):
        source = self.audio()
        for name in ['', 'CON', 'LPT1', 'x' * 121]:
            response = self.client.post('/api/voices', json={'name': name, 'source_path': str(source)})
            self.assertEqual(response.status_code, 400, response.text)
        # A supplied filename may not resolve into another directory, even on Windows.
        response = self.client.delete('/api/voices/..%5Coutside.wav')
        self.assertEqual(response.status_code, 400, response.text)
        self.assertTrue(source.exists())

    def test_bulk_delete_only_audio_and_own_sidecars(self):
        self.audio()
        (paths.VOICES_DIR / 'reference.txt').write_text('transcript', encoding='utf-8')
        notes = paths.VOICES_DIR / 'notes.txt'
        notes.write_text('keep', encoding='utf-8')
        nested = self.audio('voices/subfolder/keep.wav')
        self.assertEqual(self.client.delete('/api/voices').json(), {'deleted': 1})
        self.assertTrue(notes.exists())
        self.assertTrue(nested.exists())
        self.assertFalse((paths.VOICES_DIR / 'reference.txt').exists())

    def test_adapter_discovery_requires_config(self):
        final = paths.TRAINING_OUTPUTS_DIR / 'demo/final_adapter'
        checkpoint = paths.TRAINING_OUTPUTS_DIR / 'demo/checkpoint-100'
        empty = paths.TRAINING_OUTPUTS_DIR / 'demo/checkpoint-200'
        for folder in [final, checkpoint, empty]:
            folder.mkdir(parents=True)
        for folder in [final, checkpoint]:
            (folder / 'adapter_config.json').write_text('{}', encoding='utf-8')
        response = self.client.get('/api/adapters')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([row['kind'] for row in response.json()], ['final', 'checkpoint'])
        self.assertEqual(response.json()[0]['label'], 'demo · final')
