"""Validation/proofreader GUI regressions through the real API, not Gradio AST.

Old cases map to: unbound suggested-folder readiness -> project+proofreader
session; validation update isolation -> real manifest/update; rejected switch
restoration -> unchanged server session after409; persisted switch/interval ->
project disk metadata; start preflight -> the real manager with a tiny trainer.
React pending-context/draft restoration belongs to the ProofreadPanel tests.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from indextts_training.dataset import load_project


class ValidationGuiTests(GuiApiTestCase):
    def setUp(self):
        super().setUp()
        self.training_audio = self.audio('corpus/train.wav', .7)
        self.training_audio.with_suffix('.txt').write_text('训练语料原文。', encoding='utf-8')
        response = self.client.post('/api/projects', json={'name': 'demo', 'source_folder': str(self.training_audio.parent), 'language': 'ZH'})
        self.assertEqual(response.status_code, 201, response.text)
        self.client.put('/api/projects/demo/training', json={'adapter_name': 'demo', 'training_steps': 2,
            'save_steps': 1, 'eval_enabled': False, 'seed': 9876})
        prepared = self.client.put('/api/projects/demo/dataset', json={'prepare': True})
        self.assertEqual(prepared.status_code, 200, prepared.text)
        self.project = prepared.json()
        self.manifest = Path(self.project['manifest'])
        self.project_file = paths.TRAINING_PROJECTS_DIR / 'demo/project.json'
        session = self.client.post('/proofread/api/session')
        self.assertEqual(session.status_code, 200, session.text)
        self.headers = {'X-Session-Token': session.json()['token']}

    def select(self, role, directory, project='demo', language='ZH'):
        return self.client.post('/proofread/api/select', headers=self.headers,
                                json={'directory': str(directory), 'role': role, 'project': project, 'language': language})

    def session(self):
        response = self.client.get('/proofread/api/session', headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def make_validation(self):
        directory = paths.VALIDATION_DIR / 'demo'
        response = self.client.put('/api/projects/demo/validation', json={'source_folder': str(directory)})
        self.assertEqual(response.status_code, 200, response.text)
        audio = self.audio('validation/demo/validation.wav', .9)
        audio.with_suffix('.txt').write_text('不同的验证语料。', encoding='utf-8')
        return directory

    def test_first_validation_folder_binding_keeps_training_settings(self):
        project = self.client.get('/api/projects/demo').json()
        suggested = str(paths.VALIDATION_DIR / 'demo')
        self.assertEqual(project['validation_suggested_folder'], suggested)
        self.assertEqual(project['validation']['source_folder'], '')
        self.assertFalse(project['validation']['enabled'])
        self.assertIsNone(project['validation']['interval_steps'])
        self.assertFalse(Path(suggested).exists())
        context = self.select('validation', project['validation']['source_folder'])
        self.assertEqual(context.status_code, 200, context.text)
        context = context.json()
        self.assertEqual((context['directory'], context['role'], context['project'], context['language']),
                         ('', 'validation', 'demo', 'ZH'))
        status = self.client.get('/proofread/api/manifest/status', headers=self.headers).json()
        self.assertFalse(status['available'])
        before = load_project('demo', paths.TRAINING_DIR)
        before_manifest = self.manifest.read_bytes()
        saved = self.client.put('/api/projects/demo/validation', json={'source_folder': suggested})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertTrue(Path(suggested).is_dir())
        self.assertEqual(saved.json()['validation']['source_folder'], suggested)
        self.assertTrue(saved.json()['validation']['enabled'])
        persisted = load_project('demo', paths.TRAINING_DIR)
        self.assertEqual({key: persisted[key] for key in before}, before)
        self.assertEqual(self.manifest.read_bytes(), before_manifest)
        self.assertEqual(persisted['language'], 'ZH')
        self.assertEqual(persisted['training']['seed'], 9876)
        # Only the explicit advertised default can be created; an arbitrary
        # mistyped source directory is rejected without manufacturing a corpus.
        missing = self.root / 'mistyped-validation-path'
        rejected = self.client.put('/api/projects/demo/validation', json={'source_folder': str(missing)})
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertFalse(missing.exists())
        self.assertEqual(load_project('demo', paths.TRAINING_DIR), persisted)

    def test_validation_workbench_updates_never_replace_training_manifest(self):
        directory = self.make_validation()
        before = load_project('demo', paths.TRAINING_DIR)
        training_bytes = self.manifest.read_bytes()
        self.assertEqual(self.select('validation', directory).status_code, 200)
        response = self.client.post('/proofread/api/manifest/update', headers=self.headers, json={'role': 'validation'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['status']['role'], 'validation')
        persisted = load_project('demo', paths.TRAINING_DIR)
        for key, value in before.items():
            if key != 'validation':
                self.assertEqual(persisted[key], value, key)
        self.assertEqual(self.manifest.read_bytes(), training_bytes)
        validation_manifest = Path(persisted['validation']['manifest'])
        self.assertNotEqual(validation_manifest, self.manifest)
        self.assertTrue(validation_manifest.is_file())
        validation_bytes = validation_manifest.read_bytes()
        # A stale training-tab response must not act on the validation session.
        rejected = self.client.post('/proofread/api/manifest/update', headers=self.headers, json={'role': 'training'})
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertEqual(self.manifest.read_bytes(), training_bytes)
        self.assertEqual(validation_manifest.read_bytes(), validation_bytes)
        self.assertEqual(load_project('demo', paths.TRAINING_DIR), persisted)

    def test_rejected_switch_keeps_role_project_language_and_directory(self):
        # Add a missing transcript so the ASR worker remains at a deterministic barrier.
        self.audio('corpus/missing.wav', .7)
        selected = self.select('training', self.training_audio.parent)
        self.assertEqual(selected.status_code, 200, selected.text)
        before = selected.json()
        entered = threading.Event()
        original = self.state.asr.transcribe
        def pending(audio, cancel_event=None):
            entered.set()
            if cancel_event is not None:
                cancel_event.wait(8)
            return '识别结果'
        self.state.asr.transcribe = pending
        started = self.client.post('/proofread/api/asr/start', headers=self.headers, json={})
        self.assertEqual(started.status_code, 200, started.text)
        try:
            self.assertTrue(entered.wait(5))
            target = paths.VALIDATION_DIR / 'demo'
            rejected = self.select('validation', target, project='another-project', language='EN')
            self.assertEqual(rejected.status_code, 409, rejected.text)
            after = self.session()
            for key in ['token', 'directory', 'role', 'project', 'language']:
                self.assertEqual(after[key], before[key], key)
        finally:
            self.client.post('/proofread/api/asr/stop', headers=self.headers, json={})
            self.wait_until(lambda: not self.client.get('/proofread/api/asr/status', headers=self.headers).json()['job']['running'])
            self.state.asr.transcribe = original
        # A fresh, valid switch after rejection succeeds: no rejected target
        # was latched into the server's session or leaked into project metadata.
        valid_directory = self.make_validation()
        selected = self.select('validation', valid_directory)
        self.assertEqual(selected.status_code, 200, selected.text)
        self.assertEqual(selected.json()['role'], 'validation')
        self.assertEqual(load_project('demo', paths.TRAINING_DIR)['source_folder'], str(self.training_audio.parent))

    def test_project_save_persists_validation_switch_interval_and_folder(self):
        directory = self.make_validation()
        for interval, saved_interval in [(None, None), (0, None), (7, 7)]:
            with self.subTest(interval=interval):
                response = self.client.put('/api/projects/demo/validation', json={
                    'source_folder': str(directory), 'enabled': False, 'interval_steps': interval})
                self.assertEqual(response.status_code, 200, response.text)
                training = self.client.put('/api/projects/demo/training', json={'learning_rate': .001, 'seed': 1234})
                self.assertEqual(training.status_code, 200, training.text)
                saved = load_project('demo', paths.TRAINING_DIR)
                self.assertEqual(saved['validation']['interval_steps'], saved_interval)
                self.assertFalse(saved['validation']['enabled'])
                self.assertEqual(saved['validation']['source_folder'], str(directory))
                reloaded = self.client.get('/api/projects/demo').json()
                self.assertEqual(reloaded['validation']['interval_steps'], saved_interval)
                self.assertFalse(reloaded['validation']['enabled'])
                self.assertEqual(reloaded['training']['seed'], 1234)
        # UI displays None as 0; wire and disk never accidentally persist a
        # disabled validation set as enabled just because its directory exists.
        self.assertEqual(load_project('demo', paths.TRAINING_DIR)['training']['learning_rate'], .001)

    def test_start_allows_no_validation_folder_only_when_switch_is_off(self):
        captures = []
        def tiny_trainer(config, callback, stop_event):
            captures.append(config)
            callback(step=1, total=1, epoch=1, loss=.5)
            return {'output': '', 'stopped': False}
        self.state.training.train_fn = tiny_trainer
        self.client.put('/api/projects/demo/validation', json={'enabled': True})
        output = paths.TRAINING_OUTPUTS_DIR / 'demo'
        self.assertFalse(output.exists())
        rejected = self.client.post('/api/training/start', json={'project': 'demo'})
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertIn('Select and save a validation source folder', rejected.json()['error']['message'])
        self.assertEqual(captures, [])
        self.assertFalse(output.exists())
        self.client.put('/api/projects/demo/validation', json={'enabled': False})
        started = self.client.post('/api/training/start', json={'project': 'demo'})
        self.assertEqual(started.status_code, 202, started.text)
        self.wait_until(lambda: not self.state.training.snapshot()['running'])
        self.state.training.thread.join(10)
        self.assertEqual(len(captures), 1)
        self.assertIsNone(captures[0]['validation'])
        self.assertEqual(self.client.get('/api/training/status').json()['status'], 'succeeded')
        saved = json.loads((output / 'training_config.json').read_text(encoding='utf-8'))
        self.assertIsNone(saved['validation'])
        self.assertEqual(saved['manifest'], str(self.manifest))
        self.assertEqual(saved['seed'], 9876)
