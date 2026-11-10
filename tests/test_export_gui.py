"""Export UI contracts at the real FastAPI/service boundary (no Gradio AST).

The old adapter-info, mode, start/cancel and snapshot callback cases now exercise
HTTP plus the real ExportManager/fake subprocess. Browser-only visibility and
Save-dialog cancellation belong to the React export/FileBrowser tests;
the API guarantees browsing/info/status never starts or changes an export.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from gui_api_support import GuiApiTestCase
from tools.gui_server import paths
from tools.gui_server.fake import _write_adapter


class ExportGuiTests(GuiApiTestCase):
    def setUp(self):
        super().setUp()
        self.adapter = paths.TRAINING_OUTPUTS_DIR / 'voice' / 'checkpoint-step-000100'
        _write_adapter(self.adapter, {'rank': 16, 'alpha': 32})
        self.model_dir = self.state.extras['model_dir']

    def info(self, adapter=None):
        response = self.client.get('/api/export/adapter-info', params={'adapter': str(self.adapter) if adapter is None else adapter})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def finish(self):
        self.wait_until(lambda: self.state.export.snapshot()['status'] not in ('running', 'cancelling'), timeout=20)
        self.state.export.thread.join(10)
        return self.client.get('/api/export/status').json()

    def test_recorded_training_directory_is_metadata_not_a_base_checkpoint(self):
        historical = self.root / 'historical-model-directory'
        historical.mkdir()
        (historical / 'gpt.pth').write_bytes(b'historical placeholder')
        training_config = self.adapter.parent / 'training_config.json'
        for recorded in [historical, self.root / 'missing-old-model']:
            with self.subTest(recorded=str(recorded)):
                training_config.write_text(json.dumps({'model_dir': str(recorded)}), encoding='utf-8')
                data = self.info()
                self.assertEqual(data['filename'], 'voice_checkpoint-step-000100.pth')
                self.assertEqual(data['training_base'], str(recorded))
                self.assertEqual(data['base'], str(self.model_dir / 'gpt.pth'))
                self.assertEqual(data['config'], str(self.model_dir / 'config.yaml'))
                self.assertIn(str(recorded), data['notice'])
                self.assertIn('does not verify', data['notice'])
                self.assertNotIn('unavailable', data['notice'])
        self.assertEqual(self.state.export.snapshot()['status'], 'idle')

    def test_adapter_selection_and_no_selection(self):
        self.assertTrue(self.info()['valid'])
        for selection in ['', '__NONE__', str(self.root / 'missing-adapter')]:
            with self.subTest(selection=selection):
                invalid = self.info(selection)
                self.assertFalse(invalid['valid'])
                self.assertTrue(invalid['error'])
                self.assertEqual(invalid['notice'], invalid['error'])
        self.assertEqual(self.state.export.snapshot()['status'], 'idle')

    def test_browser_and_path_modes_are_explicit_and_browsing_is_read_only(self):
        schema = self.app.openapi()['components']['schemas']['ExportStart']
        self.assertEqual(schema['properties']['mode']['enum'], ['browser', 'path'])
        with patch.object(self.state.export, 'start', wraps=self.state.export.start) as start:
            self.assertEqual(self.client.get('/api/fs/list', params={'path': str(self.root), 'kind': 'pth'}).status_code, 200)
            self.info()
            self.assertEqual(self.client.get('/api/fs/list', params={'path': str(self.root / 'not-found')}).status_code, 404)
            self.assertEqual(self.client.get('/api/export/status').json()['status'], 'idle')
            self.assertEqual(self.state.export.snapshot()['output'], '')
            start.assert_not_called()
        for mode, destination in [('path', 'relative.pth'), ('path', ''), ('browser', None), ('browser', '../escape.pth'), ('desktop', 'export.pth')]:
            response = self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'mode': mode, 'destination': destination})
            self.assertEqual(response.status_code, 400, response.text)
        output = self.root / 'exports chosen by user' / 'voice.pth'
        response = self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'mode': 'path', 'destination': str(output)})
        self.assertEqual(response.status_code, 202, response.text)
        done = self.finish()
        self.assertEqual(done['status'], 'succeeded', done)
        self.assertEqual(Path(done['output']), output)
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix('.merge.json').is_file())

    def test_explicit_base_and_expected_base_are_forwarded_without_cpu_fallback(self):
        chosen_base = self.root / 'custom_fp32.pth'
        chosen_base.write_bytes(b'user selected source')
        expected = self.root / 'expected_training_base.pth'
        expected.write_bytes(b'user selected identity target')
        config = self.root / 'custom_config.yaml'
        config.write_text('fake: true\n', encoding='utf-8')
        # Adapter metadata/info are suggestions, never authority over an explicit choice.
        (self.adapter.parent / 'training_config.json').write_text(json.dumps({'model_dir': str(self.root / 'moved')}), encoding='utf-8')
        self.info()
        with patch('indextts_training.export.subprocess.Popen', wraps=subprocess.Popen) as spawn:
            response = self.client.post('/api/export/start', json={
                'adapter': str(self.adapter), 'mode': 'browser', 'destination': 'explicit.pth',
                'base': str(chosen_base), 'config': str(config), 'expected_base': str(expected)})
            self.assertEqual(response.status_code, 202, response.text)
            done = self.finish()
        self.assertEqual(done['status'], 'succeeded', done)
        command = spawn.call_args.args[0]
        for flag, expected_path in [('--adapter', self.adapter), ('--base', chosen_base),
                                    ('--config', config), ('--expected-base', expected)]:
            self.assertEqual(Path(command[command.index(flag) + 1]), expected_path)
        self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('--skip-cpu-forward', command)
        self.assertEqual(done['report']['base'], str(chosen_base))
        self.assertEqual(done['report']['config'], str(config))
        self.assertEqual(chosen_base.read_bytes(), b'user selected source')
        self.assertEqual(expected.read_bytes(), b'user selected identity target')

    def test_missing_explicit_base_never_falls_back_to_default(self):
        for field in ['base', 'expected_base']:
            with self.subTest(field=field):
                response = self.client.post('/api/export/start', json={
                    'adapter': str(self.adapter), 'destination': 'voice.pth',
                    field: str(self.root / 'does-not-exist.pth')})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn('does-not-exist.pth', response.json()['error']['message'])
                self.assertEqual(self.state.export.snapshot()['status'], 'idle')

    def test_running_finished_and_cancelled_state_control_downloads(self):
        response = self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'destination': 'voice.pth'})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()['status'], 'running')
        self.assertEqual(response.json()['downloads'], {})
        self.assertEqual(self.client.post('/api/export/start', json={'adapter': str(self.adapter)}).status_code, 409)
        cancelled = self.client.post('/api/export/cancel')
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertIn(cancelled.json()['status'], ('cancelling', 'cancelled'))
        self.assertEqual(cancelled.json()['downloads'], {})
        done = self.finish()
        self.assertEqual(done['status'], 'cancelled', done)
        self.assertEqual(done['downloads'], {})
        self.assertFalse(Path(response.json()['output']).exists())
        self.assertEqual(self.client.post('/api/export/start', json={'adapter': str(self.adapter), 'destination': 'again.pth'}).status_code, 202)
        done = self.finish()
        self.assertEqual(done['status'], 'succeeded', done)
        self.assertEqual(set(done['downloads']), {'model', 'report'})
        for url in done['downloads'].values():
            self.assertEqual(self.client.get(url).status_code, 200)
        # Keep every original terminal/active-state case, including diagnostic
        # text. Completed download records must not leak into a newer failure.
        for status in ('idle', 'running', 'cancelling', 'failed', 'cancelled'):
            with self.subTest(status=status):
                with self.state.export.lock:
                    self.state.export.state.update(status=status, error='Full diagnostic text' if status == 'failed' else '')
                data = self.client.get('/api/export/status').json()
                self.assertEqual(data['status'], status)
                self.assertEqual(data['downloads'], {})
                if status == 'failed':
                    self.assertIn('Full diagnostic text', data['error'])

    def test_snapshot_urls_are_encoded_and_json_keeps_untrusted_values_as_data(self):
        with self.state.export.lock:
            self.state.export.state.update(status='succeeded', phase='completed',
                output='model"><script>.pth', task_id='id/with space', token='tok&"',
                report={'base_sha256': 'a' * 64, 'output_sha256': 'b' * 64, 'tensor_count': 456,
                        'lora_target_count': 96, 'changed_tensor_count': 96, 'rank': 4, 'alpha': 8, 'dtype': 'float32',
                        'cpu_forward_max_abs_difference': 0, 'elapsed_seconds': .1,
                        'source_files_unchanged': True, 'round_trip_tensors_identical': False})
        response = self.client.get('/api/export/status')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers['content-type'].startswith('application/json'))
        data = response.json()
        self.assertEqual(data['output'], 'model"><script>.pth')
        self.assertEqual(data['downloads']['model'], '/exports/id%2Fwith%20space/model?token=tok%26%22')
        self.assertEqual(data['downloads']['report'], '/exports/id%2Fwith%20space/report?token=tok%26%22')
        for key, value in self.state.export.snapshot()['report'].items():
            self.assertEqual(data['report'][key], value)
        self.assertIs(data['report']['round_trip_tensors_identical'], False)
        with self.state.export.lock:
            self.state.export.state['report'] = {}
        self.assertEqual(self.client.get('/api/export/status').json()['report'], {})

    def test_all_phases_preserve_codes_and_chinese_notice_translation(self):
        phases = ('idle', 'starting', 'hashing', 'building', 'checking', 'saving', 'verifying', 'publishing', 'completed')
        for phase in phases:
            with self.subTest(phase=phase):
                with self.state.export.lock:
                    self.state.export.state.update(status='running', phase=phase)
                    self.state.export.started = time.monotonic() - 1.25
                response = self.client.get('/api/export/status')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['phase'], phase)
                self.assertGreaterEqual(response.json()['elapsed'], 1.2)
                self.assertEqual(response.json()['downloads'], {})
        with self.state.export.lock:
            self.state.export.state.update(status='idle', phase='idle')
        (self.adapter.parent / 'training_config.json').write_text(json.dumps({'model_dir': '历史路径'}), encoding='utf-8')
        chinese = self.client.get('/api/export/adapter-info', params={'adapter': str(self.adapter)}, headers={'X-UI-Language': 'zh_CN'}).json()
        self.assertIn('历史路径', chinese['notice'])
        self.assertNotIn('does not verify', chinese['notice'])
        self.assertNotEqual(chinese['notice'], self.info()['notice'])
