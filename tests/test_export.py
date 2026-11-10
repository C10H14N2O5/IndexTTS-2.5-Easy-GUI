"""CPU export orchestration checks; dummy subprocesses never import model code."""
from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from indextts_training import activity
from indextts_training.export import ExportManager, adapter_details, choose_export_path, create_export_router
from indextts_training.manager import TrainingManager


FAKE_CLI = r'''
import argparse, json, pathlib, sys, time
p=argparse.ArgumentParser()
for name in ('adapter','base','config','output'):
    p.add_argument('--'+name, required=True)
p.add_argument('--expected-base')
a=p.parse_args()
cfg=json.loads(pathlib.Path(a.config).read_text(encoding='utf-8'))
print('EXPORT_PHASE:merging',flush=True)
print('中文导出日志',flush=True)
mode=cfg.get('mode','success')
if mode=='sleep': time.sleep(30)
if mode=='exit': sys.exit(7)
out=pathlib.Path(a.output)
out.write_bytes(b'DUMMY-GPT-CHECKPOINT-0123456789')
report=out.with_suffix('.merge.json')
if mode=='missing-report': sys.exit(0)
if mode=='malformed-report': report.write_text('{',encoding='utf-8');sys.exit(0)
report.write_text(json.dumps({'output':str(out),'source_files_unchanged':True,
    'round_trip_tensors_identical':mode!='incomplete-report','base_sha256':'test',
    'tensor_count':2}),encoding='utf-8')
print('EXPORT_PHASE:completed',flush=True)
'''


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="export_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文 项目"
        (self.root / "tools").mkdir(parents=True)
        (self.root / "tools/merge_lora_to_pth.py").write_text(FAKE_CLI, encoding="utf-8")
        self.adapter = self.root / "训练/角色/final_adapter"
        self.adapter.mkdir(parents=True)
        (self.adapter / "adapter_config.json").write_text(json.dumps({"peft_type":"LORA","bias":"none"}), encoding="utf-8")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"dummy adapter")
        self.base = self.root / "原始模型.pth"
        self.base.write_bytes(b"original base")
        self.config = self.root / "config.json"
        self.config.write_text('{}', encoding="utf-8")
        self.manager = ExportManager(self.root)
        self.addCleanup(self.stop_manager)

    def stop_manager(self):
        self.manager.cancel()
        if self.manager.thread is not None:
            self.manager.thread.join(8)
            self.assertFalse(self.manager.thread.is_alive(), "Export worker did not stop")
        token = activity.acquire('test-idle-check')
        activity.release(token)

    def start(self, mode='browser', destination='角色.pth'):
        return self.manager.start(str(self.adapter), mode, destination, str(self.base), str(self.config))

    def finish(self):
        self.manager.thread.join(8)
        self.assertFalse(self.manager.thread.is_alive())
        return self.manager.snapshot()

    def write_mode(self, mode):
        self.config.write_text(json.dumps({'mode':mode}), encoding="utf-8")

    def wait_for_phase(self, phase):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.manager.snapshot()
            if state['phase'] == phase:
                return
            if state['status'] not in ('running','cancelling'):
                self.fail(state)
            time.sleep(.02)
        self.fail(f'Export did not reach phase {phase}')

    def assert_clean(self, parent):
        self.assertEqual(list(parent.glob('.gpt-export-*')), [])

    def test_success_chinese_paths_logs_and_sources_preserved(self):
        before = self.base.read_bytes()
        self.start()
        result = self.finish()
        self.assertEqual(result['status'], 'succeeded', result)
        out = Path(result['output'])
        self.assertEqual(out.name, '角色.pth')
        self.assertEqual(out.read_bytes(), b'DUMMY-GPT-CHECKPOINT-0123456789')
        report = json.loads(out.with_suffix('.merge.json').read_text(encoding='utf-8'))
        self.assertEqual(report['output'], str(out))
        self.assertIn('中文导出日志', result['log'])
        self.assertEqual(self.base.read_bytes(), before)
        self.assert_clean(out.parent)

    def test_absolute_destination_and_no_overwrite(self):
        destination = self.root / '导出目录/自定义名字.pth'
        self.start('path', str(destination))
        self.assertEqual(self.finish()['status'], 'succeeded')
        original = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.start('path', str(destination))
        self.assertEqual(destination.read_bytes(), original)

    def test_invalid_filename_and_destination_rejected_before_running(self):
        for filename in ('../escape.pth', r'a\escape.pth', 'CON.pth', 'NUL.pth', 'bad.wav', '', 'a?.pth'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.start(destination=filename)
        for destination in ('relative.pth', str(self.base), str(self.adapter/'output.pth')):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.start('path', destination)
        self.assertEqual(self.manager.snapshot()['status'], 'idle')

    def test_existing_report_prevents_export(self):
        output = self.root / 'result.pth'
        output.with_suffix('.merge.json').write_text('existing', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.start('path', str(output))
        self.assertFalse(output.exists())
        self.assertEqual(output.with_suffix('.merge.json').read_text(), 'existing')

    def test_incomplete_reports_and_process_failure_do_not_publish(self):
        for mode in ('exit', 'missing-report', 'malformed-report', 'incomplete-report'):
            with self.subTest(mode=mode):
                self.write_mode(mode)
                self.start()
                result = self.finish()
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['error'])
                output = Path(result['output'])
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix('.merge.json').exists())
                self.assert_clean(output.parent)

    def test_cancel_terminates_subprocess_and_releases_activity(self):
        self.write_mode('sleep')
        self.start()
        self.wait_for_phase('merging')
        self.manager.cancel()
        result = self.finish()
        self.assertEqual(result['status'], 'cancelled')
        self.assertFalse(Path(result['output']).exists())
        self.assert_clean(Path(result['output']).parent)
        self.assertIsNone(self.manager.process)
        self.write_mode('success')
        self.start()
        self.assertEqual(self.finish()['status'], 'succeeded')

    def test_worker_start_failure_releases_gate_and_cleans_staging(self):
        with patch('indextts_training.export.threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'thread unavailable'):
                self.start()
        self.manager.thread = None  # An unstarted Thread cannot be joined.
        result = self.manager.snapshot()
        self.assertEqual(result['status'], 'failed')
        self.assert_clean(Path(result['output']).parent)
        token = activity.acquire('training')
        activity.release(token)

    def test_training_cannot_start_during_export(self):
        self.write_mode('sleep')
        self.start()
        self.wait_for_phase('merging')
        training = TrainingManager()
        with self.assertRaisesRegex(RuntimeError, 'export is already running'):
            training.start({'manifest': str(self.base), 'project':None})
        self.assertFalse(training.snapshot()['running'])
        self.manager.cancel()
        self.finish()

    def test_export_cannot_start_during_training_and_after_stop_request(self):
        entered, finish = threading.Event(), threading.Event()
        def fake_train(config, callback, stop_event):
            entered.set()
            finish.wait(5)
            return {'output':'dummy', 'stopped':stop_event.is_set()}
        fake_module = types.ModuleType('indextts_training.trainer')
        fake_module.train_lora = fake_train
        training = TrainingManager()
        with patch.dict('sys.modules', {'indextts_training.trainer':fake_module}):
            training.start({'manifest':str(self.base), 'project':None})
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaisesRegex(RuntimeError, 'training is already running'):
                    self.start()
                training.stop()
                with self.assertRaisesRegex(RuntimeError, 'training is already running'):
                    self.start()
            finally:
                finish.set()
                training.thread.join(5)
        self.start()
        self.assertEqual(self.finish()['status'], 'succeeded')

    def test_download_token_range_report_and_missing_file(self):
        self.start()
        state = self.finish()
        self.assertEqual(state['status'], 'succeeded')
        app = FastAPI()
        app.include_router(create_export_router(self.manager))
        path = f"/exports/{state['task_id']}"
        with TestClient(app) as client:
            self.assertEqual(client.get(path+'/model').status_code,404)
            self.assertEqual(client.get(path+'/model?token=wrong').status_code,404)
            self.assertEqual(client.get(path+'/unknown?token='+state['token']).status_code,404)
            response = client.get(path+'/model?token='+state['token'], headers={'Range':'bytes=0-4'})
            self.assertEqual(response.status_code,206)
            self.assertEqual(response.content,b'DUMMY')
            self.assertIn('attachment', response.headers['content-disposition'])
            response = client.get(path+'/report?token='+state['token'])
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.json()['source_files_unchanged'])
            Path(state['output']).unlink()
            self.assertEqual(client.get(path+'/model?token='+state['token']).status_code,404)

    def test_old_download_remains_bound_after_next_export(self):
        self.start(destination='first.pth')
        first = self.finish()
        self.start(destination='second.pth')
        second = self.finish()
        self.assertNotEqual(first['token'], second['token'])
        app = FastAPI()
        app.include_router(create_export_router(self.manager))
        with TestClient(app) as client:
            url=f"/exports/{first['task_id']}/model"
            self.assertEqual(client.get(url+'?token='+first['token']).status_code,200)
            self.assertEqual(client.get(url+'?token='+second['token']).status_code,404)

    def test_publication_failure_leaves_no_download(self):
        with patch('tools.merge_lora_to_pth.publish_export', side_effect=OSError('publish failed')):
            self.start()
            result = self.finish()
        self.assertEqual(result['status'],'failed')
        self.assertNotIn(result['task_id'], self.manager.completed)
        self.assertFalse(Path(result['output']).exists())
        self.assert_clean(Path(result['output']).parent)

    def test_adapter_configuration_validation(self):
        self.assertTrue(adapter_details(str(self.adapter))['valid'])
        config = self.adapter / 'adapter_config.json'
        config.write_text(json.dumps({'peft_type':'LORA','bias':'none','use_dora':True}), encoding='utf-8')
        self.assertFalse(adapter_details(str(self.adapter))['valid'])
        with self.assertRaisesRegex(ValueError,'Unsupported adapter option'):
            self.start()

    def test_process_teardown_error_still_releases_activity(self):
        staging = self.root / '.gpt-export-fault'
        staging.mkdir()
        output = self.root / 'cleanup-fault.pth'
        logfile = self.root / 'cleanup-fault.log'
        proc = Mock()
        proc.stdout = io.StringIO('')
        proc.wait.return_value = 0
        proc.poll.return_value = None
        proc.kill.side_effect = OSError('simulated process cleanup failure')
        token = activity.acquire('export')
        try:
            with patch('indextts_training.export.subprocess.Popen', return_value=proc):
                try:
                    self.manager._worker(['fake'], staging, output, logfile, token)
                except OSError:
                    pass
            probe = activity.acquire('training')
            activity.release(probe)
        finally:
            activity.release(token)
            self.manager.process = None

    def test_training_thread_constructor_failure_releases_activity(self):
        training = TrainingManager()
        try:
            with patch('indextts_training.manager.threading.Thread', side_effect=RuntimeError('constructor failed')):
                with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                    training.start({'manifest':str(self.base),'project':None})
            probe = activity.acquire('export')
            activity.release(probe)
        finally:
            # A failed assertion must not poison later unrelated tests.
            with activity._LOCK:
                if activity._ACTIVE is not None:
                    activity.release(activity._ACTIVE[1])


class SaveAsTests(unittest.TestCase):
    def test_cancel_returns_empty_and_destroys_dialog(self):
        root = Mock()
        with patch('tkinter.Tk', return_value=root), patch('tkinter.filedialog.asksaveasfilename', return_value='') as dialog:
            selected, error = choose_export_path(r'D:\导出\角色.pth')
        self.assertEqual((selected,error), ('',''))
        root.destroy.assert_called_once()
        self.assertEqual(dialog.call_args.kwargs['defaultextension'], '.pth')
        self.assertTrue(dialog.call_args.kwargs['confirmoverwrite'])

    def test_selected_chinese_path_returned_exactly(self):
        path = r'D:\中文目录\新角色.pth'
        with patch('tkinter.Tk', return_value=Mock()), patch('tkinter.filedialog.asksaveasfilename', return_value=path):
            self.assertEqual(choose_export_path('initial.pth'), (path,''))

    def test_native_dialog_failure_is_explicit_and_retains_value(self):
        root = Mock()
        with patch('tkinter.Tk', return_value=root), patch('tkinter.filedialog.asksaveasfilename', side_effect=RuntimeError('dialog unavailable')):
            selected, error = choose_export_path('keep.pth')
        self.assertEqual(selected,'keep.pth')
        self.assertIn('dialog unavailable',error)
        root.destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
