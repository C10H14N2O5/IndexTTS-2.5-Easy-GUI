"""Isolated ZIP updater acceptance tests. No network, real environment, or model writes."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.update_core import Updater, UpdateError, REQUIRED, MANIFEST, digest, managed_path

SHA_A = 'a' * 40
SHA_B = 'b' * 40


class SimulatedPowerLoss(BaseException):
    pass


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='zip_update_')
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent/'中文 部署'
        self.root.mkdir()
        self.environment = ['pyproject.toml', 'uv.lock', '.python-version', '1- install.bat']
        self.original = {path: ('original ' + path + '\n').encode() for path in REQUIRED}
        self.original['tools/old.py'] = b'old program\n'
        self.set_manifest(self.original)
        for path, data in self.original.items():
            target = self.root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.protected = {
            'models/model.pth': b'base weights', 'models/asr/model.gguf': b'asr weights',
            'training/outputs/voice/adapter_model.safetensors': b'adapter',
            'training/datasets/voice/manifest.jsonl': b'data', 'voices/sample.wav': b'voice',
            'outputs/result.pth': b'export', '.venv/keep.txt': b'environment',
            '.runtime/gui_settings.json': b'{"custom":1}', 'local-settings.bat': b'proxy config',
            'my_corpus/audio.wav': b'audio', 'my_corpus/audio.txt': b'transcript',
            'tools/custom_user_script.py': b'user code',
        }
        for path, data in self.protected.items():
            target = self.root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.checks = []
        self.updater = Updater(self.root, healthcheck=lambda: self.checks.append('checked'))
        self.initial = self.updater.initialize('local')

    def set_manifest(self, files, environment=None):
        files[MANIFEST] = json.dumps(dict(schema=1, files=sorted(files), environment=environment or self.environment)).encode()

    def package(self, files=None, extra=None):
        files = dict(files if files is not None else self.target())
        path = self.parent / f'package-{len(list(self.parent.glob("package-*")))}.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, data in files.items():
                archive.writestr('repo-commit/' + name, data)
            for name, data in (extra or {}).items():
                archive.writestr(name, data)
        return path

    def target(self):
        files = dict(self.original)
        files['easy_gui.py'] = b'new GUI\n'
        del files['tools/old.py']
        files['tools/new.py'] = b'new code\n'
        self.set_manifest(files)
        return files

    def assert_protected(self):
        for path, data in self.protected.items():
            self.assertEqual((self.root/path).read_bytes(), data, path)

    def assert_original(self):
        for path, data in self.original.items():
            self.assertEqual((self.root/path).read_bytes(), data, path)
        self.assertFalse((self.root/'tools/new.py').exists())
        self.assertEqual(self.updater._state(), self.initial)
        self.assert_protected()

    def test_success_check_no_change_and_manual_rollback(self):
        archive = self.package()
        result = self.updater.update(archive, SHA_A, check=True)
        self.assertEqual(result['status'], 'available')
        self.assert_original()
        self.assertEqual(self.checks, [])
        result = self.updater.update(archive, SHA_A)
        self.assertEqual(result['status'], 'updated')
        self.assertEqual(self.checks, ['checked'])
        self.assertFalse((self.root/'tools/old.py').exists())
        self.assertEqual((self.root/'tools/new.py').read_bytes(), b'new code\n')
        self.assertEqual(self.updater.update(self.parent/'does-not-exist.zip', SHA_A)['status'], 'unchanged')
        self.assert_protected()
        self.assertEqual(self.updater.rollback()['status'], 'rolled_back')
        self.assert_original()
        self.assertFalse(self.updater.last_path.exists())

    def test_dependencies_and_environment_inventory_changes_are_blocked(self):
        for change in ('content', 'inventory'):
            with self.subTest(change=change):
                target = self.target()
                if change == 'content':
                    target['uv.lock'] += b'new dependency'
                else:
                    target['tools/extra-env.py'] = b'env'
                    self.set_manifest(target, self.environment + ['tools/extra-env.py'])
                with self.assertRaisesRegex(UpdateError, 'Dependency/environment'):
                    self.updater.update(self.package(target), SHA_A)
                self.assert_original()
                self.assertFalse(self.updater.journal_path.exists())

    def test_environment_line_endings_are_normalized(self):
        target = self.target()
        for path in self.environment:
            target[path] = target[path].replace(b'\n', b'\r\n')
        self.assertEqual(self.updater.update(self.package(target), SHA_A)['status'], 'updated')

    def test_local_managed_changes_or_new_path_collision_stop_update(self):
        (self.root/'easy_gui.py').write_bytes(b'user edit')
        with self.assertRaisesRegex(UpdateError, 'Local program changes'):
            self.updater.update(self.package(), SHA_A)
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'user edit')
        (self.root/'easy_gui.py').write_bytes(self.original['easy_gui.py'])
        (self.root/'tools/new.py').write_bytes(b'unmanaged')
        with self.assertRaisesRegex(UpdateError, 'unmanaged local file'):
            self.updater.update(self.package(), SHA_A)
        self.assertEqual((self.root/'tools/new.py').read_bytes(), b'unmanaged')
        self.assert_protected()

    def test_rollback_refuses_post_update_local_changes(self):
        self.updater.update(self.package(), SHA_A)
        (self.root/'easy_gui.py').write_bytes(b'changed after update')
        with self.assertRaisesRegex(UpdateError, 'Local program changes'):
            self.updater.rollback()
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'changed after update')
        self.assertTrue(self.updater.last_path.exists())

    def test_protected_paths_cannot_be_claimed_by_manifest(self):
        for path in self.protected:
            if path == 'tools/custom_user_script.py':
                continue
            with self.subTest(path=path):
                target = self.target()
                target[path] = b'attack'
                self.set_manifest(target)
                with self.assertRaises(UpdateError):
                    self.updater.update(self.package(target), SHA_A)
                self.assert_original()

    def test_archive_validation_rejects_traversal_case_collisions_and_multiple_roots(self):
        for name in ('repo-commit/../escape.py', 'repo-commit/C:/escape.py', 'repo-commit/CON',
                     'repo-commit/tools/eVIL. ', 'repo-commit/EASY_GUI.PY', 'other/file.py', 'repo-commit/tools/new.py/child',
                     'repo-commit/Tools/another.py'):
            with self.subTest(name=name):
                with self.assertRaises(UpdateError):
                    self.updater.update(self.package(extra={name:b'bad'}), SHA_A)
                self.assert_original()

    def test_archive_symlink_and_truncated_zip_are_rejected(self):
        archive = self.package()
        with zipfile.ZipFile(archive, 'a') as package:
            info = zipfile.ZipInfo('repo-commit/tools/link.py')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, '../../outside')
        with self.assertRaisesRegex(UpdateError, 'link or special'):
            self.updater.update(archive, SHA_A)
        archive.write_bytes(b'not a ZIP')
        with self.assertRaises(zipfile.BadZipFile):
            self.updater.update(archive, SHA_A)
        self.assert_original()

    def test_missing_manifest_and_missing_required_file_rejected(self):
        for name in (MANIFEST, 'easy_gui.py'):
            target = self.target()
            del target[name]
            with self.assertRaises(UpdateError):
                self.updater.update(self.package(target), SHA_A)
            self.assert_original()

    def test_health_failure_restores_every_program_file(self):
        def fail():
            raise RuntimeError('import check failed')
        updater = Updater(self.root, healthcheck=fail)
        with self.assertRaisesRegex(UpdateError, 'previous version was restored'):
            updater.update(self.package(), SHA_A)
        self.assert_original()
        self.assertFalse(updater.journal_path.exists())

    def test_partial_copy_failure_restores_old_and_removes_new_files(self):
        def fail(event, detail):
            if event == 'file' and detail['path'] == 'tools/new.py':
                raise OSError('simulated disk failure')
        updater = Updater(self.root, healthcheck=lambda: None, fault=fail)
        with self.assertRaisesRegex(UpdateError, 'previous version was restored'):
            updater.update(self.package(), SHA_A)
        self.assert_original()

    def test_power_loss_recovery_and_check_refuses_pending_journal(self):
        def interrupt(event, detail):
            if event == 'file':
                raise SimulatedPowerLoss()
        updater = Updater(self.root, healthcheck=lambda: None, fault=interrupt)
        with self.assertRaises(SimulatedPowerLoss):
            updater.update(self.package(), SHA_A)
        self.assertTrue(updater.journal_path.exists())
        with self.assertRaisesRegex(UpdateError, 'Interrupted update'):
            self.updater.update(self.package(), SHA_A, check=True)
        self.assertEqual(self.updater.recover()['status'], 'recovered')
        self.assert_original()

    def test_killed_process_recovers_files_and_cleans_only_owned_staging(self):
        archive = self.package()
        unknown = self.root/'.runtime/update/staging'/('f' * 32)
        unknown.mkdir(parents=True)
        (unknown/'user.txt').write_bytes(b'leave me')
        script = (
            'import os,sys; from pathlib import Path; from tools.update_core import Updater; '
            'fault=lambda event,detail: os._exit(77) if event=="file" else None; '
            'Updater(Path(sys.argv[1]),healthcheck=lambda:None,fault=fault).update(Path(sys.argv[2]),sys.argv[3])'
        )
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', script, str(self.root), str(archive), SHA_A],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 77, result.stderr)
        self.assertTrue(self.updater.journal_path.exists())
        self.assertGreater(len(list((self.root/'.runtime/update/staging').iterdir())), 1)
        self.assertEqual(self.updater.recover()['status'], 'recovered')
        self.assertEqual(list((self.root/'.runtime/update/staging').iterdir()), [unknown])
        self.assertEqual((unknown/'user.txt').read_bytes(), b'leave me')
        self.assert_original()

    def test_power_loss_after_metadata_restores_previous_version_and_previous_backup(self):
        self.updater.update(self.package(), SHA_A)
        first_last = self.updater.last_path.read_bytes()
        target = self.target()
        target['easy_gui.py'] = b'third GUI'
        def interrupt(event, detail):
            if event == 'metadata':
                raise SimulatedPowerLoss()
        updater = Updater(self.root, healthcheck=lambda: None, fault=interrupt)
        with self.assertRaises(SimulatedPowerLoss):
            updater.update(self.package(target), SHA_B)
        self.assertEqual(self.updater.recover()['status'], 'recovered')
        self.assertEqual(self.updater._state()['commit'], SHA_A)
        self.assertEqual(self.updater.last_path.read_bytes(), first_last)
        self.updater.rollback()
        self.assert_original()

    def test_recovery_conflict_preserves_journal_backups_and_foreign_file(self):
        def interrupt(event, detail):
            if event == 'file':
                raise SimulatedPowerLoss()
        updater = Updater(self.root, healthcheck=lambda: None, fault=interrupt)
        with self.assertRaises(SimulatedPowerLoss):
            updater.update(self.package(), SHA_A)
        (self.root/'easy_gui.py').write_bytes(b'foreign edit after crash')
        with self.assertRaisesRegex(UpdateError, 'Could not restore'):
            self.updater.recover()
        self.assertTrue(self.updater.journal_path.exists())
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'foreign edit after crash')
        self.assertTrue(list((self.root/'.runtime/update/backups').iterdir()))

    def test_only_most_recent_successful_backup_is_retained(self):
        self.updater.update(self.package(), SHA_A)
        first = json.loads(self.updater.last_path.read_text())['id']
        target = self.target()
        target['easy_gui.py'] = b'next GUI'
        self.updater.update(self.package(target), SHA_B)
        backups = list((self.root/'.runtime/update/backups').iterdir())
        self.assertEqual(len(backups), 1)
        self.assertNotEqual(backups[0].name, first)
        self.updater.rollback()
        self.assertEqual(self.updater._state()['commit'], SHA_A)
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'new GUI\n')

    def test_committed_cleanup_error_never_reports_rollback(self):
        self.updater.update(self.package(), SHA_A)
        target = self.target()
        target['easy_gui.py'] = b'next GUI'
        original_remove = self.updater._remove_owned
        def fail_old_backup(category, identifier):
            if category == 'backups':
                raise PermissionError('old backup locked')
            return original_remove(category, identifier)
        with patch.object(self.updater, '_remove_owned', side_effect=fail_old_backup):
            result = self.updater.update(self.package(target), SHA_B)
        self.assertEqual(result['status'], 'updated')
        self.assertEqual(self.updater._state()['commit'], SHA_B)
        self.assertFalse(self.updater.journal_path.exists())

    def test_committed_journal_unlink_error_is_recoverable_without_false_rollback(self):
        original_unlink = Path.unlink
        journal_path = self.updater.journal_path
        def fail_journal(path, *args, **kwargs):
            if path == journal_path:
                raise PermissionError('journal locked')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail_journal):
            with self.assertRaisesRegex(UpdateError, 'no rollback was performed'):
                self.updater.update(self.package(), SHA_A)
        self.assertEqual(self.updater._state()['commit'], SHA_A)
        self.assertEqual(self.updater.recover()['status'], 'completed')
        self.assertFalse(self.updater.journal_path.exists())
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'new GUI\n')

    def test_corrupt_backup_preserves_incomplete_recovery_record(self):
        def interrupt(event, detail):
            if event == 'file':
                raise SimulatedPowerLoss()
        updater = Updater(self.root, healthcheck=lambda: None, fault=interrupt)
        with self.assertRaises(SimulatedPowerLoss):
            updater.update(self.package(), SHA_A)
        journal = json.loads(updater.journal_path.read_text())
        (self.root/'.runtime/update/backups'/journal['id']/'easy_gui.py').write_bytes(b'damaged backup')
        with self.assertRaisesRegex(UpdateError, 'Backup is missing or modified'):
            self.updater.recover()
        self.assertTrue(self.updater.journal_path.exists())

    def test_failed_manual_rollback_restores_new_version_and_keeps_backup(self):
        self.updater.update(self.package(), SHA_A)
        last = self.updater.last_path.read_bytes()
        def fail_check():
            raise RuntimeError('old version health check failed')
        updater = Updater(self.root, healthcheck=fail_check)
        with self.assertRaisesRegex(UpdateError, 'previous version was restored'):
            updater.rollback()
        self.assertEqual(self.updater._state()['commit'], SHA_A)
        self.assertEqual(self.updater.last_path.read_bytes(), last)
        self.assertEqual((self.root/'easy_gui.py').read_bytes(), b'new GUI\n')

    def test_git_worktree_is_refused_and_uninitialized_recovery_is_clean(self):
        (self.root/'.git').mkdir()
        with self.assertRaisesRegex(UpdateError, 'Git working'):
            Updater(self.root)
        (self.root/'.git').rmdir()
        self.updater.current_path.unlink()
        self.assertEqual(self.updater.recover()['status'], 'clean')
        self.assertFalse(self.updater.current_path.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows file-sharing behavior')
    def test_locked_destination_fails_without_losing_original_files(self):
        with (self.root/'easy_gui.py').open('rb'):
            with self.assertRaises(UpdateError):
                self.updater.update(self.package(), SHA_A)
        self.assert_original()

    def test_cli_initialize_runs_without_git_in_path(self):
        self.updater.current_path.unlink()
        script = Path(__file__).resolve().parents[1]/'tools/update_core.py'
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(script), '--root', str(self.root), '--initialize'],
            capture_output=True, text=True, encoding='utf-8', timeout=30,
            env=dict(os.environ, PATH='', PYTHONUTF8='1'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'initialized')
        self.assert_original()


if __name__ == '__main__':
    unittest.main()
