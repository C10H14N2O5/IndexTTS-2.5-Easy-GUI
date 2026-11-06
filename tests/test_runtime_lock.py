"""Real Windows mutex interoperability and guarded installer checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.runtime_lock import ApplicationLock, lock_name, require_clean_update

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which('powershell.exe')


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='runtime-lock-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'中文 空格'
        self.root.mkdir()

    def test_process_conflict_and_release(self):
        command = [sys.executable, '-B', '-c',
                   'import sys; from pathlib import Path; from tools.runtime_lock import ApplicationLock; '
                   'lease=ApplicationLock(Path(sys.argv[1])); lease.__enter__(); print("acquired"); lease.close()', str(self.root)]
        with ApplicationLock(self.root):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('acquired', result.stdout)

    def test_same_process_conflict_and_independent_roots(self):
        with ApplicationLock(self.root):
            with self.assertRaises(RuntimeError):
                with ApplicationLock(self.root):
                    pass
            with ApplicationLock(self.root.parent):
                pass

    def test_unfinished_transaction_blocks_startup(self):
        require_clean_update(self.root)
        journal = self.root/'.runtime/update/transaction.json'
        journal.parent.mkdir(parents=True)
        journal.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'interrupted update'):
            require_clean_update(self.root)

    def test_launcher_keeps_lease_through_preflight_and_gui(self):
        tools = self.root/'tools'
        tools.mkdir()
        for filename in ('run_gui.py', 'runtime_lock.py'):
            shutil.copyfile(ROOT/'tools'/filename, tools/filename)
        for filename in ('ensure_ffmpeg.py', 'verify_cuda.py', 'verify_numpy.py'):
            (tools/filename).write_text("from pathlib import Path\nPath('preflight.txt').write_text('checked')\n")
        (self.root/'easy_gui.py').write_text("from tools.runtime_lock import gui_application_lock\nfrom pathlib import Path\ngui_application_lock(Path.cwd())\nPath('gui.txt').write_text('started')\n")
        command = [sys.executable, '-B', str(tools/'run_gui.py')]
        with ApplicationLock(self.root):
            result = subprocess.run(command, cwd=self.root, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root/'preflight.txt').exists())
        result = subprocess.run(command, cwd=self.root, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertTrue((self.root/'gui.txt').exists())

    @unittest.skipUnless(POWERSHELL, 'Windows PowerShell required')
    def test_powershell_same_mutex_and_installer_no_execution_when_busy(self):
        tools = self.root/'tools'
        tools.mkdir()
        shutil.copyfile(ROOT/'tools/install.ps1', tools/'install.ps1')
        (tools/'install_runtime.bat').write_text('@echo off\r\necho executed>executed.txt\r\nexit /b 7\r\n')
        command = [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(tools/'install.ps1'), '-Root', str(self.root)]
        with ApplicationLock(self.root):
            result = subprocess.run(command, cwd=self.root, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root/'executed.txt').exists())
        result = subprocess.run(command, cwd=self.root, capture_output=True)
        self.assertEqual(result.returncode, 7, result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))
        self.assertTrue((self.root/'executed.txt').exists())

    @unittest.skipUnless(POWERSHELL, 'Windows PowerShell required')
    def test_installer_blocks_pending_transaction(self):
        tools = self.root/'tools'
        tools.mkdir()
        shutil.copyfile(ROOT/'tools/install.ps1', tools/'install.ps1')
        (tools/'install_runtime.bat').write_text('@echo off\r\necho executed>executed.txt\r\n')
        journal = self.root/'.runtime/update/transaction.json'
        journal.parent.mkdir(parents=True)
        journal.write_text('{}')
        result = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(tools/'install.ps1'), '-Root', str(self.root)], cwd=self.root, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root/'executed.txt').exists())


if __name__ == '__main__':
    unittest.main()
