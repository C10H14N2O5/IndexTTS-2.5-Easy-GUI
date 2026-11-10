"""Thin launcher compatibility without a model, GPU, browser or live server."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_import_is_model_and_ui_free(self):
        code = (
            "import sys; import easy_gui; "
            "assert callable(easy_gui.main); "
            "assert not any(x in sys.modules for x in "
            "('torch', 'gradio', 'uvicorn', 'tools.gui_server.app', 'indextts.infer_v2'))"
        )
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lease_and_update_guard_precede_launch(self):
        import easy_gui
        calls = []
        with patch('tools.runtime_lock.gui_application_lock', side_effect=lambda root: calls.append(('lock', root))), \
             patch('tools.runtime_lock.require_clean_update', side_effect=lambda root: calls.append(('guard', root))), \
             patch('tools.gui_server.launcher.main', side_effect=lambda argv: calls.append(('launch', argv))):
            easy_gui.main(['--no-browser', '--port', '7861'])
        self.assertEqual(calls, [('lock', ROOT), ('guard', ROOT),
                                 ('launch', ['--no-browser', '--port', '7861'])])

    def test_pending_update_never_starts_server(self):
        import easy_gui
        with patch('tools.runtime_lock.gui_application_lock'), \
             patch('tools.runtime_lock.require_clean_update', side_effect=RuntimeError('recovery required')), \
             patch('tools.gui_server.launcher.main') as launch:
            with self.assertRaisesRegex(RuntimeError, 'recovery required'):
                easy_gui.main([])
        launch.assert_not_called()


class LauncherTests(unittest.TestCase):
    def test_cli_forwards_explicit_fake_root_and_browser_option(self):
        from tools.gui_server import launcher
        with patch.object(launcher, 'serve') as serve:
            launcher.main(['--fake', '--data-dir', '.runtime/demo', '--no-browser',
                           '--host', '192.168.50.116', '--port', '7862'])
        serve.assert_called_once_with('192.168.50.116', 7862, fake=True,
                                      open_browser=False, data_dir='.runtime/demo')

    def test_cli_keeps_environment_selection_when_fake_flag_absent(self):
        from tools.gui_server import launcher
        with patch.object(launcher, 'serve') as serve:
            launcher.main([])
        serve.assert_called_once_with(None, None, fake=None, open_browser=True, data_dir=None)

    def test_data_redirect_is_never_allowed_in_production(self):
        from tools.gui_server import launcher
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(launcher.paths, 'use_data_root') as redirect, \
             patch.object(launcher.runtime_env, 'configure') as configure:
            with self.assertRaisesRegex(ValueError, 'only with --fake'):
                launcher.serve(fake=False, data_dir='unsafe')
        redirect.assert_not_called()
        configure.assert_not_called()

    def test_browser_address_handles_wildcards_and_ipv6(self):
        from tools.gui_server.launcher import _browser_url
        for host in ('0.0.0.0', '::', ''):
            self.assertEqual(_browser_url(host, 7860), 'http://127.0.0.1:7860/')
        self.assertEqual(_browser_url('::1', 7860), 'http://[::1]:7860/')
        self.assertEqual(_browser_url('192.168.50.116', 7860), 'http://192.168.50.116:7860/')

    def test_fake_launch_skips_model_and_uses_single_worker(self):
        from tools.gui_server import launcher
        application = object()
        with patch.dict(os.environ, {'INDEXTTS_HOST': '127.0.0.1', 'INDEXTTS_PORT': '7863'}), \
             patch.object(launcher.runtime_env, 'configure'), \
             patch.object(launcher.console, 'install'), \
             patch.object(launcher, 'ensure_model') as ensure, \
             patch.object(launcher, '_open_when_ready') as browser, \
             patch('tools.gui_server.app.create_app', return_value=application) as create, \
             patch('uvicorn.run') as run:
            launcher.serve(fake=True, open_browser=False)
        ensure.assert_not_called()
        browser.assert_not_called()
        create.assert_called_once_with(fake=True)
        self.assertIs(run.call_args.args[0], application)
        self.assertEqual(run.call_args.kwargs['workers'], 1)
        self.assertEqual(run.call_args.kwargs['host'], '127.0.0.1')
        self.assertEqual(run.call_args.kwargs['port'], 7863)

    def test_production_checks_model_before_app_and_opens_browser(self):
        from tools.gui_server import launcher
        order = []
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(launcher.runtime_env, 'configure', side_effect=lambda: order.append('env')), \
             patch.object(launcher.console, 'install', side_effect=lambda: order.append('console')), \
             patch.object(launcher, 'ensure_model', side_effect=lambda: order.append('model')), \
             patch.object(launcher, '_open_when_ready') as browser, \
             patch('tools.gui_server.app.create_app', side_effect=lambda **_: order.append('app')), \
             patch('uvicorn.run'):
            launcher.serve('0.0.0.0', 7864, fake=False)
        self.assertEqual(order, ['env', 'model', 'console', 'app'])
        browser.assert_called_once_with('0.0.0.0', 7864, 'http://127.0.0.1:7864/')


if __name__ == '__main__':
    unittest.main()
