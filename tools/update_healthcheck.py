"""Offline post-update checks; no model loading, downloads or browser launch."""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', GRADIO_ANALYTICS_ENABLED='False',
                      PYTHONDONTWRITEBYTECODE='1')
    manifest = json.loads((root/'tools/update-manifest.json').read_text(encoding='utf-8'))
    if os.name == 'nt':
        # Parse, never execute, candidate PowerShell entrypoints using Windows PS5.1.
        script = r'''
$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility
$root = $env:INDEXTTS_VALIDATE_ROOT
$manifest = [IO.File]::ReadAllText((Join-Path $root 'tools/update-manifest.json')) | ConvertFrom-Json
foreach ($relative in $manifest.files) {
    if ($relative.EndsWith('.ps1')) {
        $tokens = $null; $errors = $null
        $null = [Management.Automation.Language.Parser]::ParseFile((Join-Path $root $relative), [ref]$tokens, [ref]$errors)
        if ($errors.Count) { throw ($relative + ': ' + ($errors.Message -join '; ')) }
    }
}
'''
        shell = Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
        result = subprocess.run([str(shell), '-NoProfile', '-Command', script],
            env=dict(os.environ, INDEXTTS_VALIDATE_ROOT=str(root)), capture_output=True,
            timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError('PowerShell syntax check failed: ' + result.stderr.decode(errors='replace'))
    count = 0
    for relative in manifest['files']:
        if relative.endswith('.py'):
            path = root / relative
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            count += 1
    for name in ('indextts_training.asr', 'indextts_training.proofread',
                 'indextts_training.export', 'tools.merge_lora_to_pth'):
        importlib.import_module(name)
    import easy_gui
    if Path(easy_gui.__file__).resolve() != root/'easy_gui.py':
        raise RuntimeError('GUI imported from a different installation.')
    config = easy_gui.demo.get_config_file()
    tabs = [item['props'].get('id') for item in config['components'] if item['type'] == 'tabitem']
    if 'tab_export_checkpoint' not in tabs or 'tab_prepare_dataset' not in tabs:
        raise RuntimeError('GUI configuration is missing required pages.')
    print(json.dumps(dict(syntax_files=count, gui_config=True, model_loaded=False, browser_opened=False)), flush=True)


if __name__ == '__main__':
    main()
