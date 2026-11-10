"""Background CPU exports with task-scoped downloads and exclusive publication."""
from __future__ import annotations

import atexit
from collections import deque
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .activity import acquire, release


def adapter_details(path: str) -> dict:
    adapter = Path(str(path or '')).expanduser()
    name = f'{adapter.parent.name}_{adapter.name}'
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip('. ') or 'gpt'
    result = {'filename': name + '.pth', 'training_base': '', 'valid': False, 'error': ''}
    try:
        if not path or path == '__NONE__':
            raise ValueError('Select a LoRA adapter directory.')
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            if not (adapter / filename).is_file():
                raise ValueError(f'Missing adapter file: {adapter / filename}')
        cfg = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8-sig'))
        if cfg.get('peft_type') != 'LORA' or cfg.get('bias') != 'none':
            raise ValueError('Expected a standard, bias-free acoustic GPT LoRA.')
        for key in ('modules_to_save', 'use_dora', 'use_rslora', 'rank_pattern', 'alpha_pattern', 'lora_bias', 'use_qalora', 'layer_replication'):
            if cfg.get(key):
                raise ValueError(f'Unsupported adapter option: {key}')
        result['valid'] = True
        training_config = adapter.parent / 'training_config.json'
        if training_config.is_file():
            try:
                result['training_base'] = str(json.loads(training_config.read_text(encoding='utf-8-sig')).get('model_dir', ''))
            except (OSError, ValueError, AttributeError):
                pass
    except (OSError, ValueError, AttributeError) as exc:
        result['error'] = str(exc)
    return result


def choose_export_path(current: str = '', title: str = '') -> tuple[str, str]:
    """Native Save As runs on the server; cancellation never implies acceptance."""
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        initial = Path(current or 'gpt.pth').expanduser()
        selected = filedialog.asksaveasfilename(
            parent=root, title=title or 'Export GPT checkpoint',
            initialdir=str(initial.parent.resolve()), initialfile=initial.name,
            defaultextension='.pth', filetypes=[('GPT checkpoint', '*.pth')],
            confirmoverwrite=True,
        )
        return str(selected or ''), ''
    except Exception as exc:
        return str(current or ''), f'{type(exc).__name__}: {exc}'
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _filename(value: str) -> str:
    if not value or value != Path(value).name or re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
        raise ValueError('Enter a filename without directory separators.')
    if value.rstrip('. ') != value or Path(value).suffix.lower() != '.pth':
        raise ValueError('The output filename must end in .pth.')
    if Path(value).stem.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Reserved Windows filename.')
    return value


class ExportManager:
    def __init__(self, root: Path, script: Path | None = None):
        self.root = Path(root).resolve()
        # Development fakes inject a stand-in merge script; production uses merge_lora_to_pth.py.
        self.script = Path(script) if script else self.root / 'tools' / 'merge_lora_to_pth.py'
        self.lock = threading.RLock()
        self.thread = None
        self.process = None
        self.stop_event = threading.Event()
        self.lines = deque(maxlen=250)
        self.completed = {}
        self.state = dict(status='idle', phase='idle', elapsed=0.0, log='', output='',
                          error='', report={}, token='', task_id='', logfile='')
        self.started = 0.0
        self.on_finished = None
        atexit.register(self.cancel)

    def snapshot(self) -> dict:
        with self.lock:
            state = dict(self.state)
            state['report'] = dict(state['report'])
            state['log'] = ''.join(self.lines)[-24000:]
            if state['status'] in ('running', 'cancelling'):
                state['elapsed'] = time.monotonic() - self.started
            return state

    def start(self, adapter: str, mode: str, destination: str, base: str,
              config: str, expected_base: str = '') -> dict:
        with self.lock:
            if self.state['status'] in ('running', 'cancelling'):
                raise RuntimeError('An export is already running.')
            details = adapter_details(adapter)
            if not details['valid']:
                raise ValueError(details['error'])
            adapter_path = Path(adapter).expanduser().resolve()
            base_path, config_path = Path(base).expanduser().resolve(), Path(config).expanduser().resolve()
            for path in (base_path, config_path):
                if not path.is_file():
                    raise ValueError(f'File not found: {path}')
            expected = Path(expected_base).expanduser().resolve() if str(expected_base or '').strip() else None
            if expected is not None and not expected.is_file():
                raise ValueError(f'File not found: {expected}')
            task_id = uuid.uuid4().hex
            if mode == 'browser':
                output = self.root / 'outputs' / 'exports' / task_id / _filename(destination.strip())
            elif mode == 'path':
                raw = Path(destination.strip()).expanduser()
                if not destination.strip() or not raw.is_absolute():
                    raise ValueError('Enter an absolute output path.')
                _filename(raw.name)
                output = raw.resolve()
            else:
                raise ValueError('Unknown export mode.')
            report = output.with_suffix('.merge.json')
            if output in (base_path, config_path, expected) or output.is_relative_to(adapter_path):
                raise ValueError('Output must be separate from the base and adapter.')
            if any(os.path.lexists(p) for p in (output, report)):
                raise FileExistsError(f'Export already exists: {output}')
            activity_token = acquire('export')
            staging = None
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix='.gpt-export-', dir=output.parent)).resolve()
                logdir = self.root / '.runtime' / 'export' / 'logs'
                logdir.mkdir(parents=True, exist_ok=True)
                logfile = logdir / f'{task_id}.log'
                self.lines.clear()
                self.stop_event.clear()
                self.started = time.monotonic()
                self.state = dict(status='running', phase='starting', elapsed=0.0, log='', output=str(output),
                                  error='', report={}, token=secrets.token_urlsafe(32), task_id=task_id, logfile=str(logfile))
                command = [sys.executable, '-B', '-u', str(self.script),
                           '--adapter', str(adapter_path), '--base', str(base_path), '--config', str(config_path),
                           '--output', str(staging / output.name)]
                if expected:
                    command += ['--expected-base', str(expected)]
                self.thread = threading.Thread(target=self._worker,
                    args=(command, staging, output, logfile, activity_token), daemon=True, name='gpt-export')
                self.thread.start()
            except Exception:
                release(activity_token)
                self.state.update(status='failed', phase='failed', error='Could not start export.')
                if staging is not None:
                    self._clean_staging(staging, output.parent)
                raise
            return self.snapshot()

    @staticmethod
    def _clean_staging(staging: Path, parent: Path) -> None:
        # Only the unique private directory allocated by this task may be removed.
        if staging.is_symlink() or staging.resolve().parent != parent.resolve() or not staging.name.startswith('.gpt-export-'):
            raise RuntimeError('Unexpected export staging directory.')
        if staging.exists():
            shutil.rmtree(staging)

    def _worker(self, command, staging, output, logfile, activity_token):
        proc = None
        try:
            environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='')
            with logfile.open('w', encoding='utf-8') as log:
                with self.lock:
                    if self.stop_event.is_set():
                        raise InterruptedError('Export cancelled.')
                    proc = subprocess.Popen(command, cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    self.process = proc
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        self.lines.append(line)
                        if line.startswith('EXPORT_PHASE:'):
                            phase = line.split(':', 1)[1].strip()
                            if phase != 'completed':
                                self.state['phase'] = phase
                proc.stdout.close()
                code = proc.wait()
                if self.stop_event.is_set():
                    raise InterruptedError('Export cancelled.')
                if code:
                    raise RuntimeError(f'Export process exited with code {code}. See the export log.')
                staged_output = staging / output.name
                staged_report = staged_output.with_suffix('.merge.json')
                report = json.loads(staged_report.read_text(encoding='utf-8'))
                if not staged_output.is_file() or report.get('source_files_unchanged') is not True or report.get('round_trip_tensors_identical') is not True:
                    raise RuntimeError('Export validation report is incomplete.')
                report['output'] = str(output)
                staged_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                from tools.merge_lora_to_pth import publish_export
                with self.lock:
                    if self.stop_event.is_set():
                        raise InterruptedError('Export cancelled.')
                    self.state['phase'] = 'publishing'
                    publish_export(staged_output, staged_report, output)
                    self.state.update(status='succeeded', phase='completed', report=report)
                    self.completed[self.state['task_id']] = dict(token=self.state['token'], model=output,
                        report=output.with_suffix('.merge.json'))
        except Exception as exc:
            cancelled = self.stop_event.is_set() or isinstance(exc, InterruptedError)
            with self.lock:
                message = f'{type(exc).__name__}: {exc}'
                self.state.update(status='cancelled' if cancelled else 'failed',
                                  phase='cancelled' if cancelled else 'failed', error='' if cancelled else message)
                self.lines.append(message + '\n')
            try:
                with logfile.open('a', encoding='utf-8') as log:
                    log.write(message + '\n')
            except OSError:
                pass
        finally:
            try:
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    if proc is not None and proc.stdout is not None:
                        proc.stdout.close()
                finally:
                    self._clean_staging(staging, output.parent)
            except Exception as exc:
                with self.lock:
                    self.lines.append(f'Export cleanup failed: {exc}\n')
            finally:
                with self.lock:
                    self.process = None
                    self.state['elapsed'] = time.monotonic() - self.started
                release(activity_token)
                if self.on_finished is not None:
                    try:
                        self.on_finished(self.snapshot())
                    except Exception:
                        pass

    def cancel(self) -> dict:
        with self.lock:
            if self.state['status'] in ('running', 'cancelling'):
                self.stop_event.set()
                self.state['status'] = 'cancelling'
                proc = self.process
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
            return self.snapshot()


def create_export_router(manager: ExportManager):
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse

    router = APIRouter(prefix='/exports')

    @router.get('/{task_id}/{kind}')
    def download(task_id: str, kind: str, token: str = ''):
        with manager.lock:
            record = manager.completed.get(task_id)
            if record is None or not secrets.compare_digest(record['token'], token) or kind not in ('model', 'report'):
                raise HTTPException(status_code=404, detail='Export not found.')
            path = record[kind]
            if not path.is_file():
                raise HTTPException(status_code=404, detail='Export file no longer exists.')
            return FileResponse(path, filename=path.name,
                media_type='application/octet-stream' if kind == 'model' else 'application/json',
                headers={'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer'})

    return router
