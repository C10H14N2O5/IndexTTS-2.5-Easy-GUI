"""Git-free program-file updates. Standard library only; user data is never managed."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile

MANIFEST = 'tools/update-manifest.json'
REQUIRED = {MANIFEST, 'easy_gui.py', 'tools/update_core.py', 'tools/runtime_lock.py',
            'tools/update_healthcheck.py', '1- install.bat', '2- run.bat', '3- update.bat',
            'tools/update.ps1', 'tools/run_gui.py', 'tools/install.ps1', 'tools/install_runtime.bat',
            'pyproject.toml', 'uv.lock', '.python-version'}
ALLOWED_DIRS = {'tools', 'indextts', 'indextts_training', 'backends', 'tests', 'cli_tests',
                'docs', 'archive', 'assets', 'examples', 'accelerator_wheels'}
ALLOWED_ROOT = {'easy_gui.py', '1- install.bat', '2- run.bat', '3- update.bat', 'readme.md',
                'license', 'license_zh.txt', 'disclaimer', 'manifest.in', 'pyproject.toml',
                'uv.lock', '.python-version', '.gitignore', '.gitattributes', 'local-settings.example.bat'}
WEIGHTS = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.gguf', '.t7'}
AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
MAX_ARCHIVE_BYTES = 4 * 1024**3


class UpdateError(RuntimeError):
    pass


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise UpdateError(f'Unsafe package path: {value!r}')
    parts = value.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    for part in parts:
        if not part or part in ('.', '..') or part.rstrip(' .') != part or re.search(r'[<>:"|?*\x00-\x1f]', part):
            raise UpdateError(f'Unsafe package path: {value!r}')
        if part.split('.')[0].upper() in reserved:
            raise UpdateError(f'Reserved Windows path: {value!r}')
    return '/'.join(parts)


def managed_path(value: str) -> str:
    path = safe_relative(value)
    parts = path.split('/')
    folded = [part.casefold() for part in parts]
    if len(parts) == 1:
        allowed = folded[0] in ALLOWED_ROOT
    else:
        allowed = folded[0] in ALLOWED_DIRS
    if (not allowed or any(part in {'.git', '.runtime', '.venv', '__pycache__', '.proofread', '.aws', '.codex', '.agents'} for part in folded)
            or Path(path).suffix.lower() in WEIGHTS
            or (Path(path).suffix.lower() in AUDIO and folded[0] != 'assets')
            or (folded[0] == 'accelerator_wheels' and Path(path).suffix.lower() == '.whl')):
        raise UpdateError(f'Protected or non-program path in update manifest: {path}')
    return path


def linked(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def contained(root: Path, relative: str) -> Path:
    relative = safe_relative(relative)
    root = root.absolute()
    if linked(root):
        raise UpdateError(f'Linked root is not supported: {root}')
    current = root
    for part in relative.split('/'):
        current /= part
        if linked(current):
            raise UpdateError(f'Linked path is not supported: {current}')
    if not current.resolve().is_relative_to(root.resolve()):
        raise UpdateError(f'Path escapes its root: {relative}')
    return current


def digest(path: Path, normalize: bool = False) -> str | None:
    if not path.exists():
        return None
    if linked(path) or not path.is_file():
        raise UpdateError(f'Expected a regular file: {path}')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        if normalize:
            hasher.update(stream.read().replace(b'\r\n', b'\n'))
        else:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with pending.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise UpdateError(f'Cannot read update metadata {path}: {exc}') from exc


def parse_manifest(data) -> dict:
    if not isinstance(data, dict) or data.get('schema') != 1:
        raise UpdateError('Unsupported or missing update manifest schema.')
    files, environment = data.get('files'), data.get('environment')
    if not isinstance(files, list) or not isinstance(environment, list) or not files or not environment:
        raise UpdateError('Update manifest requires file and environment lists.')
    normalized = [managed_path(path) for path in files]
    lowered = {path.casefold() for path in normalized}
    if len(lowered) != len(normalized):
        raise UpdateError('Case-insensitive duplicate in update manifest.')
    if not REQUIRED <= set(normalized):
        raise UpdateError('Update package is missing required program files.')
    for path in normalized:
        if any(parent.as_posix().casefold() in lowered for parent in PurePosixPath(path).parents if str(parent) != '.'):
            raise UpdateError(f'File/directory collision in update manifest: {path}')
    environment = [managed_path(path) for path in environment]
    if len(set(environment)) != len(environment) or not set(environment) <= set(normalized):
        raise UpdateError('Environment files must be distinct managed files.')
    if not {'pyproject.toml', 'uv.lock', '.python-version', '1- install.bat'} <= set(environment):
        raise UpdateError('Update manifest does not protect dependency declarations.')
    return dict(schema=1, files=normalized, environment=environment)


class Updater:
    """Caller holds ApplicationLock. The injected hooks are for isolated tests only."""
    def __init__(self, root, healthcheck=None, fault=None):
        self.root = Path(root).absolute()
        if linked(self.root) or not self.root.is_dir():
            raise UpdateError('Deployment root must be an existing ordinary directory.')
        if os.path.lexists(self.root / '.git'):
            raise UpdateError('ZIP updates are disabled in Git working directories.')
        self.runtime = contained(self.root, '.runtime/update')
        self.current_path = contained(self.root, '.runtime/update/current.json')
        self.journal_path = contained(self.root, '.runtime/update/transaction.json')
        self.last_path = contained(self.root, '.runtime/update/last-success.json')
        self.healthcheck = healthcheck or self._healthcheck
        self.fault = fault or (lambda event, detail: None)

    def _healthcheck(self):
        python = self.root / '.venv/Scripts/python.exe'
        if not python.is_file():
            raise UpdateError('The existing project Python environment is required; run 1- install.bat separately.')
        result = subprocess.run([str(python), '-B', '-X', 'utf8', str(self.root/'tools/update_healthcheck.py'), '--root', str(self.root)],
            cwd=self.root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding='utf-8', errors='replace', timeout=120, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode:
            raise UpdateError('Updated program failed its health check.')

    def _state(self, path=None):
        return self._validate_state(read_json(path or self.current_path))

    @staticmethod
    def _validate_state(state):
        if not isinstance(state, dict) or state.get('schema') != 1 or not isinstance(state.get('files'), dict) or not isinstance(state.get('environment'), dict):
            raise UpdateError('Invalid installed version record.')
        for path, value in state['files'].items():
            managed_path(path)
            if value is not None and not re.fullmatch(r'[0-9a-f]{64}', str(value)):
                raise UpdateError('Invalid file hash in installed version record.')
        if len({p.casefold() for p in state['files']}) != len(state['files']):
            raise UpdateError('Duplicate installed path.')
        if not REQUIRED <= set(state['files']) or not set(state['environment']) <= set(state['files']):
            raise UpdateError('Incomplete installed version record.')
        if not {'pyproject.toml', 'uv.lock', '.python-version', '1- install.bat'} <= set(state['environment']):
            raise UpdateError('Incomplete installed environment record.')
        if not isinstance(state.get('commit'), str) or not state['commit']:
            raise UpdateError('Missing installed commit identity.')
        for value in state['environment'].values():
            if not re.fullmatch(r'[0-9a-f]{64}', str(value)):
                raise UpdateError('Invalid environment hash.')
        return state

    def initialize(self, commit='local'):
        if self.journal_path.exists():
            raise UpdateError('An interrupted update must be recovered first.')
        if self.current_path.exists():
            state = self._state()
            self._verify_current(state)
            return state
        manifest = parse_manifest(read_json(contained(self.root, MANIFEST)))
        files = {path: digest(contained(self.root, path)) for path in manifest['files']}
        missing = [path for path in REQUIRED | set(manifest['environment']) if files.get(path) is None]
        if missing:
            raise UpdateError('Cannot initialize an incomplete deployment: ' + ', '.join(sorted(missing)))
        state = dict(schema=1, commit=commit, files=files,
            environment={path: digest(contained(self.root, path), True) for path in manifest['environment']})
        atomic_json(self.current_path, state)
        return state

    def _verify_current(self, state):
        conflicts = []
        for path, expected in state['files'].items():
            if digest(contained(self.root, path)) != expected:
                conflicts.append(path)
        if conflicts:
            raise UpdateError('Local program changes detected; update/rollback stopped: ' + ', '.join(conflicts))

    def verify(self):
        if self.journal_path.exists():
            raise UpdateError('Interrupted update detected; run 3- update.bat to recover it.')
        state = self._state()
        self._verify_current(state)
        return dict(status='verified', commit=state['commit'])

    def _owned_folder(self, category, identifier):
        if category not in ('staging', 'backups') or not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise UpdateError('Invalid updater-owned directory identifier.')
        return contained(self.root, f'.runtime/update/{category}/{identifier}')

    def _remove_owned(self, category, identifier):
        folder = self._owned_folder(category, identifier)
        if folder.exists():
            for directory, dirs, files in os.walk(folder, followlinks=False):
                for name in dirs + files:
                    if linked(Path(directory)/name):
                        raise UpdateError(f'Refusing cleanup through a linked path: {directory}/{name}')
            shutil.rmtree(folder)

    def _clean_abandoned(self):
        """Only updater-created UUID directories carrying our ownership marker."""
        keep = set()
        if self.last_path.exists():
            last = read_json(self.last_path)
            self._owned_folder('backups', last.get('id', ''))
            keep.add(last['id'])
        if self.journal_path.exists():
            pending = self._validate_journal(read_json(self.journal_path))
            keep.add(pending['id'])
        for category in ('staging', 'backups'):
            parent = contained(self.root, f'.runtime/update/{category}')
            if not parent.exists():
                continue
            for folder in parent.iterdir():
                if not re.fullmatch(r'[0-9a-f]{32}', folder.name) or folder.name in keep:
                    continue
                marker = contained(self.root, f'.runtime/update/{category}/{folder.name}/.updater-owned.json')
                if marker.is_file() and read_json(marker) == dict(schema=1, category=category, id=folder.name):
                    try:
                        self._remove_owned(category, folder.name)
                    except OSError as exc:
                        print(f'[WARNING] Abandoned updater directory cleanup failed: {exc}', flush=True)

    @contextmanager
    def _package(self, archive):
        identifier = uuid.uuid4().hex
        folder = self._owned_folder('staging', identifier)
        folder.mkdir(parents=True)
        atomic_json(folder/'.updater-owned.json', dict(schema=1, category='staging', id=identifier))
        try:
            with zipfile.ZipFile(archive) as package:
                if len(package.infolist()) > 100_000:
                    raise UpdateError('Source archive has too many entries.')
                members, roots, identities, components = {}, set(), set(), {}
                total = 0
                for member in package.infolist():
                    name = member.filename.rstrip('/') if member.is_dir() else member.filename
                    safe_relative(name)
                    identity = name.casefold()
                    if identity in identities:
                        raise UpdateError('Case-insensitive duplicate archive member: ' + name)
                    identities.add(identity)
                    parts = name.split('/')
                    for index in range(1, len(parts) + 1):
                        prefix = '/'.join(parts[:index])
                        previous = components.setdefault(prefix.casefold(), prefix)
                        if previous != prefix:
                            raise UpdateError('Case-conflicting archive directory: ' + prefix)
                    roots.add(parts[0])
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                        raise UpdateError('Archive contains a link or special file: ' + name)
                    if member.flag_bits & 1:
                        raise UpdateError('Encrypted ZIP entries are unsupported.')
                    if len(parts) == 1:
                        if not member.is_dir():
                            raise UpdateError('Expected one directory enclosing the source archive.')
                        continue
                    relative = '/'.join(parts[1:])
                    if not member.is_dir():
                        total += member.file_size
                        if total > MAX_ARCHIVE_BYTES or member.file_size > MAX_ARCHIVE_BYTES:
                            raise UpdateError('Source archive exceeds the 4 GiB extraction limit.')
                        members[relative] = member
                if len(roots) != 1 or MANIFEST not in members:
                    raise UpdateError('ZIP must contain one source directory with an update manifest.')
                if members[MANIFEST].file_size > 2 * 1024**2:
                    raise UpdateError('Update manifest exceeds the 2 MiB limit.')
                lowered_members = {path.casefold() for path in members}
                for name in members:
                    for parent in PurePosixPath(name).parents:
                        if str(parent) != '.' and parent.as_posix().casefold() in lowered_members:
                            raise UpdateError('Archive file/directory collision: ' + name)
                manifest = parse_manifest(json.loads(package.read(members[MANIFEST]).decode('utf-8-sig')))
                for relative in manifest['files']:
                    if relative not in members:
                        raise UpdateError('Missing manifest file in archive: ' + relative)
                    destination = contained(folder, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(members[relative]) as incoming, destination.open('xb') as stream:
                        shutil.copyfileobj(incoming, stream, 1024 * 1024)
            yield folder, manifest
        finally:
            try:
                self._remove_owned('staging', identifier)
            except OSError as exc:
                print(f'[WARNING] Temporary package cleanup failed: {exc}', flush=True)

    def update(self, archive, commit, check=False):
        if not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
            raise UpdateError('A full 40-character GitHub commit SHA is required.')
        if self.journal_path.exists():
            if check:
                raise UpdateError('Interrupted update detected; run 3- update.bat to recover it.')
            self.recover()
        state = self._state() if self.current_path.exists() else None
        if state is None:
            if check:
                raise UpdateError('No installed baseline. Run 3- update.bat once to initialize it.')
            state = self.initialize()
        self._verify_current(state)
        if state.get('commit') == commit.lower():
            return dict(status='unchanged', commit=commit.lower())
        with self._package(archive) as (folder, manifest):
            environment = {path: digest(contained(folder, path), True) for path in manifest['environment']}
            if environment != state['environment']:
                changed = sorted(path for path in set(environment) | set(state['environment']) if environment.get(path) != state['environment'].get(path))
                raise UpdateError('Dependency/environment definitions changed; no program files were modified. Upgrade the environment separately: ' + ', '.join(changed))
            files = {path: digest(contained(folder, path)) for path in manifest['files']}
            for path in files:
                if path not in state['files'] and os.path.lexists(contained(self.root, path)):
                    raise UpdateError('New package file conflicts with an unmanaged local file: ' + path)
            target = dict(schema=1, commit=commit.lower(), files=files, environment=environment)
            changed = sorted(path for path in set(files) | set(state['files']) if files.get(path) != state['files'].get(path))
            if check:
                return dict(status='available', current=state.get('commit'), commit=commit.lower(), changed=changed)
            sources = {path: contained(folder, path) for path in files}
            self._execute(state, target, sources, rollback=False)
            return dict(status='updated', previous=state.get('commit'), commit=commit.lower(), changed=changed)

    def _atomic_copy(self, source, destination, temporary):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(temporary):
            raise UpdateError('Unexpected temporary update file: ' + str(temporary))
        try:
            with source.open('rb') as incoming, temporary.open('xb') as stream:
                shutil.copyfileobj(incoming, stream, 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _execute(self, before, after, sources, rollback):
        self._verify_current(before)
        identifier = uuid.uuid4().hex
        backup = self._owned_folder('backups', identifier)
        backup.mkdir(parents=True)
        atomic_json(backup/'.updater-owned.json', dict(schema=1, category='backups', id=identifier))
        operations = []
        previous_last = read_json(self.last_path) if self.last_path.exists() else None
        journal_written = False
        committed = False
        try:
            for path in sorted(set(before['files']) | set(after['files'])):
                old_hash, new_hash = before['files'].get(path), after['files'].get(path)
                if old_hash == new_hash:
                    continue
                destination = contained(self.root, path)
                if digest(destination) != old_hash:
                    raise UpdateError('Program changed while preparing update: ' + path)
                if old_hash is not None:
                    saved = contained(backup, path)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('rb') as incoming, saved.open('xb') as stream:
                        shutil.copyfileobj(incoming, stream, 1024 * 1024)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if digest(saved) != old_hash:
                        raise UpdateError('Backup verification failed: ' + path)
                temporary = str(PurePosixPath(path).with_name('.' + PurePosixPath(path).name + '.update-' + identifier + '.tmp'))
                if os.path.lexists(contained(self.root, temporary)):
                    raise UpdateError('Unexpected transaction temporary file: ' + temporary)
                operations.append(dict(path=path, before=old_hash, after=new_hash, temporary=temporary))
            journal = dict(schema=1, id=identifier, phase='applying', before=before, after=after,
                           operations=operations, previous_last=previous_last, rollback=rollback)
            atomic_json(self.journal_path, journal)
            journal_written = True
            self.fault('journal', journal)
            for operation in operations:
                path = operation['path']
                destination = contained(self.root, path)
                if digest(destination) != operation['before']:
                    raise UpdateError('Program changed during update: ' + path)
                if operation['after'] is None:
                    destination.unlink()
                else:
                    self._atomic_copy(sources[path], destination, contained(self.root, operation['temporary']))
                    if digest(destination) != operation['after']:
                        raise UpdateError('Installed file verification failed: ' + path)
                self.fault('file', operation)
            self.healthcheck()
            self.fault('healthcheck', journal)
            self._verify_current(after)
            if rollback:
                self.last_path.unlink(missing_ok=True)
            else:
                atomic_json(self.last_path, dict(schema=1, id=identifier, before=before, after=after, operations=operations))
            atomic_json(self.current_path, after)
            self.fault('metadata', journal)
            journal['phase'] = 'committed'
            atomic_json(self.journal_path, journal)
            committed = True
            self._finish_committed(journal)
        except Exception as exc:
            if committed:
                raise UpdateError(f'Program files and version record were committed, but finalization is incomplete: {exc}. Run the updater again; no rollback was performed.') from exc
            if journal_written:
                try:
                    self.recover()
                except Exception as restore:
                    raise UpdateError(f'Update failed ({exc}); recovery is incomplete ({restore}). Keep .runtime/update backups and run the updater again.') from restore
                raise UpdateError(f'Update failed and the previous version was restored: {exc}') from exc
            self._remove_owned('backups', identifier)
            raise

    def _validate_journal(self, journal):
        if not isinstance(journal, dict) or journal.get('schema') != 1:
            raise UpdateError('Invalid recovery journal.')
        self._owned_folder('backups', journal.get('id', ''))
        if journal.get('phase') not in ('applying', 'committed'):
            raise UpdateError('Unknown recovery phase.')
        before = self._validate_state(journal.get('before'))
        after = self._validate_state(journal.get('after'))
        operations = journal.get('operations')
        if not isinstance(operations, list):
            raise UpdateError('Invalid recovery operations.')
        expected_paths = {path for path in set(before['files']) | set(after['files']) if before['files'].get(path) != after['files'].get(path)}
        seen = set()
        for operation in journal.get('operations', []):
            path = managed_path(operation['path'])
            if path in seen or operation.get('before') != before['files'].get(path) or operation.get('after') != after['files'].get(path):
                raise UpdateError('Invalid recovery file hashes.')
            seen.add(path)
            expected = str(PurePosixPath(path).with_name('.' + PurePosixPath(path).name + '.update-' + journal['id'] + '.tmp'))
            if operation.get('temporary') != expected:
                raise UpdateError('Invalid recovery temporary path.')
        if seen != expected_paths:
            raise UpdateError('Recovery journal does not cover every changed file.')
        return journal

    def _finish_committed(self, journal):
        # Cleanup is not part of program publication. Never claim a committed
        # update was rolled back merely because an obsolete backup is locked.
        try:
            if journal.get('previous_last'):
                self._remove_owned('backups', journal['previous_last']['id'])
            if journal['rollback']:
                self._remove_owned('backups', journal['id'])
        except OSError as exc:
            print(f'[WARNING] Update committed, but obsolete backup cleanup failed: {exc}', flush=True)
        self.journal_path.unlink(missing_ok=True)

    def recover(self):
        if not self.journal_path.exists():
            self._clean_abandoned()
            return dict(status='clean')
        journal = self._validate_journal(read_json(self.journal_path))
        if journal['phase'] == 'committed':
            self._finish_committed(journal)
            self._clean_abandoned()
            return dict(status='completed', commit=journal['after'].get('commit'))
        backup = self._owned_folder('backups', journal['id'])
        failures = []
        for operation in reversed(journal['operations']):
            path = operation['path']
            try:
                destination = contained(self.root, path)
                current = digest(destination)
                if current not in (operation['before'], operation['after'], None):
                    raise UpdateError('Local change conflicts with recovery: ' + path)
                temporary = contained(self.root, operation['temporary'])
                temporary.unlink(missing_ok=True)
                if current != operation['before']:
                    if operation['before'] is None:
                        destination.unlink(missing_ok=True)
                    else:
                        saved = contained(backup, path)
                        if digest(saved) != operation['before']:
                            raise UpdateError('Backup is missing or modified: ' + path)
                        self._atomic_copy(saved, destination, temporary)
            except Exception as exc:
                failures.append(f'{path}: {exc}')
        if failures:
            raise UpdateError('Could not restore: ' + '; '.join(failures))
        self._verify_current(journal['before'])
        atomic_json(self.current_path, journal['before'])
        if journal['previous_last'] is None:
            self.last_path.unlink(missing_ok=True)
        else:
            atomic_json(self.last_path, journal['previous_last'])
        self.journal_path.unlink()
        self._remove_owned('backups', journal['id'])
        self._clean_abandoned()
        return dict(status='recovered', commit=journal['before'].get('commit'))

    def rollback(self):
        recovery = self.recover()
        state = self._state()
        self._verify_current(state)
        if not self.last_path.exists():
            raise UpdateError('No successful update backup is available for rollback.')
        last = read_json(self.last_path)
        self._validate_state(last.get('before'))
        self._validate_state(last.get('after'))
        if state != last.get('after'):
            raise UpdateError('Rollback record does not match the installed version.')
        backup = self._owned_folder('backups', last.get('id', ''))
        previous = last['before']
        sources = {}
        for path, value in previous['files'].items():
            managed_path(path)
            if value is not None and value != state['files'].get(path):
                source = contained(backup, path)
                if digest(source) != value:
                    raise UpdateError('Rollback backup is missing or modified: ' + path)
                sources[path] = source
        self._execute(state, previous, sources, rollback=True)
        return dict(status='rolled_back', previous=state.get('commit'), commit=previous.get('commit'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--commit', default='local')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--rollback', action='store_true')
    action.add_argument('--initialize', action='store_true')
    action.add_argument('--recover', action='store_true')
    action.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    try:
        from runtime_lock import ApplicationLock
        with ApplicationLock(args.root):
            updater = Updater(args.root)
            if args.recover:
                result = updater.recover()
            elif args.verify:
                result = updater.verify()
            elif args.initialize:
                result = dict(status='initialized', commit=updater.initialize(args.commit).get('commit'))
            elif args.rollback:
                result = updater.rollback()
            else:
                if args.archive is None:
                    raise UpdateError('--archive is required when checking or applying an update.')
                result = updater.update(args.archive, args.commit, args.check)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
