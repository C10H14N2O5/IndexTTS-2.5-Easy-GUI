"""Compatible flat voice+TXT and emotion-description libraries."""
from __future__ import annotations

import contextlib
import shutil
import threading
from pathlib import Path

from .. import paths, status
from ..errors import Conflict, NotFound, ServiceError
from ..i18n import msg
from .uploads import require_audio, audio_duration

LOCK = threading.RLock()


@contextlib.contextmanager
def mutation(kind='library update'):
    """Prevent replacing/removing audio while inference or training owns it."""
    from indextts_training.activity import acquire, release
    token = acquire(kind)
    try:
        with LOCK:
            yield
    finally:
        release(token)


def clean_name(value: str) -> str:
    name = paths.safe_name(value)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if not name or len(name) > 120 or name.upper() in reserved:
        raise ServiceError('Enter a valid library name (1–120 characters).')
    return name


def _entry(directory: Path, name: str, suffixes: set[str]) -> Path:
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        raise ServiceError('Invalid library filename.')
    path = directory / name
    if path.suffix.lower() not in suffixes or path.resolve().parent != directory.resolve():
        raise ServiceError('Invalid library filename.')
    if not path.is_file():
        raise NotFound('Library item not found: {0}', name)
    return path


def transcript(path: Path) -> str:
    sidecar = path.with_suffix('.txt')
    if sidecar.resolve().parent != paths.VOICES_DIR.resolve():
        return ''
    try:
        return sidecar.read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        return ''


def voice_info(path: Path) -> dict:
    stat = path.stat()
    return {'name': path.name, 'label': path.stem.replace('_', ' '), 'path': str(path),
            'transcript': transcript(path), 'duration': audio_duration(path),
            'size': stat.st_size, 'modified': stat.st_mtime}


def voices() -> list[dict]:
    if not paths.VOICES_DIR.is_dir():
        return []
    with LOCK:
        return [voice_info(p) for p in sorted(paths.VOICES_DIR.iterdir(), key=lambda p: p.name.casefold())
                if p.is_file() and not p.is_symlink() and p.suffix.lower() in paths.AUDIO_SUFFIXES]


def get_voice(name: str) -> dict:
    with LOCK:
        return voice_info(_entry(paths.VOICES_DIR, name, paths.AUDIO_SUFFIXES))


def _save_transcript(path: Path, text: str) -> None:
    sidecar = path.with_suffix('.txt')
    if sidecar.resolve().parent != paths.VOICES_DIR.resolve():
        raise ServiceError('Invalid library filename.')
    text = text.strip()
    if text:
        sidecar.write_text(text + '\n', encoding='utf-8')
    else:
        sidecar.unlink(missing_ok=True)


def save_voice(name: str, source_path: str, text: str = '', overwrite: bool = False) -> dict:
    source = require_audio(source_path)
    audio_duration(source, validate=True)
    safe = clean_name(name)
    destination = paths.VOICES_DIR / f'{safe}{source.suffix.lower()}'
    with mutation():
        paths.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        if destination.resolve().parent != paths.VOICES_DIR.resolve():
            raise ServiceError('Invalid library filename.')
        if destination.exists() and not overwrite:
            raise Conflict('A library item with this name already exists.', extra={'name': destination.name})
        if source != destination.resolve():
            shutil.copy2(source, destination)
        _save_transcript(destination, text)
        result = voice_info(destination)
    status.publish('Reference library', msg('Saved reference voice: {0}', safe), 'success')
    return result


def update_voice(name: str, new_name: str | None, text: str | None) -> dict:
    with mutation():
        source = _entry(paths.VOICES_DIR, name, paths.AUDIO_SUFFIXES)
        destination = source
        if new_name is not None:
            destination = paths.VOICES_DIR / f'{clean_name(new_name)}{source.suffix.lower()}'
            if destination != source:
                if destination.exists() or destination.with_suffix('.txt').exists():
                    raise Conflict('A library item with this name already exists.')
                if destination.resolve().parent != paths.VOICES_DIR.resolve():
                    raise ServiceError('Invalid library filename.')
                saved_text = transcript(source)
                source.rename(destination)
                source.with_suffix('.txt').unlink(missing_ok=True)
                _save_transcript(destination, saved_text)
        if text is not None:
            _save_transcript(destination, text)
        result = voice_info(destination)
    status.publish('Reference library', msg('Saved reference voice: {0}', destination.stem), 'success')
    return result


def delete_voice(name: str) -> int:
    with mutation():
        path = _entry(paths.VOICES_DIR, name, paths.AUDIO_SUFFIXES)
        sidecar = path.with_suffix('.txt')
        if sidecar.resolve().parent != paths.VOICES_DIR.resolve():
            raise ServiceError('Invalid library filename.')
        path.unlink()
        sidecar.unlink(missing_ok=True)
    status.publish('Reference library', msg('Deleted reference voice: {0}', name), 'success')
    return 1


def delete_all_voices() -> int:
    with mutation():
        selected = [p for p in paths.VOICES_DIR.iterdir() if p.is_file() and not p.is_symlink()
                    and p.suffix.lower() in paths.AUDIO_SUFFIXES]
        for path in selected:
            path.unlink()
            sidecar = path.with_suffix('.txt')
            if not sidecar.is_symlink():
                sidecar.unlink(missing_ok=True)
    status.publish('Reference library', msg('Deleted {0} reference voice file(s).', len(selected)), 'success')
    return len(selected)


def description_info(path: Path) -> dict:
    return {'name': path.name, 'label': path.stem.replace('_', ' '), 'path': str(path),
            'text': path.read_text(encoding='utf-8').strip()}


def descriptions() -> list[dict]:
    with LOCK:
        return [description_info(p) for p in sorted(paths.EMOTION_DESCRIPTIONS_DIR.glob('*.txt'))
                if p.is_file() and not p.is_symlink()]


def description_text(value: str | None) -> str:
    if not value or value == paths.NONE_SENTINEL:
        return ''
    candidate = Path(value)
    if candidate.resolve().parent != paths.EMOTION_DESCRIPTIONS_DIR.resolve():
        raise ServiceError('Invalid library filename.')
    return description_info(_entry(paths.EMOTION_DESCRIPTIONS_DIR, candidate.name, {'.txt'}))['text']


def save_description(name: str, text: str, overwrite: bool = False) -> dict:
    safe = clean_name(name)
    text = text.strip()
    if not text:
        raise ServiceError('Enter an emotion description to save.')
    destination = paths.EMOTION_DESCRIPTIONS_DIR / f'{safe}.txt'
    with mutation():
        if destination.resolve().parent != paths.EMOTION_DESCRIPTIONS_DIR.resolve():
            raise ServiceError('Invalid library filename.')
        if destination.exists() and not overwrite:
            raise Conflict('A library item with this name already exists.')
        destination.write_text(text + '\n', encoding='utf-8')
        result = description_info(destination)
    status.publish('Emotion library', msg('Saved emotion description: {0}', destination.name), 'success')
    return result


def delete_description(name: str) -> int:
    with mutation():
        path = _entry(paths.EMOTION_DESCRIPTIONS_DIR, name, {'.txt'})
        path.unlink()
    status.publish('Emotion library', msg('Deleted emotion description: {0}', name), 'success')
    return 1
