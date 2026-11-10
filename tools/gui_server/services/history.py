"""Output history with additive <wav>.json parameter sidecars."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import paths, status
from ..errors import NotFound, ServiceError
from ..i18n import msg
from .library import mutation
from .uploads import audio_duration


def remember(path: Path, kind: str, params: dict) -> None:
    sidecar = Path(str(path) + '.json')
    pending = sidecar.with_suffix('.json.tmp')
    try:
        pending.write_text(json.dumps({'kind': kind, 'created': time.time(), 'params': params},
                                      ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pending.replace(sidecar)
    finally:
        pending.unlink(missing_ok=True)


def list_history(limit: int = 100) -> list[dict]:
    if not paths.OUTPUTS_DIR.exists():
        return []
    result = []
    files = sorted((p for p in paths.OUTPUTS_DIR.iterdir()
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in paths.AUDIO_SUFFIXES),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:max(1, min(limit, 1000))]:
        meta = {}
        try:
            candidate = Path(str(path) + '.json')
            if not candidate.is_symlink():
                value = json.loads(candidate.read_text(encoding='utf-8'))
                if isinstance(value, dict):
                    meta = value
        except (OSError, ValueError):
            pass
        stat = path.stat()
        result.append({'name': path.name, 'path': str(path), 'created': meta.get('created', stat.st_mtime),
                       'duration': audio_duration(path), 'size': stat.st_size,
                       'kind': meta.get('kind', 'dialogue' if '_dialogue_' in path.name else 'synthesis'),
                       'params': meta.get('params') if isinstance(meta.get('params'), dict) else None})
    return result


def delete(name: str) -> int:
    if not name or '/' in name or '\\' in name or Path(name).name != name:
        raise ServiceError('Invalid output filename.')
    path = paths.OUTPUTS_DIR / name
    if path.suffix.lower() not in paths.AUDIO_SUFFIXES or path.resolve().parent != paths.OUTPUTS_DIR.resolve():
        raise ServiceError('Invalid output filename.')
    with mutation('output cleanup'):
        if not path.is_file():
            raise NotFound('Audio file not found: {0}', str(path))
        path.unlink()
        sidecar = Path(str(path) + '.json')
        if not sidecar.is_symlink():
            sidecar.unlink(missing_ok=True)
    return 1


def delete_all() -> int:
    with mutation('output cleanup'):
        files = [p for p in paths.OUTPUTS_DIR.iterdir() if p.is_file() and not p.is_symlink()]
        for path in files:
            path.unlink()
    status.publish('Outputs', msg('Deleted {0} output file(s).', len(files)), 'success')
    return len(files)
