"""Read-only file browser and whitelisted media paths for a trusted-local GUI."""
from __future__ import annotations

import os
from pathlib import Path

from .. import paths
from ..errors import NotFound, ServiceError
from ..i18n import t


def roots() -> list[dict]:
    values = [("Project", paths.ROOT), ("Reference library", paths.VOICES_DIR),
              ("Outputs", paths.OUTPUTS_DIR), ("Training", paths.TRAINING_DIR),
              ("Validation Set", paths.VALIDATION_DIR)]
    if os.name == 'nt':
        values.extend((f'{letter}:', Path(f'{letter}:/')) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                      if Path(f'{letter}:/').is_dir())
    else:
        values.append(('/', Path('/')))
    return [{'label': t(label), 'path': str(path.resolve())} for label, path in values if path.is_dir()]


def list_directory(path: str, kind: str = 'dir') -> dict:
    if kind not in {'dir', 'pth', 'audio'}:
        raise ServiceError('Invalid file browser filter.')
    directory = Path(path).expanduser() if path else paths.ROOT
    if not directory.is_absolute():
        directory = paths.ROOT / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise NotFound('Directory not found: {0}', str(directory))
    entries = []
    for item in directory.iterdir():
        try:
            is_dir = item.is_dir()
            if not is_dir and (kind == 'dir' or
                               (kind == 'pth' and item.suffix.lower() != '.pth') or
                               (kind == 'audio' and item.suffix.lower() not in paths.AUDIO_SUFFIXES)):
                continue
            stat = item.stat()
            entries.append({'name': item.name, 'path': str(item), 'type': 'dir' if is_dir else 'file',
                            'size': None if is_dir else stat.st_size, 'modified': stat.st_mtime})
        except OSError:
            continue
    entries.sort(key=lambda item: (item['type'] != 'dir', item['name'].casefold()))
    return {'path': str(directory), 'parent': str(directory.parent) if directory.parent != directory else None,
            'entries': entries}


def media_path(value: str) -> Path:
    if not value.strip():
        raise ServiceError('Choose an audio file first.')
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = paths.ROOT / candidate
    candidate = candidate.resolve()
    allowed = (paths.VOICES_DIR, paths.OUTPUTS_DIR, paths.TRAINING_OUTPUTS_DIR,
               paths.TRAINING_DATASETS_DIR, paths.VALIDATION_DIR, paths.TEMP_DIR, paths.ASSETS_DIR)
    # Training settings copy uploaded evaluation audio into the project folder.
    # Permit that single known artifact, not arbitrary project files. Checking
    # the resolved path prevents a project file/symlink escaping this exception.
    project_reference = (candidate.parent.parent == paths.TRAINING_PROJECTS_DIR.resolve()
                         and candidate.stem == 'eval_reference'
                         and candidate.suffix.lower() in paths.AUDIO_SUFFIXES
                         and (candidate.parent / 'project.json').is_file())
    if not project_reference and not any(paths.is_within(candidate, root) for root in allowed):
        raise ServiceError('This media path is outside the allowed folders.', status=403, code='forbidden')
    suffixes = paths.AUDIO_SUFFIXES | {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    if candidate.suffix.lower() not in suffixes:
        raise ServiceError('This file type cannot be served as media.', status=403, code='forbidden')
    if not candidate.is_file():
        raise NotFound('Audio file not found: {0}', str(candidate))
    return candidate
