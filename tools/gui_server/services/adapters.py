"""Discover 1.x-compatible final adapters and checkpoints, in project order."""
from __future__ import annotations

from .. import paths
from ..i18n import t


def list_adapters() -> list[dict]:
    result = []
    for project in sorted(paths.TRAINING_OUTPUTS_DIR.iterdir()):
        if not project.is_dir() or project.is_symlink():
            continue
        final = project / 'final_adapter'
        candidates = ([final] if final.is_dir() else []) + sorted(project.glob('checkpoint-*'))
        for path in candidates:
            if path.is_symlink() or not (path / 'adapter_config.json').is_file():
                continue
            is_final = path.name == 'final_adapter'
            label = t('{0} · final', project.name) if is_final else f'{project.name} · {path.name}'
            result.append({'label': label,
                           'path': str(path), 'project': project.name, 'checkpoint': path.name,
                           'kind': 'final' if is_final else 'checkpoint'})
    return result
