"""Discover 1.x-compatible final adapters and checkpoints, in project order."""
from __future__ import annotations

from .. import paths
from ..i18n import t


def list_adapters() -> list[dict]:
    result = []
    for project in sorted(paths.TRAINING_OUTPUTS_DIR.iterdir(), key=lambda item: paths.natural_key(item.name)):
        if not project.is_dir() or project.is_symlink():
            continue
        named = [project / 'final_adapter', project / 'best_adapter']
        candidates = [path for path in named if path.is_dir()] + sorted(project.glob('checkpoint-*'))
        for path in candidates:
            if path.is_symlink() or not (path / 'adapter_config.json').is_file():
                continue
            kind = {'final_adapter': 'final', 'best_adapter': 'best'}.get(path.name, 'checkpoint')
            label = (t('{0} · final', project.name) if kind == 'final' else
                     t('{0} · best', project.name) if kind == 'best' else f'{project.name} · {path.name}')
            result.append({'label': label,
                           'path': str(path), 'project': project.name, 'checkpoint': path.name,
                           'kind': kind})
    return result
