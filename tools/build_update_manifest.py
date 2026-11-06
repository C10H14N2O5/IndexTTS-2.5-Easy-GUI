"""Maintainer helper: build the ZIP updater inventory from a tracked-file list.

The updater itself never needs Git. Feed this tool a UTF-8 file containing the
repository's tracked paths plus files being introduced in the same change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .update_core import managed_path, UpdateError
except ImportError:
    from update_core import managed_path, UpdateError


ENVIRONMENT = (
    '.python-version', 'pyproject.toml', 'uv.lock', '1- install.bat',
    'tools/install_runtime.bat', 'tools/install.ps1',
)


def build_manifest(paths):
    files = set()
    for path in paths:
        path = path.strip().replace('\\', '/')
        if not path:
            continue
        try:
            managed_path(path)
        except UpdateError:
            continue
        files.add(path)
    files.add('tools/update-manifest.json')
    environment = set(ENVIRONMENT)
    environment.update(path for path in files if path.startswith('tools/verify_') and path.endswith('.py'))
    environment.update(path for path in files if path.startswith('tools/nemo_tn') and path.endswith('.py'))
    if not environment <= files:
        raise ValueError(f'Missing environment files in inventory: {sorted(environment - files)}')
    return dict(schema=1, files=sorted(files), environment=sorted(environment))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--files-from', type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.files_from.read_text(encoding='utf-8-sig').splitlines())
    target = args.root/'tools/update-manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {len(manifest["files"])} managed paths; {len(manifest["environment"])} environment inputs.')


if __name__ == '__main__':
    main()
