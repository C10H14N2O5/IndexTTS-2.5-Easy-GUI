"""Offline post-update checks; no model loading, downloads or browser launch."""
from __future__ import annotations

import argparse
import ast
import contextlib
from html.parser import HTMLParser
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


REQUIRED_ROUTES = {
    ("GET", "/api/bootstrap"), ("GET", "/api/events"), ("GET", "/api/system/status"),
    ("POST", "/api/synthesis"), ("POST", "/api/dialogue"), ("GET", "/api/voices"),
    ("GET", "/api/projects"), ("PUT", "/api/projects/{name}/dataset"),
    ("PUT", "/api/projects/{name}/validation"), ("GET", "/api/projects/{name}/metrics"),
    ("POST", "/api/training/start"), ("POST", "/api/training/stop"),
    ("GET", "/api/abx/sessions"), ("POST", "/api/abx/sessions/{session_id}/generate"),
    ("POST", "/api/export/start"), ("POST", "/api/export/cancel"),
    ("POST", "/proofread/api/session"), ("GET", "/exports/{task_id}/{kind}"),
}


class _EntryAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.modules = 0
        self.root = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id") == "root":
            self.root = True
        if tag == "script" and values.get("src"):
            self.refs.append(values["src"])
            if values.get("type") == "module":
                self.modules += 1
        if tag == "link" and values.get("href"):
            self.refs.append(values["href"])


def validate_web_assets(root: Path) -> int:
    """Validate the SPA entry and its local JS/CSS/font references, including lazy chunks."""
    web = (root / "tools/gui_web").resolve()
    index = web / "index.html"
    if not index.is_file():
        raise RuntimeError("Web interface is missing tools/gui_web/index.html.")
    parser = _EntryAssets()
    parser.feed(index.read_text(encoding="utf-8"))
    if not parser.root or not parser.modules:
        raise RuntimeError("Web interface entry must contain #root and a module script.")
    visited = {index}
    queue = []

    def reference(value: str, owner: Path):
        value = value.strip()
        if value.startswith(("data:", "#")):
            return
        url = urlsplit(value)
        if url.scheme or url.netloc:
            raise RuntimeError(f"Web interface assets must be local: {value}")
        local = unquote(url.path)
        target = (web / local.lstrip("/") if local.startswith("/") or local.startswith("assets/")
                  else owner.parent / local).resolve()
        if not target.is_relative_to(web):
            raise RuntimeError(f"Web interface asset escapes build directory: {value}")
        if not target.is_file():
            raise RuntimeError(f"Web interface asset is missing: {target.relative_to(web)}")
        if target not in visited:
            visited.add(target)
            queue.append(target)

    for ref in parser.refs:
        reference(ref, index)
    # Validate every built JS/CSS, not only the eagerly loaded entry. Vite emits
    # lazy chunks referenced in map-dependency arrays rather than HTML tags.
    for asset in sorted((web / "assets").glob("*")):
        if asset.is_file() and asset.suffix in {".js", ".css"} and asset not in visited:
            visited.add(asset)
            queue.append(asset)
    while queue:
        owner = queue.pop()
        if owner.suffix == ".css":
            for ref in re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", owner.read_text(encoding="utf-8")):
                reference(ref, owner)
        elif owner.suffix == ".js":
            text = owner.read_text(encoding="utf-8")
            for ref in re.findall(r"['\"]((?:\./|\.\./|/assets/|assets/)[^'\"\s]+\.(?:js|css|woff2?|svg|png|jpe?g))['\"]", text):
                reference(ref, owner)
    return len(visited)


def validate_routes(app) -> int:
    routes = {(method, route.path) for route in app.routes for method in getattr(route, "methods", ())}
    missing = REQUIRED_ROUTES - routes
    if missing:
        raise RuntimeError("GUI server is missing required routes: " + ", ".join(f"{method} {path}" for method, path in sorted(missing)))
    slices = set(app.state.gui.providers)
    if not {"system", "training", "export"} <= slices:
        raise RuntimeError("GUI server is missing live snapshot providers.")
    # Builds the Pydantic response schemas too, catching unresolved imports/types.
    app.openapi()
    return len(routes)


def validate_runtime(root: Path) -> int:
    """Construct the real FastAPI router tree in temporary storage, without its lifespan."""
    for name in ("indextts_training.asr", "indextts_training.proofread", "indextts_training.export",
                 "tools.merge_lora_to_pth", "tools.gui_server.launcher"):
        importlib.import_module(name)
    module = importlib.import_module("tools.gui_server.app")
    if Path(module.__file__).resolve() != root / "tools/gui_server/app.py":
        raise RuntimeError("GUI server imported from a different installation.")
    from tools.gui_server import paths
    original_root = paths.ROOT
    # Do not create user libraries, temp files or model placeholders while
    # checking an update. All runtime-writable paths are isolated and removed.
    with contextlib.ExitStack() as stack:
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="indextts-health-")))
        for key, value in vars(paths).copy().items():
            if isinstance(value, Path) and key != "WEB_DIR":
                try:
                    replacement = scratch / value.relative_to(original_root)
                except ValueError:
                    continue
                stack.enter_context(patch.object(paths, key, replacement))
        app = module.create_app(fake=False, background=False)
        count = validate_routes(app)
        app.state.gui.export.cancel()
        return count


def validate_syntax(root: Path, manifest: dict) -> int:
    if os.name == "nt":
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
        shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        result = subprocess.run([str(shell), "-NoProfile", "-Command", script],
                                env=dict(os.environ, INDEXTTS_VALIDATE_ROOT=str(root)), capture_output=True,
                                timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError("PowerShell syntax check failed: " + result.stderr.decode(errors="replace"))
    count = 0
    for relative in manifest["files"]:
        path = root / relative
        if relative.startswith("tools/gui_web/") and not path.is_file():
            raise RuntimeError(f"Managed SPA file is missing: {relative}")
        if relative.endswith(".py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args(argv).root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1")
    manifest = json.loads((root / "tools/update-manifest.json").read_text(encoding="utf-8"))
    count = validate_syntax(root, manifest)
    assets = validate_web_assets(root)
    routes = validate_runtime(root)
    print(json.dumps(dict(syntax_files=count, gui_config=True, spa_assets=assets, api_routes=routes,
                          model_loaded=False, browser_opened=False)), flush=True)


if __name__ == "__main__":
    main()
