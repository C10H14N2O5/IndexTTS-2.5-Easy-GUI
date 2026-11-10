"""Compatibility entry point for IndexTTS Easy GUI 2.0.

The existing Windows launcher and ZIP updater still invoke this file. Keep the
application lease and interrupted-update guard here, before importing the web
server; importing this module itself must not load models or create UI state.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> None:
    from tools.runtime_lock import gui_application_lock, require_clean_update

    # run_gui.py already owns this same-process lease during its environment
    # preflight. gui_application_lock is intentionally idempotent for this root.
    gui_application_lock(ROOT)
    require_clean_update(ROOT)

    from tools.gui_server.launcher import main as launch
    launch(argv)


if __name__ == "__main__":
    main()
