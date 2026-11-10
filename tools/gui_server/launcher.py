"""Start the Easy GUI web server (called by easy_gui.py, i.e. 2- run.bat).

Order: application lock and pending-update check (done by the caller), process
environment, model check (skipped by the fake engine), console mirror, then a
single-worker uvicorn server. The browser opens once the port accepts
connections. ``python -m tools.gui_server.launcher --fake --host 0.0.0.0``
starts a development/demo server without the lock or the model.
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser

from . import console, paths, runtime_env


def _browser_url(host: str, port: int) -> str:
    shown = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    if ":" in shown and not shown.startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/"


def _open_when_ready(host: str, port: int, url: str, timeout: float = 120.0) -> None:
    probe = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host

    def wait_and_open():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((probe, port), timeout=1.0):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            return
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=wait_and_open, name="open-browser", daemon=True).start()


def ensure_model() -> None:
    from tools.ensure_model import main as ensure_model_main
    if ensure_model_main() != 0:
        raise SystemExit("Automatic IndexTTS 2.5 model setup failed. Review the console output and retry.")


def serve(host: str | None = None, port: int | None = None, *, fake: bool | None = None,
          open_browser: bool = True) -> None:
    host = host or os.environ.get("INDEXTTS_HOST", "127.0.0.1")
    port = int(port or os.environ.get("INDEXTTS_PORT", "7860"))
    if fake is not None:
        os.environ["INDEXTTS_FAKE_ENGINE"] = "1" if fake else "0"
    runtime_env.configure()
    fake = runtime_env.fake_engine_enabled()
    if not fake:
        ensure_model()
    console.install()
    import uvicorn
    from .app import create_app
    app = create_app(fake=fake)
    url = _browser_url(host, port)
    print(f"[gui] IndexTTS 2.5 Easy GUI: {url}" + ("  (fake engine)" if fake else ""), flush=True)
    if open_browser:
        _open_when_ready(host, port, url)
    # One worker and one process: models, jobs and the event hub are in-process singletons.
    uvicorn.run(app, host=host, port=port, log_level="warning", workers=1, access_log=False,
                timeout_graceful_shutdown=3)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="IndexTTS 2.5 Easy GUI web server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fake", action="store_true", help="simulate synthesis, training, ASR and export")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(args.host, args.port, fake=True if args.fake else None, open_browser=not args.no_browser)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.ROOT))
    main()
