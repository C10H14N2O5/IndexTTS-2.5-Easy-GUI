"""FastAPI application factory for the Easy GUI web interface.

``create_app(fake=None, background=True)``:

* builds the shared :class:`~tools.gui_server.state.AppState`;
* wires the singletons (ASR, export manager, training manager), or their
  development fakes when ``INDEXTTS_FAKE_ENGINE=1``;
* includes every router module listed in ``ROUTER_MODULES`` (each exposes
  ``router`` and may expose ``register(state)``), the unchanged proofreading
  router (``/proofread/api/*``) and the export download router (``/exports``);
* serves the built single-page app from ``tools/gui_web``.

``background=False`` skips the lifespan ticker and the CUDA warm-up; the update
health check and unit tests use it.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from tools.version import __version__ as APP_VERSION

from . import i18n, paths, runtime_env
from .errors import ServiceError, error_payload
from .security import SPA_CSP, GuardMiddleware
from .state import AppState, build, install

ROUTER_MODULES = (
    "core", "system", "library", "synthesis", "history", "projects", "training",
    "abx", "export", "files", "media",
)
API_PREFIXES = ("/api/", "/proofread/api/", "/exports/")


class _ImmutableStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["cache-control"] = "public, max-age=31536000, immutable"
        return response


def _wire_services(state: AppState) -> None:
    from indextts_training.export import ExportManager
    from indextts_training.manager import TRAINING_MANAGER

    state.training = TRAINING_MANAGER
    if state.fake:
        from . import fake
        fake.install(state)
    else:
        from indextts_training.asr import AsrService
        state.asr = AsrService(paths.ROOT)
        state.export = ExportManager(paths.ROOT)


def _language(request: Request) -> str | None:
    return request.headers.get("x-ui-language") or request.query_params.get("lang")


def _install_error_handlers(app: FastAPI) -> None:
    async def handle(request: Request, exc: Exception):
        status, body = error_payload(exc, i18n.normalize(_language(request)) if _language(request) else None)
        if status >= 500:
            import traceback
            traceback.print_exception(exc)
        return JSONResponse(i18n.render(body), status_code=status)

    for exc_type in (ServiceError, ValueError, KeyError, RuntimeError, OSError, Exception):
        app.add_exception_handler(exc_type, handle)


def _mount_spa(app: FastAPI) -> None:
    web = paths.WEB_DIR
    app.mount("/assets", _ImmutableStatic(directory=web / "assets", check_dir=False), name="assets")
    spa_headers = {"Cache-Control": "no-store", "Content-Security-Policy": SPA_CSP}

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if ("/" + full_path).startswith(API_PREFIXES) or full_path in {"api", "exports"}:
            return JSONResponse({"error": {"code": "not_found", "message": i18n.t("Not found."), "detail": None}},
                                status_code=404)
        candidate = (web / full_path).resolve() if full_path else None
        if (candidate and candidate.is_file() and candidate.parent == web.resolve()
                and candidate.name != "index.html"):
            return FileResponse(candidate, headers={"Cache-Control": "public, max-age=3600"})
        index = web / "index.html"
        if not index.is_file():
            return HTMLResponse("<!doctype html><meta charset=utf-8><title>IndexTTS Easy GUI</title>"
                                "<p>The web interface is not built. Run <code>pnpm build</code> in "
                                "<code>frontend/</code>.</p>", status_code=503, headers=spa_headers)
        return HTMLResponse(index.read_text(encoding="utf-8"), headers=spa_headers)


async def _ticker(state: AppState) -> None:
    """Publish console batches (4 Hz) and changed provider snapshots."""
    last: dict[str, str] = {}
    due: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(0.25)
        lines = state.console.drain()
        if lines is not None:
            state.publish("console", lines)
        now = loop.time()
        for name, provider in list(state.providers.items()):
            if provider.interval <= 0 or now < due.get(name, 0.0):
                continue
            due[name] = now + provider.interval
            try:
                snapshot = await asyncio.to_thread(provider.fn)
            except Exception as exc:  # noqa: BLE001
                print(f"[gui] snapshot provider {name} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            key = json.dumps(i18n.render(snapshot, "en_US"), sort_keys=True, default=str)
            if provider.publish_unchanged or last.get(name) != key:
                last[name] = key
                state.publish(name, snapshot)


def create_app(*, fake: bool | None = None, background: bool = True) -> FastAPI:
    fake = runtime_env.fake_engine_enabled() if fake is None else fake
    paths.ensure_dirs()
    state = build(fake)
    install(state)
    _wire_services(state)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.hub.attach_loop(asyncio.get_running_loop())
        task = None
        if background:
            from .services import system
            threading.Thread(target=system.warm_up, name="cuda-warm-up", daemon=True).start()
            task = asyncio.create_task(_ticker(state))
        try:
            yield
        finally:
            state.hub.close()
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    app = FastAPI(title="IndexTTS 2.5 Easy GUI", version=APP_VERSION, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")
    app.state.gui = state
    _install_error_handlers(app)

    for name in ROUTER_MODULES:
        module = importlib.import_module(f"{__package__}.routes.{name}")
        if hasattr(module, "register"):
            module.register(state)
        app.include_router(module.router)

    from indextts_training.export import create_export_router
    from indextts_training.proofread import create_proofread_router
    app.include_router(create_proofread_router(paths.ROOT, state.asr,
                                               lambda: bool(state.training.snapshot().get("running"))))
    app.include_router(create_export_router(state.export))
    _mount_spa(app)
    app.add_middleware(GuardMiddleware)
    return app


def web_assets_ok(web: Path | None = None) -> tuple[bool, str]:
    """True when index.html exists and every /assets/* it references exists."""
    import re
    web = web or paths.WEB_DIR
    index = web / "index.html"
    if not index.is_file():
        return False, f"missing {index}"
    missing = [ref for ref in re.findall(r'(?:src|href)="/(assets/[^"]+)"', index.read_text(encoding="utf-8"))
               if not (web / ref).is_file()]
    return (not missing), ("missing " + ", ".join(missing)) if missing else "ok"
