"""Bootstrap, settings, server-sent events, activity and generic job routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from tools.version import __version__ as APP_VERSION

from .. import i18n, paths
from ..schemas import Activity, Bootstrap, ConsoleSnapshot, JobInfo, Settings, SettingsPatch
from ..services import catalog, settings as settings_service, system as system_service
from ..state import AppState, current

router = APIRouter(prefix="/api", tags=["core"])


def system_snapshot() -> dict:
    from ..services import engine
    state = current()
    return {
        "model_present": True if state.fake else system_service.model_files_present(),
        "model_dir": str(paths.MODEL_DIR),
        "model": engine.loaded_info(),
        "cuda": system_service.cuda_info() if system_service.cuda_ready() else None,
        "vram": system_service.vram_metrics(),
        "fake_engine": state.fake,
    }


def _training_defaults() -> dict:
    try:
        from ..services import projects
        return dict(getattr(projects, "TRAINING_DEFAULTS", {}))
    except ImportError:
        return {}


def register(state: AppState) -> None:
    state.register_snapshot("system", system_snapshot, interval=3.0)


def full_snapshot() -> dict:
    state = current()
    data = {
        "jobs": state.jobs.list(),
        "status": state.status.snapshot(),
        "console": state.console.snapshot(),
        "settings": settings_service.load(),
    }
    for name, provider in state.providers.items():
        try:
            data[name] = provider.fn()
        except Exception as exc:  # noqa: BLE001
            data[name] = None
            print(f"[gui] snapshot provider {name} failed: {type(exc).__name__}: {exc}", flush=True)
    return data


@router.get("/bootstrap", response_model=Bootstrap)
def bootstrap():
    state = current()
    return i18n.render({
        "version": APP_VERSION,
        "fake_engine": state.fake,
        "settings": settings_service.load(),
        "capabilities": system_service.acceleration_capabilities(),
        "system": system_snapshot(),
        "catalog": catalog.bootstrap_catalog(),
        "training_defaults": _training_defaults(),
    })


@router.get("/settings", response_model=Settings)
def get_settings():
    return settings_service.load()


@router.put("/settings", response_model=Settings)
def put_settings(patch: SettingsPatch):
    updated = settings_service.update(patch.model_dump(exclude_none=True))
    current().publish("settings", updated)
    return updated


@router.get("/events", include_in_schema=False)
async def events(request: Request):
    state = current()
    language = i18n.normalize(request.query_params.get("lang"))
    stream = state.hub.stream(language, full_snapshot, request.is_disconnected)
    return StreamingResponse(stream, media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.get("/activity", response_model=Activity)
def activity():
    state = current()
    data = {"jobs": state.jobs.list(), "status": state.status.snapshot()}
    for name in ("training", "export"):
        provider = state.providers.get(name)
        data[name] = provider.fn() if provider else None
    return i18n.render(data)


@router.get("/console", response_model=ConsoleSnapshot)
def console(limit: int = 400):
    return current().console.snapshot(max(1, min(limit, 1400)))


@router.delete("/console", response_model=ConsoleSnapshot)
def clear_console():
    state = current()
    state.console.clear()
    return state.console.snapshot()


@router.get("/jobs", response_model=list[JobInfo])
def list_jobs():
    return i18n.render(current().jobs.list())


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    return i18n.render(current().jobs.get(job_id).to_dict())


@router.post("/jobs/{job_id}/cancel", response_model=JobInfo)
def cancel_job(job_id: str):
    return i18n.render(current().jobs.cancel(job_id).to_dict())
