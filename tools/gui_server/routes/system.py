"""Runtime diagnostics and model unloading without importing the model eagerly."""
from __future__ import annotations

from fastapi import APIRouter
from .. import paths, status
from ..i18n import render, msg
from ..schemas import ApiModel, SystemSnapshot, Capabilities, Ok
from ..services import engine, settings, system
from ..services.library import mutation
from ..state import current
from .core import system_snapshot

router = APIRouter(prefix='/api', tags=['system'])


class AsrStatus(ApiModel):
    available: bool
    message: str
    backend: str
    model: str
    executable: str | None = None


class RuntimeSettings(ApiModel):
    use_bf16: bool
    use_gpt_latent: bool
    use_torch_compile: bool
    use_flash_accel: bool


class SystemStatus(SystemSnapshot):
    capabilities: Capabilities
    asr: AsrStatus
    runtime: RuntimeSettings
    completion_sound_path: str


@router.get('/system/status', response_model=SystemStatus)
def get_system_status():
    state = current()
    return render({**system_snapshot(), 'capabilities': system.acceleration_capabilities(),
                   'asr': state.asr.status(), 'runtime': settings.runtime(),
                   'completion_sound_path': str(paths.ASSETS_DIR / 'inference_training_done.wav')})


@router.post('/model/unload', response_model=Ok)
def unload_model():
    with mutation('model unload'):
        engine.unload()
    message = msg('Model unloaded. VRAM released where possible.')
    status.publish('Runtime', message, 'success')
    return render({'ok': True, 'message': message})
