"""Pydantic models shared by several routers (the OpenAPI source for TS types).

Area-specific request/response models live next to their router. Keep field
names snake_case; the browser uses them unchanged.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenModel(BaseModel):
    """Response model that tolerates extra keys from legacy snapshot dicts."""

    model_config = ConfigDict(extra="allow")


class ErrorBody(ApiModel):
    code: str
    message: str
    detail: Any = None


class ErrorResponse(OpenModel):
    error: ErrorBody


class Ok(ApiModel):
    ok: bool = True
    message: Optional[str] = None


class Capability(ApiModel):
    ok: bool
    detail: str


class Capabilities(ApiModel):
    triton: Capability
    flash_attn: Capability


class CudaInfo(ApiModel):
    available: bool
    torch: Optional[str] = None
    cuda: Optional[str] = None
    device: Optional[str] = None
    total_gb: Optional[float] = None
    error: Optional[str] = None


class Settings(ApiModel):
    ui_language: Literal["zh_CN", "en_US"]
    theme: Literal["system", "light", "dark"]
    completion_sound: bool
    notifications: bool
    use_bf16: bool
    use_gpt_latent: bool
    use_torch_compile: bool
    use_flash_accel: bool


class SettingsPatch(ApiModel):
    ui_language: Optional[Literal["zh_CN", "en_US"]] = None
    theme: Optional[Literal["system", "light", "dark"]] = None
    completion_sound: Optional[bool] = None
    notifications: Optional[bool] = None
    use_bf16: Optional[bool] = None
    use_gpt_latent: Optional[bool] = None
    use_torch_compile: Optional[bool] = None
    use_flash_accel: Optional[bool] = None


class Vram(ApiModel):
    allocated_gb: float
    reserved_gb: float
    total_gb: Optional[float] = None


class LoadedModel(ApiModel):
    loaded: bool
    adapter: Optional[str] = None
    options: Optional[dict[str, Any]] = None


class SystemSnapshot(ApiModel):
    model_present: bool
    model_dir: str
    model: LoadedModel
    cuda: Optional[CudaInfo] = None
    vram: Vram
    fake_engine: bool


class ParamRange(ApiModel):
    default: float
    min: float
    max: float
    step: float


class LanguageChoice(ApiModel):
    label: str
    code: str


class Catalog(ApiModel):
    languages: list[LanguageChoice]
    default_language: str
    emotions: list[str]
    emotion_modes: list[str]
    emotion_presets: dict[str, list[float]]
    default_preset: str
    dialogue_emotion_sources: list[str]
    dialogue_max_turns: int
    chunk_modes: list[str]
    vram_presets: list[str]
    lora_ranks: list[int]
    synthesis_params: dict[str, ParamRange]
    synthesis_flags: dict[str, bool]


class Bootstrap(ApiModel):
    version: str
    fake_engine: bool
    settings: Settings
    capabilities: Capabilities
    system: SystemSnapshot
    catalog: Catalog
    training_defaults: dict[str, Any] = Field(default_factory=dict)


JobState = Literal["running", "succeeded", "failed", "cancelled"]


class JobError(ApiModel):
    message: str
    detail: Any = None
    code: Optional[str] = None


class JobInfo(ApiModel):
    id: str
    kind: str
    title: str
    state: JobState
    progress: Optional[float] = None
    stage: Optional[str] = None
    started_at: float
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[JobError] = None
    cancellable: bool
    cancel_requested: bool
    gpu: bool
    meta: dict[str, Any] = Field(default_factory=dict)


class StatusItem(ApiModel):
    id: int
    time: float
    phase: str
    message: str
    level: Literal["info", "working", "success", "error"]
    detail: Optional[str] = None


class ConsoleLine(ApiModel):
    id: int
    text: str
    level: str


class ConsoleSnapshot(ApiModel):
    lines: list[ConsoleLine]
    current: str
    reset: bool = False


class Activity(OpenModel):
    jobs: list[JobInfo]
    status: list[StatusItem]
    training: Optional[dict[str, Any]] = None
    export: Optional[dict[str, Any]] = None
