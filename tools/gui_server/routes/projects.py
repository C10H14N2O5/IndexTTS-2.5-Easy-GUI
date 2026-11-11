"""Version-compatible project, corpus and training configuration APIs."""
from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, Query
from pydantic import Field, field_validator
from ..i18n import render
from ..schemas import ApiModel, JobInfo, Ok
from ..services import projects, metrics, posthoc
from ..services.projects import ProjectInfo, TrainingConfig, CheckpointInfo

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ValidationRunAccepted(ApiModel):
    job: JobInfo


class ProjectCreate(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    language: str = "ZH"
    source_folder: str = ""

    @field_validator("language")
    @classmethod
    def valid_language(cls, value):
        return projects.language_code(value)


class ProjectClone(ApiModel):
    name: str | None = Field(None, min_length=1, max_length=120)


class ProjectDelete(ApiModel):
    confirm_name: str


class DatasetPatch(ApiModel):
    source_folder: str | None = None
    language: str | None = None
    manifest: str | None = None
    prepare: bool = False

    @field_validator("language")
    @classmethod
    def valid_language(cls, value):
        return projects.language_code(value) if value is not None else value


class ValidationPatch(ApiModel):
    source_folder: str | None = None
    enabled: bool | None = None
    interval_steps: int | None = Field(None, ge=0)
    dense_enabled: bool | None = None
    dense_start_step: int | None = Field(None, ge=1)
    dense_end_step: int | None = Field(None, ge=1)
    dense_interval_steps: int | None = Field(None, ge=1)
    prepare: bool = False


class AutotuneRequest(ApiModel):
    vram_preset: Literal["24 GB", "32 GB+"] = "24 GB"


class AutotuneResponse(ApiModel):
    training: TrainingConfig
    changed: list[str]
    minutes: float


@router.get("", response_model=list[ProjectInfo])
def list_projects():
    return render(projects.list_projects())


@router.post("", response_model=ProjectInfo, status_code=201)
def create(request: ProjectCreate):
    return render(projects.create(**request.model_dump()))


@router.get("/{name}", response_model=ProjectInfo)
def get(name: str):
    return render(projects.get(name))


@router.delete("/{name}", response_model=Ok)
def delete(name: str, request: ProjectDelete):
    return render(projects.delete(name, request.confirm_name))


@router.post("/{name}/clone", response_model=ProjectInfo, status_code=201)
def clone(name: str, request: ProjectClone):
    return render(projects.clone(name, request.name))


@router.put("/{name}/dataset", response_model=ProjectInfo)
def dataset(name: str, request: DatasetPatch):
    return render(projects.update_dataset(name, request.model_dump(exclude_none=True, exclude_unset=True)))


@router.put("/{name}/validation", response_model=ProjectInfo)
def validation(name: str, request: ValidationPatch):
    values = request.model_dump(exclude_unset=True)
    nullable = {"interval_steps", "dense_start_step", "dense_end_step", "dense_interval_steps"}
    values = {k: v for k, v in values.items() if v is not None or k in nullable}
    return render(projects.update_validation(name, values))


@router.put("/{name}/training", response_model=ProjectInfo)
def training(name: str, request: TrainingConfig):
    return render(projects.update_training(name, request.model_dump(exclude_unset=True)))


@router.post("/{name}/autotune", response_model=AutotuneResponse)
def autotune(name: str, request: AutotuneRequest):
    return projects.autotune(name, request.vram_preset)


@router.get("/{name}/checkpoints", response_model=list[CheckpointInfo])
def checkpoints(name: str, adapter: str | None = None):
    return projects.checkpoints(name, adapter)


@router.get("/{name}/adapters", response_model=list[projects.AdapterFolder])
def adapters(name: str):
    return projects.adapter_folders(name)


@router.get("/{name}/validation-results", response_model=posthoc.PosthocResults)
def validation_results(name: str, adapter: str | None = None):
    return render(posthoc.results(name, adapter))


@router.post("/{name}/validation-runs", response_model=ValidationRunAccepted, status_code=202)
def validation_run(name: str, request: posthoc.ValidationRunRequest):
    return render({"job": posthoc.start(name, request).to_dict()})


@router.get("/{name}/metrics", response_model=metrics.MetricsResponse)
def read_metrics(name: str, adapter: str | None = None, max_points: int = Query(2000, ge=1, le=2000)):
    return metrics.read(name, adapter, max_points)
