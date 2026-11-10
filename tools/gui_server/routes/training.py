"""Trainer lifecycle, defaults and live status."""
from __future__ import annotations
from fastapi import APIRouter
from pydantic import Field
from ..i18n import render
from ..schemas import ApiModel
from ..services import training, tensorboard
from ..services.projects import TrainingConfig, TRAINING_DEFAULTS

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingStart(ApiModel):
    project: str
    training: TrainingConfig | None = None


class TensorboardRequest(ApiModel):
    project: str
    adapter: str | None = None
    reload: bool = False
    port: int = Field(6006, ge=1024, le=65535)


@router.get("/defaults", response_model=TrainingConfig)
def defaults():
    return TRAINING_DEFAULTS


@router.get("/status", response_model=training.TrainingSnapshot)
def status():
    return render(training.snapshot())


@router.post("/start", response_model=training.TrainingSnapshot, status_code=202)
def start(request: TrainingStart):
    return render(training.start(request.project, request.training.model_dump(exclude_unset=True) if request.training else None))


@router.post("/stop", response_model=training.TrainingSnapshot)
def stop():
    return render(training.stop())


@router.post("/tensorboard", response_model=tensorboard.TensorboardResponse)
def tensorboard_start(request: TensorboardRequest):
    return render(tensorboard.start(**request.model_dump()))


def register(state):
    training.register(state)
