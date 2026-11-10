"""Synthesis jobs. Full request schemas are part of the generated OpenAPI."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import Field, AliasChoices

from ..i18n import render
from ..schemas import ApiModel, JobInfo
from ..services import synthesis, dialogue, asr_jobs

router = APIRouter(prefix='/api', tags=['synthesis'])


class JobAccepted(ApiModel):
    job: JobInfo


class AudioResult(ApiModel):
    path: str
    name: str
    duration: float | None
    seconds: float
    saved: bool
    params: dict
    turns: int | None = None


class SynthesisJob(JobInfo):
    result: AudioResult | None = None


class SynthesisAccepted(ApiModel):
    job: SynthesisJob


class AsrRequest(ApiModel):
    path: str = Field(validation_alias=AliasChoices('path', 'voice_path', 'prompt_audio'))


class AsrResult(ApiModel):
    text: str


class AsrJob(JobInfo):
    result: AsrResult | None = None


class AsrAccepted(ApiModel):
    job: AsrJob


@router.post('/synthesis', response_model=SynthesisAccepted, status_code=202)
def create_synthesis(body: synthesis.SynthesisRequest):
    return render({'job': synthesis.submit(body).to_dict()})


@router.post('/dialogue', response_model=SynthesisAccepted, status_code=202)
def create_dialogue(body: dialogue.DialogueRequest):
    return render({'job': dialogue.submit(body).to_dict()})


@router.post('/asr/transcribe', response_model=AsrAccepted, status_code=202)
@router.post('/asr/reference', response_model=AsrAccepted, status_code=202)
def transcribe(body: AsrRequest):
    return render({'job': asr_jobs.submit(body.path).to_dict()})
