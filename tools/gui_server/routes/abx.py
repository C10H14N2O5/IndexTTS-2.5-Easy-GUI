"""Saved blind comparisons, resumable GPU generation, ratings and reveal."""
from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, Query
from pydantic import Field
from ..i18n import render
from ..schemas import ApiModel, JobInfo
from ..services import abx

router = APIRouter(prefix="/api/abx", tags=["abx"])


class AbxRating(ApiModel):
    index: int = Field(ge=0)
    score: Literal[-2, -1, 0, 1, 2] | None


class AbxJobResponse(ApiModel):
    job: JobInfo


@router.get("/sessions", response_model=list[abx.AbxSummary])
def sessions():
    return abx.sessions()


@router.post("/sessions", response_model=abx.AbxSession, status_code=201)
def create(request: abx.AbxCreate):
    return abx.create(request)


@router.get("/sessions/{session_id}", response_model=abx.AbxSession)
def get(session_id: str, trial: int = Query(0, ge=0)):
    return abx.detail(abx.load(session_id), trial)


@router.post("/sessions/{session_id}/generate", response_model=AbxJobResponse, status_code=202)
def generate(session_id: str):
    return render({"job": abx.generate(session_id).to_dict()})


@router.put("/sessions/{session_id}/rate", response_model=abx.AbxSession)
def rate(session_id: str, request: AbxRating):
    return abx.rate(session_id, request.index, request.score)


@router.post("/sessions/{session_id}/reveal", response_model=abx.AbxSession)
def reveal(session_id: str, trial: int = Query(0, ge=0)):
    return abx.reveal(session_id, trial)
