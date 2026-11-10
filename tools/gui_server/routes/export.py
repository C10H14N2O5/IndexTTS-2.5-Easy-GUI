"""Checkpoint export lifecycle; downloads retain the token-scoped 1.x route."""
from __future__ import annotations
from typing import Literal
from fastapi import APIRouter
from ..i18n import render
from ..schemas import ApiModel
from ..services import export

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportStart(ApiModel):
    adapter: str
    mode: Literal["browser", "path"] = "browser"
    destination: str = "voice_checkpoint.pth"
    base: str = ""
    config: str = ""
    expected_base: str = ""


@router.get("/adapter-info", response_model=export.AdapterInfo)
def adapter_info(adapter: str = ""):
    return render(export.info(adapter))


@router.get("/status", response_model=export.ExportSnapshot)
def status():
    return render(export.snapshot())


@router.post("/start", response_model=export.ExportSnapshot, status_code=202)
def start(request: ExportStart):
    return render(export.start(**request.model_dump()))


@router.post("/cancel", response_model=export.ExportSnapshot)
def cancel():
    return render(export.cancel())


def register(state):
    export.register(state)
