"""Saved output history and safe top-level output cleanup."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Query

from ..schemas import ApiModel
from ..services import history
from .library import Deleted

router = APIRouter(prefix='/api', tags=['history'])


class HistoryItem(ApiModel):
    name: str
    path: str
    created: float
    duration: float | None
    size: int
    kind: str
    params: dict[str, Any] | None


@router.get('/history', response_model=list[HistoryItem])
def list_history(limit: int = Query(default=100, ge=1, le=1000)):
    return history.list_history(limit)


@router.delete('/history', response_model=Deleted)
@router.delete('/outputs', response_model=Deleted)
def delete_all():
    return {'deleted': history.delete_all()}


@router.delete('/history/{name}', response_model=Deleted)
def delete_one(name: str):
    return {'deleted': history.delete(name)}
