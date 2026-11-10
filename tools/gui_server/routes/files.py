"""Read-only local filesystem discovery (no arbitrary mutation endpoints)."""
from __future__ import annotations

from typing import Literal
from fastapi import APIRouter
from ..schemas import ApiModel
from ..services import files

router = APIRouter(prefix='/api', tags=['files'])


class FileRoot(ApiModel):
    label: str
    path: str


class FileEntry(ApiModel):
    name: str
    path: str
    type: Literal['dir', 'file']
    size: int | None
    modified: float


class DirectoryListing(ApiModel):
    path: str
    parent: str | None
    entries: list[FileEntry]


@router.get('/fs/roots', response_model=list[FileRoot])
def roots():
    return files.roots()


@router.get('/fs/list', response_model=DirectoryListing)
def list_directory(path: str = '', kind: Literal['dir', 'pth', 'audio'] = 'dir'):
    return files.list_directory(path, kind)
