"""Voice and emotion library CRUD, bounded audio uploads and adapters."""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File
from pydantic import Field

from ..schemas import ApiModel
from ..services import library, uploads, adapters

router = APIRouter(prefix='/api', tags=['library'])


def register(state):
    uploads.purge_expired()


class AudioUpload(ApiModel):
    path: str
    name: str
    size: int
    duration: float | None


class Voice(AudioUpload):
    label: str
    transcript: str
    modified: float


class SaveVoice(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    source_path: str
    transcript: str = Field(default='', max_length=100_000)
    overwrite: bool = False


class UpdateVoice(ApiModel):
    new_name: str | None = Field(default=None, min_length=1, max_length=120)
    transcript: str | None = Field(default=None, max_length=100_000)


class Deleted(ApiModel):
    deleted: int


class EmotionDescription(ApiModel):
    name: str
    label: str
    path: str
    text: str


class SaveDescription(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=100_000)
    overwrite: bool = False


class Adapter(ApiModel):
    label: str
    path: str
    project: str
    checkpoint: str
    kind: str


@router.post('/uploads', response_model=AudioUpload)
async def upload(file: UploadFile = File(...)):
    return await uploads.save_upload(file)


@router.get('/voices', response_model=list[Voice])
def list_voices():
    return library.voices()


@router.post('/voices', response_model=Voice, status_code=201)
def save_voice(body: SaveVoice):
    return library.save_voice(body.name, body.source_path, body.transcript, body.overwrite)


@router.delete('/voices', response_model=Deleted)
def delete_all_voices():
    return {'deleted': library.delete_all_voices()}


@router.get('/voices/{name}', response_model=Voice)
def get_voice(name: str):
    return library.get_voice(name)


@router.patch('/voices/{name}', response_model=Voice)
def update_voice(name: str, body: UpdateVoice):
    return library.update_voice(name, body.new_name, body.transcript)


@router.delete('/voices/{name}', response_model=Deleted)
def delete_voice(name: str):
    return {'deleted': library.delete_voice(name)}


@router.get('/emotion-descriptions', response_model=list[EmotionDescription])
def list_descriptions():
    return library.descriptions()


@router.post('/emotion-descriptions', response_model=EmotionDescription, status_code=201)
def save_description(body: SaveDescription):
    return library.save_description(body.name, body.text, body.overwrite)


@router.delete('/emotion-descriptions/{name}', response_model=Deleted)
def delete_description(name: str):
    return {'deleted': library.delete_description(name)}


@router.get('/adapters', response_model=list[Adapter])
def list_adapters():
    return adapters.list_adapters()
