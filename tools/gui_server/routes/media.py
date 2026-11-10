"""Whitelisted audio/image delivery with Starlette byte-range support."""
from __future__ import annotations

import mimetypes
from fastapi import APIRouter
from fastapi.responses import FileResponse
from ..services.files import media_path

router = APIRouter(prefix='/api', tags=['media'])


@router.head('/media', response_class=FileResponse, include_in_schema=False)
@router.get('/media', response_class=FileResponse,
            responses={200: {'content': {'application/octet-stream': {}}},
                       206: {'description': 'Requested byte range'}})
def media(path: str, download: bool = False):
    file = media_path(path)
    media_type = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
    return FileResponse(file, media_type=media_type, filename=file.name,
                        content_disposition_type='attachment' if download else 'inline',
                        headers={'Cache-Control': 'private, no-cache'})
