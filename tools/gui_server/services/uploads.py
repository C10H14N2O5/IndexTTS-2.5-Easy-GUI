"""Audio import validation, bounded uploads and 24-hour temporary-file expiry."""
from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from .. import paths
from ..errors import NotFound, ServiceError

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_TTL_SECONDS = 24 * 60 * 60


def require_audio(value: str | None) -> Path:
    if not value or value == paths.NONE_SENTINEL:
        raise ServiceError('Choose or record a reference audio file first.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = paths.ROOT / path
    path = path.resolve()
    if path.suffix.lower() not in paths.AUDIO_SUFFIXES:
        raise ServiceError('Only audio files are supported.')
    if not path.is_file():
        raise NotFound('Audio file not found: {0}', str(path))
    return path


def audio_duration(path: Path, *, validate: bool = False) -> float | None:
    try:
        import soundfile
        info = soundfile.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError('empty audio')
        return float(info.duration)
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    # libsndfile cannot decode all supported formats (notably M4A/AAC/WMA).
    # Ask the already-installed FFmpeg runtime, never download a new executable.
    if validate or path.suffix.lower() in {'.m4a', '.aac', '.wma'}:
        try:
            from indextts_training.asr import AsrService
            ffmpeg = AsrService(paths.ROOT)._ffmpeg()
            result = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-v', 'info', '-i', str(path),
                                     '-map', '0:a:0', '-t', '0.02', '-f', 'null', '-'],
                                    capture_output=True, timeout=20, check=False,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode == 0:
                found = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', result.stderr.decode(errors='replace'))
                if found:
                    h, m, s = found.groups()
                    return int(h) * 3600 + int(m) * 60 + float(s)
                return None
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass
    if validate:
        raise ServiceError('The file does not contain readable audio.')
    return None


def purge_expired() -> None:
    directory = paths.UPLOADS_DIR
    if not directory.is_dir():
        return
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for item in directory.iterdir():
        # Never follow a symlink into another directory or remove non-upload files.
        if item.is_symlink() or not item.is_file():
            continue
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink()
        except OSError:
            pass


async def save_upload(upload) -> dict:
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix not in paths.AUDIO_SUFFIXES:
        raise ServiceError('Only audio files are supported.')
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise ServiceError('Audio uploads must not exceed 200 MB.', status=413, code='too_large')
    paths.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    purge_expired()
    path = paths.UPLOADS_DIR / f'{uuid.uuid4().hex}{suffix}'
    size = 0
    try:
        with path.open('xb') as stream:
            while block := await upload.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise ServiceError('Audio uploads must not exceed 200 MB.', status=413, code='too_large')
                stream.write(block)
        from starlette.concurrency import run_in_threadpool
        duration = await run_in_threadpool(audio_duration, path, validate=True)
    except BaseException:
        # This is solely the new, task-owned partial upload.
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return {'path': str(path), 'name': Path(upload.filename or path.name).name, 'size': size, 'duration': duration}
