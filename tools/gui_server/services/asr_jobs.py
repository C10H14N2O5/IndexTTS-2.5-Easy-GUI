"""Serialized reference transcription exposed as a cancellable CPU job."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .. import paths, status
from ..i18n import msg, raw
from ..jobs import JobCancelled
from ..state import current
from .uploads import require_audio


def submit(path: str):
    audio = require_audio(path)
    state = current()
    if not state.asr.status().get('available'):
        from ..errors import ServiceError
        raise ServiceError(raw(state.asr.status().get('message')), status=409, code='asr_unavailable')

    def run(ctx):
        from indextts_training.asr import AsrCancelled
        ctx.check_cancelled()
        ctx.progress(0.1, 'Recognizing audio with SenseVoiceSmall Q8 (CPU).')
        try:
            # A CPU ASR job must not block the GPU lane; take a short locked copy
            # so a subsequent library rename/delete cannot invalidate its input.
            from .library import LOCK
            with tempfile.TemporaryDirectory(prefix='asr_reference_', dir=paths.TEMP_DIR) as temp:
                frozen = Path(temp) / audio.name
                with LOCK:
                    shutil.copyfile(audio, frozen)
                ctx.check_cancelled()
                text = state.asr.transcribe(str(frozen), cancel_event=ctx.job.cancel_event)
                ctx.check_cancelled()
            status.publish('Reference transcription', msg('Reference transcript ready.'), 'success')
            return {'text': text}
        except AsrCancelled as exc:
            raise JobCancelled() from exc
        except Exception as exc:
            status.publish('Reference transcription', raw(str(exc)), 'error')
            raise
    return state.jobs.submit('asr_reference', 'Reference transcription', run, meta={'path': str(audio)})
