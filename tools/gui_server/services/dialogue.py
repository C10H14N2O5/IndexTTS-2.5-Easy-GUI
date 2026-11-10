"""Up to twelve library-backed dialogue turns with cooperative cancellation."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field

from .. import paths, status
from ..errors import ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel
from ..state import current
from . import catalog, engine, library
from .synthesis import GenerationParams, concat_wavs, finish_audio, infer_kwargs, runtime_options
from .uploads import require_audio


class DialogueTurn(ApiModel):
    voice_path: str = Field(default='', validation_alias=AliasChoices('voice_path', 'speaker_path'))
    text: str = Field(default='', max_length=100_000)
    emotion_source: Literal['Emotion Vector Preset', 'Emotion Description'] = Field(
        default='Emotion Vector Preset', validation_alias=AliasChoices('emotion_source', 'emotion_mode'))
    preset: str = Field(default='Neutral / balanced', validation_alias=AliasChoices('preset', 'vector_preset'))
    description_path: str | None = None


class DialogueRequest(GenerationParams):
    turns: list[DialogueTurn] = Field(min_length=1, max_length=12)
    dialogue_silence: float = Field(default=0.5, ge=0, le=5)


def submit(params: DialogueRequest):
    active = []
    for index, turn in enumerate(params.turns, 1):
        if not turn.text.strip():
            continue
        try:
            voice = require_audio(turn.voice_path)
        except ServiceError as exc:
            raise ServiceError('Dialogue turn {0} requires a saved voice-library speaker.', index) from exc
        if voice.parent != paths.VOICES_DIR.resolve():
            raise ServiceError('Dialogue turn {0} requires a saved voice-library speaker.', index)
        text = library.description_text(turn.description_path) if turn.emotion_source == 'Emotion Description' else ''
        if turn.emotion_source == 'Emotion Description' and not text:
            raise ServiceError('Dialogue turn {0}: choose a saved Emotion Description.', index)
        if turn.emotion_source == 'Emotion Vector Preset' and turn.preset not in catalog.EMOTION_PRESETS:
            raise ServiceError('Unknown emotion preset: {0}', turn.preset)
        active.append((turn, voice, text))
    if not active:
        raise ServiceError('Dialogue has no turns with text.')
    if params.adapter_path and params.adapter_path != paths.NONE_SENTINEL and not (Path(params.adapter_path) / 'adapter_config.json').is_file():
        raise ServiceError('LoRA adapter is missing adapter_config.json: {0}', params.adapter_path)
    runtime = runtime_options(params)
    params = params.model_copy(deep=True)
    for key, value in runtime.items():
        setattr(params, key, value)

    def run(ctx):
        started = time.perf_counter()
        ctx.progress(0.02, 'Validating dialogue')
        ctx.check_cancelled()
        ctx.progress(0.10, 'Loading IndexTTS 2.5 runtime')
        needs_qwen = any(turn.emotion_source == 'Emotion Description' for turn, _, _ in active)
        tts = engine.load_model(**runtime, use_qwen_emo=needs_qwen, adapter_path=params.adapter_path)
        status.publish('Dialogue inference', msg('Generating {0} dialogue turn(s).', len(active)), 'working')
        try:
            with tempfile.TemporaryDirectory(prefix='dialogue_', dir=paths.TEMP_DIR) as directory:
                directory = Path(directory)
                segments = []
                for index, (turn, voice, description) in enumerate(active, 1):
                    ctx.check_cancelled()
                    ctx.progress(0.16 + .72 * (index - 1) / len(active), 'Dialogue turn {0}/{1}', index, len(active))
                    output = directory / f'segment_{index:02d}.wav'
                    kwargs = dict(emo_alpha=0.65, use_random=False, emo_vector=None, use_emo_text=False, emo_text=None)
                    if turn.emotion_source == 'Emotion Description':
                        kwargs.update(use_emo_text=True, emo_text=description)
                    else:
                        kwargs['emo_vector'] = list(catalog.EMOTION_PRESETS[turn.preset])
                    tts.infer(spk_audio_prompt=str(voice), text=turn.text.strip(), output_path=str(output),
                              interval_silence=params.interval_silence, **kwargs, **infer_kwargs(params))
                    if not output.is_file():
                        raise ServiceError('IndexTTS did not create audio for chunk {0}.', index)
                    segments.append(output)
                    ctx.progress(0.16 + .72 * index / len(active), 'Completed turn {0}/{1}', index, len(active))
                ctx.check_cancelled()
                ctx.progress(.91, 'Assembling dialogue WAV')
                output = directory / 'complete.wav'
                concat_wavs(segments, output, params.dialogue_silence)
                ctx.check_cancelled()
                result = finish_audio(output, params, 'dialogue', started)
                result['turns'] = len(active)
            status.publish('Dialogue inference', msg('Dialogue generation complete.'), 'success')
            return result
        except Exception as exc:
            from ..jobs import JobCancelled
            if not isinstance(exc, JobCancelled):
                status.publish('Dialogue inference', msg('Dialogue generation failed: {0}', type(exc).__name__), 'error', raw(str(exc)))
            raise
    return current().jobs.submit('dialogue', 'Dialogue inference', run, gpu=True, meta={'turns': len(active)})
