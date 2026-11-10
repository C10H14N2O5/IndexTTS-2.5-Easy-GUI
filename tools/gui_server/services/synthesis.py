"""Single-reference synthesis, preserving 1.x IndexTTS parameter semantics."""
from __future__ import annotations

import shutil
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from indextts.utils.text_chunking import CHUNK_CHOICES, split_long_text

from .. import paths, status
from ..errors import ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel
from ..state import current
from . import catalog, engine, history, library, settings
from .uploads import require_audio, audio_duration


class GenerationParams(ApiModel):
    adapter_path: str = ''
    language: str = 'ZH'
    duration_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    interval_silence: int = Field(default=200, ge=0, le=1000)
    text_normalization: bool = True
    max_text_tokens: int = Field(default=120, ge=20, le=300)
    do_sample: bool = True
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=30, ge=0, le=200)
    temperature: float = Field(default=0.8, ge=0.1, le=2)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=0.1, le=30)
    length_penalty: float = Field(default=0.0, ge=-3, le=3)
    max_mel_tokens: int = Field(default=1500, ge=50, le=3000)
    use_bf16: bool | None = None
    use_gpt_latent: bool | None = None
    use_torch_compile: bool | None = None
    use_flash_accel: bool | None = None
    save_outputs: bool = True

    @field_validator('language')
    @classmethod
    def validate_language(cls, value):
        code = catalog.LANGUAGES.get(value, value.upper())
        if code not in catalog.LANGUAGE_LABELS:
            raise ValueError('Unknown synthesis language.')
        return code


class SynthesisRequest(GenerationParams):
    voice_path: str = Field(default='', validation_alias=AliasChoices('voice_path', 'prompt_audio'))
    quick_sample: str | None = None
    text: str = Field(min_length=1, max_length=100_000)
    reference_transcript: str = ''
    reference_auto_transcribe: bool = False
    emotion_mode: Literal['Same as voice reference', 'Emotion reference audio', 'Emotion vector', 'Emotion description'] = 'Same as voice reference'
    emotion_audio: str | None = None
    emotion_alpha: float = Field(default=0.65, ge=0, le=1)
    emotion_vector: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0.55], min_length=8, max_length=8)
    emotion_text: str = ''
    emotion_description: str | None = None
    random_emotion: bool = False
    chunk_mode: str = 'None'
    chunk_silence: float = Field(default=0.5, ge=0, le=5)

    @field_validator('emotion_vector')
    @classmethod
    def validate_vector(cls, value):
        if any(not 0 <= n <= 1 for n in value):
            raise ValueError('Emotion vector values must be between 0 and 1.')
        return value

    @field_validator('chunk_mode')
    @classmethod
    def validate_chunk_mode(cls, value):
        if value not in CHUNK_CHOICES:
            raise ValueError('Unknown text chunk mode.')
        return value


def runtime_options(params: GenerationParams) -> dict:
    stored = settings.runtime()
    for key in settings.RUNTIME_KEYS:
        explicit = getattr(params, key)
        if explicit is not None:
            stored[key] = explicit
    return stored


def infer_kwargs(params: GenerationParams) -> dict:
    result = {key: getattr(params, key) for key in (
        'duration_factor', 'text_normalization', 'do_sample', 'top_p', 'top_k',
        'temperature', 'num_beams', 'repetition_penalty', 'length_penalty', 'max_mel_tokens')}
    return {**result, 'lang': params.language, 'max_text_tokens_per_segment': params.max_text_tokens,
            'verbose': True, 'stream_return': False}


def concat_wavs(inputs: list[Path], output: Path, silence_seconds: float) -> None:
    if not inputs:
        raise ServiceError('No dialogue segments were generated.')
    with wave.open(str(inputs[0]), 'rb') as first:
        params = first.getparams()
    silence = b'\x00' * max(0, int(silence_seconds * params.framerate)) * params.nchannels * params.sampwidth
    with wave.open(str(output), 'wb') as out:
        out.setparams(params)
        for index, source in enumerate(inputs):
            with wave.open(str(source), 'rb') as inp:
                cur = inp.getparams()
                if (cur.nchannels, cur.sampwidth, cur.framerate, cur.comptype) != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
                    raise ServiceError('Dialogue segment {0} has incompatible WAV parameters.', index + 1)
                # Bound memory while joining long outputs.
                while block := inp.readframes(65536):
                    out.writeframes(block)
            if index + 1 < len(inputs):
                out.writeframes(silence)


def output_path(kind: str, saved: bool) -> Path:
    if not saved:
        path = paths.TEMP_DIR / ('preview.wav' if kind == 'synthesis' else 'dialogue_preview.wav')
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        prefix = 'indextts_2_5_' if kind == 'synthesis' else 'indextts_2_5_dialogue_'
        path = paths.OUTPUTS_DIR / f'{prefix}{stamp}.wav'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def finish_audio(staged: Path, params: GenerationParams, kind: str, started: float) -> dict:
    path = output_path(kind, params.save_outputs)
    # Write metadata before publishing the WAV. A sidecar failure must not leave
    # an output in history while reporting the synthesis as failed.
    request = params.model_dump()
    if params.save_outputs:
        history.remember(path, kind, request)
    try:
        staged.replace(path)
    except BaseException:
        if params.save_outputs:
            Path(str(path) + '.json').unlink(missing_ok=True)
        raise
    return {'path': str(path), 'name': path.name, 'duration': audio_duration(path),
            'seconds': time.perf_counter() - started, 'saved': params.save_outputs, 'params': request}


def validate(params: SynthesisRequest) -> tuple[Path, list[str], str | None]:
    if not params.text.strip():
        raise ServiceError('Enter target text to synthesize.')
    if params.quick_sample and params.quick_sample != paths.NONE_SENTINEL:
        voice = require_audio(params.quick_sample)
        if voice.parent != paths.VOICES_DIR.resolve():
            raise ServiceError('The selected Quick Sample no longer exists. Choose another saved sample or select None for a custom upload.')
    else:
        voice = require_audio(params.voice_path)
    chunks = split_long_text(params.text.strip(), params.chunk_mode)
    if not chunks:
        raise ServiceError('The target text did not produce any usable chunks.')
    emotion_audio = None
    if params.emotion_mode == 'Emotion reference audio':
        if not params.emotion_audio:
            raise ServiceError('Emotion reference mode requires a separate emotion audio file.')
        emotion_audio = str(require_audio(params.emotion_audio))
    if params.emotion_description:
        library.description_text(params.emotion_description)
    if params.adapter_path and params.adapter_path != paths.NONE_SENTINEL:
        if not (Path(params.adapter_path) / 'adapter_config.json').is_file():
            raise ServiceError('LoRA adapter is missing adapter_config.json: {0}', params.adapter_path)
    return voice, chunks, emotion_audio


def submit(params: SynthesisRequest):
    voice, chunks, emotion_audio = validate(params)
    state = current()
    runtime = runtime_options(params)
    emotion_text = (library.description_text(params.emotion_description) if params.emotion_description else params.emotion_text).strip()
    params = params.model_copy(deep=True)
    params.text = params.text.strip()
    params.voice_path = str(voice)
    for key, value in runtime.items():
        setattr(params, key, value)

    def run(ctx):
        started = time.perf_counter()
        ctx.progress(0.02, 'Validating inference inputs')
        ctx.check_cancelled()
        reference = params.reference_transcript.strip()
        if params.reference_auto_transcribe and not reference:
            ctx.progress(0.08, 'Recognizing audio with SenseVoiceSmall Q8 (CPU).')
            try:
                reference = state.asr.transcribe(str(voice), cancel_event=ctx.job.cancel_event)
            except Exception:
                ctx.check_cancelled()
                raise
            params.reference_transcript = reference
        ctx.check_cancelled()
        ctx.progress(0.12, 'Loading IndexTTS 2.5 runtime')
        status.publish('Single inference', msg('Loading IndexTTS 2.5 runtime'), 'working')
        tts = engine.load_model(**runtime, use_qwen_emo=params.emotion_mode == 'Emotion description', adapter_path=params.adapter_path)
        ctx.check_cancelled()
        kwargs = {**infer_kwargs(params), 'spk_audio_prompt': str(voice), 'reference_text': reference,
                  'emo_audio_prompt': emotion_audio, 'emo_alpha': params.emotion_alpha,
                  'emo_vector': params.emotion_vector if params.emotion_mode == 'Emotion vector' else None,
                  'use_emo_text': params.emotion_mode == 'Emotion description',
                  'emo_text': (emotion_text or params.text) if params.emotion_mode == 'Emotion description' else None,
                  'use_random': params.random_emotion}
        paths.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix='inference_', dir=paths.TEMP_DIR) as temp:
                temp = Path(temp)
                output = temp / 'complete.wav'
                explicit = params.chunk_mode != 'None' and len(chunks) > 1
                if explicit:
                    generated = []
                    for index, text in enumerate(chunks, 1):
                        ctx.check_cancelled()
                        ctx.progress(0.34 + 0.50 * (index - 1) / len(chunks), 'Synthesizing chunk {0}/{1}', index, len(chunks))
                        target = temp / f'chunk_{index:04d}.wav'
                        actual = tts.infer(text=text, output_path=str(target), interval_silence=0, **kwargs)
                        target = target if target.is_file() else Path(str(actual))
                        if not target.is_file():
                            raise ServiceError('IndexTTS did not create audio for chunk {0}.', index)
                        generated.append(target)
                    ctx.check_cancelled()
                    ctx.progress(0.86, 'Joining generated chunks')
                    concat_wavs(generated, output, params.chunk_silence)
                else:
                    ctx.progress(0.40, 'Synthesizing audio')
                    actual = tts.infer(text=params.text, output_path=str(output), interval_silence=params.interval_silence, **kwargs)
                    if not output.is_file() and actual and Path(str(actual)).is_file():
                        shutil.copyfile(str(actual), output)
                ctx.check_cancelled()
                if not output.is_file():
                    raise ServiceError('IndexTTS returned without creating an output WAV.')
                ctx.progress(0.92, 'Saving output WAV')
                result = finish_audio(output, params, 'synthesis', started)
            status.publish('Single inference', msg('Generation complete.'), 'success')
            return result
        except Exception as exc:
            from ..jobs import JobCancelled
            if not isinstance(exc, JobCancelled):
                status.publish('Single inference', msg('Generation failed: {0}', type(exc).__name__), 'error', raw(str(exc)))
            raise
    return state.jobs.submit('synthesis', 'Single inference', run, gpu=True,
                             meta={'chunks': len(chunks), 'language': params.language})
