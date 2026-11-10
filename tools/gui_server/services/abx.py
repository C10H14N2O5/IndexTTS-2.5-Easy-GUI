"""Blind trials: never expose randomized side assignments before reveal."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Literal
from pydantic import Field, field_validator
from indextts_training import abx as native
from .. import paths
from ..errors import NotFound, ServiceError
from ..i18n import msg
from ..schemas import ApiModel, OpenModel
from ..state import current
from . import engine, library
from .projects import language_code
from .synthesis import GenerationParams, runtime_options, infer_kwargs
from .uploads import require_audio


class AbxCreate(ApiModel):
    name: str = Field("", max_length=160)
    adapter_a: str = ""
    adapter_b: str = ""
    voice: str
    language: str = "ZH"
    texts: str = Field(min_length=1, max_length=100_000)
    repetitions: int = Field(2, ge=1, le=5)
    settings: GenerationParams = Field(default_factory=GenerationParams)

    @field_validator("language")
    @classmethod
    def valid_language(cls, value):
        return language_code(value)


class AbxSummary(ApiModel):
    id: str
    name: str
    a: str
    b: str
    clips_done: int
    clips_total: int
    rated: int
    trials: int
    revealed: bool


class AbxTrial(ApiModel):
    index: int
    total: int
    text: str
    rep: int
    reference: str
    sample_1: str
    sample_2: str
    score: int | None


class AbxSession(AbxSummary):
    generation_complete: bool
    texts: list[str]
    repetitions: int
    language: str
    next_unrated: int | None
    trial: AbxTrial | None
    results: dict[str, Any] | None


def load(session_id: str) -> dict:
    try:
        return native.load_session(paths.ABX_DIR, session_id)
    except native.AbxError as exc:
        if exc.template == "Blind test not found: {0}":
            raise NotFound(exc.template, *exc.args_) from exc
        raise


def _summary(session: dict) -> dict:
    return dict(id=session["id"], name=session.get("name", ""), a=session["a"]["label"], b=session["b"]["label"],
                clips_done=len(session["clips"]) - len(native.pending_clips(session)), clips_total=len(session["clips"]),
                rated=native.rated_count(session), trials=native.trial_count(session), revealed=bool(session.get("revealed")))


def detail(session: dict, index: int = 0) -> dict:
    complete = native.generation_complete(session)
    return {**_summary(session), "generation_complete": complete, "texts": session["texts"],
            "repetitions": session["repetitions"], "language": session["language"],
            "next_unrated": native.next_unrated(session), "trial": native.trial_view(session, index) if complete else None,
            "results": native.summarize(session) if session.get("revealed") else None}


def sessions() -> list[dict]:
    return native.list_sessions(paths.ABX_DIR)


def _side(path: str) -> dict:
    if not path or path == paths.NONE_SENTINEL:
        return {"label": "Base model (no adapter)", "path": None}
    folder = Path(path).expanduser().resolve()
    if not (folder / "adapter_config.json").is_file():
        raise ServiceError("LoRA adapter is missing adapter_config.json: {0}", path)
    return {"label": f"{folder.parent.name} / {folder.name}", "path": str(folder)}


def create(request: AbxCreate) -> dict:
    voice = require_audio(request.voice)
    if voice.parent != paths.VOICES_DIR.resolve():
        raise ServiceError("Choose a saved reference voice.")
    params = request.settings.model_copy(deep=True)
    params.language = request.language
    frozen = {**params.model_dump(), **runtime_options(params)}
    session = native.create_session(paths.ABX_DIR, name=request.name, a=_side(request.adapter_a), b=_side(request.adapter_b),
                                    reference_path=voice, transcript=library.transcript(voice), language=request.language,
                                    texts=request.texts, repetitions=request.repetitions, settings=frozen)
    return detail(session)


def _synthesize(clip: dict, session: dict, output: Path):
    import numpy as np
    import torch
    # Old sessions retain their original decoding settings and labels.
    saved = {k: v for k, v in session["settings"].items() if k in GenerationParams.model_fields}
    params = GenerationParams.model_validate({**saved, "language": language_code(session["language"])})
    runtime = runtime_options(params)
    with engine.MODEL_LOCK:
        model = engine.load_model(**runtime, use_qwen_emo=False, adapter_path=session[clip["side"]]["path"] or "")
        seed = int(clip["seed"])
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = model.infer(spk_audio_prompt=str(native.reference_path(session)),
                             text=session["texts"][clip["text_index"]], output_path=str(output),
                             reference_text=session["reference"].get("transcript") or "",
                             emo_audio_prompt=None, emo_alpha=0.65, emo_vector=None, use_emo_text=False,
                             emo_text=None, use_random=False, interval_silence=params.interval_silence,
                             **infer_kwargs(params))
    if not output.is_file() and result and Path(str(result)).is_file():
        shutil.move(str(result), str(output))


def generate(session_id: str):
    session = load(session_id)
    if session.get("revealed") and not native.generation_complete(session):
        raise ServiceError("Results are revealed; ratings can no longer be changed.")
    state = current()

    def run(ctx):
        for progress in native.generate_clips(session, _synthesize, lambda: ctx.cancel_requested):
            ctx.check_cancelled()
            ctx.progress(progress["done"] / max(1, progress["total"]), "Blind test clip {0}/{1}",
                         min(progress["done"] + 1, progress["total"]), progress["total"])
        ctx.check_cancelled()
        return {"session_id": session_id, "clips_done": len(session["clips"]), "clips_total": len(session["clips"])}

    return state.jobs.submit("abx_generate", msg("Generating blind test clips."), run, gpu=True,
                             meta={"session_id": session_id})


def rate(session_id: str, index: int, score: int | None) -> dict:
    # One lock covers load/edit/save, avoiding lost updates from two rating tabs.
    with native._LOCK:
        session = load(session_id)
        native.rate(session, index, score)
        return detail(session, index)


def reveal(session_id: str, index: int = 0) -> dict:
    with native._LOCK:
        session = load(session_id)
        native.reveal(session)
        return detail(session, index)
