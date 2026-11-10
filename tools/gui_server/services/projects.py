"""Project metadata and training defaults; preserves the 1.x project.json format."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from indextts_training import dataset
from .. import paths
from ..errors import Busy, Conflict, NotFound, ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel, OpenModel
from ..state import current
from .catalog import LANGUAGES, LANGUAGE_LABELS


class TrainingConfig(ApiModel):
    adapter_name: str = ""
    vram_preset: Literal["24 GB", "32 GB+"] = "24 GB"
    mode: Literal["Steps", "Epochs"] = "Steps"
    training_steps: int = Field(1000, ge=1, le=10000)
    training_epochs: int = Field(20, ge=1, le=500)
    save_steps: int = Field(100, ge=1, le=2000)
    save_epochs: int = Field(2, ge=1, le=100)
    lora_rank: Literal[4, 8, 16, 32, 64, 128, 256] = 16
    lora_alpha: int = Field(32, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0, le=0.2)
    learning_rate: float = Field(0.00004, ge=0.000001, le=0.001)
    micro_batch: int = Field(1, ge=1, le=64)
    grad_accum: int = Field(8, ge=1, le=64)
    grad_clip: float = Field(0.5, gt=0, le=100)
    seed: int = Field(1234, ge=0, le=2**32-1)
    resume_checkpoint: str = ""
    eval_enabled: bool = True
    eval_text: str = "This is an IndexTTS 2.5 training preview."
    eval_reference: str = ""
    eval_reference_transcript: str = ""
    eval_language: str = "English"
    eval_whisper_model: str = "sensevoice-small-q8"
    eval_whisper_language: str = "Auto-detect"
    semantic_loss_weight: float = 0.0
    reference_text_conditioning: bool = False
    phonetic_ctc_enabled: bool = False
    phonetic_ctc_weight: float = 0.0
    legacy_phonetic_ctc_enabled: bool = False
    differentiable_asr_enabled: bool = False
    differentiable_asr_weight: float = 0.0
    differentiable_asr_model: str = "large-v3"
    asr_cfm_steps: int = 4
    asr_cfm_temperature: float = 1.0

    @field_validator("eval_language")
    @classmethod
    def evaluation_language(cls, value):
        code = language_code(value)
        return LANGUAGE_LABELS[code]  # 1.x stores the label, not the language code.


def language_code(value: str) -> str:
    code = LANGUAGES.get(value, value.upper())
    if code not in LANGUAGE_LABELS:
        raise ValueError("Invalid language.")
    return code


TRAINING_DEFAULTS = TrainingConfig().model_dump()
_DISABLED_AUXILIARY = {key: TRAINING_DEFAULTS[key] for key in (
    "semantic_loss_weight", "reference_text_conditioning", "phonetic_ctc_enabled", "phonetic_ctc_weight",
    "legacy_phonetic_ctc_enabled", "differentiable_asr_enabled", "differentiable_asr_weight")}


class ManifestStatus(OpenModel):
    stale: bool
    managed: bool
    manifest: str
    status: str
    message: str
    stats: dict[str, Any] = Field(default_factory=dict)


class ValidationConfig(OpenModel):
    source_folder: str = ""
    manifest: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = False
    interval_steps: int | None = None


class ProjectInfo(OpenModel):
    project_name: str
    source_folder: str = ""
    manifest: str = ""
    language: str = "EN"
    stats: dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig
    validation: ValidationConfig
    validation_suggested_folder: str
    dataset_status: ManifestStatus
    validation_status: ManifestStatus
    modified: float


class CheckpointInfo(ApiModel):
    name: str
    path: str
    adapter: str
    kind: Literal["final", "checkpoint"]
    step: int | None = None
    modified: float
    evaluation: str | None = None
    resumable: bool = False


def normalized_name(value: str, *, adapter: bool = False) -> str:
    safe = paths.safe_name(value)
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if not safe or len(safe) > 120 or safe.upper() in reserved:
        raise ServiceError("Enter an Adapter Name." if adapter else "Enter a New Project Name.")
    return safe


def project_file(name: str) -> Path:
    return dataset._project_file(name, paths.TRAINING_DIR)


def load_raw(name: str) -> dict:
    file = project_file(name)
    if not file.is_file():
        raise NotFound("Project not found: {0}", name)
    data = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ServiceError("Could not read project: {0}", name)
    return data


def save_raw(name: str, data: dict) -> None:
    dataset._atomic_write(project_file(name), json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def ensure_mutable() -> None:
    if current().training.snapshot().get("running"):
        raise Busy("Stop training before editing the project.")


def training_values(meta: dict) -> dict:
    stored = meta.get("training") or {}
    values = {**TRAINING_DEFAULTS, **{k: v for k, v in stored.items() if k in TRAINING_DEFAULTS}, **_DISABLED_AUXILIARY}
    values["adapter_name"] = values.get("adapter_name") or meta.get("project_name", "")
    values["eval_language"] = LANGUAGE_LABELS.get(values["eval_language"], values["eval_language"])
    values["resume_checkpoint"] = "" if values["resume_checkpoint"] == paths.NONE_SENTINEL else values["resume_checkpoint"]
    return values


def get(name: str) -> dict:
    meta = load_raw(name)
    meta["project_name"] = name
    meta["training"] = training_values(meta)
    meta["validation"] = dataset.validation_settings(meta)
    meta["validation_suggested_folder"] = str((paths.VALIDATION_DIR / name).resolve())
    meta["dataset_status"] = dataset.dataset_manifest_status(name, paths.TRAINING_DIR)
    meta["validation_status"] = dataset.dataset_manifest_status(name, paths.TRAINING_DIR, role="validation")
    for key in ("dataset_status", "validation_status"):
        meta[key]["message"] = raw(meta[key]["message"])
    meta["modified"] = project_file(name).stat().st_mtime
    return meta


def list_projects() -> list[dict]:
    result = []
    for file in sorted(paths.TRAINING_PROJECTS_DIR.glob("*/project.json")):
        if file.is_file() and not file.parent.is_symlink():
            try:
                result.append(get(file.parent.name))
            except (OSError, ValueError, ServiceError):
                continue
    return result


def create(name: str, language: str = "ZH", source_folder: str = "") -> dict:
    safe = normalized_name(name)
    with dataset.DATASET_LOCK:
        if project_file(safe).exists():
            raise Conflict("Project already exists: {0}", safe)
        folder = _source_folder(source_folder)
        save_raw(safe, {"project_name": safe, "source_folder": folder, "manifest": "",
                        "language": language_code(language), "stats": {}})
    current().status.publish("Dataset project", msg("Project created: {0}", safe), "success")
    return get(safe)


def _source_folder(value: str) -> str:
    if not value.strip():
        return ""
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise ServiceError("Source Audio Folder does not exist.")
    return str(folder)


def update_dataset(name: str, patch: dict) -> dict:
    with dataset.DATASET_LOCK:
        ensure_mutable()
        meta = load_raw(name)
        if "source_folder" in patch:
            meta["source_folder"] = _source_folder(patch["source_folder"])
        if "language" in patch:
            meta["language"] = language_code(patch["language"])
        if "manifest" in patch:
            manifest = str(patch["manifest"] or "")
            if manifest and not Path(manifest).is_file():
                raise ServiceError("File not found: {0}", manifest)
            meta["manifest"] = manifest
        save_raw(name, meta)
        if patch.get("prepare"):
            dataset.refresh_project_manifest(name, paths.TRAINING_DIR)
    return get(name)


def update_validation(name: str, patch: dict) -> dict:
    with dataset.DATASET_LOCK:
        ensure_mutable()
        load_raw(name)
        values = {k: v for k, v in patch.items() if k in {"enabled", "interval_steps", "source_folder"}}
        if "source_folder" in values:
            requested = str(values["source_folder"] or "").strip()
            suggested = paths.VALIDATION_DIR / name
            if requested and Path(requested).expanduser().resolve() == suggested.resolve():
                if suggested.resolve().parent != paths.VALIDATION_DIR.resolve():
                    raise ServiceError("Invalid project path.")
                suggested.mkdir(parents=True, exist_ok=True)
            values["source_folder"] = _source_folder(requested)
        dataset.configure_validation(name, root=paths.TRAINING_DIR, **values)
        if patch.get("prepare"):
            dataset.refresh_project_manifest(name, paths.TRAINING_DIR, role="validation")
    return get(name)


def update_training(name: str, patch: dict) -> dict:
    with dataset.DATASET_LOCK:
        ensure_mutable()
        meta = load_raw(name)
        saved = meta.get("training") or {}
        values = TrainingConfig.model_validate({**training_values(meta), **patch}).model_dump()
        values.update(_DISABLED_AUXILIARY)
        values["adapter_name"] = normalized_name(values["adapter_name"], adapter=True)
        reference = values["eval_reference"]
        if reference:
            source = Path(reference).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in paths.AUDIO_SUFFIXES:
                raise ServiceError("Select an Eval Reference Audio file first.")
            target = project_file(name).parent / ("eval_reference" + source.suffix.lower())
            if source != target.resolve():
                shutil.copy2(source, target)
            values["eval_reference"] = str(target.resolve())
        meta["training"] = {**saved, **values}
        save_raw(name, meta)
    current().status.publish("LoRA project", msg("Training project saved: {0}", name), "success")
    return get(name)


def clone(name: str, new_name: str | None = None) -> dict:
    with dataset.DATASET_LOCK:
        ensure_mutable()
        load_raw(name)
        safe = normalized_name(new_name) if new_name is not None else None
        if safe and (project_file(safe).parent.exists() or (paths.TRAINING_DATASETS_DIR / safe).exists()):
            raise Conflict("Project already exists: {0}", safe)
        meta = dataset.clone_project(name, paths.TRAINING_DIR, new_name=safe)
    return get(meta["project_name"])


def delete(name: str, confirm_name: str) -> dict:
    if confirm_name != name:
        raise ServiceError("Type the project name to confirm deletion.")
    with dataset.DATASET_LOCK:
        ensure_mutable()
        load_raw(name)
        targets = [(project_file(name).parent, paths.TRAINING_PROJECTS_DIR),
                   (paths.TRAINING_DATASETS_DIR / name, paths.TRAINING_DATASETS_DIR)]
        for target, parent in targets:
            if target.is_symlink() or target.resolve().parent != parent.resolve():
                raise ServiceError("Invalid project path.")
        for target, _ in targets:
            if target.is_dir():
                shutil.rmtree(target)
    current().status.publish("LoRA project", msg("Project deleted: {0}.", name), "success",
                             msg("Existing training outputs and adapters were kept."))
    return {"ok": True, "message": msg("Project deleted: {0}. Existing training outputs were kept.", name)}


def autotune(name: str, vram_preset: str) -> dict:
    meta = load_raw(name)
    minutes = float(meta.get("stats", {}).get("total_seconds", 0)) / 60
    rank = 16 if minutes < 20 else 32
    steps = max(500, min(4000, int(900 + minutes * 45)))
    changes = dict(vram_preset=vram_preset, mode="Steps", training_steps=steps, training_epochs=20,
                   save_steps=max(100, steps // 10), save_epochs=2, lora_rank=rank, lora_alpha=rank * 2,
                   learning_rate=0.0002 if minutes < 30 else 0.0001,
                   grad_accum=8 if vram_preset.startswith("24") else 4)
    return {"training": {**training_values(meta), **changes}, "changed": list(changes), "minutes": minutes}


def output_dir(name: str, adapter: str | None = None) -> Path:
    meta = load_raw(name)
    value = adapter or training_values(meta)["adapter_name"] or name
    if paths.safe_name(value) != value or not value:
        raise ServiceError("Enter an Adapter Name.")
    folder = paths.TRAINING_OUTPUTS_DIR / value
    if folder.resolve().parent != paths.TRAINING_OUTPUTS_DIR.resolve():
        raise ServiceError("Invalid project path.")
    return folder


def checkpoints(name: str, adapter: str | None = None) -> list[dict]:
    root = output_dir(name, adapter)
    result = []
    for folder in sorted(root.glob("*")):
        if not folder.is_dir() or not (folder / "adapter_config.json").is_file():
            continue
        final = folder.name == "final_adapter"
        if not final and not folder.name.startswith("checkpoint-"):
            continue
        suffix = folder.name.rsplit("-", 1)[-1]
        step = int(suffix) if suffix.isdigit() else None
        audio = root / "evaluation" / f"eval-step-{step:06d}.wav" if step is not None else None
        result.append(dict(name=folder.name, path=str(folder.resolve()), adapter=root.name,
                           kind="final" if final else "checkpoint", step=step, modified=folder.stat().st_mtime,
                           evaluation=str(audio.resolve()) if audio and audio.is_file() else None,
                           resumable=(folder / "trainer_state.pt").is_file()))
    return result
