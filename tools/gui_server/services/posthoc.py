"""Validation loss for saved checkpoints, computed after training (a GPU job).

Results live next to the adapter in ``validation_posthoc.json`` and are keyed
by checkpoint name and validation-manifest fingerprint. A result is *current*
when it was computed on the project's current validation manifest and the
adapter weights file has not changed since; otherwise it is kept but reported
as *outdated*.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from indextts_training import dataset, dataset_versions, validation_results
from indextts_training.schedule import read_adapter_state
from indextts_training.validation_results import BASELINE, RESULTS_FILE  # noqa: F401 - re-exported
from .. import paths
from ..errors import Busy, NotFound, ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel
from ..state import current
from . import engine, projects

TENSORBOARD_RUN = "posthoc"
SERIES_LABEL = "validation_posthoc/loss"
BASELINE_SERIES_LABEL = "validation_baseline/loss"


class ValidationRunRequest(ApiModel):
    adapter: str | None = None
    checkpoints: list[str] = Field(default_factory=list, max_length=1000)
    include_baseline: bool = False
    recompute: bool = False


class VersionRef(ApiModel):
    """A frozen dataset version at one revision."""
    id: str
    name: str
    revision: int


class PosthocItem(ApiModel):
    name: str
    kind: Literal["baseline", "final", "best", "checkpoint"]
    step: int | None = None
    loss: float
    validation_fingerprint: str
    samples: int
    computed_at: float
    status: Literal["current", "outdated"]
    same_as: str | None = None
    validation_version: VersionRef | None = None


class ValidationSetInfo(ApiModel):
    ready: bool
    message: str
    fingerprint: str | None = None
    samples: int = 0
    manifest: str = ""
    version: VersionRef | None = None


class DatasetRecord(ApiModel):
    """The dataset recorded in an adapter's training_config.json."""
    version_id: str | None = None
    name: str = ""
    revision: int | None = None
    manifest_sha256: str | None = None
    samples: int | None = None


class TrainedWith(ApiModel):
    training: DatasetRecord | None = None
    validation: DatasetRecord | None = None


class PosthocResults(ApiModel):
    project: str
    adapter: str
    validation: ValidationSetInfo
    training_fingerprint: str | None = None
    trained_with: TrainedWith | None = None
    items: list[PosthocItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- files

def _weights_file(folder: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _signature(folder: Path) -> list | None:
    weights = _weights_file(folder)
    if weights is None:
        return None
    stat = weights.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _sha256(folder: Path) -> str | None:
    weights = _weights_file(folder)
    if weights is None:
        return None
    with weights.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


_load = validation_results.load
_upsert = validation_results.upsert


def _training_config(root: Path) -> dict:
    try:
        saved = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def _training_fingerprint(root: Path) -> str | None:
    validation = _training_config(root).get("validation")
    value = validation.get("fingerprint") if isinstance(validation, dict) else None
    return value if isinstance(value, str) else None


def _trained_with(root: Path) -> dict | None:
    """Datasets recorded by the adapter's latest training run (older runs record none)."""
    saved = _training_config(root)
    if not saved:
        return None
    training = saved.get("dataset") if isinstance(saved.get("dataset"), dict) else None
    validation = saved.get("validation") if isinstance(saved.get("validation"), dict) else None

    def record(section: dict | None, fingerprint_key: str) -> dict | None:
        if section is None:
            return None
        samples = section.get("samples")
        return dict(version_id=section.get("version_id") or None, name=str(section.get("version_name") or ""),
                    revision=section.get("revision") if isinstance(section.get("revision"), int) else None,
                    manifest_sha256=section.get(fingerprint_key) if isinstance(section.get(fingerprint_key), str) else None,
                    samples=samples if isinstance(samples, int) else None)

    return dict(training=record(training, "manifest_sha256"), validation=record(validation, "fingerprint"))


def _current(name: str) -> tuple[str | None, dict | None]:
    """Fingerprint and version of the selected validation set (no folder or audio scan)."""
    return dataset_versions.active_fingerprint(name, "validation", paths.TRAINING_DIR)


def _current_fingerprint(name: str) -> str | None:
    return _current(name)[0]


def _status(entry: dict, root: Path, fingerprint: str | None, version: dict | None = None) -> str:
    if not fingerprint or entry.get("validation_fingerprint") != fingerprint:
        return "outdated"
    recorded = entry.get("validation_version")
    if version and isinstance(recorded, dict) and recorded.get("id") == version["id"] \
            and recorded.get("revision") != version["revision"]:
        return "outdated"  # A repaired version can keep its manifest bytes but not its audio.
    if entry["name"] == BASELINE:
        return "current"
    folder = root / entry["name"]
    return "current" if entry.get("weights") and _signature(folder) == entry["weights"] else "outdated"


def _version_ref(value) -> dict | None:
    if isinstance(value, dict) and isinstance(value.get("id"), str) and isinstance(value.get("revision"), int):
        return dict(id=value["id"], name=str(value.get("name") or value["id"]), revision=value["revision"])
    return None


def _items(root: Path, fingerprint: str | None, version: dict | None = None) -> list[dict]:
    result = []
    for entry in _load(root):
        kind = "baseline" if entry["name"] == BASELINE else projects.adapter_kind(entry["name"]) or "checkpoint"
        result.append(dict(name=entry["name"], kind=kind, step=entry.get("step"), loss=float(entry["loss"]),
                           validation_fingerprint=entry.get("validation_fingerprint", ""),
                           samples=int(entry.get("samples", 0)), computed_at=float(entry.get("computed_at", 0)),
                           status=_status(entry, root, fingerprint, version), same_as=entry.get("same_as"),
                           validation_version=_version_ref(entry.get("validation_version"))))
    return result


def _points(items: list[dict], *, baseline: bool = True) -> list[tuple[int, float]]:
    """Current results by step; a checkpoint wins over a final/best adapter at the same step.

    The baseline is step 0 when ``baseline`` is true (the TensorBoard run keeps it there).
    """
    by_step: dict[int, tuple[int, float]] = {}
    for item in items:
        if item["status"] != "current" or item["step"] is None or (not baseline and item["kind"] == "baseline"):
            continue
        rank = 0 if item["kind"] in ("final", "best") else 1
        step = int(item["step"])
        if step not in by_step or rank >= by_step[step][0]:
            by_step[step] = (rank, item["loss"])
    return sorted((step, value) for step, (_, value) in by_step.items())


def series(name: str, root: Path) -> list[dict]:
    """Chart series for :mod:`metrics`.

    Checkpoint results form ``validation_posthoc/loss``; a current base-model
    result is a separate one-point ``validation_baseline/loss`` series that the
    GUI draws as a horizontal reference line.
    """
    items = _items(root, *_current(name))
    result = []
    points = _points(items, baseline=False)
    if points:
        result.append(dict(label=SERIES_LABEL, points=points, count=len(points)))
    base = next((item for item in items if item["kind"] == "baseline" and item["status"] == "current"), None)
    if base is not None:
        result.append(dict(label=BASELINE_SERIES_LABEL, points=[(0, base["loss"])], count=1))
    return result


# ---------------------------------------------------------------------- validation

def _training_rows(name: str, meta: dict) -> list[dict] | None:
    """Rows of the selected training dataset, for the overlap check."""
    if meta.get("dataset_version"):
        try:
            return dataset_versions.usable_rows(name, "training", meta["dataset_version"], paths.TRAINING_DIR)["rows"]
        except dataset_versions.VersionError:
            return None  # A damaged training version blocks training, not validation.
    if meta.get("manifest") and Path(meta["manifest"]).is_file():
        try:
            return dataset._read_manifest_data(meta["manifest"], "training")[1]
        except ValueError as exc:
            raise ServiceError(raw(str(exc))) from exc
    return None


def _validation_snapshot(name: str, *, check_overlap: bool) -> dict:
    meta = projects.load_raw(name)
    settings = dataset.validation_settings(meta)
    if settings.get("version"):
        try:
            frozen = dataset_versions.usable_rows(name, "validation", settings["version"], paths.TRAINING_DIR)
        except dataset_versions.VersionError as exc:
            raise ServiceError(msg(exc.template, *exc.args_)) from exc
        rows = frozen["rows"]
        if check_overlap:
            training_rows = _training_rows(name, meta)
            if training_rows is not None:
                try:
                    dataset._check_validation_overlap(training_rows, rows)
                except ValueError as exc:
                    raise ServiceError(raw(str(exc))) from exc
        return dict(rows=rows, fingerprint=hashlib.sha256(frozen["raw"]).hexdigest(), manifest=frozen["manifest"],
                    samples=len(rows), version=frozen["ref"])
    if not settings["source_folder"]:
        raise ServiceError("Select and save a validation folder first.")
    status = dataset.dataset_manifest_status(name, paths.TRAINING_DIR, role="validation")
    if status["stale"]:
        raise ServiceError(raw(status["message"]))
    try:
        data, rows = dataset._read_manifest_data(status["manifest"], "validation")
    except ValueError as exc:
        raise ServiceError(raw(str(exc))) from exc
    fingerprint = hashlib.sha256(data).hexdigest()
    if fingerprint != settings.get("manifest_sha256"):
        raise ServiceError("Validation manifest needs updating.")
    if check_overlap:
        training_rows = _training_rows(name, meta)
        if training_rows is not None:
            try:
                dataset._check_validation_overlap(training_rows, rows)
            except ValueError as exc:
                raise ServiceError(raw(str(exc))) from exc
    return dict(rows=rows, fingerprint=fingerprint, manifest=status["manifest"], samples=len(rows), version=None)


def _validation_info(name: str) -> dict:
    try:
        snapshot = _validation_snapshot(name, check_overlap=False)
    except ServiceError as exc:
        return dict(ready=False, message=exc.message, fingerprint=None, samples=0, manifest="",
                    version=_current(name)[1])
    ready = (msg("Validation version {0} is ready.", snapshot["version"]["name"]) if snapshot["version"]
             else msg("Validation manifest is ready."))
    return dict(ready=True, message=ready, fingerprint=snapshot["fingerprint"],
                samples=snapshot["samples"], manifest=snapshot["manifest"], version=snapshot["version"])


def _adapter_root(name: str, adapter: str | None) -> Path:
    root = projects.output_dir(name, adapter)
    if not root.is_dir():
        raise NotFound("No saved adapters were found in {0}.", root.name)
    return root


def results(name: str, adapter: str | None = None) -> dict:
    root = projects.output_dir(name, adapter)
    info = _validation_info(name)
    fingerprint, version = _current(name)
    fingerprint = info["fingerprint"] or fingerprint
    return dict(project=name, adapter=root.name, validation=info, training_fingerprint=_training_fingerprint(root),
                trained_with=_trained_with(root) if root.is_dir() else None,
                items=_items(root, fingerprint, info["version"] or version) if root.is_dir() else [])


# ---------------------------------------------------------------------- tensorboard

def _write_tensorboard(root: Path, fingerprint: str, version: dict | None = None) -> None:
    """Rewrite the ``posthoc`` TensorBoard run from the current results."""
    run = root / "tensorboard" / TENSORBOARD_RUN
    points = _points(_items(root, fingerprint, version))
    try:
        if run.is_dir():
            for event in run.glob("events.out.tfevents.*"):
                if event.is_file() and not event.is_symlink():
                    event.unlink()
        if not points:
            return
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(run)) as writer:
            for step, value in points:
                writer.add_scalar("validation/loss", value, step)
    except Exception as exc:  # noqa: BLE001 - the JSON results remain authoritative
        print(f"[validation] TensorBoard post-hoc run was not updated: {type(exc).__name__}: {exc}", flush=True)


# ------------------------------------------------------------------------------ job

def _resolve_final_step(folder: Path) -> int | None:
    sidecar = read_adapter_state(folder)
    if sidecar:
        return sidecar["step"]
    state = folder / "trainer_state.pt"
    if not state.is_file():
        return None
    try:
        import torch
        value = torch.load(state, map_location="cpu", weights_only=False).get("step")
        return int(value) if value is not None else None
    except Exception:  # noqa: BLE001
        return None


def start(name: str, request: ValidationRunRequest):
    state = current()
    if state.training.snapshot().get("running"):
        raise Busy("Stop training before computing validation loss.")
    root = _adapter_root(name, request.adapter)
    available = {item["name"]: item for item in projects.checkpoints(name, root.name)}
    selected = list(dict.fromkeys(request.checkpoints))
    for item in selected:
        if item not in available:
            raise NotFound("Checkpoint not found: {0}", item)
    if not selected and not request.include_baseline:
        raise ServiceError("Select at least one checkpoint or the base model.")
    snapshot = _validation_snapshot(name, check_overlap=True)
    fingerprint = snapshot["fingerprint"]
    if not state.fake:
        from .system import cuda_info, model_files_present
        if not model_files_present():
            raise ServiceError("IndexTTS 2.5 model is missing. Restart with 2- run.bat so the automatic model downloader can repair it.")
        if not cuda_info().get("available"):
            raise ServiceError("CUDA is unavailable. This Easy GUI is configured as an NVIDIA CUDA build; rerun 1- install.bat and check your NVIDIA driver.")

    version = snapshot["version"]
    current_names = {item["name"] for item in _items(root, fingerprint, version) if item["status"] == "current"}
    targets = []
    if request.include_baseline and (request.recompute or BASELINE not in current_names):
        targets.append(dict(name=BASELINE, path=None, step=0, kind="baseline"))
    checkpoints = [available[item] for item in selected if request.recompute or item not in current_names]
    checkpoints.sort(key=lambda item: (item["kind"] != "checkpoint", item["step"] is None, item["step"] or 0))
    targets += [dict(name=item["name"], path=item["path"], step=item["step"], kind=item["kind"]) for item in checkpoints]
    if not targets:
        raise ServiceError("The selected items already have current results. Turn on Recompute to run them again.")

    model_dir = str(state.extras.get("model_dir", paths.MODEL_DIR))
    evaluate = state.extras.get("posthoc_fn")
    if evaluate is None:
        from indextts_training.posthoc import evaluate_checkpoints as evaluate
    rows = snapshot["rows"]
    samples = snapshot["samples"]

    def record(target: dict, loss: float, same_as: str | None = None) -> None:
        folder = root / target["name"]
        entry = dict(name=target["name"], step=target.get("step"), loss=loss, validation_fingerprint=fingerprint,
                     samples=samples, computed_at=time.time(), model_dir=model_dir, same_as=same_as,
                     weights=None if target["name"] == BASELINE else _signature(folder))
        if version:
            entry["validation_version"] = version
        _upsert(root, entry)

    def run(ctx):
        engine.unload()
        work = []
        copies = []
        for target in targets:
            if target["kind"] in ("final", "best"):
                folder = Path(target["path"])
                digest = _sha256(folder)
                twin = next((item for item in available.values() if item["kind"] == "checkpoint" and digest
                             and _sha256(Path(item["path"])) == digest), None)
                if twin is not None:
                    target = {**target, "step": twin["step"]}
                    if twin["name"] in current_names or any(item["name"] == twin["name"] for item in work):
                        copies.append((target, twin["name"]))
                        continue
                elif target["step"] is None:
                    target = {**target, "step": _resolve_final_step(folder)}
            work.append(target)

        def progress(done, total, target, sample, count):
            label = msg("Base model (no adapter)") if target["name"] == BASELINE else target["name"]
            ctx.progress(done / max(1, total), msg("Validation {0}: sample {1}/{2}", label, min(sample + 1, count), count))

        outcome = {"completed": [], "stopped": False}
        try:
            if work:
                outcome = evaluate(model_dir, work, rows, on_progress=progress, on_result=record,
                                   cancelled=lambda: ctx.cancel_requested)
            computed = list(outcome.get("completed", []))
            if not outcome.get("stopped"):
                saved = {item["name"]: item for item in _load(root)
                         if item.get("validation_fingerprint") == fingerprint}
                for target, twin in copies:
                    if twin in saved:
                        record(target, float(saved[twin]["loss"]), same_as=twin)
                        computed.append(target["name"])
        finally:
            # Results recorded before a failure are kept, so the TensorBoard run must match them.
            _write_tensorboard(root, fingerprint, version)
        return {"project": name, "adapter": root.name, "computed": computed,
                "stopped": bool(outcome.get("stopped") or ctx.cancel_requested)}

    return state.jobs.submit("validation_posthoc", msg("Computing validation loss."), run, gpu=True,
                             meta={"project": name, "adapter": root.name, "targets": [item["name"] for item in targets]})
