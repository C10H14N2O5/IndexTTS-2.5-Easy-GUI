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
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import Field
from indextts_training import dataset
from indextts_training.schedule import read_adapter_state
from .. import paths
from ..errors import Busy, NotFound, ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel
from ..state import current
from . import engine, projects

RESULTS_FILE = "validation_posthoc.json"
BASELINE = "base_model"
TENSORBOARD_RUN = "posthoc"
SERIES_LABEL = "validation_posthoc/loss"
_LOCK = threading.RLock()


class ValidationRunRequest(ApiModel):
    adapter: str | None = None
    checkpoints: list[str] = Field(default_factory=list, max_length=1000)
    include_baseline: bool = False
    recompute: bool = False


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


class ValidationSetInfo(ApiModel):
    ready: bool
    message: str
    fingerprint: str | None = None
    samples: int = 0
    manifest: str = ""


class PosthocResults(ApiModel):
    project: str
    adapter: str
    validation: ValidationSetInfo
    training_fingerprint: str | None = None
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


def _load(root: Path) -> list[dict]:
    try:
        data = json.loads((root / RESULTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    return [entry for entry in entries or [] if isinstance(entry, dict) and "name" in entry and "loss" in entry]


def _save(root: Path, entries: list[dict]) -> None:
    payload = {"version": 1, "entries": sorted(entries, key=lambda item: (
        item.get("validation_fingerprint", ""), item["name"] != BASELINE, item.get("step") is None,
        item.get("step") or 0, item["name"]))}
    dataset._atomic_write(root / RESULTS_FILE, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))


def _upsert(root: Path, entry: dict) -> None:
    with _LOCK:
        entries = [item for item in _load(root) if not (
            item["name"] == entry["name"] and item.get("validation_fingerprint") == entry["validation_fingerprint"])]
        entries.append(entry)
        _save(root, entries)


def _training_fingerprint(root: Path) -> str | None:
    try:
        saved = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    validation = saved.get("validation") if isinstance(saved, dict) else None
    value = validation.get("fingerprint") if isinstance(validation, dict) else None
    return value if isinstance(value, str) else None


def _current_fingerprint(name: str) -> str | None:
    """Fingerprint recorded when the validation manifest was last built (no folder scan)."""
    settings = dataset.validation_settings(projects.load_raw(name))
    value = settings.get("manifest_sha256")
    return value if isinstance(value, str) and value else None


def _status(entry: dict, root: Path, fingerprint: str | None) -> str:
    if not fingerprint or entry.get("validation_fingerprint") != fingerprint:
        return "outdated"
    if entry["name"] == BASELINE:
        return "current"
    folder = root / entry["name"]
    return "current" if entry.get("weights") and _signature(folder) == entry["weights"] else "outdated"


def _items(root: Path, fingerprint: str | None) -> list[dict]:
    result = []
    for entry in _load(root):
        kind = "baseline" if entry["name"] == BASELINE else projects.adapter_kind(entry["name"]) or "checkpoint"
        result.append(dict(name=entry["name"], kind=kind, step=entry.get("step"), loss=float(entry["loss"]),
                           validation_fingerprint=entry.get("validation_fingerprint", ""),
                           samples=int(entry.get("samples", 0)), computed_at=float(entry.get("computed_at", 0)),
                           status=_status(entry, root, fingerprint), same_as=entry.get("same_as")))
    return result


def _points(items: list[dict]) -> list[tuple[int, float]]:
    """Current results by step; a checkpoint wins over a final/best adapter at the same step."""
    by_step: dict[int, tuple[int, float]] = {}
    for item in items:
        if item["status"] != "current" or item["step"] is None:
            continue
        rank = 0 if item["kind"] in ("final", "best") else 1
        step = int(item["step"])
        if step not in by_step or rank >= by_step[step][0]:
            by_step[step] = (rank, item["loss"])
    return sorted((step, value) for step, (_, value) in by_step.items())


def series(name: str, root: Path) -> dict | None:
    """Chart series for :mod:`metrics` (baseline is drawn at step 0)."""
    points = _points(_items(root, _current_fingerprint(name)))
    return dict(label=SERIES_LABEL, points=points, count=len(points)) if points else None


# ---------------------------------------------------------------------- validation

def _validation_snapshot(name: str, *, check_overlap: bool) -> dict:
    meta = projects.load_raw(name)
    settings = dataset.validation_settings(meta)
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
    if check_overlap and meta.get("manifest") and Path(meta["manifest"]).is_file():
        try:
            _, training_rows = dataset._read_manifest_data(meta["manifest"], "training")
            dataset._check_validation_overlap(training_rows, rows)
        except ValueError as exc:
            raise ServiceError(raw(str(exc))) from exc
    return dict(rows=rows, fingerprint=fingerprint, manifest=status["manifest"], samples=len(rows))


def _validation_info(name: str) -> dict:
    try:
        snapshot = _validation_snapshot(name, check_overlap=False)
    except ServiceError as exc:
        return dict(ready=False, message=exc.message, fingerprint=None, samples=0, manifest="")
    return dict(ready=True, message=msg("Validation manifest is ready."), fingerprint=snapshot["fingerprint"],
                samples=snapshot["samples"], manifest=snapshot["manifest"])


def _adapter_root(name: str, adapter: str | None) -> Path:
    root = projects.output_dir(name, adapter)
    if not root.is_dir():
        raise NotFound("No saved adapters were found in {0}.", root.name)
    return root


def results(name: str, adapter: str | None = None) -> dict:
    root = projects.output_dir(name, adapter)
    info = _validation_info(name)
    fingerprint = info["fingerprint"] or _current_fingerprint(name)
    return dict(project=name, adapter=root.name, validation=info, training_fingerprint=_training_fingerprint(root),
                items=_items(root, fingerprint) if root.is_dir() else [])


# ---------------------------------------------------------------------- tensorboard

def _write_tensorboard(root: Path, fingerprint: str) -> None:
    """Rewrite the ``posthoc`` TensorBoard run from the current results."""
    run = root / "tensorboard" / TENSORBOARD_RUN
    points = _points(_items(root, fingerprint))
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

    current_names = {item["name"] for item in _items(root, fingerprint) if item["status"] == "current"}
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
        _upsert(root, dict(name=target["name"], step=target.get("step"), loss=loss, validation_fingerprint=fingerprint,
                           samples=samples, computed_at=time.time(), model_dir=model_dir, same_as=same_as,
                           weights=None if target["name"] == BASELINE else _signature(folder)))

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
            _write_tensorboard(root, fingerprint)
        return {"project": name, "adapter": root.name, "computed": computed,
                "stopped": bool(outcome.get("stopped") or ctx.cancel_requested)}

    return state.jobs.submit("validation_posthoc", msg("Computing validation loss."), run, gpu=True,
                             meta={"project": name, "adapter": root.name, "targets": [item["name"] for item in targets]})
