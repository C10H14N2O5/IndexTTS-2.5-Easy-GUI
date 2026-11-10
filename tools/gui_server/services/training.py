"""Build native acoustic-only trainer configuration from saved widget values."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal
from pydantic import Field
from indextts_training import dataset
from .. import paths
from ..errors import Busy, ServiceError
from ..i18n import msg, raw
from ..schemas import ApiModel, OpenModel
from ..state import current
from . import engine, projects


class TrainingSnapshot(OpenModel):
    running: bool = False
    failed: bool = False
    stopped: bool = False
    status: Literal["idle", "running", "stopping", "stopped", "succeeded", "failed"] = "idle"
    step: int = 0
    total: int = 0
    epoch: float = 0
    loss: float | None = None
    elapsed: float = 0
    eta: float = 0
    message: str = ""
    output: str = ""
    output_dir: str = ""
    project: str = ""
    adapter: str = ""
    started_at: float | None = None
    finished_at: float | None = None


def snapshot(state=None) -> dict:
    state = state or current()
    data = state.training.snapshot()
    if data["running"]:
        status = "stopping" if state.training.stop_event.is_set() else "running"
    elif data["failed"]:
        status = "failed"
    elif data.get("finished_at"):
        status = "stopped" if state.training.stop_event.is_set() else "succeeded"
    else:
        status = "idle"
    return {**data, "status": status, "stopped": status == "stopped", "message": raw(data.get("message", ""))}


def start(name: str, overrides: dict | None = None) -> dict:
    state = current()
    with dataset.DATASET_LOCK:
        meta = projects.load_raw(name)
        values = projects.TrainingConfig.model_validate({**projects.training_values(meta), **(overrides or {})}).model_dump()
        manifest = meta.get("manifest") or ""
        if not manifest or not Path(manifest).is_file():
            raise ServiceError("Prepare/select a valid dataset project first.")
        adapter = projects.normalized_name(values["adapter_name"], adapter=True)
        output = projects.output_dir(name, adapter)
        resume = values["resume_checkpoint"]
        if resume:
            folder = Path(resume).resolve()
            if not folder.is_dir() or not (folder / "adapter_config.json").is_file():
                raise ServiceError("LoRA adapter is missing adapter_config.json: {0}", resume)
        if values["eval_enabled"] and values["eval_reference"] and not Path(values["eval_reference"]).is_file():
            raise ServiceError("Select an Eval Reference Audio file first.")
        busy = state.jobs.running_gpu_job()
        if busy is not None:
            raise Busy("The GPU is busy with another job.", extra={"busy_job": busy.to_dict()})
        if not state.fake:
            from .system import cuda_info, model_files_present
            if not model_files_present():
                raise ServiceError("IndexTTS 2.5 model is missing. Restart with 2- run.bat so the automatic model downloader can repair it.")
            if not cuda_info().get("available"):
                raise ServiceError("CUDA is unavailable. This Easy GUI is configured as an NVIDIA CUDA build; rerun 1- install.bat and check your NVIDIA driver.")
        updates = max(1, math.ceil(max(1, int(meta.get("stats", {}).get("valid", 1))) /
                                  (values["micro_batch"] * values["grad_accum"])))
        cfg = dict(manifest=manifest, model_dir=str(state.extras.get("model_dir", paths.MODEL_DIR)),
                   output_dir=str(output), mode=values["mode"], steps=values["training_steps"],
                   epochs=values["training_epochs"], save_every=values["save_steps"] if values["mode"] == "Steps"
                   else values["save_epochs"] * updates, rank=values["lora_rank"], alpha=values["lora_alpha"],
                   dropout=values["lora_dropout"], lr=values["learning_rate"], batch=values["micro_batch"],
                   grad_accum=values["grad_accum"], grad_clip=values["grad_clip"], seed=values["seed"],
                   vram=values["vram_preset"], resume=resume or None, language=meta.get("language", "EN"),
                   project=name, training_root=str(paths.TRAINING_DIR), eval_enabled=values["eval_enabled"],
                   eval_text=values["eval_text"].strip(), eval_reference=values["eval_reference"],
                   eval_reference_transcript=values["eval_reference_transcript"].strip(),
                   eval_language=projects.language_code(values["eval_language"]),
                   **projects._DISABLED_AUXILIARY)
        cfg.update(differentiable_asr_model="large-v3", asr_cfm_steps=4, asr_cfm_temperature=1.0)
        state.training.start(cfg)
    state.status.publish("LoRA training", msg("Training started."), "working")
    result = snapshot(state)
    state.publish("training", result)
    return result


def stop() -> dict:
    state = current()
    if state.training.snapshot().get("running"):
        state.training.stop()
        state.status.publish("LoRA training", msg("Stop requested."), "working")
    result = snapshot(state)
    state.publish("training", result)
    return result


def register(state) -> None:
    state.training.before_start = engine.unload

    def finished(outcome, data):
        key = {"succeeded": "Training complete.", "stopped": "Training stopped.", "failed": "Training failed."}[outcome]
        state.status.publish("LoRA training", msg(key), "error" if outcome == "failed" else "info" if outcome == "stopped" else "success", raw(data["message"]))
        state.publish("training", snapshot(state))

    state.training.on_finished = finished
    state.register_snapshot("training", lambda: snapshot(state), 1)
