"""CPU export API adapters and successful-task-only download links."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from pydantic import Field
from indextts_training.export import adapter_details
from .. import paths
from ..errors import Busy
from ..i18n import msg, raw
from ..schemas import ApiModel, OpenModel
from ..state import current


class AdapterInfo(ApiModel):
    filename: str
    training_base: str
    valid: bool
    error: str
    base: str
    config: str
    notice: str


class ExportSnapshot(OpenModel):
    status: Literal["idle", "running", "cancelling", "succeeded", "failed", "cancelled"]
    phase: str
    elapsed: float
    log: str
    output: str
    error: str
    report: dict[str, Any] = Field(default_factory=dict)
    token: str
    task_id: str
    logfile: str
    downloads: dict[str, str] = Field(default_factory=dict)


def info(adapter: str) -> dict:
    data = adapter_details(adapter)
    directory = Path(current().extras.get("model_dir", paths.MODEL_DIR))
    data.update(base=str(directory / "gpt.pth"), config=str(directory / "config.yaml"))
    if not data["valid"]:
        data["notice"] = raw(data["error"])
    elif data["training_base"]:
        data["notice"] = msg("Recorded training model directory: {0}. This historical path does not verify the identity of its weights; confirm the original FP32 base below.", data["training_base"])
    else:
        data["notice"] = msg("Ready to export: {0}", adapter)
    return data


def snapshot(state=None) -> dict:
    state = state or current()
    data = state.export.snapshot()
    data["downloads"] = {}
    if data["status"] == "succeeded":
        data["downloads"] = {kind: f"/exports/{quote(data['task_id'], safe='')}/{kind}?token={quote(data['token'], safe='')}"
                             for kind in ("model", "report")}
    data["error"] = raw(data["error"]) if data["error"] else ""
    return data


def start(adapter: str, mode: str, destination: str, base: str = "", config: str = "", expected_base: str = "") -> dict:
    state = current()
    busy = state.jobs.running_gpu_job()
    if busy is not None:
        raise Busy("The GPU is busy with another job.", extra={"busy_job": busy.to_dict()})
    directory = Path(state.extras.get("model_dir", paths.MODEL_DIR))
    state.export.start(adapter, mode, destination, base or str(directory / "gpt.pth"),
                       config or str(directory / "config.yaml"), expected_base)
    state.status.publish("Checkpoint export", msg("Export running"), "working")
    result = snapshot(state)
    state.publish("export", result)
    return result


def cancel() -> dict:
    state = current()
    state.export.cancel()
    result = snapshot(state)
    state.publish("export", result)
    return result


def register(state) -> None:
    def finished(data):
        key = {"succeeded": "Export complete", "failed": "Export failed", "cancelled": "Export cancelled"}.get(data["status"])
        if key:
            state.status.publish("Checkpoint export", msg(key), "error" if data["status"] == "failed" else "info" if data["status"] == "cancelled" else "success",
                                 raw(data["error"]) if data["error"] else "")
            state.publish("export", snapshot(state))
    state.export.on_finished = finished
    state.register_snapshot("export", lambda: snapshot(state), 1)
