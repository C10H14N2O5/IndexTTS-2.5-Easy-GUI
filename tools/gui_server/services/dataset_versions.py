"""Frozen dataset versions of a project's training and validation manifests."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from indextts_training import dataset, dataset_versions as native
from .. import paths
from ..errors import Busy, Conflict, NotFound, ServiceError
from ..i18n import Message, msg, raw
from ..schemas import ApiModel, OpenModel
from ..state import current

Role = Literal["training", "validation"]


class VersionIntegrity(ApiModel):
    status: Literal["ready", "damaged", "invalid"]
    message: str
    missing: int = 0
    changed: int = 0


class VersionDiff(ApiModel):
    added: int
    removed: int
    text_changed: int


class DatasetVersionInfo(ApiModel):
    id: str
    role: Role
    name: str
    note: str = ""
    revision: int
    created_at: float
    updated_at: float
    language: str = "EN"
    source_folder: str = ""
    manifest_sha256: str
    stats: dict[str, Any] = Field(default_factory=dict)
    integrity: VersionIntegrity
    diff: VersionDiff | None = None
    used_by: list[str] = Field(default_factory=list)
    active: bool = False


class WorkingManifest(OpenModel):
    stale: bool
    managed: bool
    manifest: str
    status: str
    message: str
    stats: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionList(ApiModel):
    role: Role
    active: str | None = None
    working: WorkingManifest
    freeze_reason: str | None = None
    next_name: str
    versions: list[DatasetVersionInfo] = Field(default_factory=list)


class VersionFreeze(ApiModel):
    name: str | None = Field(None, max_length=native.NAME_MAX)
    note: str = Field("", max_length=native.NOTE_MAX)
    activate: bool = True


class VersionPatch(ApiModel):
    name: str | None = Field(None, max_length=native.NAME_MAX)
    note: str | None = Field(None, max_length=native.NOTE_MAX)


class VersionSelect(ApiModel):
    version_id: str | None = None


class VersionRepair(ApiModel):
    changed: Literal["accept", "drop"] = "accept"
    scope: Literal["version", "affected"] = "version"


class IntegrityEntry(ApiModel):
    audio: str
    relocated_to: str | None = None


class AffectedVersion(ApiModel):
    id: str
    name: str


class IntegrityReport(ApiModel):
    id: str
    name: str
    status: Literal["ready", "damaged", "invalid"]
    message: str
    missing: list[IntegrityEntry] = Field(default_factory=list)
    changed: list[IntegrityEntry] = Field(default_factory=list)
    missing_count: int = 0
    changed_count: int = 0
    relocatable: int = 0
    affected: list[AffectedVersion] = Field(default_factory=list)


class RepairResult(ApiModel):
    message: str
    versions: list[DatasetVersionInfo] = Field(default_factory=list)


def message(value) -> Message:
    """Render a native message: ``(template, args)`` tuples or plain backend strings."""
    if isinstance(value, tuple):
        template, args = value
        return msg(template, *args)
    return raw(value)


def call(function, *args, **kwargs):
    """Map native version errors to API errors with translatable messages."""
    try:
        return function(*args, **kwargs)
    except native.VersionError as exc:
        text = msg(exc.template, *exc.args_)
        if exc.kind == "not_found":
            raise NotFound(text) from exc
        if exc.kind == "conflict":
            raise Conflict(text) from exc
        raise ServiceError(text) from exc
    except ValueError as exc:
        raise ServiceError(raw(str(exc))) from exc


def _project(name: str) -> dict:
    from . import projects
    return projects.load_raw(name)


def _ensure_idle(action: Message) -> None:
    if current().training.snapshot().get("running"):
        raise Busy(action)


def _info(name: str, role: str, info: dict, *, active: str | None, working: list | None) -> dict:
    try:
        report = native.verify(name, role, info["id"], paths.TRAINING_DIR)
        integrity = dict(status=report["status"], message=message(report["message"]),
                         missing=len(report["missing"]), changed=len(report["changed"]))
        rows = report["rows"]
    except native.VersionError as exc:
        integrity = dict(status="invalid", message=msg(exc.template, *exc.args_), missing=0, changed=0)
        rows = []
    return dict(id=info["id"], role=role, name=info["name"], note=info.get("note", ""),
                revision=int(info.get("revision", 1)), created_at=float(info.get("created_at", 0)),
                updated_at=float(info.get("updated_at", 0)), language=info.get("language", "EN"),
                source_folder=info.get("source_folder", ""), manifest_sha256=info.get("manifest_sha256", ""),
                stats=info.get("stats") or {}, integrity=integrity,
                diff=native.diff(rows, working) if rows else None,
                used_by=native.used_by(name, role, info["id"], paths.TRAINING_DIR), active=active == info["id"])


def listing(name: str, role: str) -> dict:
    meta = _project(name)
    with dataset.DATASET_LOCK:
        working = dataset.dataset_manifest_status(name, paths.TRAINING_DIR, role=role)
        working["message"] = raw(working["message"])
        active = native.active_id(meta, role)
        rows = native.working_rows(name, role, paths.TRAINING_DIR) if working.get("managed") else None
        blocker = call(native.freeze_blocker, name, role, paths.TRAINING_DIR, working_status=working)
        versions = [_info(name, role, info, active=active, working=rows)
                    for info in call(native.list_versions, name, role, paths.TRAINING_DIR)]
        return dict(role=role, active=active, working=working,
                    freeze_reason=msg(blocker[0], *blocker[1]) if blocker else None,
                    next_name=call(native.next_default_name, name, role, paths.TRAINING_DIR), versions=versions)


def _single(name: str, role: str, info: dict) -> dict:
    meta = _project(name)
    rows = native.working_rows(name, role, paths.TRAINING_DIR)
    return _info(name, role, info, active=native.active_id(meta, role), working=rows)


def freeze(name: str, role: str, request: VersionFreeze) -> dict:
    _project(name)
    if request.activate:
        _ensure_idle(msg("Stop training before changing the dataset version."))
    info = call(native.freeze, name, role, name=request.name, note=request.note, activate=request.activate,
                root=paths.TRAINING_DIR)
    current().status.publish("Dataset versions", msg("Dataset version frozen: {0}", info["name"]), "success")
    return _single(name, role, info)


def update(name: str, role: str, version_id: str, request: VersionPatch) -> dict:
    _project(name)
    info = call(native.update, name, role, version_id, name=request.name, note=request.note, root=paths.TRAINING_DIR)
    return _single(name, role, info)


def select(name: str, role: str, version_id: str | None) -> None:
    _project(name)
    _ensure_idle(msg("Stop training before changing the dataset version."))
    call(native.select, name, role, version_id, paths.TRAINING_DIR)


def delete(name: str, role: str, version_id: str) -> dict:
    _project(name)
    _ensure_idle(msg("Stop training before deleting a dataset version."))
    result = call(native.delete, name, role, version_id, paths.TRAINING_DIR)
    label = result["info"]["name"]
    text = (msg("Dataset version deleted: {0}. The working manifest is used from now on.", label)
            if result["was_active"] else msg("Dataset version deleted: {0}.", label))
    current().status.publish("Dataset versions", text, "success")
    return {"ok": True, "message": text}


def integrity(name: str, role: str, version_id: str) -> dict:
    _project(name)
    report = call(native.integrity_report, name, role, version_id, paths.TRAINING_DIR)
    return {**report, "message": message(report["message"])}


def repair(name: str, role: str, version_id: str, request: VersionRepair) -> dict:
    _project(name)
    _ensure_idle(msg("Stop training before repairing a dataset version."))
    repaired = call(native.repair, name, role, version_id, changed=request.changed, scope=request.scope,
                    root=paths.TRAINING_DIR)
    names = ", ".join(info["name"] for info in repaired)
    text = msg("Repaired dataset versions: {0}.", names) if repaired else msg("No dataset version needed repair.")
    if repaired:
        current().status.publish("Dataset versions", text, "success")
    return {"message": text, "versions": [_single(name, role, info) for info in repaired]}


def selection(name: str, role: str, *, working_status: dict | None = None) -> dict:
    """The ProjectInfo summary of the dataset a run would use."""
    result = native.resolve_active(name, role, paths.TRAINING_DIR, working_status=working_status)
    return {**result, "message": message(result["message"])}
