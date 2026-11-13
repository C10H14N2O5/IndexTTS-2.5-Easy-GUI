"""Frozen, read-only versions of a project's prepared manifests.

A version is a snapshot of the working ``manifest.jsonl`` / ``validation.jsonl``
taken by the user. Audio is never copied: the snapshot keeps the manifest rows
(including the transcript text at freeze time) and a fingerprint of every
referenced audio file (size, modification time and SHA-256). Training and
validation can then use the snapshot while the working corpus keeps growing.

Layout, per project and role (``training`` / ``validation``)::

    datasets/<project>/versions/<role>/index.json         {"next": <sequence>}
    datasets/<project>/versions/<role>/<id>/version.json   metadata and history
    datasets/<project>/versions/<role>/<id>/manifest.jsonl byte-identical to the frozen manifest
    datasets/<project>/versions/<role>/<id>/audio.jsonl    one audio fingerprint per manifest row

Version ids (``train-001``, ``val-001``) are never reused. A version is only
changed by an explicit repair, which keeps the id, increments ``revision`` and
records what changed in ``history``.

User-facing failures raise :class:`VersionError` with an English template and
arguments so the GUI can translate them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import statistics
import threading
import time
from pathlib import Path

from .dataset import (DATASET_LOCK, _atomic_write, _decode_manifest, _duration, _manifest_path, _project_file,
                      audio_files, dataset_manifest_status, load_project, validation_settings)

SCHEMA = 1
ROLES = ("training", "validation")
PREFIX = {"training": "train", "validation": "val"}
NAME_MAX = 60
NOTE_MAX = 500
MAX_LISTED = 200
_ID = re.compile(r"^(train|val)-(\d{3,})$")
_HASHES: dict[tuple, str] = {}
_HASHES_LOCK = threading.Lock()


class VersionError(ValueError):
    """User-facing failure; ``template``/``args_`` allow translation, ``kind`` maps to an HTTP status."""

    def __init__(self, template: str, *args, kind: str = "invalid"):
        self.template = template
        self.args_ = args
        self.kind = kind
        super().__init__(template.format(*args) if args else template)


# ----------------------------------------------------------------------------- helpers

def _role(role: str) -> str:
    if role not in ROLES:
        raise VersionError("Unknown dataset role.")
    return role


def versions_root(project: str, role: str, root="training") -> Path:
    _project_file(project, root)  # validates the project name
    datasets = Path(root) / "datasets"
    folder = datasets / project / "versions" / _role(role)
    if not folder.resolve().is_relative_to(datasets.resolve()):
        raise VersionError("Invalid project path.")
    return folder


def _version_dir(project: str, role: str, version_id, root="training") -> Path:
    match = _ID.match(str(version_id or ""))
    if not match or match.group(1) != PREFIX[_role(role)]:
        raise VersionError("Dataset version not found: {0}", version_id, kind="not_found")
    return versions_root(project, role, root) / str(version_id)


def _sequence(version_id: str) -> int:
    match = _ID.match(version_id)
    return int(match.group(2)) if match else 0


def _sha256_file(path: Path, stat=None) -> str:
    stat = stat or path.stat()
    key = (os.path.normcase(str(path)), stat.st_size, stat.st_mtime_ns)
    with _HASHES_LOCK:
        cached = _HASHES.get(key)
    if cached:
        return cached
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with _HASHES_LOCK:
        if len(_HASHES) > 100_000:
            _HASHES.clear()
        _HASHES[key] = digest
    return digest


def _fingerprint(audio: str) -> dict:
    path = Path(audio)
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("Expected a regular audio file.")
        stat = path.stat()
        return dict(audio=audio, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=_sha256_file(path, stat))
    except OSError as exc:
        raise VersionError("Unable to read audio listed in the manifest: {0}", audio) from exc


def _encode_rows(rows: list[dict]) -> bytes:
    # Same encoding as dataset.prepare_dataset, so a freeze is byte-identical.
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")


def _encode_audio(records: list[dict]) -> bytes:
    return ("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n").encode("utf-8")


def _row_stats(rows: list[dict], source_stats: dict | None = None) -> dict:
    durations = [float(row.get("duration") or 0.0) for row in rows]
    ordered = sorted(durations)
    source_stats = source_stats or {}
    return dict(samples=len(rows), valid=len(rows), missing_text=0, total_seconds=sum(durations),
                median_seconds=statistics.median(durations) if durations else 0.0,
                p90_seconds=ordered[max(0, int(len(ordered) * .9) - 1)] if ordered else 0.0,
                auto_min_seconds=source_stats.get("auto_min_seconds", 0.0),
                auto_max_seconds=source_stats.get("auto_max_seconds", 0.0))


def _clean_name(value) -> str:
    name = " ".join(str(value or "").split())
    if len(name) > NAME_MAX:
        raise VersionError("Version names can have at most {0} characters.", NAME_MAX)
    return name


def _clean_note(value) -> str:
    note = str(value or "").strip()
    if len(note) > NOTE_MAX:
        raise VersionError("Version notes can have at most {0} characters.", NOTE_MAX)
    return note


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data) -> None:
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))


def _save_project(project: str, meta: dict, root) -> None:
    _write_json(_project_file(project, root), meta)


# ------------------------------------------------------------------------------ reading

def load_version(project: str, role: str, version_id, root="training") -> tuple[Path, dict]:
    folder = _version_dir(project, role, version_id, root)
    info = _read_json(folder / "version.json")
    if not isinstance(info, dict) or info.get("schema") != SCHEMA or info.get("id") != folder.name \
            or info.get("role") != role or folder.is_symlink():
        raise VersionError("Dataset version not found: {0}", version_id, kind="not_found")
    return folder, info


def list_versions(project: str, role: str, root="training") -> list[dict]:
    base = versions_root(project, role, root)
    if not base.is_dir():
        return []
    result = []
    for folder in base.iterdir():
        if not folder.is_dir() or not _ID.match(folder.name):
            continue
        try:
            result.append(load_version(project, role, folder.name, root)[1])
        except VersionError:
            continue
    return sorted(result, key=lambda info: _sequence(info["id"]))


def active_id(meta: dict, role: str):
    value = meta.get("dataset_version") if _role(role) == "training" else validation_settings(meta).get("version")
    return value if isinstance(value, str) and value else None


def _read_audio_index(folder: Path) -> list[dict]:
    try:
        lines = (folder / "audio.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError):
        return []
    return [record for record in records if isinstance(record, dict)]


def verify(project: str, role: str, version_id, root="training") -> dict:
    """Check the frozen manifest and every referenced audio file.

    ``status`` is ``ready``, ``damaged`` (audio missing or changed) or
    ``invalid`` (the frozen files themselves were edited outside the GUI).
    Same-size files with a different modification time are hashed, so a copy
    with identical content still counts as unchanged.
    """
    folder, info = load_version(project, role, version_id, root)
    base = dict(id=info["id"], name=info["name"], info=info, folder=folder, missing=[], changed=[], rows=[],
                audio=[], raw=b"")
    try:
        raw = (folder / "manifest.jsonl").read_bytes()
        rows = _decode_manifest(raw)
    except (OSError, ValueError):
        return dict(base, status="invalid", ok=False,
                    message=("The frozen manifest of version {0} is unreadable.", (info["name"],)))
    audio = _read_audio_index(folder)
    if hashlib.sha256(raw).hexdigest() != info.get("manifest_sha256") or len(audio) != len(rows) or not rows:
        return dict(base, status="invalid", ok=False,
                    message=("The files of version {0} were modified outside the GUI.", (info["name"],)))
    missing, changed = [], []
    for record in audio:
        path = Path(str(record.get("audio") or ""))
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError
            stat = path.stat()
        except OSError:
            missing.append(record)
            continue
        if stat.st_size != record.get("size"):
            changed.append(record)
        elif stat.st_mtime_ns != record.get("mtime_ns"):
            try:
                if _sha256_file(path, stat) != record.get("sha256"):
                    changed.append(record)
            except OSError:
                missing.append(record)
    result = dict(base, rows=rows, audio=audio, raw=raw, missing=missing, changed=changed)
    if not missing and not changed:
        return dict(result, status="ready", ok=True, message=("Version {0} is ready.", (info["name"],)))
    return dict(result, status="damaged", ok=False,
                message=("Version {0}: {1} audio file(s) are missing and {2} were changed since it was frozen. Repair the version or choose another one.",
                         (info["name"], len(missing), len(changed))))


def usable_rows(project: str, role: str, version_id, root="training") -> dict:
    """Rows of a verified version for training or validation, else VersionError."""
    report = verify(project, role, version_id, root)
    if not report["ok"]:
        template, args = report["message"]
        raise VersionError(template, *args)
    info = report["info"]
    return dict(raw=report["raw"], rows=report["rows"], info=info, manifest=str((report["folder"] / "manifest.jsonl").resolve()),
                ref=version_ref(info))


def version_ref(info: dict | None) -> dict | None:
    if not info:
        return None
    return dict(id=info["id"], name=info["name"], revision=int(info.get("revision", 1)))


def resolve_active(project: str, role: str, root="training", *, working_status: dict | None = None) -> dict:
    """The dataset a training run or validation uses: the selected version or the working manifest.

    ``message`` is either a plain backend string (working set) or a
    ``(template, args)`` tuple (versions).
    """
    meta = load_project(project, root)
    settings = meta if role == "training" else validation_settings(meta)
    version_id = active_id(meta, role)
    if version_id is None:
        status = working_status or dataset_manifest_status(project, root, role=role)
        return dict(role=role, version_id=None, name="", revision=None, status=status["status"],
                    stale=bool(status["stale"]), message=status["message"], manifest=status["manifest"],
                    manifest_sha256=settings.get("manifest_sha256") if status.get("managed") else None,
                    stats=status.get("stats") or {}, language=meta.get("language", "EN"))
    try:
        report = verify(project, role, version_id, root)
    except VersionError:
        return dict(role=role, version_id=version_id, name="", revision=None, status="missing", stale=True,
                    message=("The selected dataset version {0} no longer exists. Choose another version.", (version_id,)),
                    manifest="", manifest_sha256=None, stats={}, language=meta.get("language", "EN"))
    info = report["info"]
    return dict(role=role, version_id=version_id, name=info["name"], revision=int(info.get("revision", 1)),
                status=report["status"], stale=not report["ok"], message=report["message"],
                manifest=str((report["folder"] / "manifest.jsonl").resolve()),
                manifest_sha256=info.get("manifest_sha256"), stats=info.get("stats") or {},
                language=info.get("language") or meta.get("language", "EN"))


def active_fingerprint(project: str, role: str, root="training") -> tuple[str | None, dict | None]:
    """Manifest fingerprint and version reference of the selected dataset, without scanning audio."""
    meta = load_project(project, root)
    version_id = active_id(meta, role)
    if version_id is None:
        settings = meta if role == "training" else validation_settings(meta)
        value = settings.get("manifest_sha256")
        return (value if isinstance(value, str) and value else None), None
    try:
        _, info = load_version(project, role, version_id, root)
    except VersionError:
        return None, None
    return info.get("manifest_sha256"), version_ref(info)


def working_rows(project: str, role: str, root="training") -> list[dict] | None:
    meta = load_project(project, root)
    settings = meta if role == "training" else validation_settings(meta)
    manifest = settings.get("manifest")
    if not manifest:
        return None
    try:
        return _decode_manifest(Path(manifest).read_bytes())
    except (OSError, ValueError):
        return None


def diff(rows: list[dict], working: list[dict] | None) -> dict | None:
    """Compare a version with the working manifest by audio path and transcript."""
    if working is None:
        return None
    key = lambda row: os.path.normcase(str(row.get("audio", "")))  # noqa: E731
    mine = {key(row): row.get("text") for row in rows}
    theirs = {key(row): row.get("text") for row in working}
    return dict(added=len(theirs.keys() - mine.keys()), removed=len(mine.keys() - theirs.keys()),
                text_changed=sum(1 for item in mine.keys() & theirs.keys() if mine[item] != theirs[item]))


def used_by(project: str, role: str, version_id: str, root="training") -> list[str]:
    """Adapter folders whose latest training run recorded this version."""
    outputs = Path(root) / "outputs"
    if not outputs.is_dir():
        return []
    result = []
    for config in outputs.glob("*/training_config.json"):
        data = _read_json(config)
        if not isinstance(data, dict) or data.get("project") != project:
            continue
        section = data.get("dataset") if role == "training" else data.get("validation")
        if isinstance(section, dict) and section.get("version_id") == version_id:
            result.append(config.parent.name)
    return sorted(result)


# ----------------------------------------------------------------------------- mutations

def freeze_blocker(project: str, role: str, root="training", *, working_status: dict | None = None):
    """Why the working manifest cannot be frozen now, as ``(template, args, kind)``; None when it can."""
    status = working_status or dataset_manifest_status(project, root, role=_role(role))
    if not status.get("managed"):
        return ("An external training manifest is selected. Only manifests built by this project can be frozen.", (), "invalid")
    if status["stale"]:
        return (f"Build an up-to-date {role} manifest before freezing it.", (), "invalid")
    settings = load_project(project, root)
    settings = settings if role == "training" else validation_settings(settings)
    digest = settings.get("manifest_sha256")
    duplicate = next((info for info in list_versions(project, role, root) if info.get("manifest_sha256") == digest), None)
    if duplicate is not None:
        return ("The current manifest is identical to version {0}.", (duplicate["name"],), "conflict")
    return None


def next_default_name(project: str, role: str, root="training") -> str:
    versions = list_versions(project, role, root)
    names = {info["name"].casefold() for info in versions}
    sequence = _next_sequence(project, role, root)
    name = f"v{sequence}"
    number = 2
    while name.casefold() in names:
        name, number = f"v{sequence}-{number}", number + 1
    return name


def _next_sequence(project: str, role: str, root) -> int:
    base = versions_root(project, role, root)
    index = _read_json(base / "index.json")
    stored = index.get("next") if isinstance(index, dict) and isinstance(index.get("next"), int) else 1
    existing = [_sequence(folder.name) for folder in base.iterdir() if _ID.match(folder.name)] if base.is_dir() else []
    return max([stored, 1, *(value + 1 for value in existing)])


def _unique_name(project: str, role: str, name: str, root, *, exclude: str | None = None) -> None:
    for info in list_versions(project, role, root):
        if info["id"] != exclude and info["name"].casefold() == name.casefold():
            raise VersionError("A version named {0} already exists.", name, kind="conflict")


def freeze(project: str, role: str, *, name=None, note="", activate=False, root="training") -> dict:
    """Snapshot the up-to-date working manifest as a new read-only version."""
    from .proofread import directory_asr_running, directory_lock
    _role(role)
    with DATASET_LOCK:
        meta = load_project(project, root)
        if not meta:
            raise VersionError("Project not found: {0}", project, kind="not_found")
        settings = meta if role == "training" else validation_settings(meta)
        folder = str(settings.get("source_folder") or "")
        with directory_lock(folder or None):
            if folder and directory_asr_running(folder):
                raise VersionError("Recognition is running for this directory. Finish recognition before freezing.")
            blocker = freeze_blocker(project, role, root)
            if blocker is not None:
                raise VersionError(blocker[0], *blocker[1], kind=blocker[2])
            manifest = _manifest_path(root, project, role)
            raw = manifest.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != settings.get("manifest_sha256"):
                raise VersionError(f"Build an up-to-date {role} manifest before freezing it.")
            rows = _decode_manifest(raw)
            if not rows:
                raise VersionError(f"Build an up-to-date {role} manifest before freezing it.")
            audio = [_fingerprint(str(row.get("audio") or "")) for row in rows]
            sequence = _next_sequence(project, role, root)
            version_id = f"{PREFIX[role]}-{sequence:03d}"
            requested = _clean_name(name)
            if requested:
                _unique_name(project, role, requested, root)
            label = requested or next_default_name(project, role, root)
            now = time.time()
            info = dict(schema=SCHEMA, id=version_id, role=role, name=label, note=_clean_note(note), revision=1,
                        created_at=now, updated_at=now, source_folder=folder,
                        language=str(rows[0].get("language") or meta.get("language", "EN")),
                        manifest_bounds=settings.get("manifest_bounds") or {}, manifest_sha256=digest,
                        stats=_row_stats(rows, settings.get("stats")), source_stats=settings.get("stats") or {},
                        source_signature=settings.get("source_signature"), history=[])
            base = versions_root(project, role, root)
            base.mkdir(parents=True, exist_ok=True)
            target = base / version_id
            if target.exists():
                raise VersionError("A version named {0} already exists.", version_id, kind="conflict")
            pending = base / f".{version_id}.{secrets.token_hex(6)}.tmp"
            try:
                pending.mkdir()
                (pending / "manifest.jsonl").write_bytes(raw)
                (pending / "audio.jsonl").write_bytes(_encode_audio(audio))
                (pending / "version.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(pending, target)
            finally:
                if pending.exists():
                    shutil.rmtree(pending, ignore_errors=True)
            _write_json(base / "index.json", {"next": sequence + 1})
            if activate:
                _set_active(project, role, version_id, root)
            return info


def _set_active(project: str, role: str, version_id, root) -> dict:
    meta = load_project(project, root)
    if not meta:
        raise VersionError("Project not found: {0}", project, kind="not_found")
    if role == "training":
        meta["dataset_version"] = version_id
    else:
        settings = validation_settings(meta)
        settings["version"] = version_id
        meta["validation"] = settings
    _save_project(project, meta, root)
    return meta


def select(project: str, role: str, version_id, root="training") -> dict:
    """Choose the dataset used for training/validation (None = working manifest)."""
    _role(role)
    with DATASET_LOCK:
        if version_id is not None:
            load_version(project, role, version_id, root)
        return _set_active(project, role, version_id, root)


def update(project: str, role: str, version_id, *, name=None, note=None, root="training") -> dict:
    with DATASET_LOCK:
        folder, info = load_version(project, role, version_id, root)
        if name is not None:
            label = _clean_name(name)
            if not label:
                raise VersionError("Enter a version name.")
            _unique_name(project, role, label, root, exclude=info["id"])
            info["name"] = label
        if note is not None:
            info["note"] = _clean_note(note)
        info["updated_at"] = time.time()
        _write_json(folder / "version.json", info)
        return info


def delete(project: str, role: str, version_id, root="training") -> dict:
    """Delete a version; the selection falls back to the working manifest when it was selected."""
    with DATASET_LOCK:
        folder, info = load_version(project, role, version_id, root)
        base = versions_root(project, role, root)
        if folder.resolve().parent != base.resolve():
            raise VersionError("Invalid project path.")
        shutil.rmtree(folder)
        meta = load_project(project, root)
        was_active = active_id(meta, role) == info["id"]
        if was_active:
            _set_active(project, role, None, root)
        return dict(info=info, was_active=was_active)


# ---------------------------------------------------------------------------- integrity

def _relocations(folder: str, missing: list[dict], taken: set[str]) -> dict[str, str]:
    """Find moved files: same size and SHA-256 inside the version's source folder."""
    if not folder or not missing or not Path(folder).is_dir():
        return {}
    by_size: dict[int, list[dict]] = {}
    for record in missing:
        by_size.setdefault(record.get("size"), []).append(record)
    found: dict[str, str] = {}
    try:
        candidates = audio_files(folder)
    except (OSError, ValueError):
        return {}
    for path in candidates:
        resolved = str(path.resolve())
        if os.path.normcase(resolved) in taken:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        group = by_size.get(stat.st_size)
        if not group:
            continue
        try:
            digest = _sha256_file(path, stat)
        except OSError:
            continue
        for record in group:
            if record["audio"] not in found and record.get("sha256") == digest:
                found[record["audio"]] = resolved
                taken.add(os.path.normcase(resolved))
                break
    return found


def integrity_report(project: str, role: str, version_id, root="training") -> dict:
    report = verify(project, role, version_id, root)
    info = report["info"]
    taken = {os.path.normcase(str(record.get("audio"))) for record in report["audio"]}
    moved = _relocations(info.get("source_folder", ""), report["missing"], taken) if report["status"] == "damaged" else {}
    affected = []
    for other in list_versions(project, role, root):
        if other["id"] == info["id"]:
            continue
        try:
            if verify(project, role, other["id"], root)["status"] == "damaged":
                affected.append(dict(id=other["id"], name=other["name"]))
        except VersionError:
            continue
    return dict(id=info["id"], name=info["name"], status=report["status"], message=report["message"],
                missing=[dict(audio=record["audio"], relocated_to=moved.get(record["audio"]))
                         for record in report["missing"][:MAX_LISTED]],
                changed=[dict(audio=record["audio"], relocated_to=None) for record in report["changed"][:MAX_LISTED]],
                missing_count=len(report["missing"]), changed_count=len(report["changed"]),
                relocatable=len(moved), affected=affected)


def _repair_one(project: str, role: str, version_id: str, changed: str, root) -> dict | None:
    report = verify(project, role, version_id, root)
    if report["status"] == "ready":
        return None
    if report["status"] != "damaged":
        template, args = report["message"]
        raise VersionError(template, *args)
    folder, info = report["folder"], report["info"]
    missing = {record["audio"] for record in report["missing"]}
    altered = {record["audio"] for record in report["changed"]}
    taken = {os.path.normcase(str(record.get("audio"))) for record in report["audio"]}
    moved = _relocations(info.get("source_folder", ""), report["missing"], taken)
    rows, records = [], []
    dropped, accepted, relocated = [], [], []
    for row, record in zip(report["rows"], report["audio"]):
        audio = record["audio"]
        if audio in missing:
            if audio not in moved:
                dropped.append(audio)
                continue
            row = {**row, "audio": moved[audio]}
            record = _fingerprint(moved[audio])
            relocated.append(dict(source=audio, target=moved[audio]))
        elif audio in altered:
            if changed == "drop":
                dropped.append(audio)
                continue
            record = _fingerprint(audio)
            row = {**row, "duration": _duration(Path(audio))}
            accepted.append(audio)
        rows.append(row)
        records.append(record)
    if not rows:
        raise VersionError("Repairing version {0} would remove every clip. Delete the version instead.", info["name"])
    raw = _encode_rows(rows)
    now = time.time()
    previous = info.get("manifest_sha256")
    info = dict(info, revision=int(info.get("revision", 1)) + 1, updated_at=now,
                manifest_sha256=hashlib.sha256(raw).hexdigest(), stats=_row_stats(rows, info.get("source_stats")))
    info["history"] = [*info.get("history", []), dict(
        revision=info["revision"], at=now, action="repair", previous_manifest_sha256=previous,
        relocated=relocated, dropped=dropped, accepted=accepted)]
    base = folder.parent
    pending = base / f".{folder.name}.{secrets.token_hex(6)}.tmp"
    backup = base / f".{folder.name}.{secrets.token_hex(6)}.old"
    try:
        pending.mkdir()
        (pending / "manifest.jsonl").write_bytes(raw)
        (pending / "audio.jsonl").write_bytes(_encode_audio(records))
        (pending / "version.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(folder, backup)
        try:
            os.replace(pending, folder)
        except OSError:
            os.replace(backup, folder)
            raise
    finally:
        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    return info


def repair(project: str, role: str, version_id, *, changed="accept", scope="version", root="training") -> list[dict]:
    """Repair damaged versions in place: relocate moved audio by content, drop missing
    clips, and accept or drop changed audio. Each repaired version gets a new revision."""
    if changed not in ("accept", "drop"):
        raise VersionError("Choose how to handle changed audio.")
    with DATASET_LOCK:
        _, info = load_version(project, role, version_id, root)
        targets = [info["id"]]
        if scope == "affected":
            targets += [other["id"] for other in list_versions(project, role, root) if other["id"] != info["id"]]
        repaired = []
        for target in targets:
            try:
                result = _repair_one(project, role, target, changed, root)
            except VersionError:
                if target == info["id"]:
                    raise
                continue  # Other affected versions that cannot be repaired stay unchanged.
            if result is not None:
                repaired.append(result)
        return repaired
