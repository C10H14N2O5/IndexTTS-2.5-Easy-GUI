"""Local, session-scoped proofreading and ASR routes for the Easy GUI."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .dataset import AUDIO_EXTS, DATASET_LOCK


class Conflict(ValueError):
    pass


# All filesystem transactions share a lock because recursive roots can overlap.
# Audio conversion and model inference run outside this short critical section.
_FILESYSTEM_LOCK = threading.RLock()
_PREVIEW_LOCK = threading.Lock()
_ACTIVE_BATCHES: set[str] = set()
_BATCHES_LOCK = threading.Lock()


def _directory_key(directory):
    return os.path.normcase(str(Path(directory).resolve()))


def directory_lock(directory):
    """Serialize file transactions, including parent/child selected roots."""
    return _FILESYSTEM_LOCK


def _overlapping_roots(first, second):
    first, second = Path(first), Path(second)
    return first.is_relative_to(second) or second.is_relative_to(first)


def directory_asr_running(directory):
    """Shared across page sessions; caller may already hold directory_lock."""
    if not directory:
        return False
    key = _directory_key(directory)
    with _BATCHES_LOCK:
        return any(_overlapping_roots(key, active) for active in _ACTIVE_BATCHES)


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def natural(name):
    return [(1, int(v)) if v.isdigit() else (0, v.casefold()) for v in re.split(r"(\d+)", name)]


def safe_inside(path: Path, root: Path):
    """Reject links at any level, including a swapped ancestor/junction."""
    path = Path(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("文件路径不在当前目录内 / Path is outside the selected directory")
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("校对文件路径不允许使用链接 / Linked paths are not allowed")
    return path


class Library:
    """One page's file IDs, with directory-wide locking and merged disk state."""
    def __init__(self, directory=None):
        self.directory = None
        self.files = {}
        self.lock = threading.RLock()
        if directory:
            self.select(directory)

    def select(self, directory):
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("所选路径不是目录 / Select an existing directory")
        self.directory = root
        self.lock = directory_lock(root)
        with self.lock:
            self._read_state()
            self.rescan()
            return self.session()

    def rescan(self):
        if self.directory is None:
            return
        old = {p: ident for ident, p in self.files.items()}
        found = []
        for parent, dirs, names in os.walk(self.directory, followlinks=False):
            parent = Path(parent)
            dirs[:] = [d for d in dirs if d.casefold() != ".proofread" and not (parent / d).is_symlink()
                       and not getattr(parent / d, "is_junction", lambda: False)()]
            for name in names:
                p = parent / name
                if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
                    continue
                try:
                    safe_inside(p, self.directory)
                except ValueError:
                    continue
                found.append(p)
        found.sort(key=lambda p: natural(p.relative_to(self.directory).as_posix()))
        self.files = {old.get(p, secrets.token_hex(12)): p for p in found}

    def path(self, ident):
        if ident not in self.files or self.directory is None:
            raise ValueError("文件已失效，请重新加载目录 / Reload the directory")
        p = safe_inside(self.files[ident], self.directory)
        if not p.is_file():
            raise ValueError("音频已不存在 / Audio file no longer exists")
        return p

    def text_path(self, ident):
        return safe_inside(self.path(ident).with_suffix(".txt"), self.directory)

    def name(self, ident):
        return self.path(ident).relative_to(self.directory).as_posix()

    def metadata_dir(self):
        if self.directory is None:
            raise ValueError("请先选择目录 / Select a directory first")
        return safe_inside(self.directory / ".proofread", self.directory)

    def _read_state(self):
        path = safe_inside(self.metadata_dir() / "state.json", self.directory)
        if not path.exists():
            return {"reviewed": {}, "last_name": None}
        state = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(state, dict) or not isinstance(state.get("reviewed"), dict):
            raise ValueError("校对进度文件格式错误 / Invalid .proofread/state.json")
        return state

    def _persist(self, state):
        path = safe_inside(self.metadata_dir() / "state.json", self.directory)
        atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"))

    def item(self, ident):
        with self.lock:
            path = self.text_path(ident)
            raw = path.read_bytes() if path.exists() else None
            version = hashlib.sha256(raw).hexdigest() if raw is not None else "missing"
            text = raw.decode("utf-8-sig") if raw is not None else ""
            name = self.name(ident)
            return {"id": ident, "name": name, "text": text, "version": version,
                    "reviewed": bool(text.strip()) and self._read_state()["reviewed"].get(name) == version}

    def session(self):
        if self.directory is None:
            return {"directory": "", "items": [], "last_id": None}
        with self.lock:
            state = self._read_state()
            items = []
            for ident in self.files:
                try:
                    item = self.item(ident)
                    items.append({"id": ident, "name": item["name"], "reviewed": item["reviewed"],
                                  "missing_text": not bool(item["text"].strip())})
                except UnicodeError:
                    items.append({"id": ident, "name": self.name(ident), "reviewed": False,
                                  "missing_text": False, "error": "TXT is not valid UTF-8"})
            return {"directory": str(self.directory), "items": items,
                    "last_id": next((i for i in self.files if self.name(i) == state.get("last_name")), None)}

    def check(self, ident, version):
        if signature(self.text_path(ident)) != version:
            raise Conflict("TXT 已被其他程序修改；请保留编辑内容并重新加载 / TXT changed; reload before saving")

    def save(self, ident, text, version, reviewed=False):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("文本为空，不能保存 / Cannot save empty text")
        with self.lock:
            self.check(ident, version)
            target = self.text_path(ident)
            previous = target.read_bytes() if target.exists() else None
            if previous is not None:
                backup = safe_inside(self.metadata_dir() / "backups" / target.relative_to(self.directory), self.directory)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    with backup.open("xb") as stream:
                        stream.write(previous)
            state = self._read_state()
            self.check(ident, version)
            atomic_write(target, text.encode("utf-8"))
            name = self.name(ident)
            if reviewed:
                state["reviewed"][name] = signature(target)
            else:
                state["reviewed"].pop(name, None)
            state["last_name"] = name
            try:
                self._persist(state)
            except Exception:
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_write(target, previous)
                raise
            return self.item(ident)

    def review(self, ident, reviewed, version):
        with self.lock:
            self.check(ident, version)
            item = self.item(ident)
            if reviewed and not item["text"].strip():
                raise ValueError("空文本不能标记已校对 / Empty text cannot be reviewed")
            state = self._read_state()
            if reviewed:
                state["reviewed"][item["name"]] = version
            else:
                state["reviewed"].pop(item["name"], None)
            self._persist(state)
            return self.item(ident)

    def position(self, ident):
        with self.lock:
            state = self._read_state()
            state["last_name"] = self.name(ident)
            self._persist(state)


class ProofreadSession:
    def __init__(self, root, asr_service):
        self.root = Path(root)
        self.asr = asr_service
        self.token = secrets.token_urlsafe(32)
        self.library = Library()
        self.project = ""
        self.language = "ZH"
        self.role = "training"
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.last_access = time.monotonic()
        self.job = self._empty_job()

    @staticmethod
    def _empty_job():
        return dict(running=False, total=0, completed=0, succeeded=0, skipped=0,
                    failed=0, current="", error="", cancelled=False)

    def select(self, directory, project="", language="ZH", role="training"):
        if role not in ("training", "validation"):
            raise ValueError("无效的语料类型 / Invalid dataset role")
        with self.lock:
            if self.job["running"]:
                raise Conflict("请先停止批量识别 / Stop batch recognition before switching folders")
            candidate = Library(directory) if directory else Library()
            self.library = candidate
            self.project = str(project or "")
            self.language = str(language or "ZH")
            self.role = role
            self.job = self._empty_job()
            return self.snapshot()

    def asr_status(self):
        with self.lock:
            return {**self.asr.status(), "job": dict(self.job),
                    "directory_running": directory_asr_running(self.library.directory)}

    def snapshot(self):
        with self.lock:
            return {**self.library.session(), "token": self.token, "project": self.project,
                    "language": self.language, "role": self.role, "asr": self.asr_status()}

    def start_batch(self):
        with self.lock:
            if self.job["running"]:
                raise Conflict("识别任务正在运行 / Recognition is already running")
            if self.library.directory is None:
                raise ValueError("请先选择目录 / Select a directory first")
            status = self.asr.status()
            if not status["available"]:
                raise ValueError(status["message"])
            with self.library.lock:
                self.library.rescan()
                ids = list(self.library.files)
                directory = self.library.directory
                key = _directory_key(directory)
                with _BATCHES_LOCK:
                    if any(_overlapping_roots(key, active) for active in _ACTIVE_BATCHES):
                        raise Conflict("此目录或其父子目录已有识别任务 / This directory or an overlapping folder already has an active transcription batch")
                    _ACTIVE_BATCHES.add(key)
                try:
                    self.cancel.clear()
                    self.job = {**self._empty_job(), "running": True, "total": len(ids)}
                    self.thread = threading.Thread(target=self._batch, args=(ids, key), daemon=True)
                    self.thread.start()
                except Exception:
                    self.job["running"] = False
                    with _BATCHES_LOCK:
                        _ACTIVE_BATCHES.discard(key)
                    raise
            return self.asr_status()

    def _batch(self, ids, directory_key):
        try:
            for ident in ids:
                if self.cancel.is_set():
                    break
                with self.library.lock:
                    item = self.library.item(ident)
                    audio = self.library.path(ident)
                with self.lock:
                    self.job["current"] = item["name"]
                if item["text"].strip():
                    with self.lock:
                        self.job["skipped"] += 1
                        self.job["completed"] += 1
                    continue
                text = self.asr.transcribe(audio, cancel_event=self.cancel)
                if self.cancel.is_set():
                    break
                with self.library.lock:
                    # A concurrent edit always wins, including an edit to blank text.
                    latest = self.library.item(ident)
                    if latest["version"] != item["version"] or latest["text"].strip():
                        skipped = True
                    else:
                        self.library.save(ident, text, item["version"], reviewed=False)
                        skipped = False
                with self.lock:
                    self.job["skipped" if skipped else "succeeded"] += 1
                    self.job["completed"] += 1
        except Exception as exc:
            with self.lock:
                if not self.cancel.is_set():
                    self.job["failed"] += 1
                    self.job["completed"] += 1
                    self.job["error"] = str(exc)
        finally:
            with self.lock:
                self.job["running"] = False
                self.job["cancelled"] = self.cancel.is_set()
                with _BATCHES_LOCK:
                    _ACTIVE_BATCHES.discard(directory_key)

    def retranscribe(self, ident, version):
        with self.lock:
            library = self.library
            with library.lock:
                library.check(ident, version)
                audio = library.path(ident)
        text = self.asr.transcribe(audio)
        with self.lock, library.lock:
            if self.library is not library:
                raise Conflict("目录已改变 / Directory changed during recognition")
            library.check(ident, version)
            return {"id": ident, "text": text, "version": version}

    def bound_project(self):
        from .dataset import load_project, validation_settings
        project = self.project
        if not project or Path(project).name != project or project in (".", "..") or "/" in project or "\\" in project:
            raise ValueError("请先选择并保存项目 / Select and save a project first")
        meta = load_project(project, str(self.root / "training"))
        dataset = validation_settings(meta) if self.role == "validation" else meta
        source = dataset.get("source_folder")
        if not source or self.library.directory is None or Path(source).resolve() != self.library.directory:
            if self.role == "validation":
                raise ValueError("请先保存项目的验证语料目录 / Save the selected validation folder first")
            raise ValueError("请先保存项目的当前语料目录 / Save the project with the selected source folder first")
        if meta.get("language", "EN") != self.language:
            raise ValueError("请先保存项目的当前语言 / Save the project with the selected language first")
        return project


def create_proofread_router(root, asr_service, training_running=lambda: False):
    root = Path(root).resolve()
    router = APIRouter(prefix="/proofread")
    sessions = {}
    sessions_lock = threading.Lock()
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}

    def session_for(request):
        token = request.headers.get("X-Session-Token") or request.query_params.get("token")
        with sessions_lock:
            session = sessions.get(token)
        if session is None:
            raise PermissionError("校对页面已失效，请刷新 / Refresh the proofreading page")
        session.last_access = time.monotonic()
        return session

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != request.url.scheme or parsed.netloc != request.headers.get("host"):
                raise PermissionError("请求来源验证失败 / Invalid request origin")
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise PermissionError("请求来源验证失败 / Invalid request origin")

    async def body_for(request):
        raw = await request.body()
        if len(raw) > 2_000_000:
            raise ValueError("请求过大 / Request is too large")
        body = json.loads(raw or b"{}")
        if not isinstance(body, dict):
            raise ValueError("Invalid request body")
        return body

    def error(exc):
        status = 403 if isinstance(exc, PermissionError) else 409 if isinstance(exc, Conflict) else 400
        return JSONResponse({"error": str(exc)}, status_code=status, headers=headers)

    @router.get("/", response_class=HTMLResponse)
    def page():
        html_headers = {**headers, "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; frame-ancestors 'self'"}
        return HTMLResponse((root / "tools/proofread/index.html").read_text(encoding="utf-8"), headers=html_headers)

    @router.post("/api/session")
    def new_session(request: Request):
        try:
            same_origin(request)
            session = ProofreadSession(root, asr_service)
            with sessions_lock:
                now = time.monotonic()
                for token, old in list(sessions.items()):
                    if not old.job["running"] and now - old.last_access > 86400:
                        sessions.pop(token, None)
                sessions[session.token] = session
            return JSONResponse(session.snapshot(), headers=headers)
        except Exception as exc:
            return error(exc)

    @router.get("/api/{operation:path}")
    def get_operation(operation: str, request: Request):
        try:
            same_origin(request)
            session = session_for(request)
            library = session.library
            ident = request.query_params.get("id")
            if operation == "session":
                with session.lock, session.library.lock:
                    session.library.rescan()
                    result = session.snapshot()
            elif operation == "item":
                result = library.item(ident)
            elif operation == "asr/status":
                result = session.asr_status()
            elif operation == "manifest/status":
                from .dataset import dataset_manifest_status
                with session.lock, DATASET_LOCK:
                    try:
                        project = session.bound_project()
                        result = dataset_manifest_status(project, root=str(root / "training"), role=session.role)
                        result["available"] = True
                    except ValueError as exc:
                        result = {"available": False, "stale": True, "message": str(exc), "manifest": ""}
                    result["role"] = session.role
                    result["training_running"] = bool(training_running())
                    result["asr_running"] = directory_asr_running(session.library.directory)
            elif operation == "audio":
                path = library.path(ident)
                if path.suffix.lower() not in {".wav", ".mp3", ".ogg", ".flac"}:
                    stamp = path.stat()
                    key = hashlib.sha256(f"{path}:{stamp.st_size}:{stamp.st_mtime_ns}".encode()).hexdigest()
                    preview = root / ".runtime/proofread/preview" / (key + ".wav")
                    with _PREVIEW_LOCK:
                        if not preview.exists():
                            preview.parent.mkdir(parents=True, exist_ok=True)
                            ffmpeg = root / ".runtime/bin/ffmpeg.exe"
                            temp = preview.with_suffix(".tmp.wav")
                            try:
                                subprocess.run([str(ffmpeg), "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_s16le", str(temp)],
                                               check=True, capture_output=True, timeout=60,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                                os.replace(temp, preview)
                            finally:
                                temp.unlink(missing_ok=True)
                    path = preview
                return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "audio/wav", headers=headers)
            else:
                return JSONResponse({"error": "Not found"}, status_code=404)
            return JSONResponse(result, headers=headers)
        except Exception as exc:
            return error(exc)

    @router.post("/api/{operation:path}")
    async def post_operation(operation: str, request: Request):
        try:
            same_origin(request)
            session = session_for(request)
            body = await body_for(request)
            # Blocking filesystem/model work belongs in a worker, not the ASGI loop.
            from starlette.concurrency import run_in_threadpool
            def dispatch():
                library = session.library
                if operation == "select":
                    return session.select(body.get("directory", ""), body.get("project", ""),
                                          body.get("language", "ZH"), body.get("role", "training"))
                if operation == "save":
                    return library.save(body["id"], body["text"], body["version"], body.get("reviewed") is True)
                if operation == "review":
                    return library.review(body["id"], body.get("reviewed") is True, body["version"])
                if operation == "position":
                    library.position(body["id"])
                    return {}
                if operation == "asr/start":
                    return session.start_batch()
                if operation == "asr/stop":
                    session.cancel.set()
                    return session.asr_status()
                if operation == "asr/item":
                    return session.retranscribe(body["id"], body["version"])
                if operation == "manifest/update":
                    from .dataset import dataset_manifest_status, refresh_project_manifest
                    with session.lock, DATASET_LOCK:
                        role = session.role
                        if body.get("role", role) != role:
                            raise Conflict("校对语料已切换 / The selected dataset has changed")
                        if training_running():
                            raise Conflict("训练运行中，不能更新清单 / Stop training before updating the manifest")
                        if directory_asr_running(session.library.directory):
                            raise Conflict("识别运行中，请完成后更新清单 / Finish recognition before updating the manifest")
                        project = session.bound_project()
                        meta = refresh_project_manifest(project, root=str(root / "training"), role=role)
                        status = dataset_manifest_status(project, root=str(root / "training"), role=role)
                        return {"meta": meta, "status": {**status, "role": role}}
                raise ValueError("Unknown operation")
            return JSONResponse(await run_in_threadpool(dispatch), headers=headers)
        except Exception as exc:
            return error(exc)

    return router
