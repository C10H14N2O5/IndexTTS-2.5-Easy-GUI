"""Cooperative background jobs with one atomic GPU/training/export lease.

``fn(ctx)`` runs on a daemon thread. Cancellation is checked between chunks,
turns or clips; an in-flight CUDA kernel is never forcibly interrupted. Job
records are detached snapshots so queued SSE events cannot mutate afterwards.
"""
from __future__ import annotations

import contextvars
import copy
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Any, Callable

from .errors import Busy, NotFound, ServiceError
from .i18n import Message, msg, raw

KEEP_FINISHED = 50
PROGRESS_INTERVAL = 0.1


class JobCancelled(Exception):
    """Raised at a cooperative stop point."""


class Job:
    def __init__(self, kind: str, title: Message, *, gpu: bool, cancellable: bool, meta: dict | None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.gpu = gpu
        self.cancellable = cancellable
        self.meta = copy.deepcopy(meta or {})
        self.state = "running"
        self.progress: float | None = None
        self.stage: Message | None = msg("Starting…")
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.result: Any = None
        self.error: dict | None = None
        self.cancel_event = threading.Event()
        self._last_publish = 0.0
        self._lock = threading.RLock()

    def to_dict(self) -> dict:
        with self._lock:
            return copy.deepcopy({
                "id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
                "progress": self.progress, "stage": self.stage, "started_at": self.started_at,
                "finished_at": self.finished_at, "result": self.result, "error": self.error,
                "cancellable": self.cancellable and self.state == "running",
                "cancel_requested": self.cancel_event.is_set(), "gpu": self.gpu, "meta": self.meta,
            })


class JobContext:
    def __init__(self, manager: "JobManager", job: Job):
        self._manager = manager
        self.job = job

    @property
    def cancel_requested(self) -> bool:
        return self.job.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelled()

    def progress(self, fraction: float | None, stage: str | Message | None = None, *args) -> None:
        with self.job._lock:
            self.job.progress = None if fraction is None else max(0.0, min(1.0, float(fraction)))
            if stage is not None and not self.cancel_requested:
                self.job.stage = stage if isinstance(stage, Message) else msg(stage, *args)
        self._manager._publish(self.job, throttle=True)

    def update_meta(self, **values) -> None:
        with self.job._lock:
            self.job.meta.update(copy.deepcopy(values))
        self._manager._publish(self.job)


class JobManager:
    def __init__(self, publish: Callable[[str, Any], None] | None = None,
                 on_finished: Callable[[Job], None] | None = None):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._publish_event = publish or (lambda _type, _data: None)
        self._on_finished = on_finished
        self._closed = False

    def _publish(self, job: Job, throttle: bool = False) -> None:
        with job._lock:
            now = time.monotonic()
            if throttle and now - job._last_publish < PROGRESS_INTERVAL:
                return
            job._last_publish = now
            record = job.to_dict()
        self._publish_event("job", record)

    def running_gpu_job(self) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs.values() if job.gpu and job.state == "running"), None)

    def submit(self, kind: str, title: str | Message, fn: Callable[[JobContext], Any], *, gpu: bool = False,
               cancellable: bool = True, meta: dict | None = None) -> Job:
        from indextts_training.activity import acquire, release

        title = title if isinstance(title, Message) else msg(title)
        with self._lock:
            if self._closed:
                raise Busy("The server is shutting down. Please reconnect.")
            lease = None
            if gpu:
                busy = self.running_gpu_job()
                if busy is not None:
                    raise Busy("The GPU is busy with another task: {0}. Wait for it to finish or cancel it.",
                               busy.title, extra={"busy_job": busy.to_dict()})
                try:
                    # Acquire before returning 202, not inside the worker: training
                    # and export use this same lease so a check/start race is impossible.
                    lease = acquire("inference")
                except RuntimeError as exc:
                    raise Busy(raw(str(exc))) from exc
            job = Job(kind, title, gpu=gpu, cancellable=cancellable, meta=meta)
            self._jobs[job.id] = job
            ctx = JobContext(self, job)
            call_context = contextvars.copy_context()
            thread = threading.Thread(target=call_context.run, args=(self._run, job, ctx, fn, lease),
                                      name=f"job-{kind}-{job.id}", daemon=True)
            self._threads[job.id] = thread
            try:
                self._publish(job)
                thread.start()
            except BaseException:
                self._jobs.pop(job.id, None)
                self._threads.pop(job.id, None)
                if lease is not None:
                    release(lease)
                raise
            self._trim()
            return job

    def _run(self, job: Job, context: JobContext, fn: Callable[[JobContext], Any], lease: str | None) -> None:
        from indextts_training.activity import release

        result, error = None, None
        outcome = "failed"
        try:
            context.check_cancelled()
            result = fn(context)
            outcome = "cancelled" if (job.cancel_event.is_set() and isinstance(result, dict)
                                      and result.get("stopped")) else "succeeded"
        except JobCancelled:
            outcome = "cancelled"
        except ServiceError as exc:
            error = {"message": exc.message, "detail": exc.detail, "code": exc.code}
        except Exception as exc:
            traceback.print_exc()
            error = {"message": raw(f"{type(exc).__name__}: {exc}"), "detail": type(exc).__name__, "code": "internal"}
        finally:
            with self._lock:
                if lease is not None:
                    release(lease)
                with job._lock:
                    job.result, job.error = result, error
                    job.finished_at = time.time()
                    job.progress = 1.0 if outcome == "succeeded" else job.progress
                    job.stage = msg({"succeeded": "Done", "cancelled": "Stopped", "failed": "Failed"}[outcome])
                    job.state = outcome
                self._publish(job)
                self._trim()
            if self._on_finished is not None:
                try:
                    self._on_finished(job)
                except Exception:
                    traceback.print_exc()
            with self._lock:
                self._threads.pop(job.id, None)

    def _trim(self) -> None:
        finished = [key for key, job in self._jobs.items() if job.state != "running"]
        for key in finished[:-KEEP_FINISHED]:
            self._jobs.pop(key, None)

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFound("Task not found. It may have finished before the server restarted.")
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [job.to_dict() for job in reversed(self._jobs.values())]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with job._lock:
            if job.state == "running":
                if not job.cancellable:
                    raise ServiceError("This task cannot be cancelled.", status=409, code="conflict")
                job.cancel_event.set()
                job.stage = msg("Stopping after the current step…")
                self._publish(job)
        return job

    def shutdown(self, timeout: float = 3.0) -> None:
        """Reject new jobs, request cooperative cancellation and join with a bounded wait."""
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.state == "running":
                    job.cancel_event.set()
            threads = list(self._threads.values())
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
