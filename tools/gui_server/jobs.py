"""Background jobs for GPU and ASR work (synthesis, dialogue, ABX clips, ASR).

A job runs ``fn(ctx)`` on a daemon thread. ``ctx.progress()`` reports a fraction
and a translatable stage; ``ctx.check_cancelled()`` is the cooperative stop
point (between chunks, dialogue turns and ABX clips). Jobs that need the GPU
share one slot: submitting a second GPU job raises ``Busy`` with ``busy_job``.
Jobs live in memory only; after a restart the browser marks unknown running
jobs as interrupted.
"""
from __future__ import annotations

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
    """Raised by ``JobContext.check_cancelled`` when a stop was requested."""


class Job:
    def __init__(self, kind: str, title: Message, *, gpu: bool, cancellable: bool, meta: dict | None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.gpu = gpu
        self.cancellable = cancellable
        self.meta = dict(meta or {})
        self.state = "running"
        self.progress: float | None = None
        self.stage: Message | None = msg("Starting…")
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.result: Any = None
        self.error: dict | None = None
        self.cancel_event = threading.Event()
        self._last_publish = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
            "progress": self.progress, "stage": self.stage, "started_at": self.started_at,
            "finished_at": self.finished_at, "result": self.result, "error": self.error,
            "cancellable": self.cancellable and self.state == "running",
            "cancel_requested": self.cancel_event.is_set(), "gpu": self.gpu, "meta": self.meta,
        }


class JobContext:
    def __init__(self, manager: "JobManager", job: Job):
        self._manager = manager
        self.job = job

    @property
    def cancel_requested(self) -> bool:
        return self.job.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.job.cancel_event.is_set():
            raise JobCancelled()

    def progress(self, fraction: float | None, stage: str | Message | None = None, *args) -> None:
        job = self.job
        if fraction is not None:
            job.progress = max(0.0, min(1.0, float(fraction)))
        if stage is not None:
            job.stage = stage if isinstance(stage, Message) else msg(stage, *args)
        self._manager._publish(job, throttle=True)

    def update_meta(self, **values) -> None:
        self.job.meta.update(values)
        self._manager._publish(self.job)


class JobManager:
    def __init__(self, publish: Callable[[str, Any], None] | None = None,
                 on_finished: Callable[[Job], None] | None = None):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._publish_event = publish or (lambda _type, _data: None)
        self._on_finished = on_finished

    def _publish(self, job: Job, throttle: bool = False) -> None:
        now = time.monotonic()
        if throttle and now - job._last_publish < PROGRESS_INTERVAL:
            return
        job._last_publish = now
        self._publish_event("job", job.to_dict())

    def running_gpu_job(self) -> Job | None:
        with self._lock:
            return next((job for job in self._jobs.values() if job.gpu and job.state == "running"), None)

    def submit(self, kind: str, title: str | Message, fn: Callable[[JobContext], Any], *, gpu: bool = False,
               cancellable: bool = True, meta: dict | None = None) -> Job:
        title = title if isinstance(title, Message) else msg(title)
        with self._lock:
            if gpu:
                busy = self.running_gpu_job()
                if busy is not None:
                    raise Busy("The GPU is busy with another task: {0}. Wait for it to finish or cancel it.",
                               busy.title, extra={"busy_job": busy.to_dict()})
            job = Job(kind, title, gpu=gpu, cancellable=cancellable, meta=meta)
            self._jobs[job.id] = job
            self._trim()
        context = JobContext(self, job)
        thread = threading.Thread(target=self._run, args=(job, context, fn), name=f"job-{kind}-{job.id}", daemon=True)
        self._publish(job)
        thread.start()
        return job

    def _run(self, job: Job, context: JobContext, fn: Callable[[JobContext], Any]) -> None:
        try:
            result = fn(context)
            job.result = result
            job.state = "cancelled" if job.cancel_event.is_set() and (isinstance(result, dict) and result.get("stopped")) else "succeeded"
            job.progress = 1.0 if job.state == "succeeded" else job.progress
            job.stage = msg("Done") if job.state == "succeeded" else msg("Stopped")
        except JobCancelled:
            job.state = "cancelled"
            job.stage = msg("Stopped")
        except ServiceError as exc:
            job.state = "failed"
            job.error = {"message": exc.message, "detail": exc.detail, "code": exc.code}
            job.stage = msg("Failed")
        except Exception as exc:  # noqa: BLE001 - report every failure to the user
            traceback.print_exc()
            job.state = "failed"
            job.error = {"message": raw(f"{type(exc).__name__}: {exc}"), "detail": type(exc).__name__, "code": "internal"}
            job.stage = msg("Failed")
        finally:
            job.finished_at = time.time()
            self._publish(job)
            if self._on_finished is not None:
                try:
                    self._on_finished(job)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()

    def _trim(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if job.state != "running"]
        for job_id in finished[:-KEEP_FINISHED] if len(finished) > KEEP_FINISHED else []:
            self._jobs.pop(job_id, None)

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
        if job.state == "running":
            if not job.cancellable:
                raise ServiceError("This task cannot be cancelled.", status=409, code="conflict")
            job.cancel_event.set()
            job.stage = msg("Stopping after the current step…")
            self._publish(job)
        return job
