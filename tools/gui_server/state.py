"""Process-wide application state shared by routes and services.

``create_app`` builds one :class:`AppState` and installs it with ``install``;
routes call ``current()``. Snapshot providers feed both the first SSE event and
the 1-second change ticker (``training``, ``export``, ``system`` ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .console import CONSOLE, ConsoleBuffer
from .events import EventHub
from .jobs import JobManager
from .status import STATUS, StatusLog


@dataclass
class SnapshotProvider:
    name: str
    fn: Callable[[], Any]
    interval: float = 1.0  # seconds between change checks; 0 = snapshot only
    publish_unchanged: bool = False


@dataclass
class AppState:
    fake: bool
    hub: EventHub
    jobs: JobManager
    status: StatusLog
    console: ConsoleBuffer
    asr: Any = None
    export: Any = None
    training: Any = None
    providers: dict[str, SnapshotProvider] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def register_snapshot(self, name: str, fn: Callable[[], Any], interval: float = 1.0,
                          publish_unchanged: bool = False) -> None:
        self.providers[name] = SnapshotProvider(name, fn, interval, publish_unchanged)

    def publish(self, event_type: str, data: Any) -> None:
        self.hub.publish(event_type, data)


_STATE: AppState | None = None


def build(fake: bool) -> AppState:
    hub = EventHub()
    STATUS.bind(hub.publish)
    jobs = JobManager(publish=hub.publish)
    return AppState(fake=fake, hub=hub, jobs=jobs, status=STATUS, console=CONSOLE)


def install(state: AppState) -> None:
    global _STATE
    _STATE = state


def current() -> AppState:
    if _STATE is None:
        raise RuntimeError("GUI application state is not initialised; call create_app() first.")
    return _STATE
