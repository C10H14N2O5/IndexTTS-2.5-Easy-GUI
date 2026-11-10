"""Workflow status messages (replaces the 1.x single global status line).

``publish(phase, message, level, detail)`` keeps a short history for the
activity center and broadcasts a ``status`` event; the browser shows ``success``
and ``error`` levels as toasts. Phase/message/detail may be plain English
templates or :class:`Message` objects.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from .i18n import Message, msg, raw

LEVELS = ("info", "working", "success", "error")


def _as_message(value: Any) -> Message | None:
    if value is None or value == "":
        return None
    if isinstance(value, Message):
        return value
    return raw(value)


class StatusLog:
    def __init__(self, publish: Callable[[str, Any], None] | None = None, limit: int = 60):
        self._items: deque[dict] = deque(maxlen=limit)
        self._lock = threading.Lock()
        self._publish = publish or (lambda _type, _data: None)
        self._seq = 0

    def bind(self, publish: Callable[[str, Any], None]) -> None:
        self._publish = publish

    def publish(self, phase: str | Message, message: str | Message, level: str = "info", detail: Any = "") -> dict:
        with self._lock:
            self._seq += 1
            item = {
                "id": self._seq,
                "time": time.time(),
                "phase": phase if isinstance(phase, Message) else msg(str(phase)),
                "message": _as_message(message) or msg("Working."),
                "level": level if level in LEVELS else "info",
                "detail": _as_message(detail),
            }
            self._items.append(item)
        self._publish("status", item)
        return item

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._items)


STATUS = StatusLog()


def publish(phase, message, level="info", detail="") -> dict:
    from .state import current
    try:
        log = current().status
    except RuntimeError:
        log = STATUS
    return log.publish(phase, message, level, detail)
