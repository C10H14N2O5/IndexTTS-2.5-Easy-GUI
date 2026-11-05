"""Process-wide exclusion for training and CPU checkpoint export."""
from __future__ import annotations

import threading
import uuid

_LOCK = threading.RLock()
_ACTIVE: tuple[str, str] | None = None


def acquire(kind: str) -> str:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError(f"Cannot start {kind}: {_ACTIVE[0]} is already running.")
        token = uuid.uuid4().hex
        _ACTIVE = (kind, token)
        return token


def release(token: str) -> None:
    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None and _ACTIVE[1] == token:
            _ACTIVE = None
