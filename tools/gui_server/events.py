"""In-process event hub that fans server events out to SSE subscribers.

Publishers may run on any thread. Each event is ``{"type": str, "data": Any}``
where data may contain :class:`~tools.gui_server.i18n.Message` objects; they are
rendered per subscriber language when the event is written to the stream.

Event types: ``snapshot`` (first event of every stream), ``job``, ``status``,
``console``, ``training``, ``export``, ``system``, ``settings``, ``heartbeat``.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
from typing import Any, AsyncIterator, Callable

from .i18n import render

HEARTBEAT_SECONDS = 15.0
QUEUE_LIMIT = 2000


class _Subscriber:
    __slots__ = ("queue", "language", "dropped")

    def __init__(self, language: str):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self.language = language
        self.dropped = False


class EventHub:
    def __init__(self) -> None:
        self._subscribers: set[_Subscriber] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._seq = itertools.count(1)

    # -- lifecycle -------------------------------------------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._closed = False

    def close(self) -> None:
        """Ask every open stream to finish (used on shutdown)."""
        self._closed = True
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout, None)
        except RuntimeError:
            pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -- publishing ------------------------------------------------------
    def publish(self, event_type: str, data: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed() or self._closed:
            return
        event = {"id": next(self._seq), "type": event_type, "data": data}
        try:
            loop.call_soon_threadsafe(self._fanout, event)
        except RuntimeError:  # loop shutting down
            pass

    def _fanout(self, event: dict | None) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled browser tab: drop it; EventSource reconnects and gets a fresh snapshot.
                subscriber.dropped = True

    # -- streaming -------------------------------------------------------
    @staticmethod
    def encode(event: dict, language: str) -> str:
        payload = json.dumps({"type": event["type"], "data": render(event["data"], language)},
                             ensure_ascii=False, separators=(",", ":"), default=str)
        return f"id: {event['id']}\ndata: {payload}\n\n"

    async def stream(self, language: str, snapshot: Callable[[], Any],
                     is_disconnected: Callable[[], Any] | None = None) -> AsyncIterator[str]:
        subscriber = _Subscriber(language)
        with self._lock:
            self._subscribers.add(subscriber)
        try:
            yield "retry: 2000\n\n"
            yield self.encode({"id": next(self._seq), "type": "snapshot", "data": snapshot()}, language)
            while not self._closed:
                try:
                    event = await asyncio.wait_for(subscriber.queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    if is_disconnected is not None and await is_disconnected():
                        break
                    yield self.encode({"id": next(self._seq), "type": "heartbeat", "data": {"time": time.time()}}, language)
                    continue
                if event is None or subscriber.dropped:
                    break
                yield self.encode(event, language)
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)
