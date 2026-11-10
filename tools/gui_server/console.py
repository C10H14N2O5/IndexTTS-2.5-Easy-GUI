"""stdout/stderr mirror for the in-app console drawer (moved from easy_gui.py).

Lines keep a stable id. A carriage-return progress line (tqdm: ``%`` or
``it/s``) overwrites the previous line and keeps its id, so the browser can
replace it in place. The server classifies each line's level with the same
rules the 1.x console used for colours.
"""
from __future__ import annotations

import re
import sys
import threading
from collections import deque

MAX_LINES = 1400
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def classify(line: str) -> str:
    low = line.lower()
    if "traceback" in low or "error" in low or "exception" in low or "failed" in low or "fatal" in low:
        return "error"
    if "warn" in low or "fallback" in low:
        return "warn"
    if "[ui-progress]" in low or "%" in line or "it/s" in low:
        return "progress"
    if "[ui]" in low or "loading" in low or "starting inference" in low:
        return "info"
    if "saved" in low or "complete" in low or "done" in low or "ready" in low:
        return "success"
    if "gpt_gen_time" in low or "s2mel_time" in low or "bigvgan_time" in low or "rtf:" in low:
        return "metric"
    return "plain"


def _clean(line: str) -> str:
    return _ANSI.sub("", str(line)).rstrip()


class ConsoleBuffer:
    def __init__(self, max_lines: int = MAX_LINES):
        self.lock = threading.Lock()
        self.lines: deque[list] = deque(maxlen=max_lines)  # [id, text]
        self.current = ""
        self.overwrite = False
        self.next_id = 1
        self.dirty: set[int] = set()
        self.current_dirty = False

    def _append(self, text: str) -> None:
        self.lines.append([self.next_id, text])
        self.dirty.add(self.next_id)
        self.next_id += 1

    def write(self, data: str) -> None:
        with self.lock:
            for ch in data:
                if ch == "\r":
                    current = _clean(self.current)
                    if current.strip():
                        if self.lines and ("%" in current or "it/s" in current):
                            self.lines[-1][1] = current
                            self.dirty.add(self.lines[-1][0])
                        else:
                            self._append(current)
                    self.current = ""
                    self.overwrite = True
                    self.current_dirty = True
                    continue
                if ch == "\n":
                    line = _clean(self.current)
                    if line.strip():
                        self._append(line)
                    self.current = ""
                    self.overwrite = False
                    self.current_dirty = True
                    continue
                if self.overwrite:
                    self.current = ""
                    self.overwrite = False
                self.current += ch
                self.current_dirty = True

    @staticmethod
    def _line(entry) -> dict:
        return {"id": entry[0], "text": entry[1], "level": classify(entry[1])}

    def snapshot(self, limit: int = 400) -> dict:
        with self.lock:
            lines = [self._line(entry) for entry in list(self.lines)[-limit:]]
            current = _clean(self.current)
        return {"lines": lines, "current": current if current.strip() else ""}

    def drain(self) -> dict | None:
        """Return lines changed since the last drain, or None when nothing changed."""
        with self.lock:
            if not self.dirty and not self.current_dirty:
                return None
            oldest = self.lines[0][0] if self.lines else self.next_id
            changed = [self._line(entry) for entry in self.lines if entry[0] in self.dirty and entry[0] >= oldest]
            self.dirty.clear()
            self.current_dirty = False
            current = _clean(self.current)
        return {"lines": changed, "current": current if current.strip() else ""}

    def clear(self) -> None:
        with self.lock:
            self.lines.clear()
            self.dirty.clear()
            self.current = ""
            self.current_dirty = True


CONSOLE = ConsoleBuffer()


class _Mirror:
    _indextts_console_mirror = True

    def __init__(self, stream, buffer: ConsoleBuffer):
        self.stream = stream
        self.buffer = buffer
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        if data:
            self.buffer.write(str(data))
        return self.stream.write(data) if self.stream is not None else len(data or "")

    def flush(self):
        return self.stream.flush() if self.stream is not None else None

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def install(buffer: ConsoleBuffer = CONSOLE) -> None:
    if getattr(sys.stdout, "_indextts_console_mirror", False):
        return
    sys.stdout = _Mirror(sys.stdout, buffer)
    sys.stderr = _Mirror(sys.stderr, buffer)
