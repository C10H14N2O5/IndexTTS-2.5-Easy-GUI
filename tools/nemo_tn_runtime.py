from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".runtime" / "nemo-tn"
WORKER = ROOT / "tools" / "nemo_tn_worker.py"


def _runtime_python() -> Path:
    configured = os.environ.get("INDEXTTS_NEMO_TN_PYTHON")
    return Path(configured) if configured else DEFAULT_ENV / "python.exe"


class NemoTextNormalizationClient:
    """Persistent JSON-RPC bridge to the isolated Windows NeMo environment."""

    def __init__(self):
        self.python = _runtime_python()
        self._process: Optional[subprocess.Popen] = None
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._reported_unavailable = False

    @property
    def available(self) -> bool:
        return self.python.is_file() and WORKER.is_file()

    def _read_stdout(self, process):
        try:
            for line in process.stdout:
                try:
                    self._responses.put(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.debug("Ignoring malformed NeMo worker response: %s", exc)
        finally:
            self._responses.put(None)

    @staticmethod
    def _read_stderr(process):
        try:
            for line in process.stderr:
                line = line.strip()
                if line:
                    logger.debug("[nemo-tn] %s", line)
        except Exception:
            pass

    def _start(self) -> bool:
        if not self.available:
            if not self._reported_unavailable:
                logger.warning(
                    "Isolated NeMo Windows runtime not found at %s; using the safe fallback.",
                    self.python,
                )
                self._reported_unavailable = True
            return False

        if self._process is not None and self._process.poll() is None:
            return True

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["INDEXTTS_NEMO_TN_CACHE"] = str(ROOT / ".runtime" / "nemo-tn-cache")
        # Launching the interpreter by absolute path does not run conda's
        # activation hook. Add the DLL/script directories explicitly so
        # Pynini/OpenFst resolve correctly without changing the main process.
        env["CONDA_PREFIX"] = str(self.python.parent)
        env["PATH"] = os.pathsep.join(
            [
                str(self.python.parent),
                str(self.python.parent / "Scripts"),
                str(self.python.parent / "Library" / "bin"),
                os.environ.get("PATH", ""),
            ]
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self._responses = queue.Queue()
            self._process = subprocess.Popen(
                [str(self.python), "-u", str(WORKER)],
                cwd=str(ROOT),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(self._process,),
                name="nemo-tn-stdout",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(self._process,),
                name="nemo-tn-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            return True
        except Exception as exc:
            logger.warning("Could not start isolated NeMo runtime: %s", exc)
            self._stop()
            return False

    def _stop(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _request(self, payload, timeout: float = 90.0):
        with self._lock:
            if not self._start() or self._process is None:
                return None
            try:
                self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                response = self._responses.get(timeout=timeout)
                if response is None:
                    raise RuntimeError("NeMo worker exited")
                return response
            except Exception as exc:
                logger.warning("Isolated NeMo request failed: %s", exc)
                self._stop()
                return None

    def ping(self) -> bool:
        response = self._request({"op": "ping"}, timeout=20.0)
        return bool(response and response.get("ok"))

    def normalize(self, text: str, lang: str, input_case: str = "cased") -> Optional[str]:
        response = self._request(
            {
                "op": "normalize",
                "text": text,
                "lang": lang,
                "input_case": input_case,
            }
        )
        if not response or not response.get("ok"):
            if response:
                logger.warning("NeMo could not normalize lang=%s: %s", lang, response.get("error"))
            return None
        return response.get("text")

    def close(self):
        with self._lock:
            if self._process is not None:
                self._request({"op": "shutdown"}, timeout=2.0)
                self._stop()


_CLIENT: Optional[NemoTextNormalizationClient] = None


def get_nemo_tn_client() -> NemoTextNormalizationClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = NemoTextNormalizationClient()
        atexit.register(_CLIENT.close)
    return _CLIENT


if __name__ == "__main__":
    client = get_nemo_tn_client()
    if not client.ping():
        raise SystemExit(1)
    print("isolated NeMo text-normalization runtime OK")
