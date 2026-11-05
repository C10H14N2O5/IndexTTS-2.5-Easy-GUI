"""Local, serialized SenseVoice GGUF transcription without a Python ML runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid


_ASR_LOCK = threading.Lock()
MODEL_SHA256 = "4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5"


class AsrError(RuntimeError):
    """A failed request; any available diagnostic log is included in the message."""


class AsrCancelled(AsrError):
    pass


class AsrTimeout(AsrError):
    pass


def _cancelled(event):
    return event is not None and event.is_set()


def _decode_transcript(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="strict")
    text = re.sub(r"<\|[^<>]*\|>", "", text)
    text = " ".join(text.split()).strip()
    if not text:
        raise AsrError("识别结果为空；请检查音频是否包含清晰语音。")
    if "\ufffd" in text or re.search(
        r"\[sensevoice\]|-->|gguf_init|ggml_|<\||^usage:|^error:", text, re.IGNORECASE
    ) or any(ord(char) < 32 for char in text):
        raise AsrError("识别输出包含诊断信息或无效字符，未作为正文保存。")
    return text


class AsrService:
    def __init__(self, root, timeout=180.0):
        self.root = Path(root).resolve()
        self.executable = self.root / ".runtime/funasr/llama-funasr-sensevoice.exe"
        self.model = self.root / "models/asr/sensevoice-small-q8.gguf"
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("ASR timeout must be positive")

    def _ffmpeg(self):
        bundled = self.root / ".runtime/bin/ffmpeg.exe"
        if bundled.is_file():
            return str(bundled)
        try:
            import imageio_ffmpeg
            candidate = imageio_ffmpeg.get_ffmpeg_exe()
            if Path(candidate).is_file():
                return candidate
        except (ImportError, RuntimeError, OSError):
            pass
        raise AsrError("缺少 FFmpeg；请先运行项目安装程序。")

    def status(self):
        missing = [str(path) for path in (self.executable, self.model)
                   if not path.is_file() or path.stat().st_size == 0]
        message = "SenseVoiceSmall Q8 · CPU · 自动识别语言"
        if missing:
            message = ("ASR 尚未安装，缺少：" + ", ".join(missing)
                       + "。请使用 tools/import_funasr_runtime.py 导入已有运行时。")
        else:
            try:
                self._ffmpeg()
            except AsrError as exc:
                missing.append("ffmpeg")
                message = str(exc)
        return {"available": not missing, "message": message, "backend": "cpu",
                "model": str(self.model), "executable": str(self.executable)}

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def _run(self, args, log_dir, label, deadline, cancel_event):
        if _cancelled(cancel_event):
            raise AsrCancelled("识别已停止。")
        if time.monotonic() >= deadline:
            raise AsrTimeout(f"识别超过 {self.timeout:g} 秒，已停止。")
        stdout_path = log_dir / f"{label}.stdout.bin"
        stderr_path = log_dir / f"{label}.stderr.bin"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                args, cwd=self.root, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            try:
                while process.poll() is None:
                    if _cancelled(cancel_event):
                        raise AsrCancelled("识别已停止。")
                    if time.monotonic() >= deadline:
                        raise AsrTimeout(f"识别超过 {self.timeout:g} 秒，已停止。")
                    if cancel_event is not None:
                        cancel_event.wait(0.1)
                    else:
                        time.sleep(0.1)
                if _cancelled(cancel_event):
                    raise AsrCancelled("识别已停止。")
                if time.monotonic() >= deadline:
                    raise AsrTimeout(f"识别超过 {self.timeout:g} 秒，已停止。")
                if process.returncode:
                    raise AsrError(f"{label} 执行失败，退出码 {process.returncode}。")
            finally:
                self._stop(process)
        return stdout_path.read_bytes()

    def transcribe(self, audio_path, cancel_event=None):
        """Return text only. Never create or change a corpus TXT/audio file."""
        if _cancelled(cancel_event):
            raise AsrCancelled("识别已停止。")
        source = Path(audio_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise AsrError("请选择有效的音频文件。")
        state = self.status()
        if not state["available"]:
            raise AsrError(state["message"])
        while not _ASR_LOCK.acquire(timeout=0.1):
            if _cancelled(cancel_event):
                raise AsrCancelled("等待中的识别任务已停止。")
        work = None
        try:
            if _cancelled(cancel_event):
                raise AsrCancelled("识别已停止。")
            ident = uuid.uuid4().hex
            work = self.root / ".runtime/asr/work" / ident
            log_dir = self.root / ".runtime/asr/logs" / ident
            work.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            audio = work / "input.wav"
            metadata = {"source": str(source), "backend": "cpu", "model": str(self.model),
                        "timeout_seconds": self.timeout, "status": "running"}
            start = time.monotonic()
            deadline = start + self.timeout
            try:
                self._run([self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
                           "-i", str(source), "-vn", "-ar", "16000", "-ac", "1",
                           "-c:a", "pcm_s16le", str(audio)],
                          log_dir, "ffmpeg", deadline, cancel_event)
                # Native audio/model fopen cannot reliably consume Unicode argv on
                # Windows. The process CWD may contain Unicode; these argv do not.
                raw = self._run([str(self.executable), "-m", self.model.relative_to(self.root).as_posix(),
                                 "-a", audio.relative_to(self.root).as_posix(), "--backend", "cpu"],
                                log_dir, "sensevoice", deadline, cancel_event)
                result = _decode_transcript(raw)
                if _cancelled(cancel_event):
                    raise AsrCancelled("识别已停止。")
                metadata.update(status="ok", characters=len(result))
                return result
            except Exception as exc:
                metadata.update(status="cancelled" if isinstance(exc, AsrCancelled) else "failed",
                                error=str(exc))
                error_type = type(exc) if isinstance(exc, AsrError) else AsrError
                raise error_type(f"{exc}\n诊断日志：{log_dir}") from exc
            finally:
                metadata["elapsed_seconds"] = round(time.monotonic() - start, 3)
                try:
                    (log_dir / "request.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                except OSError:
                    pass  # A metadata failure must not conceal the original error.
        finally:
            try:
                if work is not None:
                    # Generated fixed-child path, never a source/corpus directory.
                    work_root = (self.root / ".runtime/asr/work").resolve()
                    if work.resolve().parent == work_root:
                        shutil.rmtree(work, ignore_errors=True)
            finally:
                _ASR_LOCK.release()
