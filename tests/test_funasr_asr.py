"""CPU-only ASR regression tests with temporary data and lightweight child processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from indextts_training.asr import AsrService, AsrError, AsrCancelled, AsrTimeout, _decode_transcript, _ASR_LOCK
from tools import import_funasr_runtime as importer


class OutputTests(unittest.TestCase):
    def test_utf8_and_model_tags(self):
        raw = "<|zh|><|NEUTRAL|><|Speech|>  原话，保留语气词。\r\n".encode("utf-8")
        self.assertEqual(_decode_transcript(raw), "原话，保留语气词。")

    def test_reject_empty_invalid_utf8_and_diagnostics(self):
        for raw in (b"", b" \n", b"<|nospeech|>", b"[sensevoice] model ready", b"gguf_init failed",
                    b"00:00 --> 00:02", b"<|broken", b"usage: xyz", b"text\x00more"):
            with self.subTest(raw=raw), self.assertRaises(AsrError):
                _decode_transcript(raw)
        with self.assertRaises(UnicodeDecodeError):
            _decode_transcript(b"\xff")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asr_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文 项目"
        self.root.mkdir()
        self.service = AsrService(self.root, timeout=5)
        self.source = self.root / "中文 音频.wav"
        self.source.write_bytes(b"original audio bytes")
        self.sidecar = self.source.with_suffix(".txt")
        self.sidecar.write_text("保留的人工文本", encoding="utf-8")

    def test_missing_runtime_status_does_not_create_folders(self):
        status = self.service.status()
        self.assertFalse(status["available"])
        self.assertEqual(status["backend"], "cpu")
        self.assertIn("import_funasr_runtime.py", status["message"])
        self.assertFalse((self.root / ".runtime").exists())

    def test_unicode_input_uses_unique_ascii_native_args_and_preserves_source(self):
        paths = []
        def run(args, log_dir, label, deadline, event):
            if label == "ffmpeg":
                self.assertEqual(Path(args[args.index("-i") + 1]), self.source)
                self.assertEqual(args[args.index("-ar") + 1], "16000")
                self.assertEqual(args[args.index("-ac") + 1], "1")
                Path(args[-1]).write_bytes(b"temporary PCM16")
                return b""
            self.assertEqual(args[-2:], ["--backend", "cpu"])
            model = args[args.index("-m") + 1]
            audio = args[args.index("-a") + 1]
            self.assertTrue(model.isascii())
            self.assertTrue(audio.isascii())
            self.assertFalse(Path(audio).is_absolute())
            paths.append(audio)
            self.assertTrue((self.root / audio).is_file())
            return "<|zh|>新的识别。".encode("utf-8")
        with patch.object(self.service, "status", return_value={"available": True}), \
             patch.object(self.service, "_ffmpeg", return_value="ffmpeg.exe"), \
             patch.object(self.service, "_run", side_effect=run):
            self.assertEqual(self.service.transcribe(self.source), "新的识别。")
            self.service.transcribe(self.source)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not (self.root / p).exists() for p in paths))
        self.assertEqual(self.source.read_bytes(), b"original audio bytes")
        self.assertEqual(self.sidecar.read_text(encoding="utf-8"), "保留的人工文本")
        logs = list((self.root / ".runtime/asr/logs").glob("*/request.json"))
        self.assertEqual(len(logs), 2)
        self.assertEqual(json.loads(logs[0].read_text(encoding="utf-8"))["status"], "ok")

    def test_failures_retain_log_and_release_lock(self):
        with patch.object(self.service, "status", return_value={"available": True}), \
             patch.object(self.service, "_ffmpeg", return_value="ffmpeg.exe"), \
             patch.object(self.service, "_run", side_effect=[b"", b"[sensevoice] error"]):
            with self.assertRaisesRegex(AsrError, "诊断日志"):
                self.service.transcribe(self.source)
        self.assertFalse(_ASR_LOCK.locked())
        self.assertEqual(list((self.root / ".runtime/asr/work").iterdir()), [])
        record = next((self.root / ".runtime/asr/logs").glob("*/request.json"))
        self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["status"], "failed")

    def test_cancel_before_start(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(AsrCancelled):
            self.service.transcribe(self.source, event)
        self.assertFalse((self.root / ".runtime").exists())

    def test_cancel_while_waiting_for_another_request(self):
        event = threading.Event()
        timer = threading.Timer(0.15, event.set)
        _ASR_LOCK.acquire()
        try:
            timer.start()
            with patch.object(self.service, "status", return_value={"available": True}), \
                 self.assertRaises(AsrCancelled):
                self.service.transcribe(self.source, event)
        finally:
            _ASR_LOCK.release()
            timer.cancel()
        self.assertFalse((self.root / ".runtime").exists())

    def test_child_exit_and_raw_diagnostic_logs(self):
        log = self.root / "logs"
        log.mkdir()
        with self.assertRaisesRegex(AsrError, "退出码 7"):
            self.service._run([sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr);sys.exit(7)"],
                              log, "test", time.monotonic() + 5, None)
        self.assertIn(b"out", (log / "test.stdout.bin").read_bytes())
        self.assertIn(b"err", (log / "test.stderr.bin").read_bytes())

    def test_timeout_terminates_child(self):
        log = self.root / "logs"
        log.mkdir()
        processes = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process
        with patch("indextts_training.asr.subprocess.Popen", side_effect=launch):
            with self.assertRaises(AsrTimeout):
                self.service._run([sys.executable, "-c", "import time;time.sleep(30)"],
                                  log, "test", time.monotonic() + 0.2, None)
        self.assertIsNotNone(processes[0].poll())

    def test_cancel_terminates_child(self):
        log = self.root / "logs"
        log.mkdir()
        event = threading.Event()
        processes = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process
        timer = threading.Timer(0.2, event.set)
        timer.start()
        try:
            with patch("indextts_training.asr.subprocess.Popen", side_effect=launch), \
                 self.assertRaises(AsrCancelled):
                self.service._run([sys.executable, "-c", "import time;time.sleep(30)"],
                                  log, "test", time.monotonic() + 5, event)
        finally:
            timer.cancel()
        self.assertIsNotNone(processes[0].poll())


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asr_import_test_")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.target = Path(self.temp.name) / "target"
        (self.source / "models").mkdir(parents=True)
        (self.source / "llama-funasr-sensevoice.exe").write_bytes(b"MZfixture")
        (self.source / "models/sensevoice-small-q8.gguf").write_bytes(b"GGUFfixture")
        (self.source / "README.md").write_text("notes", encoding="utf-8")
        (self.source / "LICENSE").write_text("license", encoding="utf-8")
        (self.source / "logs").mkdir()
        (self.source / "logs/private.json").write_text("private")
        self.digest = hashlib.sha256(b"GGUFfixture").hexdigest()

    def test_import_idempotent_and_preserves_notices_not_private_data(self):
        with patch.object(importer, "MODEL_SHA256", self.digest):
            first = importer.import_runtime(self.source, self.target)
            second = importer.import_runtime(self.source, self.target)
        self.assertEqual(first, second)
        self.assertEqual((self.target / "models/asr/sensevoice-small-q8.gguf").read_bytes(), b"GGUFfixture")
        self.assertTrue((self.target / ".runtime/funasr/LICENSE").is_file())
        self.assertFalse((self.target / ".runtime/funasr/logs").exists())
        self.assertTrue((self.source / "logs/private.json").is_file())

    def test_bad_model_aborts_before_any_copy(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            importer.import_runtime(self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_different_existing_runtime_requires_replace(self):
        destination = self.target / "models/asr/sensevoice-small-q8.gguf"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing model")
        with patch.object(importer, "MODEL_SHA256", self.digest), self.assertRaisesRegex(ValueError, "--replace"):
            importer.import_runtime(self.source, self.target)
        self.assertEqual(destination.read_bytes(), b"existing model")
        self.assertFalse((self.target / ".runtime").exists())


if __name__ == "__main__":
    unittest.main()
