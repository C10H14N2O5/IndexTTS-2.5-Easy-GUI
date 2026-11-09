"""Proofreading integration checks use temporary corpora only."""
import hashlib
import json
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from indextts_training.proofread import (Conflict, Library, ProofreadSession,
                                       create_proofread_router, directory_asr_running)


class FakeASR:
    def status(self):
        return dict(available=True, message="Ready", backend="cpu", model="test", executable="test")

    def transcribe(self, path, cancel_event=None):
        return "识别正文。"


class CorpusFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="proofread_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "中文 切片"
        self.corpus.mkdir()
        for name in ("10.wav", "2.wav", "1.wav", "子目录/1.wav", "子目录/3.flac"):
            p = self.corpus / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"RIFF" + bytes(range(100)))
        self.target = self.corpus / "1.txt"
        self.target.write_bytes("原文。\n".encode("utf-8-sig"))
        self.library = Library(self.corpus)
        self.ids = {self.library.name(i): i for i in self.library.files}
        self.first = self.ids["1.wav"]


class LibraryTests(CorpusFixture, unittest.TestCase):
    def test_recursive_sort_and_missing(self):
        items = self.library.session()["items"]
        self.assertEqual([i["name"] for i in items], ["1.wav", "2.wav", "10.wav", "子目录/1.wav", "子目录/3.flac"])
        self.assertFalse(items[0]["missing_text"])
        self.assertTrue(items[1]["missing_text"])

    def test_save_backup_original_utf8_and_review(self):
        original = self.target.read_bytes()
        item = self.library.item(self.first)
        saved = self.library.save(self.first, " 新文。\n", item["version"], True)
        self.assertTrue(saved["reviewed"])
        self.assertEqual(self.target.read_bytes(), " 新文。\n".encode())
        self.library.save(self.first, "二次", saved["version"])
        self.assertEqual((self.corpus / ".proofread/backups/1.txt").read_bytes(), original)
        self.assertFalse(self.library.item(self.first)["reviewed"])

    def test_nested_backup_avoids_name_collision(self):
        p = self.corpus / "子目录/1.txt"
        p.write_text("子目录原文", encoding="utf-8")
        nested = self.ids["子目录/1.wav"]
        self.library.save(nested, "修改", self.library.item(nested)["version"])
        self.library.save(self.first, "根目录修改", self.library.item(self.first)["version"])
        self.assertEqual((self.corpus / ".proofread/backups/子目录/1.txt").read_text(encoding="utf-8"), "子目录原文")
        self.assertNotEqual((self.corpus / ".proofread/backups/1.txt").read_bytes(), b"modified")

    def test_legacy_state_restores_and_external_change_invalidates(self):
        meta = self.corpus / ".proofread"
        meta.mkdir()
        (meta / "state.json").write_text(json.dumps({"reviewed": {"1.wav": hashlib.sha256(self.target.read_bytes()).hexdigest()}, "last_name": "1.wav"}), encoding="utf-8")
        self.assertTrue(self.library.item(self.first)["reviewed"])
        self.assertEqual(self.library.session()["last_id"], self.first)
        self.target.write_text("外部修改", encoding="utf-8")
        self.assertFalse(self.library.item(self.first)["reviewed"])

    def test_conflict_keeps_external_text(self):
        item = self.library.item(self.first)
        self.target.write_text("外部修改", encoding="utf-8")
        with self.assertRaises(Conflict):
            self.library.save(self.first, "旧页面", item["version"])
        self.assertEqual(self.target.read_text(encoding="utf-8"), "外部修改")

    def test_empty_save_and_review_rejected(self):
        ident = self.ids["2.wav"]
        with self.assertRaises(ValueError):
            self.library.save(ident, " \n", "missing")
        with self.assertRaises(ValueError):
            self.library.review(ident, True, "missing")
        self.assertFalse((self.corpus / "2.txt").exists())

    def test_progress_merges_across_sessions(self):
        other = Library(self.corpus)
        second = next(i for i in other.files if other.name(i) == "2.wav")
        self.library.review(self.first, True, self.library.item(self.first)["version"])
        other.save(second, "第二条", "missing", True)
        state = json.loads((self.corpus / ".proofread/state.json").read_text(encoding="utf-8"))
        self.assertEqual(set(state["reviewed"]), {"1.wav", "2.wav"})
        self.assertTrue(self.library.item(self.first)["reviewed"])

    def test_state_failure_rolls_back_text(self):
        original = self.target.read_bytes()
        with patch.object(self.library, "_persist", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.library.save(self.first, "修改", self.library.item(self.first)["version"])
            with self.assertRaises(OSError):
                self.library.save(self.ids["2.wav"], "新文件", "missing")
        self.assertEqual(self.target.read_bytes(), original)
        self.assertFalse((self.corpus / "2.txt").exists())

    def test_rescan_stable_ids_and_excludes_metadata(self):
        hidden = self.corpus / ".proofread/backups/bad.wav"
        hidden.parent.mkdir(parents=True)
        hidden.write_bytes(b"bad")
        (self.corpus / "new.mp3").write_bytes(b"audio")
        self.library.rescan()
        self.assertIn(self.first, self.library.files)
        self.assertNotIn(".proofread/backups/bad.wav", [self.library.name(i) for i in self.library.files])

    def test_non_utf8_is_not_overwritten_or_batch_missing(self):
        self.target.write_bytes(b"\xff")
        row = next(i for i in self.library.session()["items"] if i["id"] == self.first)
        self.assertFalse(row["missing_text"])
        self.assertIn("error", row)
        with self.assertRaises(UnicodeError):
            self.library.item(self.first)

    def test_batch_skips_existing_and_resume(self):
        session = ProofreadSession(self.root, FakeASR())
        session.select(self.corpus)
        session.start_batch()
        session.thread.join(5)
        self.assertFalse(session.job["running"])
        self.assertEqual(session.job["succeeded"], 4)
        self.assertEqual(session.job["skipped"], 1)
        self.assertFalse(any(i["reviewed"] for i in session.snapshot()["items"]))
        session.start_batch()
        session.thread.join(5)
        self.assertEqual(session.job["skipped"], 5)
        self.assertEqual(session.job["succeeded"], 0)

    def test_batch_concurrent_manual_save_wins(self):
        entered, release = threading.Event(), threading.Event()
        class PausedASR(FakeASR):
            def transcribe(inner, path, cancel_event=None):
                entered.set()
                release.wait(5)
                return "ASR"
        session = ProofreadSession(self.root, PausedASR())
        session.select(self.corpus)
        session.start_batch()
        self.assertTrue(entered.wait(3))
        (self.corpus / "2.txt").write_text("人工正文", encoding="utf-8")
        release.set()
        session.thread.join(5)
        self.assertEqual((self.corpus / "2.txt").read_text(encoding="utf-8"), "人工正文")
        self.assertEqual(session.job["skipped"], 2)

    def test_cancel_keeps_completed_and_stops_before_write(self):
        entered = threading.Event()
        class CancelASR(FakeASR):
            def transcribe(inner, path, cancel_event=None):
                entered.set()
                cancel_event.wait(5)
                raise RuntimeError("Cancelled")
        session = ProofreadSession(self.root, CancelASR())
        session.select(self.corpus)
        session.start_batch()
        self.assertTrue(entered.wait(3))
        with self.assertRaises(Conflict):
            session.select("")
        session.cancel.set()
        session.thread.join(5)
        self.assertTrue(session.job["cancelled"])
        self.assertEqual(session.job["failed"], 0)
        self.assertFalse((self.corpus / "2.txt").exists())

    def test_directory_registry_blocks_other_session_and_clears_on_cancel(self):
        entered = threading.Event()
        class PausedASR(FakeASR):
            def transcribe(inner, path, cancel_event=None):
                entered.set()
                cancel_event.wait(5)
                raise RuntimeError("Cancelled")
        first = ProofreadSession(self.root, PausedASR())
        second = ProofreadSession(self.root, FakeASR())
        first.select(self.corpus)
        second.select(self.corpus)
        first.start_batch()
        try:
            self.assertTrue(entered.wait(3))
            self.assertTrue(directory_asr_running(self.corpus))
            self.assertTrue(second.asr_status()["directory_running"])
            with self.assertRaises(Conflict):
                second.start_batch()
        finally:
            first.cancel.set()
            first.thread.join(5)
        self.assertFalse(directory_asr_running(self.corpus))
        second.start_batch()
        second.thread.join(5)
        self.assertFalse(directory_asr_running(self.corpus))
        self.assertEqual(second.job["succeeded"], 4)

    def test_directory_registry_cleared_when_thread_start_fails(self):
        session = ProofreadSession(self.root, FakeASR())
        session.select(self.corpus)
        with patch("indextts_training.proofread.threading.Thread.start", side_effect=RuntimeError("cannot start")):
            with self.assertRaisesRegex(RuntimeError, "cannot start"):
                session.start_batch()
        self.assertFalse(session.job["running"])
        self.assertFalse(directory_asr_running(self.corpus))

    def test_parent_child_roots_cannot_run_overlapping_batches(self):
        entered = threading.Event()
        class PausedASR(FakeASR):
            def transcribe(inner, path, cancel_event=None):
                entered.set()
                cancel_event.wait(5)
                raise RuntimeError("Cancelled")
        parent = ProofreadSession(self.root, FakeASR())
        child = ProofreadSession(self.root, PausedASR())
        parent.select(self.corpus)
        child.select(self.corpus / "子目录")
        child.start_batch()
        try:
            self.assertTrue(entered.wait(3))
            self.assertTrue(directory_asr_running(self.corpus))
            self.assertTrue(directory_asr_running(self.corpus / "子目录" / "nested"))
            self.assertFalse(directory_asr_running(self.root / "unrelated"))
            with self.assertRaises(Conflict):
                parent.start_batch()
        finally:
            child.cancel.set()
            child.thread.join(5)
        self.assertFalse(directory_asr_running(self.corpus))

    def test_parent_child_saves_share_transaction_lock(self):
        child = Library(self.corpus / "子目录")
        parent_id = self.ids["子目录/1.wav"]
        child_id = next(i for i in child.files if child.name(i) == "1.wav")
        entered, finished = threading.Event(), threading.Event()
        outcomes = []
        def save_from_child():
            entered.set()
            try:
                child.save(child_id, "child edit", "missing")
            except Conflict:
                outcomes.append("conflict")
            finally:
                finished.set()
        with self.library.lock:
            worker = threading.Thread(target=save_from_child)
            worker.start()
            self.assertTrue(entered.wait(3))
            self.assertFalse(finished.wait(.05))
            self.library.save(parent_id, "parent edit", "missing")
        worker.join(5)
        self.assertEqual(outcomes, ["conflict"])
        self.assertEqual((self.corpus / "子目录/1.txt").read_text(encoding="utf-8"), "parent edit")

    def test_batch_failure_stops_and_keeps_prior_results(self):
        class FailingASR(FakeASR):
            count = 0
            def transcribe(inner, path, cancel_event=None):
                inner.count += 1
                if inner.count > 1:
                    raise RuntimeError("model error")
                return "完成"
        session = ProofreadSession(self.root, FailingASR())
        session.select(self.corpus)
        session.start_batch()
        session.thread.join(5)
        self.assertEqual(session.job["failed"], 1)
        self.assertEqual(session.job["succeeded"], 1)
        self.assertTrue((self.corpus / "2.txt").exists())
        self.assertFalse((self.corpus / "10.txt").exists())

    def test_retranscribe_does_not_save(self):
        original = self.target.read_bytes()
        session = ProofreadSession(self.root, FakeASR())
        session.select(self.corpus)
        ident = next(i for i in session.library.files if session.library.name(i) == "1.wav")
        result = session.retranscribe(ident, session.library.item(ident)["version"])
        self.assertEqual(result["text"], "识别正文。")
        self.assertEqual(self.target.read_bytes(), original)


class RoutesTests(CorpusFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        app = FastAPI()
        app.include_router(create_proofread_router(self.root, FakeASR()))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.token = self.client.post("/proofread/api/session").json()["token"]
        self.headers = {"X-Session-Token": self.token}

    def post(self, route, body):
        return self.client.post("/proofread/api/" + route, json=body, headers=self.headers)

    def test_http_session_isolation_and_clear(self):
        first = self.post("select", {"directory": str(self.corpus)}).json()
        self.assertEqual(len(first["items"]), 5)
        self.assertEqual(first["role"], "training")
        second_token = self.client.post("/proofread/api/session").json()["token"]
        second = self.client.get("/proofread/api/session", headers={"X-Session-Token": second_token}).json()
        self.assertEqual(second["items"], [])
        self.assertEqual(self.post("select", {"directory": ""}).json()["items"], [])

    def test_http_token_and_cross_origin_rejected(self):
        self.assertEqual(self.client.get("/proofread/api/session").status_code, 403)
        self.assertEqual(self.client.post("/proofread/api/session", headers={"Origin": "http://evil.invalid"}).status_code, 403)

    def test_http_range_and_file_scope(self):
        snapshot = self.post("select", {"directory": str(self.corpus)}).json()
        ident = snapshot["items"][0]["id"]
        response = self.client.get(f"/proofread/api/audio?id={ident}&token={self.token}", headers={"Range": "bytes=4-9"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, bytes(range(6)))
        self.assertEqual(self.client.get(f"/proofread/api/audio?id=../../secret&token={self.token}").status_code, 400)

    def test_http_conflict_status_and_original_audio(self):
        snapshot = self.post("select", {"directory": str(self.corpus)}).json()
        ident = snapshot["items"][0]["id"]
        item = self.client.get(f"/proofread/api/item?id={ident}", headers=self.headers).json()
        audio = (self.corpus / "1.wav").read_bytes()
        self.target.write_text("external", encoding="utf-8")
        response = self.post("save", {"id": ident, "text": "stale", "version": item["version"]})
        self.assertEqual(response.status_code, 409)
        self.assertEqual((self.corpus / "1.wav").read_bytes(), audio)

    def test_manifest_update_rejects_batch_from_other_page(self):
        entered = threading.Event()
        class PausedASR(FakeASR):
            def transcribe(inner, path, cancel_event=None):
                entered.set()
                cancel_event.wait(5)
                raise RuntimeError("Cancelled")
        batch = ProofreadSession(self.root, PausedASR())
        batch.select(self.corpus)
        batch.start_batch()
        try:
            self.assertTrue(entered.wait(3))
            self.post("select", {"directory": str(self.corpus)})
            result = self.post("manifest/update", {})
            self.assertEqual(result.status_code, 409)
            status = self.client.get("/proofread/api/manifest/status", headers=self.headers).json()
            self.assertTrue(status["asr_running"])
        finally:
            batch.cancel.set()
            batch.thread.join(5)

    def test_validation_role_reuses_asr_and_proofreading_without_updating_training_manifest(self):
        validation = self.root / "验证集"
        validation.mkdir()
        validation_clip = validation / "unseen.wav"
        for clip, sample in ((self.corpus / "1.wav", b"\0\0"), (validation_clip, b"\1\0")):
            with wave.open(str(clip), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(sample * 16000)
        project_path = self.root / "training/projects/voice/project.json"
        project_path.parent.mkdir(parents=True)
        project_path.write_text(json.dumps({
            "source_folder": str(self.corpus), "language": "ZH",
            "validation": {"source_folder": str(validation)},
        }), encoding="utf-8")
        context = {"project": "voice", "language": "ZH"}
        self.assertEqual(self.post("select", {**context, "directory": str(self.corpus)}).status_code, 200)
        training = self.post("manifest/update", {})
        self.assertEqual(training.status_code, 200, training.text)
        training_path = Path(training.json()["status"]["manifest"])
        original_manifest = training_path.read_bytes()

        validation_context = {**context, "role": "validation"}
        self.post("select", {**validation_context, "directory": str(self.corpus)})
        self.assertEqual(self.post("manifest/update", {}).status_code, 400)
        selection = self.post("select", {**validation_context, "directory": str(validation)}).json()
        self.assertEqual(selection["role"], "validation")
        self.assertEqual(self.post("manifest/update", {"role": "training"}).status_code, 409)
        ident = selection["items"][0]["id"]
        transcript = validation_clip.with_suffix(".txt")
        response = self.post("asr/item", {"id": ident, "version": "missing"})
        self.assertEqual(response.json()["text"], "识别正文。")
        self.assertFalse(transcript.exists())
        self.assertEqual(self.post("asr/start", {}).status_code, 200)
        deadline = time.monotonic() + 5
        while True:
            job = self.client.get("/proofread/api/asr/status", headers=self.headers).json()["job"]
            if not job["running"] or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        self.assertFalse(job["running"])
        self.assertEqual(job["succeeded"], 1)
        self.assertEqual(transcript.read_text(encoding="utf-8"), "识别正文。")

        validation_result = self.post("manifest/update", {"role": "validation"})
        self.assertEqual(validation_result.status_code, 200, validation_result.text)
        status = validation_result.json()["status"]
        self.assertEqual(status["role"], "validation")
        self.assertEqual(status["stats"]["valid"], 1)
        validation_path = Path(status["manifest"])
        self.assertEqual(validation_path, training_path.with_name("validation.jsonl"))
        self.assertEqual(Path(json.loads(validation_path.read_text(encoding="utf-8"))["audio"]), validation_clip.resolve())

        version = self.client.get(f"/proofread/api/item?id={ident}", headers=self.headers).json()["version"]
        self.assertEqual(self.post("save", {"id": ident, "text": "手动校对。", "version": version}).status_code, 200)
        status = self.client.get("/proofread/api/manifest/status", headers=self.headers).json()
        self.assertTrue(status["stale"])
        self.assertEqual(status["role"], "validation")
        self.assertEqual(training_path.read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
