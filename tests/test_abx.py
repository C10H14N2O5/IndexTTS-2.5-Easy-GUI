"""Offline tests for the blind A/B adapter comparison backend.

Run with: .venv\\Scripts\\python.exe -B -m unittest discover -s tests -p test_abx.py
"""
import json
import random
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path

from indextts_training import abx


def write_wav(path, frames=160):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * frames)


class AbxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "outputs" / "abx"
        self.reference = self.base / "voice.wav"
        write_wav(self.reference)
        self.now = datetime(2026, 11, 7, 18, 0, 0)

    def create(self, texts="one\n\n two \nthree", repetitions=2, a=None, b="/adapters/b", **extra):
        return abx.create_session(
            self.root, name="demo", a={"label": "base", "path": a}, b={"label": "new", "path": b},
            reference_path=self.reference, transcript="hello", language="Chinese",
            texts=texts, repetitions=repetitions, settings={"temperature": 0.8},
            now=self.now, rng=random.Random(7), **extra,
        )

    def generate_all(self, session):
        calls = []

        def synth(clip, _session, output):
            calls.append((clip["side"], clip["text_index"], clip["rep"], clip["seed"]))
            write_wav(output)

        list(abx.generate_clips(session, synth))
        return calls

    def test_create_validates_inputs(self):
        cases = [
            dict(a="/same", b="/same"),
            dict(a=None, b=None),
            dict(texts=" \n "),
            dict(repetitions=0),
            dict(repetitions=abx.MAX_REPETITIONS + 1),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(abx.AbxError):
                self.create(**case)
        self.reference.unlink()
        with self.assertRaises(abx.AbxError):
            self.create()

    def test_create_copies_reference_and_persists(self):
        session = self.create()
        self.assertEqual(session["id"], "20261107_180000")
        self.assertEqual(session["texts"], ["one", "two", "three"])
        self.assertTrue(abx.reference_path(session).is_file())
        loaded = abx.load_session(self.root, session["id"])
        self.assertEqual(loaded["trials"], session["trials"])
        self.assertNotIn("_root", json.loads((self.root / session["id"] / "session.json").read_text(encoding="utf-8")))
        self.assertEqual(self.create()["id"], "20261107_180000_2")

    def test_trials_are_balanced_and_seeds_paired(self):
        for repetitions in (1, 2, 3):
            session = self.create(repetitions=repetitions)
            trials = session["trials"]
            self.assertEqual(len(trials), 3 * repetitions)
            first_a = sum(1 for t in trials if t["first"] == "a")
            self.assertIn(first_a, {len(trials) // 2, (len(trials) + 1) // 2})
            self.assertEqual(len({(t["text_index"], t["rep"]) for t in trials}), len(trials))
            seeds = {}
            for clip in session["clips"]:
                seeds.setdefault((clip["text_index"], clip["rep"]), set()).add(clip["seed"])
            self.assertTrue(all(len(values) == 1 for values in seeds.values()))
            self.assertEqual(len({c["id"] for c in session["clips"]}), len(session["clips"]))

    def test_generation_order_and_resume(self):
        session = self.create()
        written = []

        def synth(clip, _session, output):
            written.append(clip["side"])
            write_wav(output)

        stop_after = 4
        progress = list(abx.generate_clips(session, synth, should_stop=lambda: len(written) >= stop_after))
        self.assertTrue(progress[-1]["stopped"])
        self.assertEqual(written, ["a"] * 4)
        self.assertFalse(abx.generation_complete(session))
        with self.assertRaises(abx.AbxError):
            abx.rate(session, 0, 1)

        resumed = abx.load_session(self.root, session["id"])
        calls = self.generate_all(resumed)
        self.assertEqual([c[0] for c in calls], ["a"] * 2 + ["b"] * 6)
        self.assertTrue(abx.generation_complete(resumed))
        self.assertEqual(self.generate_all(resumed), [])

    def test_missing_output_is_reported(self):
        session = self.create(texts="one", repetitions=1)
        with self.assertRaises(abx.AbxError):
            list(abx.generate_clips(session, lambda clip, s, output: None))

    def test_trial_view_hides_sides(self):
        session = self.create()
        self.generate_all(session)
        for index in range(abx.trial_count(session)):
            view = abx.trial_view(session, index)
            self.assertNotIn("first", view)
            self.assertNotIn("side", json.dumps(view))
            self.assertTrue(Path(view["sample_1"]).is_file())
            self.assertTrue(Path(view["sample_2"]).is_file())
            self.assertNotEqual(view["sample_1"], view["sample_2"])
        with self.assertRaises(abx.AbxError):
            abx.trial_view(session, abx.trial_count(session))

    def test_rating_reveal_and_lock(self):
        session = self.create()
        self.generate_all(session)
        with self.assertRaises(abx.AbxError):
            abx.reveal(session)
        with self.assertRaises(abx.AbxError):
            abx.rate(session, 0, 3)
        abx.rate(session, 0, 2)
        abx.rate(session, 1, -1)
        abx.rate(session, 1, None)
        self.assertEqual(abx.rated_count(session), 1)
        self.assertEqual(abx.next_unrated(session, 0), 1)
        abx.reveal(session, now=self.now)
        with self.assertRaises(abx.AbxError):
            abx.rate(session, 2, 0)
        reloaded = abx.load_session(self.root, session["id"])
        self.assertTrue(reloaded["revealed"])
        self.assertEqual(reloaded["ratings"], {"0": 2})

    def test_summary_converts_positions_to_preference(self):
        session = self.create(texts="one\ntwo\nthree", repetitions=2)
        self.generate_all(session)
        for index, trial in enumerate(session["trials"]):
            # Always prefer B strongly, whichever position it is shown in.
            abx.rate(session, index, 2 if trial["first"] == "a" else -2)
        summary = abx.summarize(session)
        self.assertEqual((summary["wins_b"], summary["wins_a"], summary["ties"]), (6, 0, 0))
        self.assertAlmostEqual(summary["mean_b"], 2.0)
        self.assertAlmostEqual(summary["p_value"], 0.03125)
        self.assertEqual(summary["verdict"], "b_better")
        self.assertTrue(all(row["preference_b"] == 2 for row in summary["rows"]))

    def test_sign_test_and_ties(self):
        self.assertAlmostEqual(abx.sign_test(0, 5), 0.0625)
        self.assertAlmostEqual(abx.sign_test(6, 0), 0.03125)
        self.assertAlmostEqual(abx.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(abx.sign_test(5, 5), 1.0)
        session = self.create(texts="one", repetitions=2)
        self.generate_all(session)
        abx.rate(session, 0, 0)
        abx.rate(session, 1, 0)
        summary = abx.summarize(session)
        self.assertEqual(summary["ties"], 2)
        self.assertEqual(summary["verdict"], "no_difference")

    def test_ids_are_confined(self):
        for bad in ("../x", "..", "", "abc", "2026/11", "20261107_180000\\.."):
            with self.subTest(bad=bad), self.assertRaises(abx.AbxError):
                abx.session_dir(self.root, bad)

    def test_delete_session_removes_folder(self):
        session = self.create()
        self.generate_all(session)
        other = self.create()
        abx.delete_session(self.root, session["id"])
        self.assertFalse((self.root / session["id"]).exists())
        self.assertTrue(abx.reference_path(other).is_file())
        self.assertEqual([s["id"] for s in abx.list_sessions(self.root)], [other["id"]])
        with self.assertRaises(abx.AbxError) as caught:
            abx.load_session(self.root, session["id"])
        self.assertEqual(caught.exception.template, "Blind test not found: {0}")

    def test_delete_session_rejects_invalid_and_missing(self):
        for bad in ("../x", "..", "", "abc", "2026/11"):
            with self.subTest(bad=bad), self.assertRaises(abx.AbxError) as caught:
                abx.delete_session(self.root, bad)
            self.assertEqual(caught.exception.template, "Invalid blind test id: {0}")
        not_session = self.root / "20200101_000000"
        not_session.mkdir(parents=True)
        (not_session / "keep.txt").write_text("x", encoding="utf-8")
        (self.root / "20200102_000000").write_text("a file", encoding="utf-8")
        for missing in ("20990101_000000", "20200101_000000", "20200102_000000"):
            with self.subTest(missing=missing), self.assertRaises(abx.AbxError) as caught:
                abx.delete_session(self.root, missing)
            self.assertEqual(caught.exception.template, "Blind test not found: {0}")
        self.assertTrue((not_session / "keep.txt").is_file())
        self.assertTrue((self.root / "20200102_000000").is_file())

    @staticmethod
    def _directory_link(link, target) -> bool:
        import os
        import subprocess
        try:
            os.symlink(target, link, target_is_directory=True)
            return True
        except OSError:
            if os.name != "nt":
                return False
        # Unprivileged Windows: a directory junction needs no special rights.
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return result.returncode == 0

    def test_delete_session_never_follows_links(self):
        import os
        session = self.create()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "session.json").write_text("{}", encoding="utf-8")
        (outside / "precious.wav").write_text("x", encoding="utf-8")
        # One link escapes the root; the other points at a real session inside it.
        links = {"20200101_000000": outside, "20200102_000000": self.root / session["id"]}
        for name, target in links.items():
            if not self._directory_link(self.root / name, target):
                self.skipTest("cannot create directory links")
        for name in links:
            with self.subTest(name=name), self.assertRaises(abx.AbxError) as caught:
                abx.delete_session(self.root, name)
            # An escaping link is already an invalid id; an inner one is "not found".
            self.assertIn(caught.exception.template, {"Invalid blind test id: {0}", "Blind test not found: {0}"})
            self.assertTrue(os.path.lexists(self.root / name))
        self.assertTrue((outside / "precious.wav").is_file())
        self.assertTrue((self.root / session["id"] / "session.json").is_file())

    def test_delete_session_reports_files_in_use(self):
        from unittest import mock
        session = self.create()
        with mock.patch.object(abx.shutil, "rmtree", side_effect=PermissionError("locked")), \
                self.assertRaises(abx.AbxError) as caught:
            abx.delete_session(self.root, session["id"])
        self.assertEqual(caught.exception.template,
                         "Some files of blind test {0} are in use. Stop playback and try again.")

    def test_list_sessions_skips_broken_records(self):
        session = self.create()
        self.generate_all(session)
        abx.rate(session, 0, 1)
        broken = self.root / "20200101_000000"
        broken.mkdir(parents=True)
        (broken / "session.json").write_text("{not json", encoding="utf-8")
        (self.root / "notes").mkdir()
        listed = abx.list_sessions(self.root)
        self.assertEqual([s["id"] for s in listed], [session["id"]])
        self.assertEqual(listed[0]["rated"], 1)
        self.assertEqual(listed[0]["clips_done"], listed[0]["clips_total"])
        self.assertEqual(abx.list_sessions(self.base / "missing"), [])


if __name__ == "__main__":
    unittest.main()
