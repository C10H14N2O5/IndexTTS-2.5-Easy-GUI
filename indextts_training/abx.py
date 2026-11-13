"""Blind paired comparison of two IndexTTS adapters.

A test session is generated first and rated afterwards.  Every target text is
synthesized ``repetitions`` times by both sides with a shared seed, then each
(text, repetition) pair becomes one trial whose left/right order is randomized
and balanced.  Ratings use a five-point comparison scale relative to the shown
positions and are converted to a preference for side B when results are
revealed.

The module only depends on the standard library.  Synthesis is injected by the
GUI so tests and demos can supply a fake implementation.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import secrets
import shutil
import stat
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

SCHEMA = 1
MAX_REPETITIONS = 5
SCORES = (-2, -1, 0, 1, 2)
SIGNIFICANCE = 0.05
_ID_PATTERN = re.compile(r"^[0-9_]+$")
_LOCK = threading.RLock()


class AbxError(ValueError):
    """User-facing error; ``template`` and ``args`` allow localization."""

    def __init__(self, template: str, *args):
        self.template = template
        self.args_ = args
        super().__init__(template.format(*args) if args else template)


def abx_root(outputs_dir) -> Path:
    return Path(outputs_dir) / "abx"


def _atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def session_dir(root, session_id: str) -> Path:
    session_id = str(session_id or "")
    if not _ID_PATTERN.match(session_id):
        raise AbxError("Invalid blind test id: {0}", session_id)
    base = Path(root).resolve()
    folder = (base / session_id).resolve()
    if folder.parent != base:
        raise AbxError("Invalid blind test id: {0}", session_id)
    return folder


def split_texts(raw: str) -> list[str]:
    return [line.strip() for line in str(raw or "").splitlines() if line.strip()]


def _new_id(root: Path, now: datetime | None) -> str:
    base = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate, number = base, 2
    while (root / candidate).exists():
        candidate, number = f"{base}_{number}", number + 1
    return candidate


def _side(label, path) -> dict:
    path = str(path) if path else None
    return {"label": str(label or ("None" if path is None else Path(path).name)), "path": path}


def create_session(root, *, name: str, a: dict, b: dict, reference_path, transcript: str,
                   language: str, texts, repetitions: int, settings: dict,
                   now: datetime | None = None, rng: random.Random | None = None) -> dict:
    """Create and persist a new session.  ``a``/``b`` are ``{"label", "path"}``."""
    root = Path(root)
    a, b = _side(a.get("label"), a.get("path")), _side(b.get("label"), b.get("path"))
    if a["path"] == b["path"]:
        raise AbxError("Adapters A and B must be different.")
    texts = split_texts(texts) if isinstance(texts, str) else [str(x).strip() for x in texts if str(x).strip()]
    if not texts:
        raise AbxError("Enter at least one test sentence.")
    try:
        repetitions = int(repetitions)
    except (TypeError, ValueError):
        repetitions = 0
    if not 1 <= repetitions <= MAX_REPETITIONS:
        raise AbxError("Generations per sentence must be between 1 and {0}.", MAX_REPETITIONS)
    reference = Path(str(reference_path or ""))
    if not reference_path or not reference.is_file():
        raise AbxError("Choose a saved reference voice.")
    rng = rng or random.Random(secrets.randbits(64))

    with _LOCK:
        root.mkdir(parents=True, exist_ok=True)
        session_id = _new_id(root, now)
        folder = root / session_id
        (folder / "clips").mkdir(parents=True)
        reference_file = "reference" + (reference.suffix.lower() or ".wav")
        shutil.copy2(reference, folder / reference_file)

        base_seed = rng.randrange(1, 2**31 - 10_000_000)
        clips, used = [], set()
        for side in ("a", "b"):
            for text_index in range(len(texts)):
                for rep in range(repetitions):
                    clip_id = secrets.token_hex(4)
                    while clip_id in used:
                        clip_id = secrets.token_hex(4)
                    used.add(clip_id)
                    clips.append({
                        "id": clip_id, "side": side, "text_index": text_index, "rep": rep,
                        "seed": base_seed + text_index * 1000 + rep,
                        "file": f"clips/{clip_id}.wav", "done": False,
                    })
        pairs = [(text_index, rep) for text_index in range(len(texts)) for rep in range(repetitions)]
        firsts = ["a"] * (len(pairs) // 2) + ["b"] * (len(pairs) - len(pairs) // 2)
        rng.shuffle(firsts)
        rng.shuffle(pairs)
        session = {
            "schema": SCHEMA, "id": session_id, "name": str(name or "").strip(),
            "created": (now or datetime.now()).isoformat(timespec="seconds"),
            "a": a, "b": b,
            "reference": {"file": reference_file, "transcript": str(transcript or ""), "source": str(reference)},
            "language": str(language), "texts": texts, "repetitions": repetitions,
            "base_seed": base_seed, "settings": dict(settings or {}),
            "clips": clips,
            "trials": [{"text_index": t, "rep": r, "first": f} for (t, r), f in zip(pairs, firsts)],
            "ratings": {}, "revealed": False, "revealed_at": None,
        }
        session["_root"] = str(root)
        save_session(session)
    return session


def save_session(session: dict) -> None:
    root = Path(session["_root"])
    data = {key: value for key, value in session.items() if not key.startswith("_")}
    with _LOCK:
        _atomic_json(session_dir(root, session["id"]) / "session.json", data)


def load_session(root, session_id: str) -> dict:
    folder = session_dir(root, session_id)
    path = folder / "session.json"
    with _LOCK:
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise AbxError("Blind test not found: {0}", session_id) from None
        except (OSError, ValueError) as exc:
            raise AbxError("Blind test record is unreadable: {0}", session_id) from exc
    if not isinstance(session, dict) or session.get("schema") != SCHEMA or session.get("id") != folder.name:
        raise AbxError("Blind test record is unreadable: {0}", session_id)
    session["_root"] = str(Path(root))
    return session


def clip_path(session: dict, clip: dict) -> Path:
    return session_dir(session["_root"], session["id"]) / clip["file"]


def reference_path(session: dict) -> Path:
    return session_dir(session["_root"], session["id"]) / session["reference"]["file"]


def pending_clips(session: dict) -> list[dict]:
    pending = [c for c in session["clips"] if not c.get("done") or not clip_path(session, c).is_file()]
    return sorted(pending, key=lambda c: (c["side"], c["text_index"], c["rep"]))


def generation_complete(session: dict) -> bool:
    return not pending_clips(session)


def generate_clips(session: dict, synth: Callable[[dict, dict, Path], None],
                   should_stop: Callable[[], bool] = lambda: False) -> Iterator[dict]:
    """Synthesize missing clips, A first, then B.  Yields progress dictionaries.

    ``synth(clip, session, output_path)`` must write a WAV to ``output_path``.
    Each finished clip is persisted immediately, so a stopped run resumes.
    """
    pending = pending_clips(session)
    total = len(session["clips"])
    done = total - len(pending)
    for clip in pending:
        if should_stop():
            yield {"done": done, "total": total, "stopped": True, "side": clip["side"]}
            return
        yield {"done": done, "total": total, "stopped": False, "side": clip["side"], "starting": True}
        output = clip_path(session, clip)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        synth(clip, session, output)
        if not output.is_file():
            raise AbxError("Synthesis did not create clip {0}.", clip["id"])
        with _LOCK:
            clip["done"] = True
            save_session(session)
        done += 1
        yield {"done": done, "total": total, "stopped": False, "side": clip["side"]}


def _clip_for(session: dict, side: str, text_index: int, rep: int) -> dict:
    for clip in session["clips"]:
        if clip["side"] == side and clip["text_index"] == text_index and clip["rep"] == rep:
            return clip
    raise AbxError("Blind test record is unreadable: {0}", session["id"])


def trial_count(session: dict) -> int:
    return len(session["trials"])


def rated_count(session: dict) -> int:
    return sum(1 for key in session["ratings"] if 0 <= int(key) < trial_count(session))


def _check_index(session: dict, index) -> int:
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = -1
    if not 0 <= index < trial_count(session):
        raise AbxError("Trial {0} does not exist.", index + 1)
    return index


def trial_view(session: dict, index) -> dict:
    """Return what the listener may see: no side information before reveal."""
    index = _check_index(session, index)
    trial = session["trials"][index]
    second = "b" if trial["first"] == "a" else "a"
    first_clip = _clip_for(session, trial["first"], trial["text_index"], trial["rep"])
    second_clip = _clip_for(session, second, trial["text_index"], trial["rep"])
    return {
        "index": index, "total": trial_count(session),
        "text": session["texts"][trial["text_index"]], "rep": trial["rep"],
        "reference": str(reference_path(session)),
        "sample_1": str(clip_path(session, first_clip)), "sample_2": str(clip_path(session, second_clip)),
        "score": session["ratings"].get(str(index)),
    }


def next_unrated(session: dict, after: int = -1) -> int | None:
    count = trial_count(session)
    for offset in range(1, count + 1):
        index = (after + offset) % count
        if str(index) not in session["ratings"]:
            return index
    return None


def rate(session: dict, index, score) -> None:
    index = _check_index(session, index)
    if session.get("revealed"):
        raise AbxError("Results are revealed; ratings can no longer be changed.")
    if not generation_complete(session):
        raise AbxError("Generate all clips before rating.")
    if score is None:
        with _LOCK:
            session["ratings"].pop(str(index), None)
            save_session(session)
        return
    score = int(score)
    if score not in SCORES:
        raise AbxError("Invalid rating: {0}", score)
    with _LOCK:
        session["ratings"][str(index)] = score
        save_session(session)


def reveal(session: dict, now: datetime | None = None) -> None:
    if session.get("revealed"):
        return
    if rated_count(session) == 0:
        raise AbxError("Rate at least one trial before revealing the results.")
    with _LOCK:
        session["revealed"] = True
        session["revealed_at"] = (now or datetime.now()).isoformat(timespec="seconds")
        save_session(session)


def preference_for_b(trial: dict, score: int) -> int:
    """Convert a position-relative score (positive = sample 2 better)."""
    return score if trial["first"] == "a" else -score


def sign_test(wins_a: int, wins_b: int) -> float:
    """Two-sided exact binomial sign test, ties excluded."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins_a, wins_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def summarize(session: dict) -> dict:
    rows, preferences = [], []
    for index, trial in enumerate(session["trials"]):
        raw = session["ratings"].get(str(index))
        preference = None if raw is None else preference_for_b(trial, int(raw))
        if preference is not None:
            preferences.append(preference)
        rows.append({
            "index": index, "text": session["texts"][trial["text_index"]], "rep": trial["rep"],
            "sample_1": trial["first"].upper(), "score": raw, "preference_b": preference,
        })
    wins_b = sum(1 for p in preferences if p > 0)
    wins_a = sum(1 for p in preferences if p < 0)
    ties = sum(1 for p in preferences if p == 0)
    n = len(preferences)
    mean = sum(preferences) / n if n else 0.0
    if n > 1:
        variance = sum((p - mean) ** 2 for p in preferences) / (n - 1)
        stderr = math.sqrt(variance / n)
    else:
        stderr = 0.0
    p_value = sign_test(wins_a, wins_b)
    if wins_a + wins_b and p_value < SIGNIFICANCE:
        verdict = "b_better" if wins_b > wins_a else "a_better"
    else:
        verdict = "no_difference"
    return {
        "rated": n, "total": trial_count(session), "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "mean_b": mean, "stderr": stderr, "p_value": p_value, "verdict": verdict, "rows": rows,
    }


def _is_link_or_junction(path: Path) -> bool:
    if os.path.islink(path):
        return True
    is_junction = getattr(path, "is_junction", None)  # Python 3.12+
    if is_junction is not None:
        return bool(is_junction())
    try:
        # Python 3.11 on Windows: a junction is a mount-point reparse tag.
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except OSError:
        return False
    return bool(tag) and tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def delete_session(root, session_id: str) -> None:
    """Permanently remove one saved session folder (record, reference copy, clips).

    Only a real directory directly under ``root`` that holds ``session.json``
    is removed; links and junctions are never followed.
    """
    folder = session_dir(root, session_id)
    # The unresolved entry: ``folder`` is resolved, so a link would already point elsewhere.
    entry = Path(root).resolve() / str(session_id)
    with _LOCK:
        if (entry != folder or _is_link_or_junction(entry) or not entry.is_dir()
                or not (entry / "session.json").is_file()):
            raise AbxError("Blind test not found: {0}", session_id)
        try:
            shutil.rmtree(entry)
        except OSError as exc:  # includes PermissionError (file held open on Windows)
            raise AbxError("Some files of blind test {0} are in use. Stop playback and try again.",
                           session_id) from exc


def list_sessions(root) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    sessions = []
    for folder in root.iterdir():
        if not folder.is_dir() or not _ID_PATTERN.match(folder.name):
            continue
        try:
            session = load_session(root, folder.name)
        except AbxError:
            continue
        sessions.append({
            "id": session["id"], "name": session.get("name") or "",
            "a": session["a"]["label"], "b": session["b"]["label"],
            "clips_done": len(session["clips"]) - len(pending_clips(session)),
            "clips_total": len(session["clips"]),
            "rated": rated_count(session), "trials": trial_count(session),
            "revealed": bool(session.get("revealed")),
        })
    return sorted(sessions, key=lambda s: s["id"], reverse=True)
