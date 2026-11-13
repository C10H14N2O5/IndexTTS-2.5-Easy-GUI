"""Stored validation results next to an adapter (``validation_posthoc.json``).

Shared by the post-hoc validation job in the web GUI and the trainer, which can
record the base-model baseline before its first step. Entries are keyed by name
and validation-manifest fingerprint. Both writers run in the GUI process, so a
single module lock serializes read-modify-write cycles.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .dataset import _atomic_write

RESULTS_FILE = "validation_posthoc.json"
BASELINE = "base_model"
LOCK = threading.RLock()


def load(root) -> list[dict]:
    try:
        data = json.loads((Path(root) / RESULTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("name"), str)
            and isinstance(entry.get("loss"), (int, float)) and not isinstance(entry.get("loss"), bool)]


def save(root, entries: list[dict]) -> None:
    payload = {"version": 1, "entries": sorted(entries, key=lambda item: (
        item.get("validation_fingerprint", ""), item["name"] != BASELINE, item.get("step") is None,
        item.get("step") or 0, item["name"]))}
    _atomic_write(Path(root) / RESULTS_FILE, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))


def upsert(root, entry: dict) -> None:
    """Replace the entry with the same name and fingerprint (or add it)."""
    with LOCK:
        entries = [item for item in load(root) if not (
            item["name"] == entry["name"] and item.get("validation_fingerprint") == entry["validation_fingerprint"])]
        entries.append(entry)
        save(root, entries)


def has_baseline(root, fingerprint: str | None) -> bool:
    """True when a base-model result exists for this validation fingerprint."""
    return bool(fingerprint) and any(
        item["name"] == BASELINE and item.get("validation_fingerprint") == fingerprint for item in load(root))


def record_baseline(root, *, loss: float, fingerprint: str, samples: int, model_dir: str) -> None:
    """Store the base-model validation loss measured by the trainer."""
    Path(root).mkdir(parents=True, exist_ok=True)
    upsert(root, dict(name=BASELINE, step=0, loss=float(loss), validation_fingerprint=fingerprint,
                      samples=int(samples), computed_at=time.time(), model_dir=str(model_dir), same_as=None,
                      weights=None, source="training"))
