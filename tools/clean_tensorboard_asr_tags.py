"""Remove legacy Full-ASR training tags from TensorBoard event files.

The current IndexTTS LoRA profile is acoustic-only. This migration keeps all
other TensorBoard data, including evaluation audio and text, while removing
the obsolete auxiliary-loss series from existing event files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


STALE_TAGS = frozenset(
    {
        "train/text_semantic_loss",
        "train/phonetic_ctc_loss",
        "train/differentiable_asr_loss",
    }
)


def _event_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("events.out.tfevents.*")
        if path.is_file() and not path.name.endswith(".bak")
    )


def _clean_file(source: Path, apply: bool) -> tuple[int, set[str], Path | None]:
    # Keep the temporary event file outside the watched log directory so a
    # running TensorBoard instance never attempts to read it mid-migration.
    temp_dir = Path(tempfile.mkdtemp(prefix="tensorboard-asr-clean-", dir=source.parent.parent))
    destination: Path | None = None
    removed: set[str] = set()
    events = 0
    try:
        writer = EventFileWriter(str(temp_dir))
        try:
            for event in EventFileLoader(str(source)).Load():
                events += 1
                if event.HasField("summary") and event.summary.value:
                    kept = []
                    for value in event.summary.value:
                        if value.tag in STALE_TAGS:
                            removed.add(value.tag)
                        else:
                            kept.append(value)
                    if len(kept) != len(event.summary.value):
                        filtered = event_pb2.Event()
                        filtered.CopyFrom(event)
                        del filtered.summary.value[:]
                        filtered.summary.value.extend(kept)
                        event = filtered
                writer.add_event(event)
        finally:
            writer.close()

        generated = sorted(temp_dir.glob("events.out.tfevents.*"))
        if not generated:
            raise RuntimeError(f"TensorBoard writer produced no event file for {source}")
        destination = generated[0]
        if apply and removed:
            # Keep the backup outside the TensorBoard log directory,
            # otherwise TensorBoard can load it as a second run and the
            # removed tags would reappear in the dashboard.
            backup_dir = source.parent.parent / "tensorboard-legacy-backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{source.parent.name}-{source.name}.pre-asr-cleanup.bak"
            if backup.exists():
                raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
            shutil.copy2(source, backup)
            os.replace(destination, source)
            destination = backup
        return events, removed, destination
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("training/outputs"),
        help="TensorBoard output root or a single event file",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite matching files and retain a .pre-asr-cleanup.bak backup",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    files = [root] if root.is_file() else _event_files(root)
    if not files:
        print(f"No TensorBoard event files found under {root}")
        return 0

    for source in files:
        events, removed, backup = _clean_file(source, args.apply)
        action = "cleaned" if args.apply and removed else "checked"
        details = ", ".join(sorted(removed)) if removed else "no legacy ASR tags"
        print(f"{action}: {source} | events={events} | {details}")
        if backup is not None and args.apply and removed:
            print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
