"""Fake ``merge_lora_to_pth.py`` for INDEXTTS_FAKE_ENGINE=1 (no torch, a few seconds)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def phase(name: str) -> None:
    print(f"EXPORT_PHASE: {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("--adapter", "--base", "--config", "--output"):
        parser.add_argument(name, required=True)
    parser.add_argument("--expected-base", default="")
    args = parser.parse_args()
    output = Path(args.output)
    started = time.time()
    for name, seconds in (("hashing", 0.8), ("building", 0.8), ("checking", 0.6)):
        phase(name)
        print(f"[fake-merge] {name}…", flush=True)
        time.sleep(seconds)
    if "fail" in output.name.lower():
        print("Fake merge failure requested by the output filename.", flush=True)
        return 3
    phase("saving")
    payload = b"FAKE-MERGED-CHECKPOINT\n" + Path(args.adapter).name.encode("utf-8")
    output.write_bytes(payload)
    time.sleep(0.5)
    phase("verifying")
    time.sleep(0.4)
    report = {
        "base": args.base, "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
        "config": args.config, "adapter": args.adapter, "output": str(output),
        "output_sha256": hashlib.sha256(payload).hexdigest(), "dtype": "float32",
        "tensor_count": 456, "lora_tensor_count": 144, "lora_target_count": 72, "changed_tensor_count": 72,
        "changed_tensor_names": [], "rank": 16, "alpha": 32, "cpu_forward_max_abs_difference": 0.0,
        "source_files_unchanged": True, "round_trip_tensors_identical": True,
        "elapsed_seconds": round(time.time() - started, 2), "note": "Fake engine export (not a real checkpoint).",
    }
    output.with_suffix(".merge.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changed_tensor_names"}, indent=2), flush=True)
    phase("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
