from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "IndexTTS-2.5"
RUNTIME_DIR = ROOT / ".runtime"
CACHE_DIR = RUNTIME_DIR / "hf-cache"

for p in (MODEL_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE_DIR / "xet")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REQUIRED = ("config.yaml", "gpt.pth", "codec.pth", "s2mel.pth")


def model_ready() -> bool:
    if not all((MODEL_DIR / name).is_file() for name in REQUIRED):
        return False
    try:
        total = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
    except OSError:
        return False
    return total > 500_000_000


def main() -> int:
    sys.path.insert(0, str(ROOT))
    try:
        from indextts.utils.model_download import ensure_models_available, snapshot_download

        if model_ready():
            print(f"[Model] IndexTTS 2.5 checkpoint already present: {MODEL_DIR}")
        else:
            print("[Model] IndexTTS 2.5 checkpoint missing; downloading IndexTeam/IndexTTS-2.5...")
            snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(MODEL_DIR))
            if not model_ready():
                missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
                raise RuntimeError(f"Checkpoint download finished but required files are missing: {missing}")
            print("[Model] Main IndexTTS 2.5 checkpoint ready.")

        print("[Model] Checking auxiliary inference models...")
        ensure_models_available(str(MODEL_DIR))
        print("[Model] All IndexTTS 2.5 model assets are ready.")
        return 0
    except KeyboardInterrupt:
        print("[ERROR] Model download cancelled.")
        return 130
    except Exception as exc:
        print(f"[ERROR] Automatic model setup failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
