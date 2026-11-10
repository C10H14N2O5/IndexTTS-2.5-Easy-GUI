"""Process environment for the GUI. Call before importing torch or Hugging Face."""
from __future__ import annotations

import os

from . import paths


def configure() -> None:
    paths.ensure_dirs()
    cache = paths.CACHE_DIR
    # Keep project caches and temporary files local to this folder.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    os.environ["TMP"] = str(paths.TEMP_DIR)
    os.environ["TEMP"] = str(paths.TEMP_DIR)
    os.environ["TRITON_CACHE_DIR"] = str(paths.RUNTIME_DIR / "triton-cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(paths.RUNTIME_DIR / "torchinductor-cache")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(paths.RUNTIME_DIR / "torch-extensions")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if os.name == "nt":
        try:
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass


def fake_engine_enabled() -> bool:
    return os.environ.get("INDEXTTS_FAKE_ENGINE", "").strip().lower() in {"1", "true", "yes", "on"}
