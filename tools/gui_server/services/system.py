"""Runtime status: accelerators, model files, CUDA, VRAM (moved from easy_gui.py).

Nothing here imports torch at module import. ``cuda_info()`` imports torch on
first use and caches the result; call ``warm_up()`` from a background thread at
startup so the first request does not pay the import cost.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import threading

from .. import paths
from ..i18n import msg

_CUDA_LOCK = threading.Lock()
_CUDA_INFO: dict | None = None
_CAPS: dict | None = None


def _probe_distribution(module_name: str, distribution_name: str) -> dict:
    """Detect an optional accelerator without importing it."""
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"ok": False, "detail": msg("not installed")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": msg("metadata error: {0}: {1}", type(exc).__name__, exc)}
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": msg("module lookup error: {0}: {1}", type(exc).__name__, exc)}
    if spec is None:
        return {"ok": False, "detail": msg("{0} {1} installed, but module {2!r} was not found",
                                           distribution_name, version, module_name)}
    return {"ok": True, "detail": str(version)}


def acceleration_capabilities(refresh: bool = False) -> dict:
    global _CAPS
    if _CAPS is None or refresh:
        _CAPS = {"triton": _probe_distribution("triton", "triton-windows"),
                 "flash_attn": _probe_distribution("flash_attn", "flash-attn")}
    return _CAPS


def model_files_present() -> bool:
    model_dir = paths.MODEL_DIR
    required = ["config.yaml", "gpt.pth", "codec.pth", "s2mel.pth"]
    if not all((model_dir / name).exists() for name in required):
        return False
    try:
        return sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) > 500_000_000
    except OSError:
        return False


def cuda_info(refresh: bool = False) -> dict:
    """{available, torch, cuda, device, total_gb, error} — imports torch once."""
    global _CUDA_INFO
    with _CUDA_LOCK:
        if _CUDA_INFO is not None and not refresh:
            return _CUDA_INFO
        info = {"available": False, "torch": None, "cuda": None, "device": None, "total_gb": None, "error": None}
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
            if torch.cuda.is_available():
                info["available"] = True
                info["device"] = torch.cuda.get_device_name(0)
                info["total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
        _CUDA_INFO = info
        return info


def cuda_ready() -> bool:
    return _CUDA_INFO is not None


def vram_metrics() -> dict:
    """Process VRAM usage in GB; zeros when CUDA was never initialised."""
    info = _CUDA_INFO
    if not info or not info.get("available"):
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": info.get("total_gb") if info else None}
    try:
        import torch
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
            "total_gb": info.get("total_gb"),
        }
    except Exception:  # noqa: BLE001
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": info.get("total_gb")}


def warm_up() -> None:
    acceleration_capabilities()
    cuda_info()
