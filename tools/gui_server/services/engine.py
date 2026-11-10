"""The cached IndexTTS 2.5 model (moved from easy_gui.py ``_load_model``).

The cache key is (bf16, gpt_latent, torch_compile, flash_accel, qwen_emo,
adapter_path); any change triggers a full reload, exactly as in 1.x. In fake
engine mode a :class:`tools.gui_server.fake.FakeTTS` replaces IndexTTS2.
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path

from .. import paths
from ..errors import ServiceError
from ..paths import NONE_SENTINEL
from .system import acceleration_capabilities, model_files_present

MODEL_LOCK = threading.RLock()
_MODEL = None
_OPTIONS: tuple | None = None
_CONTEXT: tuple | None = None
OPTION_NAMES = ("use_bf16", "use_gpt_latent", "use_torch_compile", "use_flash_accel", "use_qwen_emo", "adapter")


def _fake() -> bool:
    from ..state import current
    try:
        return current().fake
    except RuntimeError:
        return False


def _context() -> tuple:
    from ..state import current
    try:
        state_id = id(current())
    except RuntimeError:
        state_id = None
    return (_fake(), str(paths.MODEL_DIR.resolve()), state_id)


def loaded_info() -> dict:
    options = _OPTIONS
    if _MODEL is None or options is None or _CONTEXT != _context():
        return {"loaded": False, "adapter": None, "options": None}
    named = dict(zip(OPTION_NAMES, options))
    return {"loaded": True, "adapter": named.pop("adapter") or None, "options": named}


def is_loaded() -> bool:
    return _MODEL is not None and _CONTEXT == _context()


def load_model(use_bf16: bool, use_gpt_latent: bool, use_torch_compile: bool, use_flash_accel: bool,
               use_qwen_emo: bool, adapter_path: str = ""):
    global _MODEL, _OPTIONS, _CONTEXT
    adapter_path = str(adapter_path or "")
    if adapter_path == NONE_SENTINEL:
        adapter_path = ""
    options = (bool(use_bf16), bool(use_gpt_latent), bool(use_torch_compile), bool(use_flash_accel),
               bool(use_qwen_emo), adapter_path)
    with MODEL_LOCK:
        context = _context()
        if _CONTEXT != context:
            unload()
        if _fake():
            from ..fake import FakeTTS
            if _MODEL is None or _OPTIONS != options:
                if adapter_path and not (Path(adapter_path) / "adapter_config.json").is_file():
                    raise ServiceError("LoRA adapter is missing adapter_config.json: {0}", adapter_path)
                _MODEL = FakeTTS(options)
                _OPTIONS = options
                _CONTEXT = context
            return _MODEL
        if not model_files_present():
            raise ServiceError("IndexTTS 2.5 model is missing. Restart with 2- run.bat so the automatic model "
                               "downloader can repair it.", status=409, code="model_missing")
        import torch
        if not torch.cuda.is_available():
            raise ServiceError("CUDA is unavailable. This Easy GUI is configured as an NVIDIA CUDA build; rerun "
                               "1- install.bat and check your NVIDIA driver.", status=409, code="cuda_unavailable")
        caps = acceleration_capabilities()
        if options[2] and not caps["triton"]["ok"]:
            raise ServiceError("torch.compile was requested but Triton is unavailable: {0}. Rerun 1- install.bat.",
                               caps["triton"]["detail"])
        if options[3] and not caps["flash_attn"]["ok"]:
            raise ServiceError("FlashAttention acceleration was requested but flash_attn is unavailable: {0}. "
                               "Rerun 1- install.bat.", caps["flash_attn"]["detail"])
        if options[3] and not caps["triton"]["ok"]:
            raise ServiceError("FlashAttention acceleration was requested but Triton is unavailable: {0}. "
                               "Rerun 1- install.bat.", caps["triton"]["detail"])
        if options[3] and not options[0]:
            raise ServiceError("FlashAttention GPT acceleration requires BF16 in this Easy GUI. Enable BF16 and retry.")
        if _MODEL is not None and _OPTIONS == options:
            return _MODEL
        unload()
        from indextts.infer_v2_5 import IndexTTS2
        model_dir = paths.MODEL_DIR
        model = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir), device="cuda:0",
            use_bf16=options[0], use_cuda_kernel=False, use_gpt_latent=options[1],
            use_torch_compile=options[2], use_deepspeed=False,
            # Build the accelerated snapshot only after PEFT has been attached.
            use_accel=options[3] and not bool(adapter_path), use_qwen_emo=options[4],
        )
        if adapter_path:
            adapter = Path(adapter_path)
            if not (adapter / "adapter_config.json").is_file():
                raise ServiceError("LoRA adapter is missing adapter_config.json: {0}", adapter)
            from peft import PeftModel
            loaded = PeftModel.from_pretrained(model.gpt.gpt, str(adapter), is_trainable=False)
            if options[3]:
                # IndexTTS's AccelInferenceEngine snapshots self.gpt.state_dict()
                # during post_init. Merge first so the snapshot contains LoRA.
                model.gpt.gpt = loaded.merge_and_unload()
                model.gpt.use_accel = True
                model.gpt.accel_engine = None
                model.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=bool(options[0]))
            else:
                model.gpt.gpt = loaded
            print(f"[adapter] Loaded IndexTTS 2.5 LoRA: {adapter}", flush=True)
        _MODEL = model
        _OPTIONS = options
        _CONTEXT = context
        return _MODEL


def unload() -> None:
    global _MODEL, _OPTIONS, _CONTEXT
    with MODEL_LOCK:
        skip_cuda_cleanup = bool(_CONTEXT[0]) if _CONTEXT is not None else _fake()
        _MODEL = None
        _OPTIONS = None
        _CONTEXT = None
        gc.collect()
        if skip_cuda_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
