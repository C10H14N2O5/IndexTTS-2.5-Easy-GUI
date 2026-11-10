"""GUI settings in ``.runtime/gui_settings.json`` (merge-updated, other keys kept)."""
from __future__ import annotations

from tools.easy_gui_i18n import read_gui_settings, update_gui_settings

from .. import paths
from ..errors import ServiceError
from ..i18n import LANGUAGES
from .system import acceleration_capabilities

RUNTIME_KEYS = ("use_bf16", "use_gpt_latent", "use_torch_compile", "use_flash_accel")
DEFAULTS = {
    "ui_language": "zh_CN",
    "theme": "system",
    "completion_sound": True,
    "notifications": True,
    "use_bf16": True,
    "use_gpt_latent": False,
    "use_torch_compile": False,
    "use_flash_accel": False,
}
THEMES = ("system", "light", "dark")


def normalize_runtime(data: dict, caps: dict | None = None) -> dict:
    """1.x rules: compile needs Triton; FlashAttention needs flash_attn and Triton and forces BF16."""
    caps = caps or acceleration_capabilities()
    data = dict(data)
    data["use_torch_compile"] = bool(data["use_torch_compile"] and caps["triton"]["ok"])
    data["use_flash_accel"] = bool(data["use_flash_accel"] and caps["flash_attn"]["ok"] and caps["triton"]["ok"])
    if data["use_flash_accel"]:
        data["use_bf16"] = True
    return data


def load() -> dict:
    stored = read_gui_settings(paths.SETTINGS_FILE)
    data = dict(DEFAULTS)
    for key, default in DEFAULTS.items():
        if key not in stored:
            continue
        value = stored[key]
        if isinstance(default, bool):
            data[key] = bool(value)
        elif key == "ui_language" and value in LANGUAGES:
            data[key] = value
        elif key == "theme" and value in THEMES:
            data[key] = value
    return normalize_runtime(data)


def runtime() -> dict:
    data = load()
    return {key: data[key] for key in RUNTIME_KEYS}


def update(patch: dict) -> dict:
    unknown = sorted(set(patch) - set(DEFAULTS))
    if unknown:
        raise ServiceError("Unknown setting: {0}", ", ".join(unknown))
    clean: dict = {}
    for key, value in patch.items():
        default = DEFAULTS[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ServiceError("Setting {0} must be true or false.", key)
            clean[key] = value
        elif key == "ui_language":
            if value not in LANGUAGES:
                raise ServiceError("Invalid interface language.")
            clean[key] = value
        elif key == "theme":
            if value not in THEMES:
                raise ServiceError("Invalid theme.")
            clean[key] = value
    caps = acceleration_capabilities()
    if clean.get("use_torch_compile") and not caps["triton"]["ok"]:
        raise ServiceError("torch.compile was requested but Triton is unavailable: {0}. Rerun 1- install.bat.",
                           caps["triton"]["detail"])
    if clean.get("use_flash_accel"):
        if not caps["flash_attn"]["ok"]:
            raise ServiceError("FlashAttention acceleration was requested but flash_attn is unavailable: {0}. Rerun 1- install.bat.",
                               caps["flash_attn"]["detail"])
        if not caps["triton"]["ok"]:
            raise ServiceError("FlashAttention acceleration was requested but Triton is unavailable: {0}. Rerun 1- install.bat.",
                               caps["triton"]["detail"])
    merged = {**load(), **clean}
    if merged["use_flash_accel"]:
        if clean.get("use_bf16") is False:
            raise ServiceError("FlashAttention GPT acceleration requires BF16 in this Easy GUI. Enable BF16 and retry.")
        merged["use_bf16"] = True
        clean["use_bf16"] = True
    update_gui_settings(paths.SETTINGS_FILE, clean)
    return load()
