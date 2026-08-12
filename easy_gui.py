from __future__ import annotations

import gc
import importlib
import json
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

if os.name == "nt":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "IndexTTS-2.5"
OUTPUTS_DIR = ROOT / "outputs"
VOICES_DIR = ROOT / "voices"
EMOTION_DESCRIPTIONS_DIR = ROOT / "emotion_descriptions"
RUNTIME_DIR = ROOT / ".runtime"
SETTINGS_FILE = RUNTIME_DIR / "gui_settings.json"
CACHE_DIR = RUNTIME_DIR / "hf-cache"
TEMP_DIR = RUNTIME_DIR / "temp"
ASSETS_DIR = ROOT / "assets"
DIALOGUE_MAX_SEGMENTS = 12
for p in (MODEL_DIR, OUTPUTS_DIR, VOICES_DIR, EMOTION_DESCRIPTIONS_DIR, CACHE_DIR, TEMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Keep project caches and temporary files local to this folder.
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE_DIR / "xet")
os.environ["HF_MODULES_CACHE"] = str(CACHE_DIR / "modules")
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TRITON_CACHE_DIR"] = str(RUNTIME_DIR / "triton-cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(RUNTIME_DIR / "torchinductor-cache")
os.environ["TORCH_EXTENSIONS_DIR"] = str(RUNTIME_DIR / "torch-extensions")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr

try:
    import winsound
except ImportError:
    winsound = None

APP_TITLE = "IndexTTS 2.5 Easy GUI"
MODEL_REPO = "IndexTeam/IndexTTS-2.5"
LANGUAGES = {
    "English": "EN",
    "Spanish": "ES",
    "Japanese": "JA",
    "Chinese": "ZH",
    "Arabic": "AR",
}
EMOTIONS = ["Happy", "Angry", "Sad", "Afraid", "Disgusted", "Melancholic", "Surprised", "Calm"]
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
NONE_SENTINEL = "__NONE__"

EMOTION_PRESETS = {
    "Neutral / balanced": [0, 0, 0, 0, 0, 0, 0, 0.55],
    "Warm / friendly": [0.30, 0, 0, 0, 0, 0, 0.05, 0.40],
    "Joyful": [0.65, 0, 0, 0, 0, 0, 0.10, 0.05],
    "Excited": [0.50, 0, 0, 0, 0, 0, 0.25, 0.05],
    "Playful": [0.40, 0, 0, 0, 0, 0, 0.25, 0.10],
    "Confident / proud": [0.20, 0.10, 0, 0, 0, 0, 0.05, 0.40],
    "Tender": [0.25, 0, 0.10, 0, 0, 0.10, 0, 0.30],
    "Romantic": [0.25, 0, 0.05, 0, 0, 0.20, 0, 0.25],
    "Hopeful": [0.30, 0, 0.05, 0.05, 0, 0.05, 0.10, 0.20],
    "Relieved": [0.20, 0, 0.05, 0, 0, 0, 0.05, 0.45],
    "Nostalgic": [0.10, 0, 0.15, 0, 0, 0.35, 0, 0.15],
    "Melancholic": [0, 0, 0.15, 0, 0, 0.55, 0, 0.10],
    "Sad": [0, 0, 0.65, 0, 0, 0.10, 0, 0.05],
    "Grieving": [0, 0, 0.55, 0.15, 0, 0.10, 0, 0],
    "Lonely": [0, 0, 0.35, 0.10, 0, 0.25, 0, 0.05],
    "Anxious": [0, 0, 0.05, 0.50, 0, 0.05, 0.10, 0.05],
    "Fearful": [0, 0, 0, 0.65, 0, 0, 0.10, 0.05],
    "Tense": [0, 0.15, 0, 0.35, 0, 0, 0.10, 0.15],
    "Angry": [0, 0.65, 0, 0.05, 0, 0, 0.05, 0.05],
    "Furious": [0, 0.75, 0, 0.05, 0, 0, 0, 0],
    "Irritated": [0, 0.40, 0, 0, 0.15, 0, 0, 0.20],
    "Disgusted": [0, 0.10, 0, 0, 0.60, 0, 0, 0.10],
    "Sarcastic": [0.10, 0.15, 0, 0, 0.25, 0, 0.10, 0.15],
    "Suspicious": [0, 0.10, 0, 0.20, 0.15, 0.10, 0.05, 0.15],
    "Surprised": [0.10, 0, 0, 0.05, 0, 0, 0.60, 0.05],
    "Amazed": [0.20, 0, 0, 0, 0, 0, 0.50, 0.05],
    "Shocked": [0, 0, 0, 0.20, 0, 0, 0.60, 0],
    "Urgent": [0, 0.25, 0, 0.25, 0, 0, 0.20, 0.10],
    "Determined": [0.10, 0.30, 0, 0, 0, 0, 0, 0.35],
    "Serious": [0, 0.10, 0.05, 0, 0, 0.05, 0, 0.55],
    "Documentary / neutral narrator": [0, 0, 0, 0, 0, 0, 0.05, 0.65],
    "News presenter": [0, 0, 0, 0, 0, 0.05, 0.05, 0.60],
    "Advertising / upbeat": [0.45, 0, 0, 0, 0, 0, 0.20, 0.10],
    "Advertising / luxury": [0.10, 0, 0, 0, 0, 0.15, 0, 0.50],
    "Storytelling / suspense": [0, 0, 0, 0.25, 0, 0.20, 0.10, 0.20],
    "Storytelling / intimate": [0.15, 0, 0.05, 0, 0, 0.15, 0, 0.40],
    "Cold / villainous": [0, 0.20, 0, 0.10, 0.20, 0.15, 0, 0.10],
}

DEFAULT_RUNTIME_SETTINGS = {
    "use_bf16": True,
    "use_gpt_latent": False,
    "use_torch_compile": False,
    "use_flash_accel": False,
}


CSS = """
.title-section { border-bottom: 2px solid #e5e7eb; margin-bottom: 20px; padding-bottom: 10px; }
.tabs { margin-top: 10px; }
.form-section { padding: 15px; border-radius: 8px; background-color: rgba(128,128,128,0.05); }
.input-field { margin-bottom: 15px; }
.button-primary { background-color: #2563eb !important; color: white !important; }
.button-stop { background-color: #ef4444 !important; color: white !important; }
.green-btn { background-color: #28a745 !important; color: white !important; border: none !important; }
.red-btn { background-color: #dc3545 !important; color: white !important; border: none !important; }
.dialogue-turn-card {
    padding: 20px 22px 18px 22px !important;
    margin: 4px 3px 16px 3px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(128,128,128,0.22) !important;
    background: rgba(128,128,128,0.055) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.025) inset !important;
}
.dialogue-turn-card > div {
    padding: 2px 3px !important;
}
.dialogue-turn-card .gr-row {
    gap: 14px !important;
}
.dialogue-turn-title {
    font-weight: 700 !important;
    opacity: .92;
    padding: 4px 0 3px 4px !important;
    margin: 0 0 7px 0 !important;
}
.dialogue-turn-title p {
    margin: 0 0 7px 0 !important;
    font-size: 1.04rem !important;
    line-height: 1.35 !important;
}
.dialogue-actions {
    padding: 5px 1px 0 10px !important;
    gap: 7px !important;
    border-left: 1px solid rgba(128,128,128,0.15);
}
.dialogue-actions button {
    min-width: 42px !important;
    margin-bottom: 3px !important;
}
.dialogue-turn-card .form,
.dialogue-turn-card .wrap {
    margin-top: 2px !important;
}
.dialogue-toolbar {
    margin: 2px 3px 14px 3px !important;
    padding: 2px 0 !important;
    gap: 8px !important;
}

.dialogue-emotion-source {
    margin-top: 10px !important;
    padding-top: 7px !important;
    border-top: 1px solid rgba(128,128,128,0.12) !important;
}
.dialogue-emotion-source label {
    margin-bottom: 5px !important;
}
.global-inference-controls { margin-top: 16px; padding: 14px; border-radius: 8px; background-color: rgba(128,128,128,0.06); }
.audio-safe-space { overflow: visible !important; padding-bottom: 28px !important; border: 0 !important; box-shadow: none !important; }
.audio-safe-space .wave, .audio-safe-space [data-testid="waveform"] { margin-bottom: 42px !important; padding-bottom: 34px !important; }
.audio-safe-space *, .audio-safe-space *::before, .audio-safe-space *::after { scrollbar-width: none !important; -ms-overflow-style: none !important; }
.audio-safe-space ::-webkit-scrollbar { width: 0 !important; height: 0 !important; display: none !important; }
.output-clean, .output-clean > div, .output-clean .wrap, .output-clean [data-testid="block-info"] { border: 0 !important; box-shadow: none !important; }
.output-path textarea { border: 0 !important; background: rgba(128,128,128,0.08) !important; min-height: 42px !important; }
.target-speech-box { position: relative; }
.compact textarea { font-family: ui-monospace, Consolas, monospace; }
.console-accordion, .console-accordion > div { border-radius: 8px !important; }
#indextts-console-body::-webkit-scrollbar { width: 5px; }
#indextts-console-body::-webkit-scrollbar-track { background: transparent; }
#indextts-console-body::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
.runtime-metrics textarea { font-family: Consolas, ui-monospace, monospace !important; }
.model-ready { padding: 9px 12px; border-radius: 8px; background: rgba(40,167,69,.10); }
.model-missing { padding: 9px 12px; border-radius: 8px; background: rgba(220,53,69,.10); }
"""

_MODEL = None
_MODEL_OPTIONS = None
_MODEL_LOCK = threading.RLock()


class _ConsoleMirror:
    def __init__(self, stream):
        self.stream = stream
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        if data:
            with CONSOLE_LOCK:
                _console_write(str(data))
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


CONSOLE_LINES = deque(maxlen=1400)
CONSOLE_LOCK = threading.Lock()
CONSOLE_CURRENT = ""
CONSOLE_OVERWRITE = False


def _clean_console_line(line: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(line)).rstrip()


def _console_commit(line: str) -> None:
    line = _clean_console_line(line)
    if line.strip():
        CONSOLE_LINES.append(line)


def _console_write(data: str) -> None:
    global CONSOLE_CURRENT, CONSOLE_OVERWRITE
    for ch in data:
        if ch == "\r":
            # tqdm/progress lines overwrite the current terminal line.
            current = _clean_console_line(CONSOLE_CURRENT)
            if current.strip():
                if CONSOLE_LINES and ("%" in current or "it/s" in current):
                    CONSOLE_LINES[-1] = current
                else:
                    CONSOLE_LINES.append(current)
            CONSOLE_CURRENT = ""
            CONSOLE_OVERWRITE = True
            continue
        if ch == "\n":
            _console_commit(CONSOLE_CURRENT)
            CONSOLE_CURRENT = ""
            CONSOLE_OVERWRITE = False
            continue
        if CONSOLE_OVERWRITE:
            CONSOLE_CURRENT = ""
            CONSOLE_OVERWRITE = False
        CONSOLE_CURRENT += ch


def _install_console_mirror():
    if getattr(sys.stdout, "_indextts_console_mirror", False):
        return

    class LockedMirror(_ConsoleMirror):
        _indextts_console_mirror = True

    sys.stdout = LockedMirror(sys.stdout)
    sys.stderr = LockedMirror(sys.stderr)


_HTML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
_ATTR_ESC = str.maketrans({"&": "&amp;", '"': "&quot;", "<": "&lt;", ">": "&gt;"})


def _console_line_color(line: str) -> str:
    low = line.lower()
    if "traceback" in low or "error" in low or "exception" in low or "failed" in low or "fatal" in low:
        return "#f87171"
    if "warn" in low or "fallback" in low:
        return "#fbbf24"
    if "[ui-progress]" in low or "%" in line or "it/s" in low:
        return "#34d399"
    if "[ui]" in low or "loading" in low or "starting inference" in low:
        return "#60a5fa"
    if "saved" in low or "complete" in low or "done" in low or "ready" in low:
        return "#4ade80"
    if "gpt_gen_time" in low or "s2mel_time" in low or "bigvgan_time" in low or "rtf:" in low:
        return "#a78bfa"
    return "#cccccc"


def console_html() -> str:
    with CONSOLE_LOCK:
        lines = list(CONSOLE_LINES)
        current = _clean_console_line(CONSOLE_CURRENT)
    if current.strip():
        lines.append(current)
    display = lines[-160:] or ["Idle. Waiting for inference activity."]

    rows = []
    for line in display:
        safe = line.translate(_HTML_ESC)
        rows.append(
            f'<div style="color:{_console_line_color(line)};white-space:pre-wrap;'
            f'overflow-wrap:anywhere;line-height:1.5">{safe}</div>'
        )
    body = "\n".join(rows)
    srcdoc = f"""<!doctype html>
<html><head><style>
html,body{{margin:0;background:#111;color:#ccc;font-family:Consolas,ui-monospace,monospace;font-size:12px;}}
#wrap{{height:285px;border-radius:8px;border:1px solid #333;overflow:hidden;box-sizing:border-box;}}
#body{{height:285px;overflow:auto;padding:9px 18px 9px 12px;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#555 transparent;}}
#body::-webkit-scrollbar{{width:5px;height:5px}}#body::-webkit-scrollbar-thumb{{background:#555;border-radius:3px}}
</style></head><body>
<div id="wrap"><div id="body">{body}<div id="anchor"></div></div></div>
<script>
const b=document.getElementById('body');
b.onscroll=()=>{{window._paused=!(b.scrollTop+b.clientHeight>=b.scrollHeight-40);}};
if(!window._paused)b.scrollTop=b.scrollHeight;
setTimeout(()=>{{if(!window._paused)b.scrollTop=b.scrollHeight;}},30);
</script></body></html>"""
    return (
        '<iframe scrolling="no" style="display:block;width:100%;height:285px;'
        'border:0;border-radius:8px;overflow:hidden;" '
        f'srcdoc="{srcdoc.translate(_ATTR_ESC)}"></iframe>'
    )


def _shared_runtime_metrics() -> str:
    try:
        import torch
        allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        return (
            f"GPU memory allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB | "
            f"model loaded: {'yes' if _MODEL is not None else 'no'}"
        )
    except Exception:
        return f"model loaded: {'yes' if _MODEL is not None else 'no'}"


def _audio_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or str(value)


def _model_files_present() -> bool:
    required = ["config.yaml", "gpt.pth", "codec.pth", "s2mel.pth"]
    if not all((MODEL_DIR / name).exists() for name in required):
        return False
    try:
        return sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file()) > 500_000_000
    except OSError:
        return False


def cuda_status() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return f"❌ CUDA unavailable | PyTorch {torch.__version__} | build CUDA: {torch.version.cuda}"
        return (
            f"✅ CUDA ready | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
            f"{torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM"
        )
    except Exception as exc:
        return f"❌ PyTorch/CUDA check failed: {type(exc).__name__}: {exc}"


def model_status() -> str:
    state = "✅ **IndexTTS 2.5 model ready**" if _model_files_present() else "⬇️ **IndexTTS 2.5 model is missing.**"
    detail = f"`{MODEL_DIR}`" if _model_files_present() else "`run.bat` downloads the official model automatically before the GUI starts."
    return f"{state}  \n{detail}  \n{cuda_status()}"


def _probe_distribution(module_name: str, distribution_name: str):
    """Detect an optional accelerator without importing it.

    Detect an optional accelerator without importing it, avoiding side effects
    during a simple status refresh.
    """
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return False, "not installed"
    except Exception as exc:
        return False, f"metadata error: {type(exc).__name__}: {exc}"

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return False, f"module lookup error: {type(exc).__name__}: {exc}"
    if spec is None:
        return False, f"{distribution_name} {version} installed, but module {module_name!r} was not found"
    return True, str(version)


def acceleration_capabilities():
    triton_ok, triton_detail = _probe_distribution("triton", "triton-windows")
    flash_ok, flash_detail = _probe_distribution("flash_attn", "flash-attn")
    return {
        "triton": (triton_ok, triton_detail),
        "flash_attn": (flash_ok, flash_detail),
    }


def acceleration_status() -> str:
    caps = acceleration_capabilities()
    def line(label, key):
        ok, detail = caps[key]
        return f"{'✅' if ok else '⚪'} **{label}:** {detail}"
    return "  \n".join([
        line("Triton / torch.compile", "triton"),
        line("FlashAttention GPT engine", "flash_attn"),
        "Optional accelerators are disabled by default. Run `install_accelerators.bat` to add or repair them.",
    ])


def _safe_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (value or "").strip())
    return safe.strip().replace(" ", "_")


def list_emotion_descriptions():
    return [p for p in sorted(EMOTION_DESCRIPTIONS_DIR.glob("*.txt")) if p.is_file()]


def emotion_description_choices(include_none=True):
    choices = [(p.stem.replace("_", " "), str(p)) for p in list_emotion_descriptions()]
    return [("None", NONE_SENTINEL), *choices] if include_none else choices


def refresh_emotion_description_dropdown(value=None):
    choices = emotion_description_choices()
    valid = {v for _, v in choices}
    selected = value if value in valid else NONE_SENTINEL
    return gr.update(choices=choices, value=selected)


def load_emotion_description(path):
    if not path or path == NONE_SENTINEL:
        return ""
    p = Path(path)
    if not p.is_file() or p.parent.resolve() != EMOTION_DESCRIPTIONS_DIR.resolve():
        return ""
    return p.read_text(encoding="utf-8").strip()


def save_emotion_description(name, description):
    safe = _safe_name(name)
    text = (description or "").strip()
    if not safe:
        raise gr.Error("Enter a reusable emotion-description name.")
    if not text:
        raise gr.Error("Enter an emotion description to save.")
    dst = EMOTION_DESCRIPTIONS_DIR / f"{safe}.txt"
    dst.write_text(text + "\n", encoding="utf-8")
    return (
        f"Saved emotion description: `{dst.name}`",
        refresh_emotion_description_dropdown(str(dst)),
        refresh_emotion_description_dropdown(str(dst)),
    )


def delete_emotion_description(path):
    if not path:
        return "No emotion description selected.", refresh_emotion_description_dropdown(), refresh_emotion_description_dropdown()
    p = Path(path)
    if p.parent.resolve() != EMOTION_DESCRIPTIONS_DIR.resolve():
        raise gr.Error("Only descriptions inside emotion_descriptions/ can be deleted here.")
    p.unlink(missing_ok=True)
    return f"Deleted emotion description: `{p.name}`", refresh_emotion_description_dropdown(), refresh_emotion_description_dropdown()


def apply_emotion_preset(name):
    values = EMOTION_PRESETS.get(name, EMOTION_PRESETS["Neutral / balanced"])
    return tuple(values)


def emotion_mode_visibility(mode):
    same = mode == "Same as voice reference"
    ref = mode == "Emotion reference audio"
    vector = mode == "Emotion vector"
    description = mode == "Emotion description"
    guided = not same
    random_ok = vector or description
    return (
        gr.update(visible=ref),
        gr.update(visible=vector),
        gr.update(visible=description),
        gr.update(visible=guided),
        gr.update(visible=random_ok),
    )


def _load_runtime_settings():
    data = dict(DEFAULT_RUNTIME_SETTINGS)
    try:
        if SETTINGS_FILE.is_file():
            loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key in data:
                if key in loaded:
                    data[key] = bool(loaded[key])
    except Exception:
        pass
    return data


def _save_runtime_settings(use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel):
    data = {
        "use_bf16": bool(use_bf16),
        "use_gpt_latent": bool(use_gpt_latent),
        "use_torch_compile": bool(use_torch_compile),
        "use_flash_accel": bool(use_flash_accel),
    }
    if data["use_flash_accel"]:
        data["use_bf16"] = True
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return "Runtime settings saved."


def normalize_runtime_settings(data, caps):
    data = dict(data)
    data["use_torch_compile"] = bool(data["use_torch_compile"] and caps["triton"][0])
    data["use_flash_accel"] = bool(data["use_flash_accel"] and caps["flash_attn"][0] and caps["triton"][0])
    if data["use_flash_accel"]:
        data["use_bf16"] = True
    return data


def on_flash_change(flash, bf16, gpt_latent, torch_compile):
    bf16 = True if flash else bool(bf16)
    _save_runtime_settings(bf16, gpt_latent, torch_compile, flash)
    return gr.update(value=bf16), "Runtime settings saved."


def _play_completion_sound():
    if winsound is None:
        return
    preferred = ASSETS_DIR / "inference_complete.wav"
    candidates = [preferred] if preferred.is_file() else sorted(ASSETS_DIR.glob("*.wav")) if ASSETS_DIR.is_dir() else []
    if not candidates:
        return
    try:
        winsound.PlaySound(str(candidates[0]), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception:
        pass


def list_voices():
    return [str(p) for p in sorted(VOICES_DIR.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]


def voice_choices(include_none=False):
    choices = [(Path(p).stem, p) for p in list_voices()]
    if include_none:
        return [("None", NONE_SENTINEL), *choices]
    return choices


def refresh_voice_dropdown(value=None, include_none=False):
    choices = voice_choices(include_none=include_none)
    valid = {v for _, v in choices}
    if include_none:
        selected = value if value in valid else NONE_SENTINEL
    else:
        selected = value if value in valid else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=selected)


def save_voice(audio, name):
    src = _audio_path(audio)
    name = (name or "").strip()
    if not src or not Path(src).exists():
        raise gr.Error("Choose or record a reference audio file first.")
    if not name:
        raise gr.Error("Enter a voice name.")
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")
    if not safe:
        raise gr.Error("Voice name contains no usable characters.")
    suffix = Path(src).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"
    dst = VOICES_DIR / f"{safe}{suffix}"
    shutil.copy2(src, dst)
    return f"Saved reference voice: `{dst}`", refresh_voice_dropdown(str(dst), include_none=True), refresh_voice_dropdown(str(dst), include_none=True)


def delete_selected_voice(path):
    if not path:
        return "No reference voice selected.", refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)
    p = Path(path)
    try:
        if p.parent.resolve() != VOICES_DIR.resolve():
            raise gr.Error("Only files inside the local voices library can be deleted here.")
        p.unlink(missing_ok=True)
    except Exception as exc:
        raise gr.Error(f"Could not delete voice: {type(exc).__name__}: {exc}")
    return f"Deleted reference voice: `{p.name}`", refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)


def delete_all_voices():
    count = 0
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
            p.unlink(missing_ok=True)
            count += 1
    return f"Deleted {count} reference voice file(s).", refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)


def delete_outputs():
    count = 0
    for p in OUTPUTS_DIR.iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)
            count += 1
    return f"Deleted {count} output file(s)."


def download_model():
    if _model_files_present():
        return model_status()
    try:
        from huggingface_hub import snapshot_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=MODEL_REPO, local_dir=str(MODEL_DIR), cache_dir=str(CACHE_DIR / "downloads"))
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(str(MODEL_DIR))
        return model_status()
    except Exception as exc:
        return f"❌ Model download failed: `{type(exc).__name__}: {exc}`"


def clear_download_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return "Local Hugging Face/Xet download cache cleared. Installed model files were kept."


def unload_model():
    global _MODEL, _MODEL_OPTIONS
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_OPTIONS = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return "Models unloaded and VRAM cleanup requested."


def delete_model():
    unload_model()
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return model_status()


def _load_model(
    use_bf16: bool, use_gpt_latent: bool, use_torch_compile: bool,
    use_flash_accel: bool, use_qwen_emo: bool,
):
    global _MODEL, _MODEL_OPTIONS
    options = (bool(use_bf16), bool(use_gpt_latent), bool(use_torch_compile), bool(use_flash_accel), bool(use_qwen_emo))
    with _MODEL_LOCK:
        if not _model_files_present():
            raise RuntimeError("IndexTTS 2.5 model is missing. Restart with run.bat so the automatic model downloader can repair it.")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. This Easy GUI is configured as an NVIDIA CUDA build; rerun install.bat and check your NVIDIA driver.")
        caps = acceleration_capabilities()
        if options[2] and not caps["triton"][0]:
            raise RuntimeError(f"torch.compile was requested but Triton is unavailable: {caps['triton'][1]}. Run install_accelerators.bat.")
        if options[3] and not caps["flash_attn"][0]:
            raise RuntimeError(f"FlashAttention acceleration was requested but flash_attn is unavailable: {caps['flash_attn'][1]}. Run install_accelerators.bat.")
        if options[3] and not caps["triton"][0]:
            raise RuntimeError(f"FlashAttention acceleration was requested but Triton is unavailable: {caps['triton'][1]}. Run install_accelerators.bat.")
        if options[3] and not options[0]:
            raise RuntimeError("FlashAttention GPT acceleration requires BF16 in this Easy GUI. Enable BF16 and retry.")
        if _MODEL is not None and _MODEL_OPTIONS == options:
            return _MODEL
        unload_model()
        from indextts.infer_v2_5 import IndexTTS2
        _MODEL = IndexTTS2(
            cfg_path=str(MODEL_DIR / "config.yaml"), model_dir=str(MODEL_DIR), device="cuda:0",
            use_bf16=options[0], use_cuda_kernel=False, use_gpt_latent=options[1],
            use_torch_compile=options[2], use_deepspeed=False, use_accel=options[3], use_qwen_emo=options[4],
        )
        _MODEL_OPTIONS = options
        return _MODEL


def generate(
    quick_sample, prompt_audio, text, language, emotion_mode, emotion_audio, emotion_alpha,
    happy, angry, sad, afraid, disgusted, melancholic, surprised, calm,
    emotion_text, random_emotion, duration_factor, interval_silence,
    text_normalization, max_text_tokens, do_sample, top_p, top_k, temperature,
    num_beams, repetition_penalty, length_penalty, max_mel_tokens,
    use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, save_outputs,
    # Do NOT enable track_tqdm here: IndexTTS S2Mel/FlowMatching owns its tqdm iterators.
    progress=gr.Progress(),
):
    progress(0.02, desc="Validating inference inputs")
    print("[ui] Preparing single inference", flush=True)
    # Quick Sample is authoritative when a saved library entry is selected.
    # The audio widget is used only when Quick Sample == None. This avoids
    # false "missing sample" validation caused by UI preview/update timing.
    if quick_sample and quick_sample != NONE_SENTINEL:
        saved_prompt = Path(str(quick_sample))
        prompt = str(saved_prompt) if saved_prompt.exists() else None
        if not prompt:
            raise gr.Error("The selected Quick Sample no longer exists. Choose another saved sample or select None for a custom upload.")
    else:
        prompt = _audio_path(prompt_audio)
        if not prompt or not Path(prompt).exists():
            raise gr.Error("Quick Sample is set to None. Upload or record a custom reference audio file before generating.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter target text to synthesize.")

    lang = LANGUAGES.get(language, "EN")
    use_qwen = emotion_mode == "Emotion description"
    progress(0.12, desc="Loading IndexTTS 2.5 runtime")
    print("[ui] Loading IndexTTS 2.5 runtime", flush=True)
    tts = _load_model(
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, use_qwen,
    )
    progress(0.28, desc="Preparing conditioning")

    emo_audio = None
    emo_vector = None
    use_emo_text = False
    emo_text = None
    if emotion_mode == "Emotion reference audio":
        emo_audio = _audio_path(emotion_audio)
        if not emo_audio or not Path(emo_audio).exists():
            raise gr.Error("Emotion reference mode requires a separate emotion audio file.")
    elif emotion_mode == "Emotion vector":
        emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    elif emotion_mode == "Emotion description":
        use_emo_text = True
        emo_text = (emotion_text or "").strip() or text

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUTS_DIR / f"indextts_2_5_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
    else:
        out = TEMP_DIR / "preview.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.34, desc="Synthesizing audio")
        print(f"[ui-progress] Synthesizing single utterance | chars={len(text)} | language={lang}", flush=True)
        start_time = time.perf_counter()
        result = tts.infer(
            spk_audio_prompt=prompt,
            text=text,
            output_path=str(out),
            lang=lang,
            emo_audio_prompt=emo_audio,
            emo_alpha=float(emotion_alpha),
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=bool(random_emotion),
            interval_silence=int(interval_silence),
            verbose=True,
            max_text_tokens_per_segment=int(max_text_tokens),
            stream_return=False,
            duration_factor=float(duration_factor),
            text_normalization=bool(text_normalization),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            max_mel_tokens=int(max_mel_tokens),
        )
        progress(0.92, desc="Saving output WAV")
        actual = out if out.exists() else Path(str(result))
        if not actual.exists():
            raise RuntimeError("IndexTTS returned without creating an output WAV.")
        saved = str(actual) if save_outputs else "Preview only (not persisted in outputs/)"
        status = (
            f"Generation complete.\nLanguage: {language} ({lang})\nEmotion mode: {emotion_mode}\n"
            f"Duration factor: {duration_factor}\nSaved WAV: {saved}"
        )
        elapsed = time.perf_counter() - start_time
        print(f"[ui] Generation complete in {elapsed:.2f}s | {actual}", flush=True)
        progress(0.97, desc="Completion notification")
        _play_completion_sound()
        progress(1.0, desc="Done")
        return str(actual), saved, status
    except Exception as exc:
        raise gr.Error(f"Generation failed: {type(exc).__name__}: {exc}")


def set_prep_voice(path):
    if not path or path == NONE_SENTINEL:
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"])
    p = Path(path)
    if not p.exists():
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"])
    return gr.update(value=str(p), interactive=False, sources=[])


def set_voice_inference(path):
    """Bind Quick Sample selection to the inference reference audio widget.

    None means custom input: clear the player and enable upload/recording.
    A saved library path means preview/use that exact file and lock editing so
    the dropdown remains the source of truth.
    """
    if not path or path == NONE_SENTINEL:
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"])
    p = Path(path)
    if not p.exists():
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"])
    return gr.update(value=str(p), interactive=False, sources=[])




def refresh_dialogue_libraries(*values):
    """Refresh choices without changing valid current row selections."""
    n = DIALOGUE_MAX_SEGMENTS
    voices = list(values[:n])
    descriptions = list(values[n:n * 2])
    updates = []
    valid_voice = {v for _, v in voice_choices(include_none=True)}
    valid_desc = {v for _, v in emotion_description_choices()}
    voice_choices_now = voice_choices(include_none=True)
    desc_choices_now = emotion_description_choices()
    for value in voices:
        selected = value if value in valid_voice else NONE_SENTINEL
        updates.append(gr.update(choices=voice_choices_now, value=selected))
    for value in descriptions:
        selected = value if value in valid_desc else NONE_SENTINEL
        updates.append(gr.update(choices=desc_choices_now, value=selected))
    return updates


def _dialogue_emotion_visibility(mode):
    return gr.update(visible=mode == "Emotion Vector Preset"), gr.update(visible=mode == "Emotion Description")


def _dialogue_default_row():
    return [NONE_SENTINEL, "", "Emotion Vector Preset", "Neutral / balanced", NONE_SENTINEL]


def _dialogue_unpack(values):
    rows = []
    for i in range(0, len(values), 5):
        row = list(values[i:i + 5])
        while len(row) < 5:
            row.append(_dialogue_default_row()[len(row)])
        rows.append(row)
    while len(rows) < DIALOGUE_MAX_SEGMENTS:
        rows.append(_dialogue_default_row())
    return rows[:DIALOGUE_MAX_SEGMENTS]


def _dialogue_updates(count, rows, message="Dialogue rows updated."):
    count = max(1, min(int(count), DIALOGUE_MAX_SEGMENTS))
    flat = []
    for row in rows[:DIALOGUE_MAX_SEGMENTS]:
        flat.extend(row)
    visibility = [gr.update(visible=i < count) for i in range(DIALOGUE_MAX_SEGMENTS)]
    return [count, *flat, *visibility, message]


def dialogue_reset_rows(*values):
    rows = [_dialogue_default_row() for _ in range(DIALOGUE_MAX_SEGMENTS)]
    return _dialogue_updates(2, rows, "Dialogue reset to two empty turns.")


def dialogue_clear_rows(count, *values):
    rows = _dialogue_unpack(values)
    for i in range(min(int(count), DIALOGUE_MAX_SEGMENTS)):
        rows[i] = _dialogue_default_row()
    return _dialogue_updates(count, rows, "Visible dialogue turns cleared.")


def dialogue_compact_rows(count, *values):
    rows = _dialogue_unpack(values)
    active = [r for r in rows[:int(count)] if str(r[1] or "").strip()]
    new_count = max(1, len(active))
    active += [_dialogue_default_row() for _ in range(DIALOGUE_MAX_SEGMENTS - len(active))]
    return _dialogue_updates(new_count, active, f"Removed empty turns. {new_count} turn(s) remain.")


def dialogue_add_after(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    if count >= DIALOGUE_MAX_SEGMENTS:
        return _dialogue_updates(count, rows, f"Maximum of {DIALOGUE_MAX_SEGMENTS} dialogue turns reached.")
    pos = min(int(index) + 1, count)
    rows.insert(pos, _dialogue_default_row())
    rows = rows[:DIALOGUE_MAX_SEGMENTS]
    return _dialogue_updates(count + 1, rows, f"Added a new turn after Turn {int(index)+1}.")


def dialogue_clone_row(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    if count >= DIALOGUE_MAX_SEGMENTS:
        return _dialogue_updates(count, rows, f"Maximum of {DIALOGUE_MAX_SEGMENTS} dialogue turns reached.")
    idx = min(max(0, int(index)), max(0, count - 1))
    rows.insert(idx + 1, list(rows[idx]))
    rows = rows[:DIALOGUE_MAX_SEGMENTS]
    return _dialogue_updates(count + 1, rows, f"Cloned Turn {idx+1}.")


def dialogue_delete_row(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    idx = min(max(0, int(index)), max(0, count - 1))
    if count <= 1:
        rows[0] = _dialogue_default_row()
        return _dialogue_updates(1, rows, "The last turn was cleared instead of removed.")
    rows.pop(idx)
    rows.append(_dialogue_default_row())
    return _dialogue_updates(count - 1, rows, f"Deleted Turn {idx+1}.")


def dialogue_clear_row(index, count, *values):
    rows = _dialogue_unpack(values)
    idx = min(max(0, int(index)), DIALOGUE_MAX_SEGMENTS - 1)
    rows[idx] = _dialogue_default_row()
    return _dialogue_updates(count, rows, f"Cleared Turn {idx+1}.")


def dialogue_move_row(index, direction, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    idx = min(max(0, int(index)), max(0, count - 1))
    target = idx + int(direction)
    if 0 <= target < count:
        rows[idx], rows[target] = rows[target], rows[idx]
        return _dialogue_updates(count, rows, f"Moved Turn {idx+1} {'up' if direction < 0 else 'down'}.")
    return _dialogue_updates(count, rows, "Turn is already at that boundary.")


def _concat_wavs(wav_paths, output_path, silence_s):
    wav_paths = [Path(x) for x in wav_paths]
    if not wav_paths:
        raise RuntimeError("No dialogue segments were generated.")
    with wave.open(str(wav_paths[0]), "rb") as first:
        params = first.getparams()
    silence_frames = max(0, int(float(silence_s) * params.framerate))
    silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for idx, src in enumerate(wav_paths):
            with wave.open(str(src), "rb") as inp:
                cur = inp.getparams()
                if (cur.nchannels, cur.sampwidth, cur.framerate, cur.comptype) != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
                    raise RuntimeError(f"Dialogue segment {idx + 1} has incompatible WAV parameters.")
                out.writeframes(inp.readframes(inp.getnframes()))
            if idx + 1 < len(wav_paths) and silence:
                out.writeframes(silence)


def generate_dialogue(language, duration_factor, interval_silence, text_normalization, max_text_tokens,
                      do_sample, top_p, top_k, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens,
                      use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, dialogue_silence, dialogue_row_count, *values,
                      # Do NOT enable track_tqdm here: it breaks IndexTTS FlowMatching tqdm on Gradio 5.x.
                      progress=gr.Progress()):
    progress(0.02, desc="Validating dialogue")
    print("[ui] Preparing Dialogue Builder inference", flush=True)
    # Per row: speaker, text, emotion_mode, vector_preset, description_path.
    rows = _dialogue_unpack(values)
    active = [r for r in rows[:max(1, min(int(dialogue_row_count), DIALOGUE_MAX_SEGMENTS))] if str(r[1] or "").strip()]
    if not active:
        raise gr.Error("Dialogue has no turns with text.")
    needs_qwen = any(r[2] == "Emotion Description" for r in active)
    progress(0.10, desc="Loading IndexTTS 2.5 runtime")
    tts = _load_model(use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, needs_qwen)
    lang = LANGUAGES.get(language, "EN")
    temp_dir = TEMP_DIR / f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    try:
        for idx, row in enumerate(active, 1):
            speaker, text, emo_mode, vector_preset, description_path = row
            base = 0.16 + 0.72 * ((idx - 1) / max(len(active), 1))
            progress(base, desc=f"Dialogue turn {idx}/{len(active)}")
            print(f"[ui-progress] Dialogue turn {idx}/{len(active)} | chars={len(str(text or ''))}", flush=True)
            if not speaker or speaker == NONE_SENTINEL or not Path(str(speaker)).is_file():
                raise gr.Error(f"Dialogue turn {idx} requires a saved voice-library speaker.")
            seg = temp_dir / f"segment_{idx:02d}.wav"
            kwargs = dict(emo_alpha=0.65, use_random=False, emo_vector=None, use_emo_text=False, emo_text=None)
            if emo_mode == "Emotion Description":
                desc = load_emotion_description(description_path)
                if not desc:
                    raise gr.Error(f"Dialogue turn {idx}: choose a saved Emotion Description.")
                kwargs.update(use_emo_text=True, emo_text=desc)
            else:
                kwargs.update(emo_vector=list(EMOTION_PRESETS.get(vector_preset, EMOTION_PRESETS["Neutral / balanced"])))
            tts.infer(
                spk_audio_prompt=str(speaker), text=(text or "").strip(), output_path=str(seg), lang=lang,
                interval_silence=int(interval_silence), verbose=True, max_text_tokens_per_segment=int(max_text_tokens),
                stream_return=False, duration_factor=float(duration_factor), text_normalization=bool(text_normalization),
                do_sample=bool(do_sample), top_p=float(top_p), top_k=int(top_k), temperature=float(temperature),
                num_beams=int(num_beams), repetition_penalty=float(repetition_penalty), length_penalty=float(length_penalty),
                max_mel_tokens=int(max_mel_tokens), **kwargs
            )
            segments.append(seg)
            progress(0.16 + 0.72 * (idx / max(len(active), 1)), desc=f"Completed turn {idx}/{len(active)}")
        progress(0.91, desc="Assembling dialogue WAV")
        out = OUTPUTS_DIR / f"indextts_2_5_dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        _concat_wavs(segments, out, dialogue_silence)
        progress(0.97, desc="Completion notification")
        _play_completion_sound()
        print(f"[ui] Dialogue complete | turns={len(active)} | {out}", flush=True)
        progress(1.0, desc="Done")
        return str(out), str(out), f"Dialogue complete. Generated {len(active)} turns."
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def _ensure_model_before_ui():
    from tools.ensure_model import main as ensure_model_main
    rc = ensure_model_main()
    if rc != 0:
        raise RuntimeError("Automatic IndexTTS 2.5 model setup failed. Review the console output and retry.")


_install_console_mirror()

if __name__ == "__main__":
    _ensure_model_before_ui()


_INITIAL_CAPS = acceleration_capabilities()
_INITIAL_RUNTIME = normalize_runtime_settings(_load_runtime_settings(), _INITIAL_CAPS)


with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=6):
            gr.Markdown("# 🗣️ IndexTTS 2.5 Easy GUI")
        with gr.Column(scale=1, min_width=110):
            gr.Markdown("[📖 IndexTTS](https://github.com/index-tts/index-tts)")
        with gr.Column(scale=3, min_width=520):
            with gr.Row():
                top_unload_btn = gr.Button("Unload Models", size="sm")
                delete_outputs_btn = gr.Button("Delete output audios", size="sm", variant="stop", elem_classes=["red-btn"])
                delete_voices_btn = gr.Button("Delete reference samples", size="sm", variant="stop", elem_classes=["red-btn"])

    top_status = gr.Markdown(model_status())

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("🎙️ Prep Samples", id="tab_prep_samples"):
            gr.Markdown(
                "### 📚 Reference Audio Library\n"
                "Manage reusable zero-shot voice references for IndexTTS 2.5. Upload or record a clean single-speaker clip and save it locally. No transcript is required by IndexTTS 2.5."
            )
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 📂 Your Samples")
                    sample_dropdown = gr.Dropdown(choices=voice_choices(include_none=True), value=NONE_SENTINEL, label="Select Sample", interactive=True, info="None enables upload/recording; saved samples load their waveform for preview.")
                    with gr.Row():
                        refresh_samples_btn = gr.Button("🔄 Refresh List", size="sm")
                        delete_sample_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop", elem_classes="red-btn")
                    sample_status = gr.Markdown()
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Reference Audio Editor")
                    prep_audio = gr.Audio(
                        label="Reference Audio (short, clean, single speaker recommended - use Trim to edit)",
                        type="filepath", interactive=True, sources=["upload", "microphone"], elem_classes=["audio-safe-space"]
                    )
                    voice_name = gr.Textbox(label="Voice Name", placeholder="e.g. narrator_es", lines=1)
                    save_voice_btn = gr.Button("💾 Save Reference Voice", variant="primary", elem_classes="green-btn")
                    gr.Markdown("Saved files are copied into `voices/` and remain available across sessions.")

            gr.Markdown("### 📝 Emotion Description Library")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 📂 Saved Descriptions")
                    prep_emo_desc_select = gr.Dropdown(choices=emotion_description_choices(), value=NONE_SENTINEL, label="Select Description", interactive=True, info="None clears the editor; saved entries display their stored text.")
                    with gr.Row():
                        refresh_emo_desc_btn = gr.Button("🔄 Refresh List", size="sm")
                        delete_emo_desc_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop", elem_classes="red-btn")
                    emo_desc_status = gr.Markdown()
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### ✍️ Emotion Description Editor")
                    prep_emo_desc_name = gr.Textbox(label="Description Name", placeholder="e.g. warm_documentary", lines=1)
                    prep_emo_desc_text = gr.Textbox(
                        label="Emotion Description", lines=4,
                        placeholder="Natural-language emotion/prosody instruction, e.g. warm, intimate, reassuring delivery with restrained excitement."
                    )
                    save_emo_desc_btn = gr.Button("💾 Save Emotion Description", variant="primary", elem_classes="green-btn")
                    gr.Markdown("Descriptions are stored as plain UTF-8 text under `emotion_descriptions/` and can be recalled from Inference.")

        with gr.Tab("🔊 Inference"):
            gr.Markdown("### 🎙️ Unified IndexTTS 2.5 Voice Synthesis")

            with gr.Group(elem_classes="global-inference-controls"):
                gr.Markdown("#### 🎛️ Global Inference Controls")
                with gr.Row():
                    language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
                    duration_factor = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Duration Factor", info="<1 faster, >1 slower")
                    text_normalization = gr.Checkbox(value=True, label="Text normalization")
                    interval_silence = gr.Slider(0, 1000, 200, step=10, label="Segment Silence (ms)")

                gr.Markdown("#### 🛠️ Advanced Engine Settings")
                with gr.Accordion("⚙️ Decoding & Sampling Parameters", open=False):
                    with gr.Row():
                        do_sample = gr.Checkbox(True, label="Sampling", info="Uses IndexTTS do_sample. The bundled 2.5 source is patched so this switch is honored instead of being hard-coded on.")
                        temperature = gr.Slider(0.1, 2.0, 0.8, step=0.05, label="Temperature")
                        top_p = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="Top P")
                        top_k = gr.Slider(0, 200, 30, step=1, label="Top K")
                    with gr.Row():
                        num_beams = gr.Slider(1, 10, 3, step=1, label="Num Beams")
                        repetition_penalty = gr.Number(value=10.0, minimum=0.1, maximum=30.0, step=0.1, label="Repetition Penalty")
                        length_penalty = gr.Number(value=0.0, minimum=-3.0, maximum=3.0, step=0.1, label="Length Penalty")
                        max_mel_tokens = gr.Slider(50, 3000, 1500, step=10, label="Max Mel Tokens")
                    max_text_tokens = gr.Slider(20, 300, 120, step=5, label="Max Text Tokens / Segment")

                with gr.Accordion("🎭 Emotion Controls", open=False):
                    emotion_mode = gr.Radio(
                        ["Same as voice reference", "Emotion reference audio", "Emotion vector", "Emotion description"],
                        value="Same as voice reference", label="Emotion Mode"
                    )
                    with gr.Group(visible=False) as emotion_strength_group:
                        emotion_alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Emotion Weight / Alpha")
                    with gr.Group(visible=False) as emotion_ref_group:
                        emotion_audio = gr.Audio(
                            label="Separate Emotion Reference", type="filepath", sources=["upload", "microphone"],
                            elem_classes="audio-safe-space"
                        )
                    with gr.Group(visible=False) as emotion_random_group:
                        random_emotion = gr.Checkbox(False, label="Random emotion sampling", info="IndexTTS use_random. May reduce strict cloning fidelity.")
                    with gr.Group(visible=False) as emotion_vector_group:
                        emotion_preset = gr.Dropdown(
                            choices=list(EMOTION_PRESETS), value="Neutral / balanced",
                            label="Emotion Vector Preset",
                            info="Convenience presets authored for this GUI. They directly set the official 8-value IndexTTS emotion vector; adjust sliders afterward if desired."
                        )
                        gr.Markdown("##### Emotion Vector — [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
                        emotion_sliders = []
                        for i in range(0, 8, 2):
                            with gr.Row():
                                emotion_sliders.append(gr.Slider(0, 1, EMOTION_PRESETS["Neutral / balanced"][i], step=0.01, label=EMOTIONS[i]))
                                emotion_sliders.append(gr.Slider(0, 1, EMOTION_PRESETS["Neutral / balanced"][i + 1], step=0.01, label=EMOTIONS[i + 1]))
                    with gr.Group(visible=False) as emotion_description_group:
                        saved_emotion_description = gr.Dropdown(
                            choices=emotion_description_choices(), value=NONE_SENTINEL, label="Saved Emotion Description",
                            info="Selecting a library entry fills the Emotion Description box below."
                        )
                        emotion_text = gr.Textbox(
                            label="Emotion Description", lines=3,
                            placeholder="Describe the desired emotion/prosody. QwenEmotion is loaded only for this mode."
                        )

                with gr.Accordion("🚀 CUDA Runtime", open=False):
                    _caps = _INITIAL_CAPS
                    _runtime_values = _INITIAL_RUNTIME
                    with gr.Row():
                        use_bf16 = gr.Checkbox(_runtime_values["use_bf16"], label="BF16", info="Recommended on supported NVIDIA GPUs. Automatically enabled by FlashAttention mode.")
                        use_gpt_latent = gr.Checkbox(_runtime_values["use_gpt_latent"], label="GPT latent path", info="IndexTTS 2.5 load-time use_gpt_latent option; temporal alignment compatibility fix is bundled.")
                        use_torch_compile = gr.Checkbox(
                            _runtime_values["use_torch_compile"], label="torch.compile (Triton)", interactive=_caps["triton"][0],
                            info="Optional S2Mel compilation. Requires triton-windows.",
                        )
                    with gr.Row():
                        use_flash_accel = gr.Checkbox(
                            _runtime_values["use_flash_accel"], label="FlashAttention GPT acceleration", interactive=_caps["flash_attn"][0] and _caps["triton"][0],
                            info="Optional IndexTTS 2.5 GPT acceleration engine. Requires the prebuilt FlashAttention Windows wheel + Triton + BF16.",
                        )
                    acceleration_status_box = gr.Markdown(acceleration_status())
                    refresh_acceleration_btn = gr.Button("🔄 Refresh Accelerator Status", size="sm")
                    runtime_settings_status = gr.Markdown("Runtime booleans are saved automatically in `.runtime/gui_settings.json`.")

            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="form-section"):
                            gr.Markdown("#### 🎙️ Voice Sample")
                            with gr.Row():
                                infer_sample_select = gr.Dropdown(choices=voice_choices(include_none=True), value=NONE_SENTINEL, label="Quick Sample Select", scale=4)
                                refresh_infer_sample_btn = gr.Button("🔄", scale=1, min_width=50)
                            infer_ref_audio = gr.Audio(
                                label="Reference Audio / Quick Sample Preview", type="filepath", value=None,
                                sources=["upload", "microphone"], interactive=True, elem_classes=["audio-safe-space"]
                            )
                            gr.Markdown(
                                "Choose **None** to upload or record a custom reference directly here. "
                                "Choosing a saved Quick Sample loads its waveform/player automatically so you can preview it before generation."
                            )

                        with gr.Column(scale=1, elem_classes=["form-section", "target-speech-box"]):
                            gr.Markdown("#### ✍️ Target Speech")
                            infer_text = gr.Textbox(label="Target Text", placeholder="Enter the text to synthesize...", lines=8, value="", elem_classes=["compact"])
                            with gr.Row():
                                infer_gen_btn = gr.Button("⚡ Generate Speech", variant="primary", size="lg", scale=4, elem_classes="button-primary")
                                infer_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, min_width=90, elem_classes="button-stop")
                                infer_save_outputs = gr.Checkbox(label="Save to outputs", value=True, scale=1, min_width=130)
                            with gr.Group(elem_classes=["output-clean"]):
                                infer_audio_out = gr.Audio(label="Generated Audio", type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                                infer_file_out = gr.Textbox(label="Saved WAV", interactive=False, elem_classes=["output-path"])
                            generation_status = gr.Textbox(label="Generation Status", interactive=False, lines=3, max_lines=3, elem_classes="compact")


                with gr.Tab("Dialogue Builder"):
                    gr.Markdown(
                        "#### 💬 Multi-Speaker Dialogue Builder\n"
                        "Modular turn workflow. Global language, decoding, timing and CUDA Runtime controls above are shared with Single Inference. "
                        "Each visible turn has its own voice, text and mutually exclusive emotion source."
                    )
                    dialogue_components = []
                    dialogue_mode_bindings = []
                    dialogue_voice_dropdowns = []
                    dialogue_description_dropdowns = []
                    dialogue_rows = []
                    dialogue_add_btns = []
                    dialogue_clone_btns = []
                    dialogue_delete_btns = []
                    dialogue_clear_btns = []
                    dialogue_up_btns = []
                    dialogue_down_btns = []
                    dialogue_row_count = gr.State(2)

                    with gr.Row(elem_classes="dialogue-toolbar"):
                        dialogue_reset_btn = gr.Button("Reset rows", size="sm", variant="secondary")
                        dialogue_clear_all_btn = gr.Button("Clear rows", size="sm", variant="secondary")
                        dialogue_compact_btn = gr.Button("Remove empty rows", size="sm", variant="secondary")
                        dialogue_refresh_libraries_btn = gr.Button("🔄 Refresh Voice + Emotion Libraries", size="sm")

                    for i in range(DIALOGUE_MAX_SEGMENTS):
                        with gr.Group(visible=i < 2, elem_classes=["dialogue-turn-card"]) as d_row:
                            with gr.Row():
                                with gr.Column(scale=2, min_width=190):
                                    gr.Markdown(f"**Turn {i+1}**", elem_classes="dialogue-turn-title")
                                    d_speaker = gr.Dropdown(
                                        choices=voice_choices(include_none=True), value=NONE_SENTINEL,
                                        label="Voice Library", interactive=True,
                                    )
                                with gr.Column(scale=7):
                                    d_text = gr.Textbox(
                                        label=f"Text {i+1}", placeholder=f"Enter dialogue text for turn {i+1}...",
                                        lines=4,
                                    )
                                with gr.Column(scale=1, min_width=92, elem_classes=["dialogue-actions"]):
                                    d_add = gr.Button("➕", variant="secondary", size="sm", elem_classes=["green-btn"])
                                    d_clone = gr.Button("📋", variant="secondary", size="sm")
                                    with gr.Row():
                                        d_up = gr.Button("⬆️", variant="secondary", size="sm", min_width=42)
                                        d_down = gr.Button("⬇️", variant="secondary", size="sm", min_width=42)
                                    d_clear = gr.Button("🧹", variant="secondary", size="sm")
                                    d_delete = gr.Button("🗑️", variant="stop", size="sm", elem_classes=["red-btn"])

                            d_mode = gr.Radio(
                                ["Emotion Vector Preset", "Emotion Description"],
                                value="Emotion Vector Preset", label="Emotion Source",
                                elem_classes=["dialogue-emotion-source"],
                            )
                            with gr.Group(visible=True) as d_vector_group:
                                d_vector = gr.Dropdown(
                                    choices=list(EMOTION_PRESETS), value="Neutral / balanced",
                                    label="Emotion Vector Preset",
                                )
                            with gr.Group(visible=False) as d_desc_group:
                                d_desc = gr.Dropdown(
                                    choices=emotion_description_choices(), value=NONE_SENTINEL,
                                    label="Emotion Description Library",
                                )
                                d_desc_preview = gr.Textbox(
                                    label="Saved Emotion Description", value="", lines=2,
                                    interactive=False,
                                )
                                d_desc.change(load_emotion_description, d_desc, d_desc_preview)

                            dialogue_components.extend([d_speaker, d_text, d_mode, d_vector, d_desc])
                            dialogue_mode_bindings.append((d_mode, d_vector_group, d_desc_group))
                            dialogue_voice_dropdowns.append(d_speaker)
                            dialogue_description_dropdowns.append(d_desc)
                            dialogue_rows.append(d_row)
                            dialogue_add_btns.append(d_add)
                            dialogue_clone_btns.append(d_clone)
                            dialogue_delete_btns.append(d_delete)
                            dialogue_clear_btns.append(d_clear)
                            dialogue_up_btns.append(d_up)
                            dialogue_down_btns.append(d_down)

                    dialogue_silence = gr.Slider(0.0, 5.0, value=0.5, step=0.1, label="Silence between dialogue turns (seconds)")
                    with gr.Row():
                        dialogue_gen_btn = gr.Button("⚡ Generate Dialogue", variant="primary", size="lg", scale=4, elem_classes="button-primary")
                        dialogue_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, min_width=90, elem_classes="button-stop")
                    with gr.Group(elem_classes=["output-clean"]):
                        dialogue_audio_out = gr.Audio(label="Generated Dialogue", type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                        dialogue_file_out = gr.Textbox(label="Saved WAV", interactive=False, elem_classes=["output-path"])
                    dialogue_status = gr.Textbox(label="Dialogue Status", interactive=False, lines=4)

            with gr.Accordion("📖 IndexTTS 2.5 Control Reference", open=False):
                gr.Markdown(
                
                "**Language** selects the language hint passed to IndexTTS (EN/ES/JA/ZH/AR). **Duration Factor** scales predicted duration: `1.0` neutral, `<1` faster/shorter, `>1` slower/longer. **Text normalization** enables the model's normalization pipeline before tokenization. **Segment Silence** inserts silence between automatically split long-text segments.\n\n"
                "**Sampling (`do_sample`)** enables stochastic GPT token selection. **Temperature** changes sampling sharpness; lower is more conservative, higher more varied. **Top P** nucleus-filters the probability mass; **Top K** limits candidates to the K most likely tokens. **Num Beams** controls beam search width. **Repetition Penalty** discourages repeated token patterns. **Length Penalty** biases beam scoring by sequence length. **Max Mel Tokens** caps generated acoustic-token length. **Max Text Tokens / Segment** controls automatic text splitting before synthesis. IndexTTS 2.5 exposes no dedicated public seed argument in this inference API.\n\n"
                "**Emotion Mode — Same as voice reference:** no separate emotion condition is supplied; prosody/emotion follows the speaker prompt. **Emotion reference audio:** uses a second audio prompt for emotion while retaining timbre from the voice reference. **Emotion vector:** sends the official eight-value vector `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`; the GUI presets are convenience combinations and the sliders remain editable. **Emotion description:** sends natural-language guidance through QwenEmotion. Saved descriptions only copy text into the box; selecting `None` clears it. **Emotion Weight / Alpha** controls conditioning strength; stronger is not always more natural. **Random emotion sampling** enables IndexTTS `use_random`, adding emotion variability at the expense of strict repeatability.\n\n"
                "**BF16** loads/executes supported model paths in bfloat16 to reduce VRAM and improve throughput on compatible NVIDIA GPUs; FlashAttention mode requires it here. **GPT latent path** enables the alternate latent conditioning path in the bundled 2.5 source; this GUI includes a temporal-length alignment fix for the observed 2× latent mismatch. **torch.compile / Triton** asks PyTorch to compile supported S2Mel/runtime graphs; first use can be slower while local kernels are compiled/cached. **FlashAttention GPT acceleration** enables the IndexTTS `use_accel` GPT path and requires the prebuilt Windows FlashAttention wheel plus Triton. It is optional and can be disabled for maximum compatibility.\n\n"
                "**BigVGAN:** BigVGAN itself remains the final neural vocoder that converts the model's acoustic/mel representation into waveform audio. Only its optional fused custom CUDA activation kernel is disabled in this fixed Windows runtime. The normal PyTorch BigVGAN path remains active, so this is a performance choice rather than a quality-mode switch.\n\n"
                "**Dialogue Builder:** `➕` inserts a new turn after the current one, `📋` clones it, `⬆️/⬇️` reorder it, `🧹` clears only that turn, and `🗑️` removes it while shifting later turns upward. `Reset rows`, `Clear rows`, and `Remove empty rows` operate on the complete dialogue. Per-turn Voice Library and emotion settings override only that turn; all generation/runtime controls above remain global and shared with Single Inference.\n\n"
                "Official references: [IndexTTS repository](https://github.com/index-tts/index-tts) · [IndexTTS 2.5 Technical Report & demos](https://index-tts.github.io/index-tts2-5.github.io/)"
                )


    with gr.Accordion("🖥️ Runtime Status", open=True, elem_classes=["console-accordion"]):
        runtime_status = gr.Markdown(cuda_status())
        with gr.Row():
            activity_status = gr.Textbox(
                label="Activity Status", value="Idle.", interactive=False, lines=1, scale=2,
            )
            runtime_metrics = gr.Textbox(
                label="Runtime Metrics", value=_shared_runtime_metrics(), interactive=False,
                lines=1, scale=3, elem_classes=["runtime-metrics"],
            )

    with gr.Accordion("🖥️ Embedded Console", open=False, elem_classes=["console-accordion"]):
        gr.Markdown(
        )
        cmd_console = gr.HTML(value=console_html())
        cmd_console_timer = gr.Timer(0.5)
        runtime_metrics_timer = gr.Timer(1.0)

    top_unload_btn.click(unload_model, outputs=runtime_status).then(model_status, outputs=top_status)
    delete_outputs_btn.click(delete_outputs, outputs=runtime_status)
    delete_voices_btn.click(delete_all_voices, outputs=[runtime_status, sample_dropdown, infer_sample_select])

    save_voice_btn.click(save_voice, [prep_audio, voice_name], [sample_status, sample_dropdown, infer_sample_select])
    refresh_samples_btn.click(lambda value: refresh_voice_dropdown(value, include_none=True), sample_dropdown, sample_dropdown)
    refresh_infer_sample_btn.click(lambda value: refresh_voice_dropdown(value, include_none=True), infer_sample_select, infer_sample_select)
    delete_sample_btn.click(delete_selected_voice, sample_dropdown, [sample_status, sample_dropdown, infer_sample_select])
    sample_dropdown.change(set_prep_voice, sample_dropdown, prep_audio)
    infer_sample_select.change(set_voice_inference, infer_sample_select, infer_ref_audio)

    save_emo_desc_btn.click(
        save_emotion_description, [prep_emo_desc_name, prep_emo_desc_text],
        [emo_desc_status, prep_emo_desc_select, saved_emotion_description],
    )
    refresh_emo_desc_btn.click(refresh_emotion_description_dropdown, prep_emo_desc_select, prep_emo_desc_select)
    delete_emo_desc_btn.click(
        delete_emotion_description, prep_emo_desc_select,
        [emo_desc_status, prep_emo_desc_select, saved_emotion_description],
    ).then(lambda: "", outputs=emotion_text)
    prep_emo_desc_select.change(lambda path: ((Path(path).stem.replace("_", " ") if path and path != NONE_SENTINEL else ""), load_emotion_description(path)), prep_emo_desc_select, [prep_emo_desc_name, prep_emo_desc_text])
    saved_emotion_description.change(load_emotion_description, saved_emotion_description, emotion_text)

    emotion_mode.change(
        emotion_mode_visibility, emotion_mode,
        [emotion_ref_group, emotion_vector_group, emotion_description_group, emotion_strength_group, emotion_random_group],
    )
    emotion_preset.change(apply_emotion_preset, emotion_preset, emotion_sliders)

    gen_inputs = [
        infer_sample_select, infer_ref_audio, infer_text, language, emotion_mode, emotion_audio, emotion_alpha,
        *emotion_sliders, emotion_text, random_emotion, duration_factor, interval_silence,
        text_normalization, max_text_tokens, do_sample, top_p, top_k, temperature,
        num_beams, repetition_penalty, length_penalty, max_mel_tokens,
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, infer_save_outputs,
    ]
    infer_gen_btn.click(lambda: "Single inference running…", outputs=activity_status, queue=False, show_progress="hidden")
    generation_event = infer_gen_btn.click(generate, gen_inputs, [infer_audio_out, infer_file_out, generation_status])
    generation_event.then(lambda status: status, generation_status, activity_status, queue=False, show_progress="hidden")
    generation_event.then(_shared_runtime_metrics, None, runtime_metrics, queue=False, show_progress="hidden")
    infer_stop_btn.click(lambda: "Single inference stop requested.", outputs=activity_status, queue=False, show_progress="hidden")
    infer_stop_btn.click(
        lambda: "Stop requested. The queued Gradio generation event was cancelled; an already-running CUDA kernel returns control before cancellation can fully complete.",
        outputs=generation_status,
        cancels=[generation_event],
    )

    for d_mode, d_vector_group, d_desc_group in dialogue_mode_bindings:
        d_mode.change(_dialogue_emotion_visibility, d_mode, [d_vector_group, d_desc_group])

    dialogue_refresh_libraries_btn.click(
        refresh_dialogue_libraries,
        inputs=[*dialogue_voice_dropdowns, *dialogue_description_dropdowns],
        outputs=[*dialogue_voice_dropdowns, *dialogue_description_dropdowns],
        queue=False, show_progress="hidden",
    )

    dialogue_edit_outputs = [dialogue_row_count, *dialogue_components, *dialogue_rows, dialogue_status]
    dialogue_edit_inputs = [dialogue_row_count, *dialogue_components]

    dialogue_reset_btn.click(
        dialogue_reset_rows, dialogue_components, dialogue_edit_outputs,
        queue=False, show_progress="hidden",
    )
    dialogue_clear_all_btn.click(
        dialogue_clear_rows, dialogue_edit_inputs, dialogue_edit_outputs,
        queue=False, show_progress="hidden",
    )
    dialogue_compact_btn.click(
        dialogue_compact_rows, dialogue_edit_inputs, dialogue_edit_outputs,
        queue=False, show_progress="hidden",
    )

    for i in range(DIALOGUE_MAX_SEGMENTS):
        idx_state = gr.State(i)
        dialogue_add_btns[i].click(
            dialogue_add_after, [idx_state, *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )
        dialogue_clone_btns[i].click(
            dialogue_clone_row, [idx_state, *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )
        dialogue_delete_btns[i].click(
            dialogue_delete_row, [idx_state, *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )
        dialogue_clear_btns[i].click(
            dialogue_clear_row, [idx_state, *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )
        dialogue_up_btns[i].click(
            dialogue_move_row, [idx_state, gr.State(-1), *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )
        dialogue_down_btns[i].click(
            dialogue_move_row, [idx_state, gr.State(1), *dialogue_edit_inputs], dialogue_edit_outputs,
            queue=False, show_progress="hidden",
        )

    dialogue_inputs = [
        language, duration_factor, interval_silence, text_normalization, max_text_tokens,
        do_sample, top_p, top_k, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens,
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, dialogue_silence,
        dialogue_row_count, *dialogue_components,
    ]
    dialogue_gen_btn.click(lambda: "Dialogue generation running…", outputs=activity_status, queue=False, show_progress="hidden")
    dialogue_event = dialogue_gen_btn.click(generate_dialogue, dialogue_inputs, [dialogue_audio_out, dialogue_file_out, dialogue_status])
    dialogue_stop_btn.click(lambda: "Dialogue stop requested.", outputs=activity_status, queue=False, show_progress="hidden")
    dialogue_stop_btn.click(lambda: "Stop requested.", outputs=dialogue_status, cancels=[dialogue_event])

    refresh_acceleration_btn.click(acceleration_status, outputs=acceleration_status_box)

    runtime_inputs = [use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel]
    for runtime_control in (use_bf16, use_gpt_latent, use_torch_compile):
        runtime_control.change(_save_runtime_settings, runtime_inputs, runtime_settings_status)
    use_flash_accel.change(
        on_flash_change, [use_flash_accel, use_bf16, use_gpt_latent, use_torch_compile],
        [use_bf16, runtime_settings_status],
    )
    dialogue_event.then(lambda status: status, dialogue_status, activity_status, queue=False, show_progress="hidden")
    dialogue_event.then(_shared_runtime_metrics, None, runtime_metrics, queue=False, show_progress="hidden")


    cmd_console_timer.tick(console_html, None, cmd_console, queue=False, show_progress="hidden")
    runtime_metrics_timer.tick(_shared_runtime_metrics, None, runtime_metrics, queue=False, show_progress="hidden")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, show_error=True)
