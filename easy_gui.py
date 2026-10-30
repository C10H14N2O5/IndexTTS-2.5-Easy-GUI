from __future__ import annotations

import gc
import importlib
import json
import math
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

from tools.easy_gui_i18n import GuiI18n, UI_LANGUAGES, update_gui_settings

_UI_I18N = GuiI18n(SETTINGS_FILE)
t = _UI_I18N.t
localized_choices = _UI_I18N.choices
localize_backend_message = _UI_I18N.backend_message
CACHE_DIR = RUNTIME_DIR / "hf-cache"
TEMP_DIR = RUNTIME_DIR / "temp"
ASSETS_DIR = ROOT / "assets"
TRAINING_DIR = ROOT / "training"
TRAINING_PROJECTS_DIR = TRAINING_DIR / "projects"
TRAINING_DATASETS_DIR = TRAINING_DIR / "datasets"
TRAINING_OUTPUTS_DIR = TRAINING_DIR / "outputs"
DIALOGUE_MAX_SEGMENTS = 12
for p in (MODEL_DIR, OUTPUTS_DIR, VOICES_DIR, EMOTION_DESCRIPTIONS_DIR, CACHE_DIR, TEMP_DIR, TRAINING_PROJECTS_DIR, TRAINING_DATASETS_DIR, TRAINING_OUTPUTS_DIR):
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

from indextts.utils.text_chunking import CHUNK_CHOICES, split_long_text

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
.training-card { padding:16px !important; border-radius:10px !important; background:rgba(128,128,128,.055) !important; }
.training-progress textarea { font-family:Consolas,monospace !important; }
.form-section { padding: 14px !important; border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important; background: var(--block-background-fill) !important; }

.training-project-bar {
    padding: 12px 14px !important;
    margin-bottom: 14px !important;
    border: 1px solid rgba(128,128,128,.24) !important;
    border-radius: 9px !important;
}
.training-project-bar button { min-height: 42px !important; }
.project-strip { padding: 12px !important; border-radius: 10px !important; border: 1px solid var(--border-color-primary) !important; margin-bottom: 12px !important; }
.global-toolbar { padding: 10px 12px !important; border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important; background: var(--block-background-fill) !important; margin-bottom: 12px !important; }
.global-toolbar button { min-height: 38px !important; }
.tab-subtitle { opacity: .82; margin: 0 0 4px 0 !important; padding: 0 !important; }
.quick-guide { border-radius: 9px !important; }
.training-wide-action button { min-height: 42px !important; font-weight: 650 !important; }
.training-orange button {
    background: #f05a00 !important;
    border-color: #f05a00 !important;
    color: white !important;
}
.training-orange button:hover { filter: brightness(1.08) !important; }
.training-red button {
    background: #c91f1f !important;
    border-color: #c91f1f !important;
    color: white !important;
}
.training-panel {
    padding: 14px !important;
    border: 1px solid rgba(128,128,128,.22) !important;
    border-radius: 9px !important;
    background: rgba(128,128,128,.055) !important;
}
.training-status-card {
    padding: 10px 14px !important;
    border: 1px solid rgba(128,128,128,.22) !important;
    border-radius: 8px !important;
}
.training-status-card textarea { min-height: 42px !important; }
.training-card h4 { margin-bottom: 8px !important; }
.fire-progress-card { padding: 12px 14px; border: 1px solid rgba(128,128,128,.22); border-radius: 9px; background: rgba(128,128,128,.055); }
.fire-progress-head, .fire-progress-foot { display:flex; justify-content:space-between; gap:12px; }
.fire-progress-title { font-weight:650; }
.fire-progress-meta, .fire-progress-foot { opacity:.78; font-size:.9em; }
.fire-train-track { width:100%; height:12px; margin:9px 0; overflow:hidden; border-radius:999px; background:rgba(128,128,128,.25); }
.fire-train-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#60a5fa); transition:width .35s ease; }
.global-status-panel {
    margin: 10px 0 14px 0 !important;
}
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


GLOBAL_STATUS_LOCK = threading.RLock()
GLOBAL_STATUS_LINE = t('Ready: select a workflow to begin.')


def _publish_global_status(phase, message, level="info", detail=""):
    """Replace the single shared status line used by every workflow tab."""
    global GLOBAL_STATUS_LINE
    normalized = re.sub(r"\s+", " ", localize_backend_message(message)).strip() or t('Working.')
    phase = re.sub(r"\s+", " ", str(phase or "Activity")).strip() or "Activity"
    detail = re.sub(r"\s+", " ", str(detail or "")).strip()
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {phase}: {normalized}"
    if detail:
        line += f" | {detail}"
    with GLOBAL_STATUS_LOCK:
        GLOBAL_STATUS_LINE = line


def _shared_runtime_metrics() -> str:
    try:
        import torch
        allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        return (
            t('GPU memory allocated: {0:.2f} GB | reserved: {1:.2f} GB | model loaded: {2}', allocated, reserved, 'yes' if _MODEL is not None else 'no')
        )
    except Exception:
        return t('model loaded: {0}', 'yes' if _MODEL is not None else 'no')


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
            return t('❌ CUDA unavailable | PyTorch {0} | build CUDA: {1}', torch.__version__, torch.version.cuda)
        return (
            t('✅ CUDA ready | PyTorch {0} | CUDA {1} | {2} | {3:.1f} GB VRAM', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
        )
    except Exception as exc:
        return t('❌ PyTorch/CUDA check failed: {0}: {1}', type(exc).__name__, exc)


def model_status() -> str:
    state = t('✅ **IndexTTS 2.5 model ready**') if _model_files_present() else t('⬇️ **IndexTTS 2.5 model is missing.**')
    detail = f"`{MODEL_DIR}`" if _model_files_present() else t('`run.bat` downloads the official model automatically before the GUI starts.')
    return f"{state}  \n{detail}  \n{cuda_status()}"


def _probe_distribution(module_name: str, distribution_name: str):
    """Detect an optional accelerator without importing it.

    Detect an optional accelerator without importing it, avoiding side effects
    during a simple status refresh.
    """
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return False, t('not installed')
    except Exception as exc:
        return False, t('metadata error: {0}: {1}', type(exc).__name__, exc)

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return False, t('module lookup error: {0}: {1}', type(exc).__name__, exc)
    if spec is None:
        return False, t('{0} {1} installed, but module {2!r} was not found', distribution_name, version, module_name)
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
        line(t('FlashAttention GPT engine'), "flash_attn"),
        t('Optional accelerators are disabled by default. Run `install_accelerators.bat` to add or repair them.'),
    ])


def _safe_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (value or "").strip())
    return safe.strip().replace(" ", "_")


def list_emotion_descriptions():
    return [p for p in sorted(EMOTION_DESCRIPTIONS_DIR.glob("*.txt")) if p.is_file()]


def emotion_description_choices(include_none=True):
    choices = [(p.stem.replace("_", " "), str(p)) for p in list_emotion_descriptions()]
    return [(t('None'), NONE_SENTINEL), *choices] if include_none else choices


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
        raise gr.Error(t('Enter a reusable emotion-description name.'))
    if not text:
        raise gr.Error(t('Enter an emotion description to save.'))
    dst = EMOTION_DESCRIPTIONS_DIR / f"{safe}.txt"
    dst.write_text(text + "\n", encoding="utf-8")
    _publish_global_status(t('Emotion library'), t('Saved emotion description: {0}', dst.name), "success")
    return (
        t('Saved emotion description: `{0}`', dst.name),
        refresh_emotion_description_dropdown(str(dst)),
        refresh_emotion_description_dropdown(str(dst)),
    )


def delete_emotion_description(path):
    if not path:
        _publish_global_status(t('Emotion library'), t('No emotion description selected.'), "info")
        return t('No emotion description selected.'), refresh_emotion_description_dropdown(), refresh_emotion_description_dropdown()
    p = Path(path)
    if p.parent.resolve() != EMOTION_DESCRIPTIONS_DIR.resolve():
        raise gr.Error(t('Only descriptions inside emotion_descriptions/ can be deleted here.'))
    p.unlink(missing_ok=True)
    _publish_global_status(t('Emotion library'), t('Deleted emotion description: {0}', p.name), "success")
    return t('Deleted emotion description: `{0}`', p.name), refresh_emotion_description_dropdown(), refresh_emotion_description_dropdown()


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
    update_gui_settings(SETTINGS_FILE, data)
    _publish_global_status(t('Runtime'), t('CUDA runtime settings saved.'), "success", t('BF16={0} · GPT latent={1} · torch.compile={2} · FlashAttention={3}', data['use_bf16'], data['use_gpt_latent'], data['use_torch_compile'], data['use_flash_accel']))
    return t('Runtime settings saved.')


def save_ui_language(language):
    if language not in {code for _, code in UI_LANGUAGES}:
        raise gr.Error(t("Invalid interface language."))
    update_gui_settings(SETTINGS_FILE, {"ui_language": language})
    message = t("Interface language saved. Restart WebUI manually to apply it.")
    _publish_global_status(t("Runtime"), message, "success")
    return message


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
    return gr.update(value=bf16), t('Runtime settings saved.')


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
        return [(t('None'), NONE_SENTINEL), *choices]
    return choices


def refresh_voice_dropdown(value=None, include_none=False):
    choices = voice_choices(include_none=include_none)
    valid = {v for _, v in choices}
    if include_none:
        selected = value if value in valid else NONE_SENTINEL
    else:
        selected = value if value in valid else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=selected)


def _voice_transcript_path(audio_path):
    path = Path(str(audio_path))
    return path.with_suffix(".txt")


def load_voice_transcript(audio_path):
    if not audio_path or audio_path == NONE_SENTINEL:
        return ""
    path = Path(str(audio_path))
    transcript = _voice_transcript_path(path)
    try:
        return transcript.read_text(encoding="utf-8", errors="ignore").strip() if transcript.is_file() else ""
    except Exception:
        return ""


def save_voice(audio, name, transcript):
    src = _audio_path(audio)
    name = (name or "").strip()
    if not src or not Path(src).exists():
        raise gr.Error(t('Choose or record a reference audio file first.'))
    if not name:
        raise gr.Error(t('Enter a voice name.'))
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")
    if not safe:
        raise gr.Error(t('Voice name contains no usable characters.'))
    suffix = Path(src).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"
    dst = VOICES_DIR / f"{safe}{suffix}"
    shutil.copy2(src, dst)
    transcript_path = _voice_transcript_path(dst)
    transcript = str(transcript or "").strip()
    if transcript:
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
    elif transcript_path.is_file():
        transcript_path.unlink()
    _publish_global_status(t('Reference library'), t('Saved reference voice: {0}', safe), "success", t('The audio and optional transcript are now available in Prep Samples and Inference.'))
    return t('Saved reference voice: `{0}`', dst), refresh_voice_dropdown(str(dst), include_none=True), refresh_voice_dropdown(str(dst), include_none=True)


def delete_selected_voice(path):
    if not path:
        return t('No reference voice selected.'), refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)
    p = Path(path)
    try:
        if p.parent.resolve() != VOICES_DIR.resolve():
            raise gr.Error(t('Only files inside the local voices library can be deleted here.'))
        p.unlink(missing_ok=True)
        _voice_transcript_path(p).unlink(missing_ok=True)
    except Exception as exc:
        raise gr.Error(t('Could not delete voice: {0}: {1}', type(exc).__name__, exc))
    _publish_global_status(t('Reference library'), t('Deleted reference voice: {0}', p.name), "success")
    return t('Deleted reference voice: `{0}`', p.name), refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)


def delete_all_voices():
    count = 0
    for p in VOICES_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
            p.unlink(missing_ok=True)
            _voice_transcript_path(p).unlink(missing_ok=True)
            count += 1
    _publish_global_status(t('Reference library'), t('Deleted {0} reference voice file(s).', count), "success")
    return t('Deleted {0} reference voice file(s).', count), refresh_voice_dropdown(include_none=True), refresh_voice_dropdown(include_none=True)


def delete_outputs():
    count = 0
    for p in OUTPUTS_DIR.iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)
            count += 1
    _publish_global_status(t('Outputs'), t('Deleted {0} output audio file(s).', count), "success")
    return t('Deleted {0} output file(s).', count)


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
        return t('❌ Model download failed: `{0}: {1}`', type(exc).__name__, exc)


def clear_download_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return t('Local Hugging Face/Xet download cache cleared. Installed model files were kept.')


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
    _publish_global_status(t('Runtime'), t('Models unloaded and VRAM cleanup requested.'), "success")
    return t('Models unloaded and VRAM cleanup requested.')


def delete_model():
    unload_model()
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return model_status()


def _load_model(
    use_bf16: bool, use_gpt_latent: bool, use_torch_compile: bool,
    use_flash_accel: bool, use_qwen_emo: bool, adapter_path: str = "",
):
    global _MODEL, _MODEL_OPTIONS
    adapter_path = str(adapter_path or "")
    if adapter_path == NONE_SENTINEL:
        adapter_path = ""
    options = (bool(use_bf16), bool(use_gpt_latent), bool(use_torch_compile), bool(use_flash_accel), bool(use_qwen_emo), adapter_path)
    with _MODEL_LOCK:
        if not _model_files_present():
            raise RuntimeError(t('IndexTTS 2.5 model is missing. Restart with run.bat so the automatic model downloader can repair it.'))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(t('CUDA is unavailable. This Easy GUI is configured as an NVIDIA CUDA build; rerun install.bat and check your NVIDIA driver.'))
        caps = acceleration_capabilities()
        if options[2] and not caps["triton"][0]:
            raise RuntimeError(t('torch.compile was requested but Triton is unavailable: {0}. Run install_accelerators.bat.', caps['triton'][1]))
        if options[3] and not caps["flash_attn"][0]:
            raise RuntimeError(t('FlashAttention acceleration was requested but flash_attn is unavailable: {0}. Run install_accelerators.bat.', caps['flash_attn'][1]))
        if options[3] and not caps["triton"][0]:
            raise RuntimeError(t('FlashAttention acceleration was requested but Triton is unavailable: {0}. Run install_accelerators.bat.', caps['triton'][1]))
        if options[3] and not options[0]:
            raise RuntimeError(t('FlashAttention GPT acceleration requires BF16 in this Easy GUI. Enable BF16 and retry.'))
        if _MODEL is not None and _MODEL_OPTIONS == options:
            return _MODEL
        unload_model()
        from indextts.infer_v2_5 import IndexTTS2
        _MODEL = IndexTTS2(
            cfg_path=str(MODEL_DIR / "config.yaml"), model_dir=str(MODEL_DIR), device="cuda:0",
            use_bf16=options[0], use_cuda_kernel=False, use_gpt_latent=options[1],
            use_torch_compile=options[2], use_deepspeed=False,
            # Build the accelerated snapshot only after PEFT has been attached.
            use_accel=options[3] and not bool(adapter_path), use_qwen_emo=options[4],
        )
        if adapter_path and adapter_path != NONE_SENTINEL:
            adapter = Path(adapter_path)
            if not (adapter / "adapter_config.json").is_file():
                raise RuntimeError(t('LoRA adapter is missing adapter_config.json: {0}', adapter))
            from peft import PeftModel
            loaded = PeftModel.from_pretrained(_MODEL.gpt.gpt, str(adapter), is_trainable=False)
            if options[3]:
                # IndexTTS's AccelInferenceEngine snapshots self.gpt.state_dict()
                # during post_init. Merge first so the snapshot contains LoRA.
                _MODEL.gpt.gpt = loaded.merge_and_unload()
                _MODEL.gpt.use_accel = True
                _MODEL.gpt.accel_engine = None
                _MODEL.gpt.post_init_gpt2_config(
                    use_deepspeed=False, kv_cache=True, half=bool(options[0])
                )
            else:
                _MODEL.gpt.gpt = loaded
            print(f"[adapter] Loaded IndexTTS 2.5 LoRA: {adapter}", flush=True)
        _MODEL_OPTIONS = options
        return _MODEL


def _resolve_inference_prompt(quick_sample, prompt_audio):
    if quick_sample and quick_sample != NONE_SENTINEL:
        saved_prompt = Path(str(quick_sample))
        prompt = str(saved_prompt) if saved_prompt.exists() else None
        if not prompt:
            raise gr.Error(t('The selected Quick Sample no longer exists. Choose another saved sample or select None for a custom upload.'))
        return prompt
    prompt = _audio_path(prompt_audio)
    if not prompt or not Path(prompt).exists():
        raise gr.Error(t('Quick Sample is set to None. Upload or record a custom reference audio file before generating.'))
    return prompt


def transcribe_inference_reference(quick_sample, prompt_audio, model_size, language, batch_size=8, progress=gr.Progress()):
    """Transcribe the exact reference audio that Single Inference will use."""
    prompt = _resolve_inference_prompt(quick_sample, prompt_audio)
    _publish_global_status(t('Reference transcription'), t('Loading Faster-Whisper for the selected reference audio.'), "working", t('Model: {0}', model_size or 'large-v3'))
    try:
        import torch
        from faster_whisper import WhisperModel
        use_cuda = bool(torch.cuda.is_available())
        progress(0.08, desc=t('Loading Faster-Whisper'))
        model = WhisperModel(
            str(model_size or "large-v3"),
            device="cuda" if use_cuda else "cpu",
            compute_type="float16" if use_cuda else "int8",
        )
        progress(0.35, desc=t('Transcribing reference audio'))
        transcribe_kwargs = dict(
            language=None if not language or language == "Auto-detect" else str(language),
            vad_filter=True,
        )
        if int(batch_size or 1) > 1:
            from faster_whisper import BatchedInferencePipeline
            segments, _ = BatchedInferencePipeline(model=model).transcribe(
                str(prompt), batch_size=int(batch_size), **transcribe_kwargs
            )
        else:
            segments, _ = model.transcribe(str(prompt), **transcribe_kwargs)
        result = " ".join(segment.text.strip() for segment in segments).strip()
        del model
        if use_cuda:
            torch.cuda.empty_cache()
        if not result:
            raise gr.Error(t('Whisper returned an empty reference transcript.'))
        progress(1.0, desc=t('Reference transcript ready'))
        _publish_global_status(t('Reference transcription'), t('Reference transcript ready.'), "success", t('{0} characters transcribed from {1}.', len(result), Path(prompt).name))
        return result
    except gr.Error:
        raise
    except Exception as exc:
        _publish_global_status(t('Reference transcription'), t('Reference transcription failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(t('Reference transcription failed: {0}: {1}', type(exc).__name__, exc))


def transcribe_audio_reference(audio_path, model_size, language, batch_size=8, progress=gr.Progress()):
    """Shared Prep Samples entry point using the same FireRed/MOSS ASR panel."""
    return transcribe_inference_reference(
        NONE_SENTINEL, audio_path, model_size, language, batch_size, progress=progress
    )


def generate(
    quick_sample, prompt_audio, text, reference_transcript, reference_auto_transcribe,
    reference_whisper_model, reference_whisper_language, reference_whisper_batch, adapter_path, language, emotion_mode, emotion_audio, emotion_alpha,
    happy, angry, sad, afraid, disgusted, melancholic, surprised, calm,
    emotion_text, random_emotion, duration_factor, interval_silence, chunk_mode, chunk_silence,
    text_normalization, max_text_tokens, do_sample, top_p, top_k, temperature,
    num_beams, repetition_penalty, length_penalty, max_mel_tokens,
    use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, save_outputs,
    # Do NOT enable track_tqdm here: IndexTTS S2Mel/FlowMatching owns its tqdm iterators.
    progress=gr.Progress(),
):
    progress(0.02, desc=t('Validating inference inputs'))
    print("[ui] Preparing single inference", flush=True)
    _publish_global_status(t('Single inference'), t('Validating reference audio and target text.'), "working")
    # Quick Sample is authoritative when a saved library entry is selected.
    # The audio widget is used only when Quick Sample == None. This avoids
    # false "missing sample" validation caused by UI preview/update timing.
    prompt = _resolve_inference_prompt(quick_sample, prompt_audio)
    text = (text or "").strip()
    if not text:
        raise gr.Error(t('Enter target text to synthesize.'))
    chunk_mode = chunk_mode if chunk_mode in CHUNK_CHOICES else "None"
    chunks = split_long_text(text, chunk_mode)
    if not chunks:
        raise gr.Error(t('The target text did not produce any usable chunks.'))

    reference_transcript = (reference_transcript or "").strip()
    if bool(reference_auto_transcribe) and not reference_transcript:
        reference_transcript = transcribe_inference_reference(
            quick_sample, prompt_audio, reference_whisper_model, reference_whisper_language,
            reference_whisper_batch, progress=progress,
        )

    lang = LANGUAGES.get(language, "EN")
    use_qwen = emotion_mode == "Emotion description"
    progress(0.12, desc=t('Loading IndexTTS 2.5 runtime'))
    print("[ui] Loading IndexTTS 2.5 runtime", flush=True)
    tts = _load_model(
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, use_qwen, adapter_path,
    )
    _publish_global_status(t('Single inference'), t('Model loaded; synthesizing target speech.'), "working", t('Language: {0} · Adapter: {1}', language, 'None' if not adapter_path or adapter_path == NONE_SENTINEL else Path(str(adapter_path)).parent.name))
    progress(0.28, desc=t('Preparing conditioning'))

    emo_audio = None
    emo_vector = None
    use_emo_text = False
    emo_text = None
    if emotion_mode == "Emotion reference audio":
        emo_audio = _audio_path(emotion_audio)
        if not emo_audio or not Path(emo_audio).exists():
            raise gr.Error(t('Emotion reference mode requires a separate emotion audio file.'))
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
        explicit_chunks = chunk_mode != "None" and len(chunks) > 1
        progress(0.34, desc=(t('Preparing text chunks') if explicit_chunks else t('Synthesizing audio')))
        print(
            f"[ui-progress] Synthesizing {'text chunks' if explicit_chunks else 'single utterance'} "
            f"| chunks={len(chunks)} | chars={len(text)} | language={lang}",
            flush=True,
        )
        start_time = time.perf_counter()
        common_infer_kwargs = dict(
            spk_audio_prompt=prompt,
            lang=lang,
            reference_text=reference_transcript,
            emo_audio_prompt=emo_audio,
            emo_alpha=float(emotion_alpha),
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=bool(random_emotion),
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
        result = None
        chunk_temp_dir = None
        try:
            if explicit_chunks:
                chunk_temp_dir = Path(tempfile.mkdtemp(prefix="inference_chunks_", dir=str(TEMP_DIR)))
                generated_chunks = []
                for index, chunk in enumerate(chunks, 1):
                    progress(0.34 + 0.50 * ((index - 1) / max(len(chunks), 1)), desc=t('Synthesizing chunk {0}/{1}', index, len(chunks)))
                    chunk_path = chunk_temp_dir / f"chunk_{index:04d}.wav"
                    result = tts.infer(
                        text=chunk,
                        output_path=str(chunk_path),
                        interval_silence=0,
                        **common_infer_kwargs,
                    )
                    actual_chunk = chunk_path if chunk_path.exists() else (Path(str(result)) if result else chunk_path)
                    if not actual_chunk.exists():
                        raise RuntimeError(t('IndexTTS did not create audio for chunk {0}.', index))
                    generated_chunks.append(actual_chunk)
                progress(0.86, desc=t('Joining generated chunks'))
                _concat_wavs(generated_chunks, out, max(0.0, float(chunk_silence)))
            else:
                progress(0.40, desc=t('Synthesizing audio'))
                result = tts.infer(
                    text=text,
                    output_path=str(out),
                    interval_silence=int(interval_silence),
                    **common_infer_kwargs,
                )
        finally:
            if chunk_temp_dir is not None:
                shutil.rmtree(chunk_temp_dir, ignore_errors=True)
        progress(0.92, desc=t('Saving output WAV'))
        actual = out if out.exists() else (Path(str(result)) if result else out)
        if not actual.exists():
            raise RuntimeError(t('IndexTTS returned without creating an output WAV.'))
        saved = str(actual) if save_outputs else t('Preview only (not persisted in outputs/)')
        status = (
            t('Generation complete.\nLanguage: {0} ({1})\nEmotion mode: {2}\nDuration factor: {3}\nChunk mode: {4} ({5} chunk{6})\nSaved WAV: {7}', language, lang, emotion_mode, duration_factor, chunk_mode, len(chunks), 's' if len(chunks) != 1 else '', saved)
        )
        elapsed = time.perf_counter() - start_time
        print(f"[ui] Generation complete in {elapsed:.2f}s | {actual}", flush=True)
        progress(0.97, desc=t('Completion notification'))
        _play_completion_sound()
        progress(1.0, desc=t('Done'))
        _publish_global_status(t('Single inference'), t('Generation complete.'), "success", t('Saved WAV: {0}', saved))
        return str(actual), saved, status
    except Exception as exc:
        _publish_global_status(t('Single inference'), t('Generation failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(t('Generation failed: {0}: {1}', type(exc).__name__, exc))


def set_prep_voice(path):
    if not path or path == NONE_SENTINEL:
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"]), ""
    p = Path(path)
    if not p.exists():
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"]), ""
    return gr.update(value=str(p), interactive=False, sources=[]), load_voice_transcript(str(p))


def set_voice_inference(path):
    """Bind Quick Sample selection to the inference reference audio widget.

    None means custom input: clear the player and enable upload/recording.
    A saved library path means preview/use that exact file and lock editing so
    the dropdown remains the source of truth.
    """
    if not path or path == NONE_SENTINEL:
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"]), ""
    p = Path(path)
    if not p.exists():
        return gr.update(value=None, interactive=True, sources=["upload", "microphone"]), ""
    return gr.update(value=str(p), interactive=False, sources=[]), load_voice_transcript(str(p))




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


def _dialogue_updates(count, rows, message=t('Dialogue rows updated.')):
    count = max(1, min(int(count), DIALOGUE_MAX_SEGMENTS))
    flat = []
    for row in rows[:DIALOGUE_MAX_SEGMENTS]:
        flat.extend(row)
    visibility = [gr.update(visible=i < count) for i in range(DIALOGUE_MAX_SEGMENTS)]
    return [count, *flat, *visibility, message]


def dialogue_reset_rows(*values):
    rows = [_dialogue_default_row() for _ in range(DIALOGUE_MAX_SEGMENTS)]
    return _dialogue_updates(2, rows, t('Dialogue reset to two empty turns.'))


def dialogue_clear_rows(count, *values):
    rows = _dialogue_unpack(values)
    for i in range(min(int(count), DIALOGUE_MAX_SEGMENTS)):
        rows[i] = _dialogue_default_row()
    return _dialogue_updates(count, rows, t('Visible dialogue turns cleared.'))


def dialogue_compact_rows(count, *values):
    rows = _dialogue_unpack(values)
    active = [r for r in rows[:int(count)] if str(r[1] or "").strip()]
    new_count = max(1, len(active))
    active += [_dialogue_default_row() for _ in range(DIALOGUE_MAX_SEGMENTS - len(active))]
    return _dialogue_updates(new_count, active, t('Removed empty turns. {0} turn(s) remain.', new_count))


def dialogue_add_after(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    if count >= DIALOGUE_MAX_SEGMENTS:
        return _dialogue_updates(count, rows, t('Maximum of {0} dialogue turns reached.', DIALOGUE_MAX_SEGMENTS))
    pos = min(int(index) + 1, count)
    rows.insert(pos, _dialogue_default_row())
    rows = rows[:DIALOGUE_MAX_SEGMENTS]
    return _dialogue_updates(count + 1, rows, t('Added a new turn after Turn {0}.', int(index) + 1))


def dialogue_clone_row(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    if count >= DIALOGUE_MAX_SEGMENTS:
        return _dialogue_updates(count, rows, t('Maximum of {0} dialogue turns reached.', DIALOGUE_MAX_SEGMENTS))
    idx = min(max(0, int(index)), max(0, count - 1))
    rows.insert(idx + 1, list(rows[idx]))
    rows = rows[:DIALOGUE_MAX_SEGMENTS]
    return _dialogue_updates(count + 1, rows, t('Cloned Turn {0}.', idx + 1))


def dialogue_delete_row(index, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    idx = min(max(0, int(index)), max(0, count - 1))
    if count <= 1:
        rows[0] = _dialogue_default_row()
        return _dialogue_updates(1, rows, t('The last turn was cleared instead of removed.'))
    rows.pop(idx)
    rows.append(_dialogue_default_row())
    return _dialogue_updates(count - 1, rows, t('Deleted Turn {0}.', idx + 1))


def dialogue_clear_row(index, count, *values):
    rows = _dialogue_unpack(values)
    idx = min(max(0, int(index)), DIALOGUE_MAX_SEGMENTS - 1)
    rows[idx] = _dialogue_default_row()
    return _dialogue_updates(count, rows, t('Cleared Turn {0}.', idx + 1))


def dialogue_move_row(index, direction, count, *values):
    rows = _dialogue_unpack(values)
    count = int(count)
    idx = min(max(0, int(index)), max(0, count - 1))
    target = idx + int(direction)
    if 0 <= target < count:
        rows[idx], rows[target] = rows[target], rows[idx]
        return _dialogue_updates(count, rows, t('Moved Turn {0} {1}.', idx + 1, 'up' if direction < 0 else 'down'))
    return _dialogue_updates(count, rows, t('Turn is already at that boundary.'))


def _concat_wavs(wav_paths, output_path, silence_s):
    wav_paths = [Path(x) for x in wav_paths]
    if not wav_paths:
        raise RuntimeError(t('No dialogue segments were generated.'))
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
                    raise RuntimeError(t('Dialogue segment {0} has incompatible WAV parameters.', idx + 1))
                out.writeframes(inp.readframes(inp.getnframes()))
            if idx + 1 < len(wav_paths) and silence:
                out.writeframes(silence)


def generate_dialogue(adapter_path, language, duration_factor, interval_silence, text_normalization, max_text_tokens,
                      do_sample, top_p, top_k, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens,
                      use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, dialogue_silence, dialogue_row_count, *values,
                      # Do NOT enable track_tqdm here: it breaks IndexTTS FlowMatching tqdm on Gradio 5.x.
                      progress=gr.Progress()):
    progress(0.02, desc=t('Validating dialogue'))
    print("[ui] Preparing Dialogue Builder inference", flush=True)
    _publish_global_status(t('Dialogue inference'), t('Validating dialogue turns.'), "working")
    # Per row: speaker, text, emotion_mode, vector_preset, description_path.
    rows = _dialogue_unpack(values)
    active = [r for r in rows[:max(1, min(int(dialogue_row_count), DIALOGUE_MAX_SEGMENTS))] if str(r[1] or "").strip()]
    if not active:
        raise gr.Error(t('Dialogue has no turns with text.'))
    needs_qwen = any(r[2] == "Emotion Description" for r in active)
    progress(0.10, desc=t('Loading IndexTTS 2.5 runtime'))
    tts = _load_model(use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, needs_qwen, adapter_path)
    _publish_global_status(t('Dialogue inference'), t('Generating {0} dialogue turn(s).', len(active)), "working", t('Language: {0}', language))
    lang = LANGUAGES.get(language, "EN")
    temp_dir = TEMP_DIR / f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    try:
        for idx, row in enumerate(active, 1):
            speaker, text, emo_mode, vector_preset, description_path = row
            base = 0.16 + 0.72 * ((idx - 1) / max(len(active), 1))
            progress(base, desc=t('Dialogue turn {0}/{1}', idx, len(active)))
            print(f"[ui-progress] Dialogue turn {idx}/{len(active)} | chars={len(str(text or ''))}", flush=True)
            if not speaker or speaker == NONE_SENTINEL or not Path(str(speaker)).is_file():
                raise gr.Error(t('Dialogue turn {0} requires a saved voice-library speaker.', idx))
            seg = temp_dir / f"segment_{idx:02d}.wav"
            kwargs = dict(emo_alpha=0.65, use_random=False, emo_vector=None, use_emo_text=False, emo_text=None)
            if emo_mode == "Emotion Description":
                desc = load_emotion_description(description_path)
                if not desc:
                    raise gr.Error(t('Dialogue turn {0}: choose a saved Emotion Description.', idx))
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
            progress(0.16 + 0.72 * (idx / max(len(active), 1)), desc=t('Completed turn {0}/{1}', idx, len(active)))
        progress(0.91, desc=t('Assembling dialogue WAV'))
        out = OUTPUTS_DIR / f"indextts_2_5_dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        _concat_wavs(segments, out, dialogue_silence)
        progress(0.97, desc=t('Completion notification'))
        _play_completion_sound()
        print(f"[ui] Dialogue complete | turns={len(active)} | {out}", flush=True)
        progress(1.0, desc=t('Done'))
        _publish_global_status(t('Dialogue inference'), t('Dialogue generation complete.'), "success", t('Generated {0} turn(s) to {1}.', len(active), out.name))
        return str(out), str(out), t('Dialogue complete. Generated {0} turns.', len(active))
    except Exception as exc:
        _publish_global_status(t('Dialogue inference'), t('Dialogue generation failed: {0}', type(exc).__name__), "error", str(exc))
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def _ensure_model_before_ui():
    from tools.ensure_model import main as ensure_model_main
    rc = ensure_model_main()
    if rc != 0:
        raise RuntimeError(t('Automatic IndexTTS 2.5 model setup failed. Review the console output and retry.'))





def _native_dialog(kind: str, current: str = "", title: str = "", filetypes=None) -> str:
    """Open the same small native Windows picker used by the training workflows."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()

        initial = Path(str(current or "")).expanduser()
        initial_dir = str(initial.parent if initial.is_file() else initial if initial.is_dir() else ROOT)

        if kind == "directory":
            selected = filedialog.askdirectory(
                parent=root,
                title=title or t('Select folder'),
                initialdir=initial_dir,
                mustexist=True,
            )
        else:
            selected = filedialog.askopenfilename(
                parent=root,
                title=title or t('Select file'),
                initialdir=initial_dir,
                filetypes=filetypes or [(t('All files'), "*.*")],
            )
        root.destroy()
        return str(selected or current or "")
    except Exception as exc:
        print(f"[ui] Native browser unavailable: {type(exc).__name__}: {exc}", flush=True)
        return str(current or "")


def browse_dataset_folder(current=""):
    return _native_dialog("directory", current, t('Select training audio folder'))


def browse_eval_audio(current=""):
    return _native_dialog(
        "file",
        current,
        t('Select evaluation reference audio'),
        [
            (t('Audio files'), "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.wma"),
            ("WAV", "*.wav"),
            ("FLAC", "*.flac"),
            (t('All files'), "*.*"),
        ],
    )


def browse_checkpoint_folder(current=""):
    selected = _native_dialog("directory", current, t('Select LoRA checkpoint folder'))
    return str(selected or current or NONE_SENTINEL)


def _training_projects():
    from indextts_training.dataset import list_projects
    return list_projects(str(TRAINING_DIR))

def _training_project_choices():
    return [(x,x) for x in _training_projects()]

def _adapter_choices():
    """Return the final adapter and every saved training checkpoint.

    The trainer writes PEFT adapters below each output directory as
    ``checkpoint-step-*`` plus ``final_adapter``.  Inference can load any of
    those folders, so the selector must expose the complete set instead of
    collapsing a project to its final export.
    """
    choices = [(t('None'), NONE_SENTINEL)]
    if not TRAINING_OUTPUTS_DIR.is_dir():
        return choices

    for project_dir in sorted(TRAINING_OUTPUTS_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not project_dir.is_dir():
            continue
        adapter_dirs = []
        final_dir = project_dir / "final_adapter"
        if final_dir.is_dir():
            adapter_dirs.append(final_dir)
        adapter_dirs.extend(
            checkpoint
            for checkpoint in sorted(project_dir.glob("checkpoint-*"), key=lambda path: path.name.lower())
            if checkpoint.is_dir()
        )
        for adapter_dir in adapter_dirs:
            if not (adapter_dir / "adapter_config.json").is_file():
                continue
            if adapter_dir.name == "final_adapter":
                label = t('{0} · final', project_dir.name)
            else:
                label = f"{project_dir.name} · {adapter_dir.name}"
            choices.append((label, str(adapter_dir.resolve())))
    return choices

def refresh_adapters(selected=NONE_SENTINEL):
    choices=_adapter_choices(); values={value for _,value in choices}
    return gr.update(choices=choices, value=selected if selected in values else NONE_SENTINEL)

def scan_training_dataset(folder):
    from indextts_training.dataset import scan_dataset, dataset_summary
    _publish_global_status(t('Dataset analysis'), t('Scanning audio files and transcript sidecars.'), "working", str(folder or t('No source folder selected.')))
    try:
        rows,stats=scan_dataset(folder)
    except Exception as exc:
        _publish_global_status(t('Dataset analysis'), t('Dataset scan failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(localize_backend_message(str(exc))) from exc
    _publish_global_status(t('Dataset analysis'), t('Dataset analysis complete.'), "success", localize_backend_message(dataset_summary(stats)))
    return localize_backend_message(dataset_summary(stats)), json.dumps(stats, ensure_ascii=False)

def prepare_training_dataset(folder, project, selected_project, language, transcribe, whisper_model, whisper_lang, asr_batch=8, progress=gr.Progress()):
    from indextts_training.dataset import prepare_dataset, dataset_summary
    project = str(project or selected_project or "").strip()
    if not project:
        raise gr.Error(t('Create or select a Training Project first.'))
    phase = t('Dataset transcription') if bool(transcribe) else t('Dataset preparation')
    _publish_global_status(t(phase), t('Scanning the dataset and calculating automatic clip bounds.'), "working", t('Project: {0}', project))
    progress(.05,desc=t('Scanning source dataset'))
    try:
        if bool(transcribe):
            _publish_global_status(t('Dataset transcription'), t('Transcribing missing transcript sidecars with Faster-Whisper.'), "working", t('Model: {0} · Language: {1}', whisper_model or 'large-v3', whisper_lang or 'Auto-detect'))
        meta=prepare_dataset(folder,project,LANGUAGES.get(language,language),None,None,bool(transcribe),whisper_model,whisper_lang,str(TRAINING_DIR))
    except Exception as exc:
        _publish_global_status(t(phase), t('Dataset preparation failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(localize_backend_message(str(exc))) from exc
    progress(1.0,desc=t('Prepared'))
    _play_completion_sound()
    summary = localize_backend_message(dataset_summary(meta.get('stats',{})))
    _publish_global_status(t('Dataset preparation'), t('Prepared dataset and project link are ready.'), "success", summary)
    linked_status=t('Loaded project: {0} · Prepared dataset linked', meta['project_name'])
    return (summary,
            gr.update(choices=_training_project_choices(),value=meta['project_name']),
            meta['manifest'],
            gr.update(choices=_training_project_choices(),value=meta['project_name']),
            meta['manifest'], linked_status)

def load_training_project(name):
    from indextts_training.dataset import load_project, dataset_summary
    meta=load_project(name,str(TRAINING_DIR)) if name else {}
    code=str(meta.get('language','EN')).upper()
    label=next((name for name, value in LANGUAGES.items() if value == code), 'English')
    if name:
        _publish_global_status(t('Dataset project'), t('Loaded project: {0}.', name), "success", t('Prepared dataset and language controls synchronized.'))
    else:
        _publish_global_status(t('Dataset project'), t('No project selected.'), "info")
    return meta.get('source_folder',''), meta.get('manifest',''), localize_backend_message(dataset_summary(meta.get('stats',{}))), label

def sync_training_project_selection(name):
    from indextts_training.dataset import load_project
    meta=load_project(name, str(TRAINING_DIR)) if name else {}
    manifest=str(meta.get('manifest') or '')
    status=t('Loaded project: {0} · Prepared dataset linked', name) if name else t('No project selected.')
    _publish_global_status(t('Dataset project'), status, "success" if name else "info", manifest or t('Prepare the dataset to create its manifest.'))
    return gr.update(choices=_training_project_choices(), value=name or None), manifest, status


def refresh_training_project_dropdown(current=None):
    choices = _training_project_choices()
    valid = {v for _, v in choices}
    return gr.update(choices=choices, value=current if current in valid else None)



def _training_project_file(project_name):
    safe = _safe_name(project_name)
    return TRAINING_DIR / "projects" / safe / "project.json" if safe else None



def create_dataset_project(project_name):
    safe = _safe_name(project_name)
    if not safe:
        raise gr.Error(t('Enter a New Project Name.'))
    project_dir = TRAINING_DIR / "projects" / safe
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "project.json"
    if not path.exists():
        path.write_text(
            json.dumps({"project_name": safe, "source_folder": "", "manifest": "", "language": "EN", "stats": {}},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    choices = _training_project_choices()
    _publish_global_status(t('Dataset project'), t('Project created: {0}', safe), "success", t('Choose a source folder and run Build / Scan Dataset.'))
    return gr.update(choices=choices, value=safe), safe, t('Project created: {0}', safe)


def save_dataset_project(project, new_name, source_folder, manifest, language):
    name = project or new_name
    safe = _safe_name(name)
    if not safe:
        raise gr.Error(t('Create or select a project first.'))
    path = _training_project_file(safe)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update({
        "project_name": safe,
        "source_folder": str(source_folder or ""),
        "manifest": str(manifest or ""),
        "language": LANGUAGES.get(language, language),
    })
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    choices = _training_project_choices()
    _publish_global_status(t('Dataset project'), t('Project saved: {0}', safe), "success", t('Prepared manifest: {0}', manifest or 'not prepared yet'))
    return gr.update(choices=choices, value=safe), safe, t('Project saved: {0}', safe)


def clone_dataset_project(project_name):
    from indextts_training.dataset import clone_project, dataset_summary
    try:
        meta=clone_project(project_name,str(TRAINING_DIR))
    except Exception as exc:
        _publish_global_status(t('Dataset project'), t('Project clone failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(t('Could not clone project: {0}', exc))
    new_name=str(meta.get('project_name',''))
    code=str(meta.get('language','EN')).upper()
    language_label=next((label for label,value in LANGUAGES.items() if value==code), 'English')
    choices=_training_project_choices()
    manifest=str(meta.get('manifest') or '')
    _publish_global_status(t('Dataset project'), t('Project cloned: {0}', new_name), "success", t('Project fields and prepared dataset copied; training outputs were not copied.'))
    return (
        gr.update(choices=choices,value=new_name), new_name,
        meta.get('source_folder',''), manifest, localize_backend_message(dataset_summary(meta.get('stats',{}))), language_label,
        gr.update(choices=choices,value=new_name), manifest,
        t('Project cloned: {0}', new_name),
    )


def save_training_project(
    project_name, source_folder, manifest, dataset_language,
    adapter_name, vram, mode, steps, epochs, save_steps, save_epochs,
    rank, alpha, dropout, lr, batch, grad_accum, grad_clip, semantic_loss_weight,
    phonetic_ctc_enabled, phonetic_ctc_weight, resume,
    eval_enabled, eval_text, eval_reference, eval_reference_transcript,
    eval_language, eval_whisper_model, eval_whisper_language,
):
    safe = _safe_name(project_name)
    if not safe:
        raise gr.Error(t('Enter or select a Training Project name.'))

    project_dir = TRAINING_DIR / "projects" / safe
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "project.json"

    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    # Keep prepared-dataset metadata intact and update the editable project fields.
    payload["project_name"] = safe
    if source_folder:
        payload["source_folder"] = str(source_folder)
    if manifest:
        payload["manifest"] = str(manifest)
    payload["language"] = LANGUAGES.get(dataset_language, dataset_language)

    persisted_eval = ""
    if eval_reference and Path(str(eval_reference)).is_file():
        src = Path(str(eval_reference))
        target = project_dir / f"eval_reference{src.suffix.lower() or '.wav'}"
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        persisted_eval = str(target.resolve())
    elif payload.get("training", {}).get("eval_reference"):
        old = Path(str(payload["training"]["eval_reference"]))
        if old.is_file():
            persisted_eval = str(old.resolve())

    payload["training"] = {
        "adapter_name": str(adapter_name or safe),
        "vram_preset": str(vram),
        "mode": str(mode),
        "training_steps": int(steps),
        "training_epochs": int(epochs),
        "save_steps": int(save_steps),
        "save_epochs": int(save_epochs),
        "lora_rank": int(rank),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "learning_rate": float(lr),
        "micro_batch": int(batch),
        "grad_accum": int(grad_accum),
        "grad_clip": float(grad_clip),
        # The acoustic-only LoRA profile deliberately disables auxiliary text
        # and pronunciation objectives. Keep the old keys for project-file
        # compatibility, but persist their inactive values so an old project
        # cannot silently reactivate Whisper/CTC training.
        "semantic_loss_weight": 0.0,
        "phonetic_ctc_enabled": False,
        "phonetic_ctc_weight": 0.0,
        "differentiable_asr_enabled": False,
        "differentiable_asr_weight": 0.0,
        "differentiable_asr_model": "large-v3",
        "asr_cfm_steps": 4,
        "asr_cfm_temperature": 1.0,
        "resume_checkpoint": "" if not resume or resume == NONE_SENTINEL else str(resume),
        "eval_enabled": bool(eval_enabled),
        "eval_text": str(eval_text or ""),
        "eval_reference": persisted_eval,
        "eval_reference_transcript": str(eval_reference_transcript or ""),
        "eval_language": str(eval_language),
        "eval_whisper_model": str(eval_whisper_model),
        "eval_whisper_language": str(eval_whisper_language),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    choices = _training_project_choices()
    _publish_global_status(t('LoRA project'), t('Training project saved: {0}', safe), "success", t('Prepared dataset: {0}', manifest or 'not linked'))
    return (
        gr.update(choices=choices, value=safe),
        gr.update(choices=choices, value=safe),
        t('Project saved: {0}', safe),
    )


def clone_lora_project(project_name):
    from indextts_training.dataset import clone_project
    try:
        meta=clone_project(project_name,str(TRAINING_DIR))
    except Exception as exc:
        _publish_global_status(t('LoRA project'), t('Project clone failed: {0}', type(exc).__name__), "error", str(exc))
        raise gr.Error(t('Could not clone project: {0}', exc))
    new_name=str(meta.get('project_name',''))
    choices=_training_project_choices()
    loaded=load_full_training_project(new_name)
    _publish_global_status(t('LoRA project'), t('Project cloned: {0}', new_name), "success", t('All project settings were copied; training outputs were not copied.'))
    return (
        gr.update(choices=choices,value=new_name), gr.update(choices=choices,value=new_name),
        new_name, str(meta.get('manifest') or ''), *loaded,
        t('Project cloned: {0}', new_name),
    )


def load_full_training_project(project_name):
    from indextts_training.dataset import load_project, dataset_summary

    meta = load_project(project_name, str(TRAINING_DIR)) if project_name else {}
    tr = dict(meta.get("training") or {})
    language_code = meta.get("language", "EN")
    language_label = next((label for label, code in LANGUAGES.items() if code == language_code), language_code)

    mode_value = tr.get("mode", "Steps")
    resume_value = tr.get("resume_checkpoint") or NONE_SENTINEL
    eval_language_code = tr.get("eval_language", "EN")
    eval_language_label = next(
        (label for label, code in LANGUAGES.items() if code == eval_language_code),
        eval_language_code if eval_language_code in LANGUAGES else "English",
    )
    if project_name:
        _publish_global_status(t('LoRA project'), t('Loaded project: {0}.', project_name), "success", t('Prepared dataset: {0}', meta.get('manifest') or 'not linked'))
    else:
        _publish_global_status(t('LoRA project'), t('No project selected.'), "info")
    return (
        meta.get("source_folder", ""),
        meta.get("manifest", ""),
        localize_backend_message(dataset_summary(meta.get("stats", {}))),
        language_label,
        tr.get("adapter_name", project_name or ""),
        tr.get("vram_preset", "24 GB"),
        mode_value,
        int(tr.get("training_steps", 1500)),
        int(tr.get("training_epochs", 20)),
        int(tr.get("save_steps", 150)),
        int(tr.get("save_epochs", 2)),
        int(tr.get("lora_rank", 16)),
        int(tr.get("lora_alpha", 32)),
        float(tr.get("lora_dropout", 0.05)),
        float(tr.get("learning_rate", 0.0002)),
        int(tr.get("micro_batch", 1)),
        int(tr.get("grad_accum", 8)),
        float(tr.get("grad_clip", 1.0)),
        0.0,
        False,
        0.0,
        gr.update(
            choices=[(t('None'), NONE_SENTINEL), *_resume_choices(tr.get("adapter_name", project_name or ""))],
            value=resume_value,
        ),
        bool(tr.get("eval_enabled", False)),
        tr.get("eval_text", ""),
        tr.get("eval_reference") or None,
        tr.get("eval_reference_transcript", ""),
        eval_language_label,
        tr.get("eval_whisper_model", "large-v3"),
        tr.get("eval_whisper_language", "Auto-detect"),
        gr.update(visible=mode_value == "Steps"),
        gr.update(visible=mode_value == "Epochs"),
        t('Loaded project: {0}', project_name) if project_name else t('No project selected.'),
    )


def delete_training_project(project_name):
    safe = _safe_name(project_name)
    if not safe:
        _publish_global_status(t('LoRA project'), t('No project selected.'), "info")
        return gr.update(choices=_training_project_choices(), value=None), gr.update(choices=_training_project_choices(), value=None), t('No project selected.')
    project_dir = TRAINING_DIR / "projects" / safe
    prepared_dir = TRAINING_DIR / "datasets" / safe
    if project_dir.is_dir():
        shutil.rmtree(project_dir)
    if prepared_dir.is_dir():
        shutil.rmtree(prepared_dir)
    # Training outputs are intentionally retained as checkpoints/adapters.
    choices = _training_project_choices()
    _publish_global_status(t('LoRA project'), t('Project deleted: {0}.', safe), "success", t('Existing training outputs and adapters were kept.'))
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        t('Project deleted: {0}. Existing training outputs were kept.', safe),
    )


def transcribe_training_eval(audio_path, model_size, language, batch_size=8, progress=gr.Progress()):
    if not audio_path or not Path(str(audio_path)).is_file():
        raise gr.Error(t('Select an Eval Reference Audio file first.'))
    from faster_whisper import WhisperModel
    _publish_global_status(t('Eval transcription'), t('Loading Faster-Whisper for evaluation audio.'), "working", t('Model: {0} · Language: {1}', model_size or 'large-v3', language or 'Auto-detect'))
    try:
        progress(0.05, desc=t('Loading Faster-Whisper'))
        model = WhisperModel(str(model_size or "large-v3"), device="cuda", compute_type="float16")
        progress(0.30, desc=t('Transcribing evaluation reference'))
        segments, _ = model.transcribe(
            str(audio_path),
            language=None if not language or language == "Auto-detect" else str(language),
            vad_filter=True,
        )
        result = " ".join(segment.text.strip() for segment in segments).strip()
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        progress(1.0, desc=t('Done'))
        _play_completion_sound()
        _publish_global_status(t('Eval transcription'), t('Evaluation reference transcription complete.'), "success", t('{0} characters transcribed.', len(result)))
        return result, t('Evaluation reference transcription complete.')
    except Exception as exc:
        _publish_global_status(t('Eval transcription'), t('Evaluation transcription failed: {0}', type(exc).__name__), "error", str(exc))
        raise


def training_autotune(project, vram):
    from indextts_training.dataset import load_project
    meta=load_project(project,str(TRAINING_DIR)); st=meta.get('stats',{}); n=max(1,int(st.get('valid',1))); mins=float(st.get('total_seconds',0))/60
    rank=16 if mins<20 else 32; alpha=rank*2; lr=2e-4 if mins<30 else 1e-4; accum=8 if str(vram).startswith('24') else 4
    steps=max(500,min(4000,int(900+mins*45)))
    save=max(100,steps//10)
    message=t('AutoTune: {0} samples / {1:.1f} min → Steps {2}, r={3}, lr={4:g}, accumulation={5}.', n, mins, steps, rank, lr, accum)
    _publish_global_status(t('LoRA AutoTune'), t('Training preset calculated.'), "success", message)
    return 'Steps',steps,20,save,2,rank,alpha,lr,accum,message

def training_mode_visibility(mode):
    return gr.update(visible=mode=='Steps'), gr.update(visible=mode=='Epochs')

def _resume_choices(adapter):
    root=TRAINING_OUTPUTS_DIR/_safe_name(adapter)
    vals=[]
    if root.is_dir():
        vals=[str(p) for p in sorted(root.glob('checkpoint-*')) if p.is_dir()]
    return [(Path(x).name,x) for x in vals]

def refresh_resume(adapter,current=None):
    ch=_resume_choices(adapter); valid={v for _,v in ch}; return gr.update(choices=[(t('None'),NONE_SENTINEL),*ch],value=current if current in valid else NONE_SENTINEL)

def start_lora_training(project,manifest,adapter,vram,mode,steps,epochs,save_steps,save_epochs,rank,alpha,dropout,lr,batch,grad_accum,grad_clip,semantic_loss_weight,phonetic_ctc_enabled,phonetic_ctc_weight,resume,eval_enabled,eval_text,eval_reference,eval_reference_transcript,eval_language):
    from indextts_training.dataset import load_project
    from indextts_training import TRAINING_MANAGER
    meta=load_project(project,str(TRAINING_DIR)) if project else {}
    manifest=str(manifest or meta.get('manifest') or '').strip()
    if not manifest or not Path(manifest).is_file():
        _publish_global_status(t('LoRA training'), t('A valid prepared dataset is required before training.'), "error")
        raise gr.Error(t('Prepare/select a valid dataset project first.'))
    safe=_safe_name(adapter)
    if not safe:
        _publish_global_status(t('LoRA training'), t('An adapter name is required before training.'), "error")
        raise gr.Error(t('Enter an Adapter Name.'))
    out=TRAINING_OUTPUTS_DIR/safe
    cfg={'manifest':manifest,'model_dir':str(MODEL_DIR),'output_dir':str(out),'mode':mode,'steps':int(steps),'epochs':int(epochs),
         'save_every':int(save_steps if mode=='Steps' else max(1,int(save_epochs))*max(1, math.ceil(max(1,int(meta.get('stats',{}).get('valid',1))) / max(1,int(batch)*int(grad_accum))))),
         'rank':int(rank),'alpha':int(alpha),'dropout':float(dropout),'lr':float(lr),'batch':int(batch),'grad_accum':int(grad_accum),'grad_clip':float(grad_clip),
         # Deliberately use the stable native acoustic objective. The legacy
         # arguments remain in the function signature for old Gradio events,
         # but cannot enable auxiliary text/ASR association anymore.
         'semantic_loss_weight':0.0,'reference_text_conditioning':False,
         'phonetic_ctc_enabled':False,'phonetic_ctc_weight':0.0,
         'legacy_phonetic_ctc_enabled':False,
         'differentiable_asr_enabled':False,'differentiable_asr_weight':0.0,
         'differentiable_asr_model':'large-v3','asr_cfm_steps':4,'asr_cfm_temperature':1.0,
         'language':str(meta.get('language','ES')),
         'resume':None if not resume or resume==NONE_SENTINEL else resume,'seed':1234,'vram':vram,'project':project,
         'eval_enabled':bool(eval_enabled),'eval_text':str(eval_text or '').strip(),'eval_reference':str(eval_reference or ''),
         'eval_reference_transcript':str(eval_reference_transcript or '').strip(),'eval_language':LANGUAGES.get(eval_language,eval_language)}
    (out/'training_config.json').parent.mkdir(parents=True,exist_ok=True); (out/'training_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    result = TRAINING_MANAGER.start(cfg)
    _publish_global_status(t('LoRA training'), t('Training started.'), "working", t('Adapter: {0} · Dataset: {1}', safe, Path(manifest).name))
    return result

def stop_lora_training():
    from indextts_training import TRAINING_MANAGER
    result = TRAINING_MANAGER.stop()
    _publish_global_status(t('LoRA training'), t('Stop requested.'), "working", t('The current training step will finish before the worker exits.'))
    return result

def training_snapshot():
    from indextts_training import TRAINING_MANAGER
    st=TRAINING_MANAGER.snapshot(); total=max(1,int(st.get('total') or 1)); step=int(st.get('step') or 0); pct=100*step/total
    loss='—' if st.get('loss') is None else f"{st['loss']:.5f}"; eta=float(st.get('eta') or 0); elapsed=float(st.get('elapsed') or 0)
    status=t('{0} | Step {1}/{2} | {3:.1f}% | Loss {4} | Elapsed {5:.1f}m | ETA {6:.1f}m', 'RUNNING' if st.get('running') else 'IDLE', step, st.get('total', 0), pct, loss, elapsed / 60, eta / 60)
    return localize_backend_message(st.get('message','')), gr.update(value=pct/100)

def training_progress_html():
    from html import escape
    from indextts_training import TRAINING_MANAGER
    st=TRAINING_MANAGER.snapshot(); total=max(0,int(st.get('total') or 0)); step=int(st.get('step') or 0)
    pct=0.0 if not total else max(0.0,min(100.0,100.0*step/total))
    loss='—' if st.get('loss') is None else f"{float(st['loss']):.5f}"
    elapsed=float(st.get('elapsed') or 0); eta=float(st.get('eta') or 0)
    def fmt(v):
        total_minutes=max(0,int(round(v/60.0)))
        return t('{0}h {1:02d}m', total_minutes // 60, total_minutes % 60)
    running=bool(st.get('running')); title=t('Training') if running else (t('Training complete') if pct>=100 else t('Training'))
    counter=t('{0:,} / {1:,} steps', step, total) if total else t('Idle')
    message=escape(localize_backend_message(st.get('message') or t('Ready')))
    return (f'<div class="fire-progress-card"><div class="fire-progress-head"><div class="fire-progress-title">{title}</div>'
            f'<div class="fire-progress-meta">{counter}</div></div><div class="fire-train-track">'
            f'<div class="fire-train-fill" style="width:{pct:.2f}%"></div></div><div class="fire-progress-foot">'
            f'<span>{pct:.1f}% · {message} · {t("Loss")} {loss}</span><span>{t("Elapsed")} {fmt(elapsed)} · {t("ETA")} {fmt(eta)}</span>'
            '</div></div>')


def global_status_text():
    """Return the latest status line for the shared one-line status box."""
    with GLOBAL_STATUS_LOCK:
        return GLOBAL_STATUS_LINE


def start_training_tensorboard(adapter):
    from indextts_training import TRAINING_MANAGER
    safe=_safe_name(adapter); log=TRAINING_OUTPUTS_DIR/safe/'tensorboard'; log.mkdir(parents=True,exist_ok=True)
    result = TRAINING_MANAGER.start_tensorboard(log)
    _publish_global_status(t('TensorBoard'), result, "success", t('Log directory: {0}', log))
    return result


def reload_training_tensorboard(project, adapter):
    from indextts_training import TRAINING_MANAGER
    safe=_safe_name(adapter or project)
    if not safe:
        _publish_global_status(t('TensorBoard'), t('Select a training project first.'), "error")
        raise gr.Error(t('Select a training project before reloading TensorBoard.'))
    log=TRAINING_OUTPUTS_DIR/safe/'tensorboard'; log.mkdir(parents=True,exist_ok=True)
    result = TRAINING_MANAGER.reload_tensorboard(log)
    _publish_global_status(t('TensorBoard'), result, "success", t('Workspace: {0} · Log directory: {1}', safe, log))
    return result

_install_console_mirror()

if __name__ == "__main__":
    _ensure_model_before_ui()


_INITIAL_CAPS = acceleration_capabilities()
_INITIAL_RUNTIME = normalize_runtime_settings(_load_runtime_settings(), _INITIAL_CAPS)


with gr.Blocks(title=t(APP_TITLE), css=CSS) as demo:
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=6):
            ui_language = gr.Dropdown(
                choices=UI_LANGUAGES, value=_UI_I18N.language,
                label="界面语言 / UI Language", scale=0,
                info=t("Changes take effect after restarting WebUI."),
            )
            ui_language_notice = gr.Markdown("")
            ui_language.change(save_ui_language, ui_language, ui_language_notice, queue=False, show_progress="hidden")
            gr.Markdown(t('# 🗣️ IndexTTS 2.5 Easy GUI'))
        with gr.Column(scale=1, min_width=110):
            gr.Markdown(t('[📖 IndexTTS](https://github.com/index-tts/index-tts)'))
        with gr.Column(scale=3, min_width=520):
            with gr.Row():
                top_unload_btn = gr.Button(t('Unload Models'), size="sm")
                delete_outputs_btn = gr.Button(t('Delete output audios'), size="sm", variant="stop", elem_classes=["red-btn"])
                delete_voices_btn = gr.Button(t('Delete reference samples'), size="sm", variant="stop", elem_classes=["red-btn"])

    top_status = gr.State("")

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab(t('🎙️ Prep Samples'), id="tab_prep_samples"):
            gr.Markdown(
                t('### 📚 Reference Audio Library\nManage reusable zero-shot voice references for IndexTTS 2.5. Upload or record a clean single-speaker clip and save it locally. A transcript is optional here, and can be supplied later during inference for pronunciation-aware adapters.')
            )
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown(t('#### 📂 Your Samples'))
                    sample_dropdown = gr.Dropdown(choices=voice_choices(include_none=True), value=NONE_SENTINEL, label=t('Select Sample'), interactive=True, info=t('None enables upload/recording; saved samples load their waveform for preview.'))
                    with gr.Row():
                        refresh_samples_btn = gr.Button(t('🔄 Refresh List'), size="sm")
                        delete_sample_btn = gr.Button(t('🗑️ Delete Selected'), size="sm", variant="stop", elem_classes="red-btn")
                    sample_status = gr.State("")
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown(t('#### 🎙️ Reference Audio Editor'))
                    prep_audio = gr.Audio(
                        label=t('Reference Audio (short, clean, single speaker recommended - use Trim to edit)'),
                        type="filepath", interactive=True, sources=["upload", "microphone"], elem_classes=["audio-safe-space"]
                    )
                    prep_reference_transcript = gr.Textbox(
                        label=t('Reference Transcript (optional)'), lines=4,
                        placeholder=t('Enter the exact words spoken in this reference audio, or transcribe them below.'),
                    )
                    with gr.Accordion(t('🛰️ Faster-Whisper Transcription (Optional)'), open=False):
                        prep_whisper_model = gr.Dropdown(
                            localized_choices(["large-v3", "medium", "small", "base", "tiny"]),
                            value="large-v3", label=t('Model'),
                        )
                        prep_whisper_language = gr.Dropdown(
                            localized_choices(["Auto-detect", "en", "es", "ja", "zh", "ar"]),
                            value="Auto-detect", label=t('Language'),
                        )
                        prep_whisper_batch = gr.Slider(1, 32, 8, step=1, label=t('Batch Size'))
                        prep_transcribe_btn = gr.Button(t('Transcribe Now'), variant="secondary")
                        prep_transcribe_status = gr.State("")
                    voice_name = gr.Textbox(label=t('Voice Name'), placeholder=t('e.g. narrator_es'), lines=1)
                    save_voice_btn = gr.Button(t('💾 Save Reference Voice'), variant="primary", elem_classes="green-btn")
                    gr.Markdown(t('Saved files are copied into `voices/` and remain available across sessions.'))

            gr.Markdown(t('### 📝 Emotion Description Library'))
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown(t('#### 📂 Saved Descriptions'))
                    prep_emo_desc_select = gr.Dropdown(choices=emotion_description_choices(), value=NONE_SENTINEL, label=t('Select Description'), interactive=True, info=t('None clears the editor; saved entries display their stored text.'))
                    with gr.Row():
                        refresh_emo_desc_btn = gr.Button(t('🔄 Refresh List'), size="sm")
                        delete_emo_desc_btn = gr.Button(t('🗑️ Delete Selected'), size="sm", variant="stop", elem_classes="red-btn")
                    emo_desc_status = gr.State("")
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown(t('#### ✍️ Emotion Description Editor'))
                    prep_emo_desc_name = gr.Textbox(label=t('Description Name'), placeholder=t('e.g. warm_documentary'), lines=1)
                    prep_emo_desc_text = gr.Textbox(
                        label=t('Emotion Description'), lines=4,
                        placeholder=t('Natural-language emotion/prosody instruction, e.g. warm, intimate, reassuring delivery with restrained excitement.')
                    )
                    save_emo_desc_btn = gr.Button(t('💾 Save Emotion Description'), variant="primary", elem_classes="green-btn")
                    gr.Markdown(t('Descriptions are stored as plain UTF-8 text under `emotion_descriptions/` and can be recalled from Inference.'))

        with gr.Tab(t('🔊 Inference')):
            gr.Markdown(t('### 🎙️ Unified IndexTTS 2.5 Voice Synthesis'))

            with gr.Group(elem_classes="global-inference-controls"):
                gr.Markdown(t('#### 🎛️ Global Inference Controls'))
                with gr.Row():
                    language = gr.Dropdown(localized_choices(list(LANGUAGES)), value="Chinese", label=t('Language'))
                    lora_adapter = gr.Dropdown(
                        choices=_adapter_choices(), value=NONE_SENTINEL, label=t('LoRA Adapter'), scale=2,
                        info=t('Optional trained IndexTTS 2.5 adapter reused by Single Inference and Dialogue Builder.'),
                    )
                    refresh_lora_adapter_btn = gr.Button(t('🔄'), size="sm", scale=0, min_width=48)
                    duration_factor = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label=t('Duration Factor'), info=t('<1 faster, >1 slower'))
                    text_normalization = gr.Checkbox(value=True, label=t('Text normalization'))
                    interval_silence = gr.Slider(0, 1000, 200, step=10, label=t('Segment Silence (ms)'))

                gr.Markdown(t('#### 🛠️ Advanced Engine Settings'))
                with gr.Accordion(t('⚙️ Decoding & Sampling Parameters'), open=False):
                    with gr.Row():
                        do_sample = gr.Checkbox(True, label=t('Sampling'), info=t('Uses IndexTTS do_sample. The bundled 2.5 source is patched so this switch is honored instead of being hard-coded on.'))
                        temperature = gr.Slider(0.1, 2.0, 0.8, step=0.05, label=t('Temperature'))
                        top_p = gr.Slider(0.0, 1.0, 0.8, step=0.01, label=t('Top P'))
                        top_k = gr.Slider(0, 200, 30, step=1, label=t('Top K'))
                    with gr.Row():
                        num_beams = gr.Slider(1, 10, 3, step=1, label=t('Num Beams'))
                        repetition_penalty = gr.Number(value=10.0, minimum=0.1, maximum=30.0, step=0.1, label=t('Repetition Penalty'))
                        length_penalty = gr.Number(value=0.0, minimum=-3.0, maximum=3.0, step=0.1, label=t('Length Penalty'))
                        max_mel_tokens = gr.Slider(50, 3000, 1500, step=10, label=t('Max Mel Tokens'))
                    max_text_tokens = gr.Slider(20, 300, 120, step=5, label=t('Max Text Tokens / Segment'))

                with gr.Accordion(t('🎭 Emotion Controls'), open=False):
                    emotion_mode = gr.Radio(
                        localized_choices(["Same as voice reference", "Emotion reference audio", "Emotion vector", "Emotion description"]),
                        value="Same as voice reference", label=t('Emotion Mode')
                    )
                    with gr.Group(visible=False) as emotion_strength_group:
                        emotion_alpha = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label=t('Emotion Weight / Alpha'))
                    with gr.Group(visible=False) as emotion_ref_group:
                        emotion_audio = gr.Audio(
                            label=t('Separate Emotion Reference'), type="filepath", sources=["upload", "microphone"],
                            elem_classes="audio-safe-space"
                        )
                    with gr.Group(visible=False) as emotion_random_group:
                        random_emotion = gr.Checkbox(False, label=t('Random emotion sampling'), info=t('IndexTTS use_random. May reduce strict cloning fidelity.'))
                    with gr.Group(visible=False) as emotion_vector_group:
                        emotion_preset = gr.Dropdown(
                            choices=localized_choices(list(EMOTION_PRESETS)), value="Neutral / balanced",
                            label=t('Emotion Vector Preset'),
                            info=t('Convenience presets authored for this GUI. They directly set the official 8-value IndexTTS emotion vector; adjust sliders afterward if desired.')
                        )
                        gr.Markdown(t('##### Emotion Vector — [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]'))
                        emotion_sliders = []
                        for i in range(0, 8, 2):
                            with gr.Row():
                                emotion_sliders.append(gr.Slider(0, 1, EMOTION_PRESETS["Neutral / balanced"][i], step=0.01, label=t(EMOTIONS[i])))
                                emotion_sliders.append(gr.Slider(0, 1, EMOTION_PRESETS["Neutral / balanced"][i + 1], step=0.01, label=t(EMOTIONS[i + 1])))
                    with gr.Group(visible=False) as emotion_description_group:
                        saved_emotion_description = gr.Dropdown(
                            choices=emotion_description_choices(), value=NONE_SENTINEL, label=t('Saved Emotion Description'),
                            info=t('Selecting a library entry fills the Emotion Description box below.')
                        )
                        emotion_text = gr.Textbox(
                            label=t('Emotion Description'), lines=3,
                            placeholder=t('Describe the desired emotion/prosody. QwenEmotion is loaded only for this mode.')
                        )

                with gr.Accordion(t('🚀 CUDA Runtime'), open=False):
                    _caps = _INITIAL_CAPS
                    _runtime_values = _INITIAL_RUNTIME
                    with gr.Row():
                        use_bf16 = gr.Checkbox(_runtime_values["use_bf16"], label=t('BF16'), info=t('Recommended on supported NVIDIA GPUs. Automatically enabled by FlashAttention mode.'))
                        use_gpt_latent = gr.Checkbox(_runtime_values["use_gpt_latent"], label=t('GPT latent path'), info=t('IndexTTS 2.5 load-time use_gpt_latent option; temporal alignment compatibility fix is bundled.'))
                        use_torch_compile = gr.Checkbox(
                            _runtime_values["use_torch_compile"], label=t('torch.compile (Triton)'), interactive=_caps["triton"][0],
                            info=t('Optional S2Mel compilation. Requires triton-windows.'),
                        )
                    with gr.Row():
                        use_flash_accel = gr.Checkbox(
                            _runtime_values["use_flash_accel"], label=t('FlashAttention GPT acceleration'), interactive=_caps["flash_attn"][0] and _caps["triton"][0],
                            info=t('Optional IndexTTS 2.5 GPT acceleration engine. Requires the prebuilt FlashAttention Windows wheel + Triton + BF16.'),
                        )
                    runtime_settings_status = gr.State("")

            with gr.Tabs():
                with gr.Tab(t('Single Inference')):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="form-section"):
                            gr.Markdown(t('#### 🎙️ Voice Sample'))
                            with gr.Row():
                                infer_sample_select = gr.Dropdown(choices=voice_choices(include_none=True), value=NONE_SENTINEL, label=t('Quick Sample Select'), scale=4)
                                refresh_infer_sample_btn = gr.Button(t('🔄'), scale=1, min_width=50)
                            infer_ref_audio = gr.Audio(
                                label=t('Reference Audio / Quick Sample Preview'), type="filepath", value=None,
                                sources=["upload", "microphone"], interactive=True, elem_classes=["audio-safe-space"]
                            )
                            gr.Markdown(
                                t('Choose **None** to upload or record a custom reference directly here. Choosing a saved Quick Sample loads its waveform/player automatically so you can preview it before generation.')
                            )
                            infer_reference_transcript = gr.Textbox(
                                label=t('Reference Transcript (optional)'), lines=4,
                                placeholder=t('Enter the exact words spoken in the reference audio, or transcribe them below.'),
                            )
                            with gr.Accordion(t('🛰️ Faster-Whisper Transcription (Optional)'), open=False):
                                infer_reference_whisper_model = gr.Dropdown(
                                    localized_choices(["large-v3", "medium", "small", "base", "tiny"]),
                                    value="large-v3", label=t('Model'),
                                )
                                infer_reference_whisper_language = gr.Dropdown(
                                    localized_choices(["Auto-detect", "en", "es", "ja", "zh", "ar"]),
                                    value="Auto-detect", label=t('Language'),
                                )
                                infer_reference_whisper_batch = gr.Slider(1, 32, 8, step=1, label=t('Batch Size'))
                                infer_reference_auto_transcribe = gr.Checkbox(
                                    False, label=t('Auto-transcribe when the field is empty'),
                                    info=t('Uses the selected Faster-Whisper model immediately before generation.'),
                                )
                                infer_reference_transcribe_btn = gr.Button(t('Transcribe Now'), variant="secondary")
                                infer_reference_transcribe_status = gr.State("")

                        with gr.Column(scale=1, elem_classes=["form-section", "target-speech-box"]):
                            gr.Markdown(t('#### ✍️ Target Speech'))
                            infer_text = gr.Textbox(label=t('Target Text'), placeholder=t('Enter the text to synthesize...'), lines=8, value="", elem_classes=["compact"])
                            with gr.Accordion(t('📚 Long Text / Chunking'), open=False):
                                with gr.Row():
                                    infer_chunk_mode = gr.Dropdown(
                                        localized_choices(CHUNK_CHOICES),
                                        value="None",
                                        label=t('Chunk Mode'),
                                        info=t('None disables splitting. Paragraph/Sentence Auto is recommended for long prose and follows orthographic sentence punctuation.'),
                                    )
                                    infer_chunk_silence = gr.Slider(
                                        0, 5, value=0.5, step=0.1,
                                        label=t('Silence Between Chunks (seconds)'),
                                    )
                                gr.Markdown(
                                    t('Each chunk is synthesized separately with the same voice and settings, then all chunks are joined into one WAV. The automatic mode keeps sentence punctuation together and only wraps an overlong sentence at clauses or words.')
                                )
                            with gr.Row():
                                infer_gen_btn = gr.Button(t('⚡ Generate Speech'), variant="primary", size="lg", scale=4, elem_classes="button-primary")
                                infer_stop_btn = gr.Button(t('⏹️ Stop'), variant="stop", scale=1, min_width=90, elem_classes="button-stop")
                                infer_save_outputs = gr.Checkbox(label=t('Save to outputs'), value=True, scale=1, min_width=130)
                            with gr.Group(elem_classes=["output-clean"]):
                                infer_audio_out = gr.Audio(label=t('Generated Audio'), type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                                infer_file_out = gr.Textbox(label=t('Saved WAV'), interactive=False, elem_classes=["output-path"])
                            generation_status = gr.State("")


                with gr.Tab(t('Dialogue Builder')):
                    gr.Markdown(
                        t('#### 💬 Multi-Speaker Dialogue Builder\nModular turn workflow. Global language, decoding, timing and CUDA Runtime controls above are shared with Single Inference. Each visible turn has its own voice, text and mutually exclusive emotion source.')
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
                        dialogue_reset_btn = gr.Button(t('Reset rows'), size="sm", variant="secondary")
                        dialogue_clear_all_btn = gr.Button(t('Clear rows'), size="sm", variant="secondary")
                        dialogue_compact_btn = gr.Button(t('Remove empty rows'), size="sm", variant="secondary")
                        dialogue_refresh_libraries_btn = gr.Button(t('🔄 Refresh Voice + Emotion Libraries'), size="sm")

                    for i in range(DIALOGUE_MAX_SEGMENTS):
                        with gr.Group(visible=i < 2, elem_classes=["dialogue-turn-card"]) as d_row:
                            with gr.Row():
                                with gr.Column(scale=2, min_width=190):
                                    gr.Markdown(t('**Turn {0}**', i + 1), elem_classes="dialogue-turn-title")
                                    d_speaker = gr.Dropdown(
                                        choices=voice_choices(include_none=True), value=NONE_SENTINEL,
                                        label=t('Voice Library'), interactive=True,
                                    )
                                with gr.Column(scale=7):
                                    d_text = gr.Textbox(
                                        label=t('Text {0}', i + 1), placeholder=t('Enter dialogue text for turn {0}...', i + 1),
                                        lines=4,
                                    )
                                with gr.Column(scale=1, min_width=92, elem_classes=["dialogue-actions"]):
                                    d_add = gr.Button(t('➕'), variant="secondary", size="sm", elem_classes=["green-btn"])
                                    d_clone = gr.Button(t('📋'), variant="secondary", size="sm")
                                    with gr.Row():
                                        d_up = gr.Button(t('⬆️'), variant="secondary", size="sm", min_width=42)
                                        d_down = gr.Button(t('⬇️'), variant="secondary", size="sm", min_width=42)
                                    d_clear = gr.Button(t('🧹'), variant="secondary", size="sm")
                                    d_delete = gr.Button(t('🗑️'), variant="stop", size="sm", elem_classes=["red-btn"])

                            d_mode = gr.Radio(
                                localized_choices(["Emotion Vector Preset", "Emotion Description"]),
                                value="Emotion Vector Preset", label=t('Emotion Source'),
                                elem_classes=["dialogue-emotion-source"],
                            )
                            with gr.Group(visible=True) as d_vector_group:
                                d_vector = gr.Dropdown(
                                    choices=localized_choices(list(EMOTION_PRESETS)), value="Neutral / balanced",
                                    label=t('Emotion Vector Preset'),
                                )
                            with gr.Group(visible=False) as d_desc_group:
                                d_desc = gr.Dropdown(
                                    choices=emotion_description_choices(), value=NONE_SENTINEL,
                                    label=t('Emotion Description Library'),
                                )
                                d_desc_preview = gr.Textbox(
                                    label=t('Saved Emotion Description'), value="", lines=2,
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

                    dialogue_silence = gr.Slider(0.0, 5.0, value=0.5, step=0.1, label=t('Silence between dialogue turns (seconds)'))
                    with gr.Row():
                        dialogue_gen_btn = gr.Button(t('⚡ Generate Dialogue'), variant="primary", size="lg", scale=4, elem_classes="button-primary")
                        dialogue_stop_btn = gr.Button(t('⏹️ Stop'), variant="stop", scale=1, min_width=90, elem_classes="button-stop")
                    with gr.Group(elem_classes=["output-clean"]):
                        dialogue_audio_out = gr.Audio(label=t('Generated Dialogue'), type="filepath", elem_classes=["audio-safe-space", "output-clean"])
                        dialogue_file_out = gr.Textbox(label=t('Saved WAV'), interactive=False, elem_classes=["output-path"])
                    dialogue_status = gr.State("")

            with gr.Accordion(t('📖 IndexTTS 2.5 Control Reference'), open=False):
                gr.Markdown(
                
                t("**Language** selects the language hint passed to IndexTTS (EN/ES/JA/ZH/AR). **Duration Factor** scales predicted duration: `1.0` neutral, `<1` faster/shorter, `>1` slower/longer. **Text normalization** enables the model's normalization pipeline before tokenization. **Segment Silence** inserts silence between automatically split long-text segments.\n\n**Sampling (`do_sample`)** enables stochastic GPT token selection. **Temperature** changes sampling sharpness; lower is more conservative, higher more varied. **Top P** nucleus-filters the probability mass; **Top K** limits candidates to the K most likely tokens. **Num Beams** controls beam search width. **Repetition Penalty** discourages repeated token patterns. **Length Penalty** biases beam scoring by sequence length. **Max Mel Tokens** caps generated acoustic-token length. **Max Text Tokens / Segment** controls automatic text splitting before synthesis. IndexTTS 2.5 exposes no dedicated public seed argument in this inference API.\n\n**Emotion Mode — Same as voice reference:** no separate emotion condition is supplied; prosody/emotion follows the speaker prompt. **Emotion reference audio:** uses a second audio prompt for emotion while retaining timbre from the voice reference. **Emotion vector:** sends the official eight-value vector `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`; the GUI presets are convenience combinations and the sliders remain editable. **Emotion description:** sends natural-language guidance through QwenEmotion. Saved descriptions only copy text into the box; selecting `None` clears it. **Emotion Weight / Alpha** controls conditioning strength; stronger is not always more natural. **Random emotion sampling** enables IndexTTS `use_random`, adding emotion variability at the expense of strict repeatability.\n\n**BF16** loads/executes supported model paths in bfloat16 to reduce VRAM and improve throughput on compatible NVIDIA GPUs; FlashAttention mode requires it here. **GPT latent path** enables the alternate latent conditioning path in the bundled 2.5 source; this GUI includes a temporal-length alignment fix for the observed 2× latent mismatch. **torch.compile / Triton** asks PyTorch to compile supported S2Mel/runtime graphs; first use can be slower while local kernels are compiled/cached. **FlashAttention GPT acceleration** enables the IndexTTS `use_accel` GPT path and requires the prebuilt Windows FlashAttention wheel plus Triton. It is optional and can be disabled for maximum compatibility.\n\n**BigVGAN:** BigVGAN itself remains the final neural vocoder that converts the model's acoustic/mel representation into waveform audio. Only its optional fused custom CUDA activation kernel is disabled in this fixed Windows runtime. The normal PyTorch BigVGAN path remains active, so this is a performance choice rather than a quality-mode switch.\n\n**Dialogue Builder:** `➕` inserts a new turn after the current one, `📋` clones it, `⬆️/⬇️` reorder it, `🧹` clears only that turn, and `🗑️` removes it while shifting later turns upward. `Reset rows`, `Clear rows`, and `Remove empty rows` operate on the complete dialogue. Per-turn Voice Library and emotion settings override only that turn; all generation/runtime controls above remain global and shared with Single Inference.\n\nOfficial references: [IndexTTS repository](https://github.com/index-tts/index-tts) · [IndexTTS 2.5 Technical Report & demos](https://index-tts.github.io/index-tts2-5.github.io/)")
                )




        with gr.Tab(t('📂 Dataset Preparation'), id="tab_prepare_dataset"):
            gr.Markdown(t('*Organize project audio/transcripts and prepare an IndexTTS 2.5 training manifest.*'), elem_classes=["tab-subtitle"])
            with gr.Accordion(t('📖 Quick Guide'), open=False, elem_classes=["quick-guide"]):
                gr.Markdown(
                    t('Create or select a project, choose the source audio folder and language, then Build / Scan Dataset. Use Missing Transcript ASR only when transcript files are absent. Prepare Dataset writes the reusable manifest.')
                )

            with gr.Row(elem_classes=["project-strip"]):
                dataset_project_select = gr.Dropdown(
                    choices=_training_project_choices(), value=None, label=t('Project'), scale=2
                )
                dataset_project_name = gr.Textbox(label=t('New Project Name'), placeholder=t('my_voice'), scale=2)
                dataset_create_project = gr.Button(t('Create Project'), scale=1)
                dataset_save_project = gr.Button(t('💾 Save Project'), scale=1)
                dataset_clone_project = gr.Button(t('Clone Project'), variant="secondary", scale=1)
                dataset_delete_project = gr.Button(
                    t('🗑️ Delete Project'), variant="stop", scale=1, elem_classes=["training-red"]
                )

            with gr.Row():
                dataset_source_folder = gr.Textbox(label=t('Source Audio Folder'), scale=8)
                dataset_browse_source = gr.Button(t('📁 Browse'), scale=1, min_width=120)
                dataset_language = gr.Dropdown(localized_choices(list(LANGUAGES)), value="English", label=t('Language / Dialect'), scale=3)

            dataset_build_scan_btn = gr.Button(
                t('Build / Scan Dataset'), variant="secondary"
            )
            dataset_summary_box = gr.State("")
            dataset_stats_state = gr.State("")

            with gr.Accordion(t('🛰️ Missing Transcript ASR'), open=True):
                dataset_transcribe = gr.Checkbox(True, label=t('Transcribe Missing .txt Files'))
                with gr.Row():
                    dataset_whisper_model = gr.Dropdown(
                        localized_choices(["large-v3", "medium", "small", "base", "tiny"]),
                        value="large-v3", label=t('Faster-Whisper Model'),
                    )
                    dataset_whisper_language = gr.Dropdown(
                        localized_choices(["Auto-detect", "en", "es", "ja", "zh", "ar"]),
                        value="Auto-detect", label=t('Whisper Language'),
                    )
                    dataset_asr_batch = gr.Slider(1, 32, 8, step=1, label=t('Batch Size'))
                gr.Markdown(t('Clip duration limits are calculated automatically from the selected dataset.'))

            dataset_prepare_btn = gr.Button(
                t('Prepare Dataset'), variant="primary",
                elem_classes=["training-wide-action", "training-orange"],
            )
            dataset_manifest = gr.Textbox(label=t('Prepared Manifest'), interactive=False)
            dataset_project_status = gr.State("")

        with gr.Tab(t('🚀 LoRA Training'), id="tab_lora_training"):
            gr.Markdown(t('*Fine-tune the IndexTTS 2.5 GPT with project persistence, AutoTune, checkpoints and TensorBoard.*'), elem_classes=["tab-subtitle"])
            with gr.Accordion(t('📖 Quick Guide'), open=False, elem_classes=["quick-guide"]):
                gr.Markdown(
                    t('Select a prepared project, choose the VRAM preset and run AutoTune. Save the project after adjusting training settings. Training adapts the IndexTTS 2.5 GPT with PEFT LoRA while the acoustic stack remains frozen. The active objective is native acoustic mel-token reconstruction; legacy text/ASR auxiliary losses are disabled.')
                )

            with gr.Row(elem_classes=["project-strip"]):
                with gr.Column(scale=4):
                    train_project = gr.Dropdown(choices=_training_project_choices(), value=None, label=t('Project'))
                with gr.Column(scale=3):
                    vram_preset = gr.Dropdown(localized_choices(["24 GB", "32 GB+"]), value="24 GB", label=t('VRAM Preset'))
                with gr.Column(scale=2, min_width=155):
                    autotune_btn = gr.Button(t('⚡ AutoTune'), variant="secondary")
                with gr.Column(scale=2, min_width=155):
                    save_training_project_btn = gr.Button(t('💾 Save Project'), variant="secondary")
                with gr.Column(scale=2, min_width=155):
                    train_clone_project = gr.Button(t('Clone Project'), variant="secondary")
                with gr.Column(scale=2, min_width=155):
                    train_delete_project = gr.Button(
                        t('🗑️ Delete Project'), variant="stop", scale=1, elem_classes=["training-red"]
                    )
            autotune_status = gr.State("")

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["form-section"]):
                    gr.Markdown(t('#### 📦 Training Setup'))
                    dataset_manifest_train = gr.Textbox(label=t('Prepared Dataset'), interactive=True)
                    adapter_name = gr.Textbox(label=t('Adapter Name'), placeholder=t('my_voice_lora'))
                    mode = gr.Dropdown(localized_choices(["Steps", "Epochs"]), value="Steps", label=t('Training Length Mode'))
                    with gr.Group(visible=True) as steps_group:
                        with gr.Row():
                            training_steps = gr.Slider(1, 10000, 1000, step=1, label=t('Training Steps'))
                            save_steps = gr.Slider(1, 2000, 100, step=1, label=t('Save Every N Steps'))
                    with gr.Group(visible=False) as epochs_group:
                        with gr.Row():
                            training_epochs = gr.Slider(1, 500, 20, step=1, label=t('Training Epochs'))
                            save_epochs = gr.Slider(1, 100, 2, step=1, label=t('Save Every N Epochs'))
                    with gr.Row():
                        resume_checkpoint = gr.Dropdown(
                            choices=localized_choices([("None", NONE_SENTINEL)]), value=NONE_SENTINEL,
                            label=t('Resume Checkpoint'), scale=5,
                        )
                        refresh_resume_btn = gr.Button(t('🔄'), min_width=52, scale=0)

                with gr.Column(scale=1, elem_classes=["form-section"]):
                    gr.Markdown(t('#### ⚙️ LoRA Hyperparameters'))
                    with gr.Row():
                        lora_rank = gr.Dropdown(
                            localized_choices([4, 8, 16, 32, 64, 128, 256]), value=16, label=t('LoRA Rank')
                        )
                        lora_alpha = gr.Slider(4, 128, 32, step=1, label=t('LoRA Alpha'))
                    lora_dropout = gr.Slider(0, .2, .05, step=.01, label=t('LoRA Dropout'))
                    learning_rate = gr.Slider(.000001, .001, .00004, step=.000001, label=t('Learning Rate'))
                    with gr.Row():
                        grad_accum = gr.Slider(1, 64, 8, step=1, label=t('Gradient Accumulation'))
                        training_seed = gr.Number(1234, precision=0, label=t('Training Seed'))
                    # Kept as hidden compatibility inputs for old project
                    # schemas. The active training profile is acoustic-only:
                    # no auxiliary text-token, ASR, CTC or reference-text loss.
                    semantic_loss_weight = gr.Number(0.0, precision=3, visible=False)
                    phonetic_ctc_enabled = gr.Checkbox(False, visible=False)
                    phonetic_ctc_weight = gr.Number(0.0, precision=3, visible=False)
                    # Kept for the trainer backend but intentionally hidden to match the FireRed control surface.
                    micro_batch = gr.Number(1, precision=0, visible=False)
                    grad_clip = gr.Number(0.5, precision=2, visible=False)
                    lora_internal_note = gr.Markdown("", visible=False)

            with gr.Accordion(t('🎧 Eval Audio + TensorBoard'), open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_enabled = gr.Checkbox(True, label=t('Generate Eval Audio at Checkpoints'))
                        eval_text = gr.Textbox(value="This is an IndexTTS 2.5 training preview.", label=t('Eval Text'), lines=3)
                        eval_language = gr.Dropdown(localized_choices(list(LANGUAGES)), value="English", label=t('Eval Language'))
                    with gr.Column(scale=1):
                        eval_reference = gr.Audio(
                            type="filepath", label=t('Eval Reference Audio'), sources=["upload", "microphone"],
                            elem_classes=["audio-safe-space"],
                        )
                        eval_reference_transcript = gr.Textbox(label=t('Eval Reference Transcript'), lines=3)
                with gr.Row():
                    eval_whisper_model = gr.Dropdown(
                        localized_choices(["large-v3", "medium", "small", "base", "tiny"]), value="large-v3", label=t('Transcription Model')
                    )
                    eval_whisper_language = gr.Dropdown(
                        localized_choices(["Auto-detect", "en", "es", "ja", "zh", "ar"]), value="Auto-detect", label=t('Speech Language')
                    )
                    eval_asr_batch = gr.Slider(1, 32, 8, step=1, label=t('Batch Size'))
                eval_transcribe_btn = gr.Button(t('🛰️ Transcribe Eval Audio'), variant="secondary")
                eval_transcribe_status = gr.State("")

            with gr.Row(elem_classes=["global-toolbar"]):
                start_training_btn = gr.Button(
                    t('🚀 Start Training'), variant="primary", elem_classes=["training-orange"], scale=1
                )
                stop_training_btn = gr.Button(
                    t('⏹️ Stop'), variant="stop", elem_classes=["training-red"], scale=1
                )
                tensorboard_btn = gr.Button(t('📊 TensorBoard'), variant="secondary", scale=1)
                reload_tensorboard_btn = gr.Button(t('↻ Reload TensorBoard'), variant="secondary", scale=1)

            training_status = gr.HTML(value=training_progress_html(), elem_classes=["training-progress"])
            training_progress = gr.State(0.0)

            tensorboard_status = gr.State("")
            training_message = gr.State("")
            training_timer = gr.Timer(1.0)

    # Internal sinks keep existing event callbacks harmless without exposing
    # a second layer of status fields in the visual workflow.
    runtime_status = gr.State("")
    activity_status = gr.State("")
    runtime_metrics = gr.State("")

    global_status_panel = gr.Textbox(
        label=t('Status'), value=global_status_text(), lines=1, max_lines=1,
        interactive=False, elem_classes=["global-status-panel"],
    )
    global_status_timer = gr.Timer(0.5)

    with gr.Accordion(t('🖥️ Embedded Console'), open=True, elem_classes=["console-accordion"]):
        gr.Markdown(
        )
        cmd_console = gr.HTML(value=console_html())
        cmd_console_timer = gr.Timer(0.5)


    dataset_browse_source.click(
        browse_dataset_folder, dataset_source_folder, dataset_source_folder,
        queue=False, show_progress="hidden",
    )
    dataset_create_project.click(
        create_dataset_project, dataset_project_name,
        [dataset_project_select, dataset_project_name, dataset_project_status],
        queue=False, show_progress="hidden",
    )
    dataset_save_project.click(
        save_dataset_project,
        [dataset_project_select, dataset_project_name, dataset_source_folder, dataset_manifest, dataset_language],
        [dataset_project_select, dataset_project_name, dataset_project_status],
        queue=False, show_progress="hidden",
    )
    dataset_clone_project.click(
        clone_dataset_project,
        dataset_project_select,
        [dataset_project_select, dataset_project_name, dataset_source_folder, dataset_manifest,
         dataset_summary_box, dataset_language, train_project, dataset_manifest_train, dataset_project_status],
        queue=False, show_progress="hidden",
    ).then(
        load_full_training_project,
        dataset_project_select,
        [
            dataset_source_folder, dataset_manifest_train, dataset_summary_box, dataset_language,
            adapter_name, vram_preset, mode, training_steps, training_epochs,
            save_steps, save_epochs, lora_rank, lora_alpha, lora_dropout,
            learning_rate, micro_batch, grad_accum, grad_clip, semantic_loss_weight,
            phonetic_ctc_enabled, phonetic_ctc_weight, resume_checkpoint,
            eval_enabled, eval_text, eval_reference, eval_reference_transcript,
            eval_language, eval_whisper_model, eval_whisper_language,
            steps_group, epochs_group, autotune_status,
        ],
        queue=False, show_progress="hidden",
    )
    dataset_delete_project.click(
        delete_training_project, dataset_project_select,
        [train_project, dataset_project_select, dataset_project_status],
        queue=False, show_progress="hidden",
    )
    dataset_project_select.change(
        load_training_project,
        dataset_project_select,
        [dataset_source_folder, dataset_manifest, dataset_summary_box, dataset_language],
        queue=False, show_progress="hidden",
    ).then(
        lambda name: t('Loaded project: {0}', name) if name else t('No project selected.'),
        dataset_project_select, dataset_project_status,
        queue=False, show_progress="hidden",
    ).then(
        sync_training_project_selection,
        dataset_project_select,
        [train_project, dataset_manifest_train, autotune_status],
        queue=False, show_progress="hidden",
    )
    dataset_build_scan_btn.click(
        scan_training_dataset,
        [dataset_source_folder],
        [dataset_summary_box, dataset_stats_state],
    )
    dataset_prepare_btn.click(
        prepare_training_dataset,
        [
            dataset_source_folder, dataset_project_name, dataset_project_select, dataset_language,
            dataset_transcribe,
            dataset_whisper_model, dataset_whisper_language, dataset_asr_batch,
        ],
        [dataset_summary_box, dataset_project_select, dataset_manifest, train_project,
         dataset_manifest_train, autotune_status],
    ).then(
        lambda summary: t('Dataset prepared. {0}', summary),
        dataset_summary_box, dataset_project_status,
        queue=False, show_progress="hidden",
    )

    train_project.change(
        load_full_training_project,
        train_project,
        [
            dataset_source_folder, dataset_manifest_train, dataset_summary_box, dataset_language,
            adapter_name, vram_preset, mode, training_steps, training_epochs,
            save_steps, save_epochs, lora_rank, lora_alpha, lora_dropout,
            learning_rate, micro_batch, grad_accum, grad_clip, semantic_loss_weight,
            phonetic_ctc_enabled, phonetic_ctc_weight, resume_checkpoint,
            eval_enabled, eval_text, eval_reference, eval_reference_transcript,
            eval_language, eval_whisper_model, eval_whisper_language,
            steps_group, epochs_group, autotune_status,
        ],
        queue=False, show_progress="hidden",
    )
    mode.change(training_mode_visibility, mode, [steps_group, epochs_group], queue=False)

    autotune_btn.click(
        training_autotune,
        [train_project, vram_preset],
        [mode, training_steps, training_epochs, save_steps, save_epochs, lora_rank, lora_alpha, learning_rate, grad_accum, autotune_status],
    )
    refresh_resume_btn.click(
        refresh_resume, [adapter_name, resume_checkpoint], resume_checkpoint,
        queue=False, show_progress="hidden",
    )
    eval_transcribe_btn.click(
        transcribe_training_eval,
        [eval_reference, eval_whisper_model, eval_whisper_language, eval_asr_batch],
        [eval_reference_transcript, eval_transcribe_status],
    )

    save_training_project_btn.click(
        save_training_project,
        [
            train_project, dataset_source_folder, dataset_manifest_train, dataset_language,
            adapter_name, vram_preset, mode, training_steps, training_epochs,
            save_steps, save_epochs, lora_rank, lora_alpha, lora_dropout,
            learning_rate, micro_batch, grad_accum, grad_clip, semantic_loss_weight,
            phonetic_ctc_enabled, phonetic_ctc_weight, resume_checkpoint,
            eval_enabled, eval_text, eval_reference, eval_reference_transcript,
            eval_language, eval_whisper_model, eval_whisper_language,
        ],
        [train_project, dataset_project_select, autotune_status],
    )
    train_clone_project.click(
        clone_lora_project,
        train_project,
        [
            train_project, dataset_project_select, dataset_project_name, dataset_manifest,
            dataset_source_folder, dataset_manifest_train, dataset_summary_box, dataset_language,
            adapter_name, vram_preset, mode, training_steps, training_epochs,
            save_steps, save_epochs, lora_rank, lora_alpha, lora_dropout,
            learning_rate, micro_batch, grad_accum, grad_clip, semantic_loss_weight,
            phonetic_ctc_enabled, phonetic_ctc_weight, resume_checkpoint,
            eval_enabled, eval_text, eval_reference, eval_reference_transcript,
            eval_language, eval_whisper_model, eval_whisper_language,
            steps_group, epochs_group, autotune_status, dataset_project_status,
        ],
        queue=False, show_progress="hidden",
    )
    train_delete_project.click(
        delete_training_project, train_project,
        [train_project, dataset_project_select, autotune_status],
        queue=False, show_progress="hidden",
    )

    start_training_btn.click(
        start_lora_training,
        [
            train_project, dataset_manifest_train, adapter_name, vram_preset, mode, training_steps,
            training_epochs, save_steps, save_epochs, lora_rank, lora_alpha,
            lora_dropout, learning_rate, micro_batch, grad_accum, grad_clip, semantic_loss_weight,
            phonetic_ctc_enabled, phonetic_ctc_weight,
            resume_checkpoint, eval_enabled, eval_text, eval_reference, eval_reference_transcript, eval_language,
        ],
        training_message,
    )
    stop_training_btn.click(stop_lora_training, outputs=training_message, queue=False)
    tensorboard_btn.click(start_training_tensorboard, adapter_name, tensorboard_status, queue=False)
    reload_tensorboard_btn.click(
        reload_training_tensorboard, [train_project, adapter_name], tensorboard_status, queue=False,
    )
    training_timer.tick(training_progress_html, None, training_status, queue=False, show_progress="hidden")
    training_timer.tick(
        training_snapshot, None, [training_message, training_progress],
        queue=False, show_progress="hidden",
    )

    dataset_project_select.change(
        lambda current: gr.update(choices=_training_project_choices(), value=current),
        dataset_project_select, train_project,
        queue=False, show_progress="hidden",
    )

    top_unload_btn.click(unload_model, outputs=runtime_status).then(model_status, outputs=top_status)
    delete_outputs_btn.click(delete_outputs, outputs=runtime_status)
    delete_voices_btn.click(delete_all_voices, outputs=[runtime_status, sample_dropdown, infer_sample_select]).then(
        lambda: ("", ""), outputs=[prep_reference_transcript, infer_reference_transcript]
    )

    save_voice_btn.click(
        save_voice, [prep_audio, voice_name, prep_reference_transcript],
        [sample_status, sample_dropdown, infer_sample_select],
    ).then(set_prep_voice, sample_dropdown, [prep_audio, prep_reference_transcript]).then(
        set_voice_inference, infer_sample_select, [infer_ref_audio, infer_reference_transcript]
    )
    prep_transcribe_btn.click(
        transcribe_audio_reference,
        [prep_audio, prep_whisper_model, prep_whisper_language, prep_whisper_batch],
        prep_reference_transcript,
    )
    refresh_samples_btn.click(lambda value: refresh_voice_dropdown(value, include_none=True), sample_dropdown, sample_dropdown)
    refresh_infer_sample_btn.click(lambda value: refresh_voice_dropdown(value, include_none=True), infer_sample_select, infer_sample_select)
    refresh_lora_adapter_btn.click(refresh_adapters, lora_adapter, lora_adapter, queue=False, show_progress="hidden")
    delete_sample_btn.click(delete_selected_voice, sample_dropdown, [sample_status, sample_dropdown, infer_sample_select]).then(
        lambda: ("", ""), outputs=[prep_reference_transcript, infer_reference_transcript]
    )
    sample_dropdown.change(set_prep_voice, sample_dropdown, [prep_audio, prep_reference_transcript])
    infer_sample_select.change(set_voice_inference, infer_sample_select, [infer_ref_audio, infer_reference_transcript])

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
    infer_reference_transcribe_btn.click(
        transcribe_inference_reference,
        [infer_sample_select, infer_ref_audio, infer_reference_whisper_model, infer_reference_whisper_language, infer_reference_whisper_batch],
        infer_reference_transcript,
    )

    gen_inputs = [
        infer_sample_select, infer_ref_audio, infer_text, infer_reference_transcript, infer_reference_auto_transcribe,
        infer_reference_whisper_model, infer_reference_whisper_language, infer_reference_whisper_batch,
        lora_adapter, language, emotion_mode, emotion_audio, emotion_alpha,
        *emotion_sliders, emotion_text, random_emotion, duration_factor, interval_silence, infer_chunk_mode, infer_chunk_silence,
        text_normalization, max_text_tokens, do_sample, top_p, top_k, temperature,
        num_beams, repetition_penalty, length_penalty, max_mel_tokens,
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, infer_save_outputs,
    ]
    infer_gen_btn.click(lambda: t('Single inference running…'), outputs=activity_status, queue=False, show_progress="hidden")
    generation_event = infer_gen_btn.click(generate, gen_inputs, [infer_audio_out, infer_file_out, generation_status])
    generation_event.then(lambda status: status, generation_status, activity_status, queue=False, show_progress="hidden")
    infer_stop_btn.click(lambda: t('Single inference stop requested.'), outputs=activity_status, queue=False, show_progress="hidden")
    infer_stop_btn.click(
        lambda: t('Stop requested. The queued Gradio generation event was cancelled; an already-running CUDA kernel returns control before cancellation can fully complete.'),
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
        lora_adapter, language, duration_factor, interval_silence, text_normalization, max_text_tokens,
        do_sample, top_p, top_k, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens,
        use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel, dialogue_silence,
        dialogue_row_count, *dialogue_components,
    ]
    dialogue_gen_btn.click(lambda: t('Dialogue generation running…'), outputs=activity_status, queue=False, show_progress="hidden")
    dialogue_event = dialogue_gen_btn.click(generate_dialogue, dialogue_inputs, [dialogue_audio_out, dialogue_file_out, dialogue_status])
    dialogue_stop_btn.click(lambda: t('Dialogue stop requested.'), outputs=activity_status, queue=False, show_progress="hidden")
    dialogue_stop_btn.click(lambda: t('Stop requested.'), outputs=dialogue_status, cancels=[dialogue_event])

    runtime_inputs = [use_bf16, use_gpt_latent, use_torch_compile, use_flash_accel]
    for runtime_control in (use_bf16, use_gpt_latent, use_torch_compile):
        runtime_control.change(_save_runtime_settings, runtime_inputs, runtime_settings_status)
    use_flash_accel.change(
        on_flash_change, [use_flash_accel, use_bf16, use_gpt_latent, use_torch_compile],
        [use_bf16, runtime_settings_status],
    )
    dialogue_event.then(lambda status: status, dialogue_status, activity_status, queue=False, show_progress="hidden")


    global_status_timer.tick(global_status_text, None, global_status_panel, queue=False, show_progress="hidden")
    cmd_console_timer.tick(console_html, None, cmd_console, queue=False, show_progress="hidden")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, server_name=os.environ.get("INDEXTTS_HOST", "127.0.0.1"), server_port=int(os.environ.get("INDEXTTS_PORT", "7860")), show_error=True)
