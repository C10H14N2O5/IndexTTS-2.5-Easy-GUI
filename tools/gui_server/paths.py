"""Directory layout shared by every service (moved from easy_gui.py)."""
from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "IndexTTS-2.5"
OUTPUTS_DIR = ROOT / "outputs"
ABX_DIR = OUTPUTS_DIR / "abx"
EXPORTS_DIR = OUTPUTS_DIR / "exports"
VOICES_DIR = ROOT / "voices"
EMOTION_DESCRIPTIONS_DIR = ROOT / "emotion_descriptions"
RUNTIME_DIR = ROOT / ".runtime"
SETTINGS_FILE = RUNTIME_DIR / "gui_settings.json"
CACHE_DIR = RUNTIME_DIR / "hf-cache"
TEMP_DIR = RUNTIME_DIR / "temp"
UPLOADS_DIR = TEMP_DIR / "uploads"
ASSETS_DIR = ROOT / "assets"
TRAINING_DIR = ROOT / "training"
TRAINING_PROJECTS_DIR = TRAINING_DIR / "projects"
TRAINING_DATASETS_DIR = TRAINING_DIR / "datasets"
TRAINING_OUTPUTS_DIR = TRAINING_DIR / "outputs"
VALIDATION_DIR = ROOT / "validation"
WEB_DIR = ROOT / "tools" / "gui_web"

MODEL_REPO = "IndexTeam/IndexTTS-2.5"
APP_TITLE = "IndexTTS 2.5 Easy GUI"
NONE_SENTINEL = "__NONE__"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

CREATED_DIRS = (MODEL_DIR, OUTPUTS_DIR, VOICES_DIR, EMOTION_DESCRIPTIONS_DIR, CACHE_DIR, TEMP_DIR,
                TRAINING_PROJECTS_DIR, TRAINING_DATASETS_DIR, TRAINING_OUTPUTS_DIR)


def ensure_dirs() -> None:
    # Read the current directory constants, not an import-time tuple: tests and
    # standalone demos can redirect paths without touching the user's data.
    for path in (MODEL_DIR, OUTPUTS_DIR, VOICES_DIR, EMOTION_DESCRIPTIONS_DIR, CACHE_DIR, TEMP_DIR,
                 TRAINING_PROJECTS_DIR, TRAINING_DATASETS_DIR, TRAINING_OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def use_data_root(directory: str | Path) -> None:
    """Redirect mutable demo data without moving shipped assets/code.

    Launcher permits this only with the fake engine. It must be called before
    importing application services, so no production cache can be reused.
    """
    global CREATED_DIRS
    old_root = ROOT
    new_root = Path(directory).expanduser().resolve()
    for name, value in list(globals().items()):
        if name in {"WEB_DIR", "ASSETS_DIR"} or not isinstance(value, Path):
            continue
        if value.is_relative_to(old_root):
            globals()[name] = new_root / value.relative_to(old_root)
    CREATED_DIRS = (MODEL_DIR, OUTPUTS_DIR, VOICES_DIR, EMOTION_DESCRIPTIONS_DIR, CACHE_DIR, TEMP_DIR,
                    TRAINING_PROJECTS_DIR, TRAINING_DATASETS_DIR, TRAINING_OUTPUTS_DIR)


def safe_name(value: str) -> str:
    """Same rule as the 1.x GUI: keep alnum, '-', '_' and spaces; spaces become '_'."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (value or "").strip())
    return safe.strip().replace(" ", "_")


def _explorer_compare():
    """Windows Explorer's name comparison (StrCmpLogicalW), or None off Windows."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        compare = ctypes.windll.shlwapi.StrCmpLogicalW
        compare.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        compare.restype = ctypes.c_int
        return compare
    except (AttributeError, OSError):
        return None


_EXPLORER_COMPARE = _explorer_compare()


def _fallback_key(value: str) -> tuple:
    # Case-insensitive natural order: digit runs compare as numbers ("v2" < "v10").
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"(\d+)", str(value).casefold()) if part)


def natural_key(value: str):
    """Sort key matching the File Explorer "Name" order (numbers compared numerically).

    ``Hazel_v2_…`` < ``Hazel_v3_…`` < ``Hazel_v3-1_…`` < ``Hazel_v10_…``.
    """
    if _EXPLORER_COMPARE is not None:
        return _ExplorerKey(str(value))
    return _fallback_key(value)


@functools.total_ordering
class _ExplorerKey:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        return _EXPLORER_COMPARE(self.value, other.value) == 0

    def __lt__(self, other):
        return _EXPLORER_COMPARE(self.value, other.value) < 0


def is_within(path: Path, parent: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(parent).resolve())
    except (OSError, ValueError):
        return False
