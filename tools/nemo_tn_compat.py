from __future__ import annotations

import difflib
import sys
import types


def install_optional_audio_diff_fallback() -> None:
    """Keep NeMo TN importable when its optional C diff extension is absent.

    ``nemo_text_processing.text_normalization`` imports the audio-based
    normalizer eagerly, although IndexTTS only uses the deterministic WFST
    normalizer. The pure-Python implementation is fully compatible for that
    unused optional path and avoids compiling an unrelated C extension.
    """
    try:
        import cdifflib  # noqa: F401
    except ImportError:
        module = types.ModuleType("cdifflib")
        module.CSequenceMatcher = difflib.SequenceMatcher
        sys.modules["cdifflib"] = module
