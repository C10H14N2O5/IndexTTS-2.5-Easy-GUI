from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

from nemo_tn_compat import install_optional_audio_diff_fallback


install_optional_audio_diff_fallback()
from nemo_text_processing.text_normalization.normalize import Normalizer


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("INDEXTTS_NEMO_TN_CACHE", ROOT / ".runtime" / "nemo-tn-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_NORMALIZERS = {}


def _get_normalizer(lang: str, input_case: str):
    key = (lang, input_case)
    if key not in _NORMALIZERS:
        # NeMo and some of its dependencies can write diagnostics to stdout
        # while compiling a grammar. Keep stdout reserved for JSON-RPC.
        with contextlib.redirect_stdout(sys.stderr):
            _NORMALIZERS[key] = Normalizer(
                input_case=input_case,
                lang=lang,
                cache_dir=str(CACHE_DIR),
                overwrite_cache=False,
                post_process=True,
            )
    return _NORMALIZERS[key]


def _response(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "ping":
                _response({"ok": True, "backend": "nemo_text_processing"})
            elif operation == "normalize":
                text = request.get("text")
                lang = str(request.get("lang") or "")
                input_case = str(request.get("input_case") or "cased")
                result = _get_normalizer(lang, input_case).normalize(
                    str(text or ""),
                    verbose=False,
                )
                _response({"ok": True, "text": result, "lang": lang})
            elif operation == "shutdown":
                _response({"ok": True})
                return 0
            else:
                _response({"ok": False, "error": f"unknown operation: {operation}"})
        except Exception as exc:
            _response({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
