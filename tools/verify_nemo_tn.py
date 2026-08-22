from __future__ import annotations

import importlib.metadata

from nemo_tn_compat import install_optional_audio_diff_fallback


install_optional_audio_diff_fallback()
from nemo_text_processing.text_normalization.normalize import Normalizer


SAMPLES = {
    "en": "The value is 25% on 12/31/2026.",
    "es": "El valor es 25% el 31/12/2026.",
    "ja": "値は25%で、2026年です。",
    "zh": "数值是25%，日期是2026年。",
    "ar": "القيمة 25٪ في عام 2026.",
}


def main() -> int:
    version = importlib.metadata.version("nemo_text_processing")
    print(f"[nemo-tn] NeMo Text Processing {version}")
    for lang, sample in SAMPLES.items():
        normalizer = Normalizer(
            input_case="cased",
            lang=lang,
            cache_dir=".runtime/nemo-tn-cache",
            overwrite_cache=False,
            post_process=True,
        )
        result = normalizer.normalize(sample, verbose=False)
        if not result or result == sample:
            raise RuntimeError(f"NeMo returned no change for {lang}: {result!r}")
        print(f"[nemo-tn] {lang}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
