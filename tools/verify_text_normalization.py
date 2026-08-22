from __future__ import annotations

import platform

from indextts.utils.nemo_tn import normalize_text


def main() -> int:
    sample = "Prueba: 25% y 2026."
    result = normalize_text(sample, "es")
    if not result or not isinstance(result, str) or "25%" in result or "2026" in result:
        raise RuntimeError(f"Spanish text normalization is unavailable or unchanged: {result!r}")
    print(f"[text-normalization] {platform.system()} Spanish normalizer OK: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
