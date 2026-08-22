"""Fail early with a useful message when NumPy is shadowed or incomplete."""
from __future__ import annotations

import sys

try:
    import numpy as np
except Exception as exc:
    raise SystemExit(f"NumPy could not be imported: {type(exc).__name__}: {exc}")

if not getattr(np, "__file__", None) or not hasattr(np, "ndarray"):
    raise SystemExit(
        "NumPy is incomplete or shadowed. Run 1- install.bat using the project .venv; "
        f"loaded module={np!r}, file={getattr(np, '__file__', None)!r}."
    )

print(f"NumPy OK: {np.__version__} ({np.__file__})")
print(f"Python: {sys.executable}")
