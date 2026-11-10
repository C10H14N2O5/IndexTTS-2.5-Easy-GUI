"""Print the OpenAPI document (used by ``pnpm gen:api`` in frontend/)."""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    os.environ.setdefault("INDEXTTS_FAKE_ENGINE", "1")
    from .app import create_app
    app = create_app(fake=True, background=False)
    sys.stdout.write(json.dumps(app.openapi(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
