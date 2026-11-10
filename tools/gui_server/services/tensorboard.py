"""Explicit, local-only TensorBoard control; never opens a server-side browser."""
from __future__ import annotations
from ..i18n import raw
from ..schemas import ApiModel
from ..state import current
from .projects import output_dir


class TensorboardResponse(ApiModel):
    url: str
    message: str
    logdir: str
    local_only: bool = True


def start(project: str, adapter: str | None = None, reload: bool = False, port: int = 6006) -> dict:
    folder = output_dir(project, adapter) / "tensorboard"
    folder.mkdir(parents=True, exist_ok=True)
    manager = current().training
    message = manager.start_tensorboard(folder, port=port, force_restart=reload, open_browser=False)
    return dict(url=f"http://127.0.0.1:{port}", message=raw(message), logdir=str(folder), local_only=True)
