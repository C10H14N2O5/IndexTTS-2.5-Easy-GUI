"""Error model shared by services and HTTP routes.

Every API error body has the shape::

    {"error": {"code": "invalid", "message": "<translated>", "detail": "<optional>"}}

Status codes: 400 invalid input, 403 forbidden, 404 not found, 409 busy or
conflict, 413 too large, 500 unexpected (the exception type is in ``detail``).
"""
from __future__ import annotations

from typing import Any

from .i18n import Message, backend, msg


class ServiceError(Exception):
    """A user-facing failure. ``template`` is an English i18n key."""

    def __init__(self, template: str | Message, *args, status: int = 400, code: str = "invalid",
                 detail: Any = None, extra: dict | None = None):
        self.message = template if isinstance(template, Message) else msg(template, *args)
        self.status = status
        self.code = code
        self.detail = detail
        self.extra = extra or {}
        super().__init__(str(self.message))


class NotFound(ServiceError):
    def __init__(self, template: str | Message, *args, **kwargs):
        kwargs.setdefault("status", 404)
        kwargs.setdefault("code", "not_found")
        super().__init__(template, *args, **kwargs)


class Busy(ServiceError):
    def __init__(self, template: str | Message, *args, **kwargs):
        kwargs.setdefault("status", 409)
        kwargs.setdefault("code", "busy")
        super().__init__(template, *args, **kwargs)


class Conflict(ServiceError):
    def __init__(self, template: str | Message, *args, **kwargs):
        kwargs.setdefault("status", 409)
        kwargs.setdefault("code", "conflict")
        super().__init__(template, *args, **kwargs)


def error_payload(exc: BaseException, language: str | None = None) -> tuple[int, dict]:
    """Map any exception to (status, JSON body) in the given language."""
    if isinstance(exc, ServiceError):
        detail = exc.detail.render(language) if isinstance(exc.detail, Message) else exc.detail
        body = {"code": exc.code, "message": exc.message.render(language), "detail": detail, **exc.extra}
        return exc.status, {"error": body}
    abx_template = getattr(exc, "template", None)
    if abx_template is not None and hasattr(exc, "args_"):
        # indextts_training.abx.AbxError keeps its template for translation.
        return 400, {"error": {"code": "invalid", "message": Message(abx_template, tuple(exc.args_)).render(language),
                               "detail": None}}
    text = str(exc) or type(exc).__name__
    if isinstance(exc, FileNotFoundError):
        return 404, {"error": {"code": "not_found", "message": backend(text, language), "detail": None}}
    if isinstance(exc, PermissionError):
        return 403, {"error": {"code": "forbidden", "message": backend(text, language), "detail": None}}
    if isinstance(exc, FileExistsError):
        return 409, {"error": {"code": "conflict", "message": backend(text, language), "detail": None}}
    if isinstance(exc, RuntimeError):
        # Training/export managers signal "already running" and activity conflicts this way.
        return 409, {"error": {"code": "busy", "message": backend(text, language), "detail": None}}
    if isinstance(exc, (ValueError, KeyError)):
        return 400, {"error": {"code": "invalid", "message": backend(text, language), "detail": type(exc).__name__}}
    return 500, {"error": {"code": "internal", "message": backend(text, language), "detail": type(exc).__name__}}
