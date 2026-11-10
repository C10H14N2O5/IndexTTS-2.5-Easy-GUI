"""Request guards: per-request UI language, same-origin writes, security headers.

Implemented as a plain ASGI middleware so streaming (SSE) responses are not
buffered. The same-origin rule matches ``indextts_training/proofread.py``: a
present ``Origin`` must match the request scheme and ``Host``; any
``Sec-Fetch-Site: cross-site`` write is rejected.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from starlette.formparsers import MultiPartException

from . import i18n, state as state_context

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Exact file-content cap is 200 MiB in services.uploads; reserve one MiB for
# multipart headers, and enforce this before Starlette spools the whole request.
MAX_UPLOAD_BODY_BYTES = 201 * 1024 * 1024
BASE_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
]
SPA_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
           "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; "
           "connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'; "
           "base-uri 'none'; form-action 'self'")


def _header(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key == name:
            return value.decode("latin-1")
    return None


def origin_allowed(scope) -> bool:
    origin = _header(scope, b"origin")
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        if parsed.scheme != scope.get("scheme", "http") or parsed.netloc != _header(scope, b"host"):
            return False
    return _header(scope, b"sec-fetch-site") != "cross-site"


def request_language(scope) -> str | None:
    value = _header(scope, b"x-ui-language")
    if value:
        return value
    query = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
    lang = query.get("lang")
    return lang[0] if lang else None


class GuardMiddleware:
    """Sets the request language and rejects cross-origin writes to the API."""

    def __init__(self, app, state=None):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        state_token = state_context.bind(self.state) if self.state is not None else None
        token = i18n.set_current_language(request_language(scope))
        try:
            async def reject(status, code, message):
                body = json.dumps({"error": {"code": code, "message": i18n.t(message), "detail": None}}).encode("utf-8")
                await send({"type": "http.response.start", "status": status,
                            "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store"), *BASE_HEADERS]})
                await send({"type": "http.response.body", "body": body})

            limited_receive = receive
            if path.rstrip("/") == "/api/uploads" and scope.get("method") == "POST":
                length = _header(scope, b"content-length")
                if length is not None:
                    try:
                        too_large = int(length) > MAX_UPLOAD_BODY_BYTES
                    except ValueError:
                        return await reject(400, "invalid", "Invalid Content-Length.")
                    if too_large:
                        return await reject(413, "too_large", "Audio uploads must not exceed 200 MB.")
                received = 0

                async def limited_receive():
                    nonlocal received
                    message = await receive()
                    if message["type"] == "http.request":
                        received += len(message.get("body", b""))
                        if received > MAX_UPLOAD_BODY_BYTES:
                            scope["gui_upload_too_large"] = True
                            # MultipartParser closes all partial spooled files on this
                            # exception, unlike an arbitrary exception from receive().
                            raise MultiPartException("Audio uploads must not exceed 200 MB.")
                    return message

            if (path.startswith("/api/") and scope.get("method", "GET") not in SAFE_METHODS
                    and not origin_allowed(scope)):
                body = json.dumps({"error": {"code": "forbidden", "message": i18n.t("Invalid request origin."),
                                             "detail": None}}).encode("utf-8")
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"application/json"), *BASE_HEADERS]})
                await send({"type": "http.response.body", "body": body})
                return

            async def send_with_headers(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers") or [])
                    present = {key.lower() for key, _ in headers}
                    headers.extend(item for item in BASE_HEADERS if item[0] not in present)
                    if path.startswith("/api/") and b"cache-control" not in present:
                        headers.append((b"cache-control", b"no-store"))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, limited_receive, send_with_headers)
        finally:
            i18n.reset_current_language(token)
            if state_token is not None:
                state_context.reset(state_token)


def resolve_allowed(path: str | Path, roots: list[Path]) -> Path | None:
    """Return the resolved path when it is inside one of ``roots``."""
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return None
    for root in roots:
        try:
            if resolved.is_relative_to(Path(root).resolve()):
                return resolved
        except (OSError, ValueError):
            continue
    return None
