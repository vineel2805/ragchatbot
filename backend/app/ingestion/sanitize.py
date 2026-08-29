from __future__ import annotations

import re

_HEADER_VALUE = re.compile(
    r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie)\s*[:=]\s*\S+(?:\s+\S+)*"
)
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_QUERY_SECRET = re.compile(r"(?i)\b(api[_-]?key|access_token|token|password|secret)=([^&\s]+)")

_MAX_ERROR_CHARS = 180

_UNSAFE_LOG_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "proxy-authorization",
        "headers",
        "body",
        "content",
        "request",
        "response",
        "cookies",
    }
)


def sanitize_error_message(exc: BaseException | str) -> str:
    if isinstance(exc, BaseException):
        prefix = type(exc).__name__
        text = str(exc)
    else:
        prefix = "Error"
        text = exc
    text = _HEADER_VALUE.sub(r"\1=<redacted>", text)
    text = _BEARER.sub("Bearer <redacted>", text)
    text = _QUERY_SECRET.sub(r"\1=<redacted>", text)
    text = text.replace("\n", " ").strip()
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "..."
    return f"{prefix}: {text}" if text else prefix


def safe_log_fields(**fields: object) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in fields.items():
        if key.lower() in _UNSAFE_LOG_KEYS:
            continue
        cleaned[key] = value
    return cleaned
