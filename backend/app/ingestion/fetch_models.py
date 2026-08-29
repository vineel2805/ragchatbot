from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FetchStatus(StrEnum):
    FETCHED = "fetched"
    SKIPPED = "skipped"
    FAILED = "failed"
    REJECTED = "rejected"
    ROBOTS_DISALLOWED = "robots_disallowed"
    GONE = "gone"


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a documentation fetch. `body` is set only on success.

    `error_message` is sanitized and must not contain response bodies or secrets.
    """

    ok: bool
    status: FetchStatus
    reason: str
    requested_url: str
    canonical_url: str | None = None
    final_url: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    body: bytes | None = None
    etag: str | None = None
    last_modified: str | None = None
    size_bytes: int | None = None
    attempts: int = 0
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class HttpExchange:
    """One completed HTTP round-trip (or a transport-level failure)."""

    status_code: int | None
    headers: dict[str, str]
    content: bytes | None
    error_type: str | None = None
    error_message: str | None = None
    size_exceeded: bool = False

    @property
    def completed(self) -> bool:
        return self.error_type is None and not self.size_exceeded and self.status_code is not None
