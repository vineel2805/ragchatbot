from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class UrlFetchStatus(StrEnum):
    PENDING = "pending"
    FETCHED = "fetched"
    SKIPPED = "skipped"
    FAILED = "failed"
    REJECTED = "rejected"
    ROBOTS_DISALLOWED = "robots_disallowed"
    GONE = "gone"


@dataclass(frozen=True)
class IngestionRun:
    id: str
    source_id: str
    status: RunStatus
    started_at: str
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class UrlRecord:
    source_id: str
    canonical_url: str
    document_id: str
    discovered_from: str | None
    fetch_status: UrlFetchStatus
    http_status: int | None
    etag: str | None
    last_modified: str | None
    content_type: str | None
    size_bytes: int | None
    content_sha256: str | None
    extracted_sha256: str | None
    chunker_version: str | None
    error_type: str | None
    error_message: str | None
    attempt_count: int
    fetched_at: str | None
    last_success_at: str | None
    last_seen_run_id: str | None
    last_touched_run_id: str | None
    consecutive_missing_success_runs: int
    is_in_corpus: bool
    duplicate_of: str | None
    created_at: str
    updated_at: str


class CatalogError(ValueError):
    """Invalid catalog operation or state transition."""
