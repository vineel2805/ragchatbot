from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ingestion.catalog_models import (
    CatalogError,
    IngestionRun,
    RunStatus,
    UrlFetchStatus,
    UrlRecord,
)
from app.ingestion.ids import make_document_id
from app.ingestion.registry import get_source
from app.ingestion.sanitize import sanitize_error_message
from app.ingestion.url_security import CanonicalizationError, canonicalize_url

SCHEMA_VERSION = 2
SOFT_DELETE_THRESHOLD = 2
DEFAULT_CATALOG_PATH = Path("data/catalog/url_catalog.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_type TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS catalog_urls (
    source_id TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    document_id TEXT NOT NULL,
    discovered_from TEXT,
    fetch_status TEXT NOT NULL DEFAULT 'pending',
    http_status INTEGER,
    etag TEXT,
    last_modified TEXT,
    content_type TEXT,
    size_bytes INTEGER,
    content_sha256 TEXT,
    extracted_sha256 TEXT,
    chunker_version TEXT,
    indexed_sha256 TEXT,
    indexed_chunker_version TEXT,
    error_type TEXT,
    error_message TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT,
    last_success_at TEXT,
    last_seen_run_id TEXT,
    last_touched_run_id TEXT,
    consecutive_missing_success_runs INTEGER NOT NULL DEFAULT 0,
    is_in_corpus INTEGER NOT NULL DEFAULT 1,
    duplicate_of TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source_id, canonical_url)
);

CREATE INDEX IF NOT EXISTS idx_catalog_urls_document
    ON catalog_urls (document_id);
CREATE INDEX IF NOT EXISTS idx_catalog_urls_source_status
    ON catalog_urls (source_id, fetch_status);
CREATE INDEX IF NOT EXISTS idx_catalog_urls_source_corpus
    ON catalog_urls (source_id, is_in_corpus);
CREATE INDEX IF NOT EXISTS idx_catalog_urls_source_seen
    ON catalog_urls (source_id, last_seen_run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source_status
    ON ingestion_runs (source_id, status);
"""


class IngestionCatalog:
    """SQLite crawl/state catalog. Does not store HTML or secrets."""

    def __init__(
        self,
        path: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
        new_id: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._new_id = new_id or (lambda: str(uuid4()))
        self._txn_depth = 0
        self._conn = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._initialize()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IngestionCatalog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._txn_depth == 0:
            self._conn.execute("BEGIN IMMEDIATE")
        self._txn_depth += 1
        try:
            yield
            if self._txn_depth == 1:
                self._conn.commit()
        except Exception:
            if self._txn_depth == 1:
                self._conn.rollback()
            raise
        finally:
            self._txn_depth -= 1

    def create_run(self, source_id: str) -> IngestionRun:
        run_id = self._new_id()
        started = self._stamp()

        def write() -> IngestionRun:
            self._conn.execute(
                """
                INSERT INTO ingestion_runs (id, source_id, status, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, source_id, RunStatus.STARTED.value, started),
            )
            return self._get_run(run_id)

        return self._write(write)

    def finish_run(
        self,
        run_id: str,
        *,
        succeeded: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> IngestionRun:
        def write() -> IngestionRun:
            run = self._get_run(run_id)
            if run.status is not RunStatus.STARTED:
                raise CatalogError(f"cannot finish run {run_id} from status {run.status}")
            status = RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED
            message = sanitize_error_message(error_message) if error_message else None
            self._conn.execute(
                """
                UPDATE ingestion_runs
                SET status = ?, finished_at = ?, error_type = ?, error_message = ?
                WHERE id = ?
                """,
                (status.value, self._stamp(), error_type, message, run_id),
            )
            return self._get_run(run_id)

        return self._write(write)

    def get_run(self, run_id: str) -> IngestionRun | None:
        row = self._conn.execute(
            "SELECT * FROM ingestion_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _run_from_row(row) if row else None

    def register_url(
        self,
        source_id: str,
        url: str,
        *,
        run_id: str | None = None,
        discovered_from: str | None = None,
    ) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)
        document_id = make_document_id(source_id, canonical)
        now = self._stamp()

        def write() -> UrlRecord:
            self._conn.execute(
                """
                INSERT INTO catalog_urls (
                    source_id, canonical_url, document_id, discovered_from,
                    fetch_status, last_seen_run_id, last_touched_run_id,
                    consecutive_missing_success_runs, is_in_corpus,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                ON CONFLICT(source_id, canonical_url) DO UPDATE SET
                    last_seen_run_id = excluded.last_seen_run_id,
                    last_touched_run_id = excluded.last_touched_run_id,
                    consecutive_missing_success_runs = 0,
                    is_in_corpus = 1,
                    discovered_from = COALESCE(catalog_urls.discovered_from, excluded.discovered_from),
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    canonical,
                    document_id,
                    discovered_from,
                    UrlFetchStatus.PENDING.value,
                    run_id,
                    run_id,
                    now,
                    now,
                ),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def get_url(self, source_id: str, url: str) -> UrlRecord | None:
        canonical = _canonical_for_source(source_id, url)
        row = self._conn.execute(
            """
            SELECT * FROM catalog_urls
            WHERE source_id = ? AND canonical_url = ?
            """,
            (source_id, canonical),
        ).fetchone()
        return _url_from_row(row) if row else None

    def mark_fetch_started(self, source_id: str, url: str, *, run_id: str | None = None) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET attempt_count = attempt_count + 1,
                    last_touched_run_id = ?,
                    fetched_at = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (run_id, self._stamp(), self._stamp(), source_id, canonical),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def mark_fetch_succeeded(
        self,
        source_id: str,
        url: str,
        *,
        run_id: str | None = None,
        http_status: int | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        content_type: str | None = None,
        size_bytes: int | None = None,
        content_sha256: str | None = None,
    ) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)
        now = self._stamp()

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET fetch_status = ?,
                    http_status = ?,
                    etag = ?,
                    last_modified = ?,
                    content_type = ?,
                    size_bytes = ?,
                    content_sha256 = ?,
                    error_type = NULL,
                    error_message = NULL,
                    last_success_at = ?,
                    fetched_at = ?,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (
                    UrlFetchStatus.FETCHED.value,
                    http_status,
                    etag,
                    last_modified,
                    content_type,
                    size_bytes,
                    content_sha256,
                    now,
                    now,
                    run_id,
                    now,
                    source_id,
                    canonical,
                ),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def mark_fetch_failed(
        self,
        source_id: str,
        url: str,
        *,
        run_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        http_status: int | None = None,
        fetch_status: UrlFetchStatus = UrlFetchStatus.FAILED,
    ) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)
        now = self._stamp()
        message = sanitize_error_message(error_message) if error_message else None

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET fetch_status = ?,
                    http_status = COALESCE(?, http_status),
                    error_type = ?,
                    error_message = ?,
                    fetched_at = ?,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (
                    fetch_status.value,
                    http_status,
                    error_type,
                    message,
                    now,
                    run_id,
                    now,
                    source_id,
                    canonical,
                ),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def needs_processing(
        self,
        source_id: str,
        url: str,
        *,
        extracted_sha256: str,
        chunker_version: str,
    ) -> bool:
        record = self.get_url(source_id, url)
        if record is None:
            return True
        if not record.extracted_sha256 or not record.chunker_version:
            return True
        return not (
            record.extracted_sha256 == extracted_sha256
            and record.chunker_version == chunker_version
        )

    def needs_indexing(
        self,
        source_id: str,
        url: str,
        *,
        extracted_sha256: str,
        chunker_version: str,
    ) -> bool:
        """Return True when the URL's Qdrant representation is stale or absent.

        A URL needs (re-)indexing when:
        - it has never been indexed (indexed_sha256 is NULL), OR
        - the extracted content has changed (new extracted_sha256), OR
        - the chunker version has changed (new chunker_version).
        """
        record = self.get_url(source_id, url)
        if record is None:
            return True
        if not record.indexed_sha256 or not record.indexed_chunker_version:
            return True
        return not (
            record.indexed_sha256 == extracted_sha256
            and record.indexed_chunker_version == chunker_version
        )

    def record_indexing(
        self,
        source_id: str,
        url: str,
        *,
        extracted_sha256: str,
        chunker_version: str,
        run_id: str | None = None,
    ) -> UrlRecord:
        """Record that this URL's chunks have been successfully upserted to Qdrant."""
        canonical = _canonical_for_source(source_id, url)
        now = self._stamp()

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET indexed_sha256 = ?,
                    indexed_chunker_version = ?,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (extracted_sha256, chunker_version, run_id, now, source_id, canonical),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def record_extraction(
        self,
        source_id: str,
        url: str,
        *,
        extracted_sha256: str,
        chunker_version: str,
        duplicate_of: str | None = None,
        run_id: str | None = None,
    ) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)
        now = self._stamp()

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET extracted_sha256 = ?,
                    chunker_version = ?,
                    duplicate_of = ?,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (
                    extracted_sha256,
                    chunker_version,
                    duplicate_of,
                    run_id,
                    now,
                    source_id,
                    canonical,
                ),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def record_missing_observation(
        self,
        source_id: str,
        url: str,
        *,
        run_id: str | None = None,
    ) -> UrlRecord:
        canonical = _canonical_for_source(source_id, url)
        now = self._stamp()

        def write() -> UrlRecord:
            self._require_url(source_id, canonical)
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET consecutive_missing_success_runs = consecutive_missing_success_runs + 1,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ? AND canonical_url = ?
                """,
                (run_id, now, source_id, canonical),
            )
            return self._get_url(source_id, canonical)

        return self._write(write)

    def record_missing_for_unseen(
        self,
        source_id: str,
        run_id: str,
    ) -> list[UrlRecord]:
        """Increment absence for in-corpus URLs not discovered in this successful run."""

        def write() -> list[UrlRecord]:
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET consecutive_missing_success_runs = consecutive_missing_success_runs + 1,
                    last_touched_run_id = ?,
                    updated_at = ?
                WHERE source_id = ?
                  AND is_in_corpus = 1
                  AND (last_seen_run_id IS NULL OR last_seen_run_id != ?)
                """,
                (run_id, self._stamp(), source_id, run_id),
            )
            rows = self._conn.execute(
                """
                SELECT * FROM catalog_urls
                WHERE source_id = ?
                  AND is_in_corpus = 1
                  AND consecutive_missing_success_runs > 0
                """,
                (source_id,),
            ).fetchall()
            return [_url_from_row(row) for row in rows]

        return self._write(write)

    def apply_soft_deletes(
        self,
        source_id: str,
        *,
        threshold: int = SOFT_DELETE_THRESHOLD,
    ) -> list[UrlRecord]:
        now = self._stamp()

        def write() -> list[UrlRecord]:
            self._conn.execute(
                """
                UPDATE catalog_urls
                SET is_in_corpus = 0,
                    updated_at = ?
                WHERE source_id = ?
                  AND is_in_corpus = 1
                  AND consecutive_missing_success_runs >= ?
                """,
                (now, source_id, threshold),
            )
            rows = self._conn.execute(
                """
                SELECT * FROM catalog_urls
                WHERE source_id = ?
                  AND is_in_corpus = 0
                  AND consecutive_missing_success_runs >= ?
                """,
                (source_id, threshold),
            ).fetchall()
            return [_url_from_row(row) for row in rows]

        return self._write(write)

    def list_document_ids(self, source_id: str) -> list[str]:
        """Return all distinct document_ids for *source_id* (including out-of-corpus entries)."""
        rows = self._conn.execute(
            "SELECT DISTINCT document_id FROM catalog_urls WHERE source_id = ?",
            (source_id,),
        ).fetchall()
        return [row["document_id"] for row in rows]

    def _initialize(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute("SELECT version FROM schema_meta").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_meta (version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] < SCHEMA_VERSION:
            self._migrate(row["version"])

    def _migrate(self, from_version: int) -> None:
        """Idempotent forward migration from *from_version* to SCHEMA_VERSION."""
        if from_version < 2:
            # Add Qdrant indexing state columns (v1 → v2).
            for stmt in (
                "ALTER TABLE catalog_urls ADD COLUMN indexed_sha256 TEXT",
                "ALTER TABLE catalog_urls ADD COLUMN indexed_chunker_version TEXT",
            ):
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # Column already exists — migration is idempotent.
            self._conn.execute("UPDATE schema_meta SET version = 2")

    def _stamp(self) -> str:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _write(self, fn: Callable[[], UrlRecord] | Callable[[], IngestionRun] | Callable[[], list[UrlRecord]]):
        if self._txn_depth:
            return fn()
        with self.transaction():
            return fn()

    def _get_run(self, run_id: str) -> IngestionRun:
        run = self.get_run(run_id)
        if run is None:
            raise CatalogError(f"unknown run {run_id}")
        return run

    def _get_url(self, source_id: str, canonical_url: str) -> UrlRecord:
        row = self._conn.execute(
            """
            SELECT * FROM catalog_urls
            WHERE source_id = ? AND canonical_url = ?
            """,
            (source_id, canonical_url),
        ).fetchone()
        if row is None:
            raise CatalogError(f"unknown URL {source_id} {canonical_url}")
        return _url_from_row(row)

    def _require_url(self, source_id: str, canonical_url: str) -> None:
        self._get_url(source_id, canonical_url)


def _canonical_for_source(source_id: str, url: str) -> str:
    keep_query = False
    try:
        keep_query = get_source(source_id).keep_query_strings
    except KeyError:
        pass
    try:
        return canonicalize_url(url, keep_query_strings=keep_query)
    except CanonicalizationError as exc:
        raise CatalogError(str(exc)) from exc


def _run_from_row(row: sqlite3.Row) -> IngestionRun:
    return IngestionRun(
        id=row["id"],
        source_id=row["source_id"],
        status=RunStatus(row["status"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_type=row["error_type"],
        error_message=row["error_message"],
    )


def _url_from_row(row: sqlite3.Row) -> UrlRecord:
    return UrlRecord(
        source_id=row["source_id"],
        canonical_url=row["canonical_url"],
        document_id=row["document_id"],
        discovered_from=row["discovered_from"],
        fetch_status=UrlFetchStatus(row["fetch_status"]),
        http_status=row["http_status"],
        etag=row["etag"],
        last_modified=row["last_modified"],
        content_type=row["content_type"],
        size_bytes=row["size_bytes"],
        content_sha256=row["content_sha256"],
        extracted_sha256=row["extracted_sha256"],
        chunker_version=row["chunker_version"],
        indexed_sha256=row["indexed_sha256"],
        indexed_chunker_version=row["indexed_chunker_version"],
        error_type=row["error_type"],
        error_message=row["error_message"],
        attempt_count=row["attempt_count"],
        fetched_at=row["fetched_at"],
        last_success_at=row["last_success_at"],
        last_seen_run_id=row["last_seen_run_id"],
        last_touched_run_id=row["last_touched_run_id"],
        consecutive_missing_success_runs=row["consecutive_missing_success_runs"],
        is_in_corpus=bool(row["is_in_corpus"]),
        duplicate_of=row["duplicate_of"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
