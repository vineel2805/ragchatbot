from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.catalog import IngestionCatalog, SOFT_DELETE_THRESHOLD
from app.ingestion.catalog_models import CatalogError, RunStatus, UrlFetchStatus
from app.ingestion.ids import CHUNKER_VERSION, make_document_id

FASTAPI_URL = "https://fastapi.tiangolo.com/tutorial/first-steps/"
FASTAPI_CANONICAL = "https://fastapi.tiangolo.com/tutorial/first-steps"
PYTHON_URL = "https://docs.python.org/3/tutorial/index.html"
STAMP = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "url_catalog.sqlite"
        self.ids = iter([f"run-{i}" for i in range(1, 50)])
        self.catalog = IngestionCatalog(
            self.path,
            now=lambda: STAMP,
            new_id=lambda: next(self.ids),
        )

    def tearDown(self) -> None:
        self.catalog.close()
        self._tmp.cleanup()

    def test_database_initialization(self) -> None:
        self.assertTrue(self.path.exists())
        row = self.catalog._conn.execute("SELECT version FROM schema_meta").fetchone()
        self.assertEqual(row["version"], 2)

    def test_schema_creation(self) -> None:
        names = {
            row[0]
            for row in self.catalog._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        self.assertIn("ingestion_runs", names)
        self.assertIn("catalog_urls", names)
        columns = {
            row[1]
            for row in self.catalog._conn.execute("PRAGMA table_info(catalog_urls)")
        }
        self.assertNotIn("html", columns)
        self.assertNotIn("body", columns)
        self.assertIn("etag", columns)
        self.assertIn("last_modified", columns)
        self.assertIn("extracted_sha256", columns)
        self.assertIn("chunker_version", columns)
        self.assertIn("indexed_sha256", columns)
        self.assertIn("indexed_chunker_version", columns)

    def test_unique_source_and_canonical_url(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        with self.assertRaises(sqlite3.IntegrityError):
            self.catalog._conn.execute(
                """
                INSERT INTO catalog_urls (
                    source_id, canonical_url, document_id, fetch_status,
                    consecutive_missing_success_runs, is_in_corpus,
                    created_at, updated_at
                ) VALUES ('fastapi', ?, 'x', 'pending', 0, 1, 't', 't')
                """,
                (FASTAPI_CANONICAL,),
            )

    def test_canonical_url_persistence(self) -> None:
        record = self.catalog.register_url(
            "fastapi",
            "https://FASTAPI.TIANGOLO.COM/tutorial/first-steps/?utm=1#frag",
        )
        self.assertEqual(record.canonical_url, FASTAPI_CANONICAL)
        self.assertEqual(
            record.document_id,
            make_document_id("fastapi", FASTAPI_CANONICAL),
        )

    def test_ingestion_run_lifecycle(self) -> None:
        run = self.catalog.create_run("fastapi")
        self.assertEqual(run.id, "run-1")
        self.assertEqual(run.status, RunStatus.STARTED)
        self.assertEqual(run.started_at, STAMP.isoformat())
        self.assertIsNone(run.finished_at)
        finished = self.catalog.finish_run(run.id, succeeded=True)
        self.assertEqual(finished.status, RunStatus.SUCCEEDED)
        self.assertEqual(finished.finished_at, STAMP.isoformat())
        with self.assertRaises(CatalogError):
            self.catalog.finish_run(run.id, succeeded=True)

    def test_url_registration(self) -> None:
        run = self.catalog.create_run("fastapi")
        record = self.catalog.register_url(
            "fastapi",
            FASTAPI_URL,
            run_id=run.id,
            discovered_from="sitemap",
        )
        self.assertEqual(record.fetch_status, UrlFetchStatus.PENDING)
        self.assertEqual(record.last_seen_run_id, run.id)
        self.assertEqual(record.discovered_from, "sitemap")
        self.assertTrue(record.is_in_corpus)

    def test_duplicate_discovery(self) -> None:
        first = self.catalog.register_url(
            "fastapi", FASTAPI_URL, run_id="run-a", discovered_from="seed"
        )
        second = self.catalog.register_url(
            "fastapi", FASTAPI_URL, run_id="run-b", discovered_from="other"
        )
        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(second.last_seen_run_id, "run-b")
        self.assertEqual(second.discovered_from, "seed")
        count = self.catalog._conn.execute(
            "SELECT COUNT(*) FROM catalog_urls WHERE source_id = 'fastapi'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_fetch_success(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        started = self.catalog.mark_fetch_started("fastapi", FASTAPI_URL, run_id="r")
        self.assertEqual(started.attempt_count, 1)
        record = self.catalog.mark_fetch_succeeded(
            "fastapi",
            FASTAPI_URL,
            run_id="r",
            http_status=200,
            etag='"abc"',
            last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
            content_type="text/html",
            size_bytes=12,
            content_sha256="aa" * 32,
        )
        self.assertEqual(record.fetch_status, UrlFetchStatus.FETCHED)
        self.assertEqual(record.http_status, 200)
        self.assertEqual(record.content_sha256, "aa" * 32)
        self.assertEqual(record.last_success_at, STAMP.isoformat())

    def test_fetch_failure(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.mark_fetch_started("fastapi", FASTAPI_URL)
        record = self.catalog.mark_fetch_failed(
            "fastapi",
            FASTAPI_URL,
            error_type="timeout",
            error_message="read timed out Authorization: Bearer secret-token",
            http_status=None,
        )
        self.assertEqual(record.fetch_status, UrlFetchStatus.FAILED)
        self.assertEqual(record.error_type, "timeout")
        self.assertNotIn("secret-token", record.error_message or "")

    def test_retry_failure_metadata(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.mark_fetch_started("fastapi", FASTAPI_URL)
        self.catalog.mark_fetch_failed("fastapi", FASTAPI_URL, error_type="timeout")
        self.catalog.mark_fetch_started("fastapi", FASTAPI_URL)
        record = self.catalog.mark_fetch_failed("fastapi", FASTAPI_URL, error_type="timeout")
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(record.error_type, "timeout")

    def test_etag_last_modified_persistence(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.mark_fetch_succeeded(
            "fastapi",
            FASTAPI_URL,
            etag='"v1"',
            last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
            content_sha256="bb" * 32,
            http_status=200,
        )
        failed = self.catalog.mark_fetch_failed(
            "fastapi",
            FASTAPI_URL,
            error_type="connection_failure",
        )
        self.assertEqual(failed.etag, '"v1"')
        self.assertEqual(failed.last_modified, "Wed, 01 Jan 2026 00:00:00 GMT")
        self.assertEqual(failed.content_sha256, "bb" * 32)
        self.assertEqual(failed.fetch_status, UrlFetchStatus.FAILED)

    def test_unchanged_detection(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.record_extraction(
            "fastapi",
            FASTAPI_URL,
            extracted_sha256="cc" * 32,
            chunker_version=CHUNKER_VERSION,
        )
        self.assertFalse(
            self.catalog.needs_processing(
                "fastapi",
                FASTAPI_URL,
                extracted_sha256="cc" * 32,
                chunker_version=CHUNKER_VERSION,
            )
        )

    def test_changed_content_detection(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.record_extraction(
            "fastapi",
            FASTAPI_URL,
            extracted_sha256="cc" * 32,
            chunker_version=CHUNKER_VERSION,
        )
        self.assertTrue(
            self.catalog.needs_processing(
                "fastapi",
                FASTAPI_URL,
                extracted_sha256="dd" * 32,
                chunker_version=CHUNKER_VERSION,
            )
        )

    def test_chunker_version_change_detection(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.record_extraction(
            "fastapi",
            FASTAPI_URL,
            extracted_sha256="cc" * 32,
            chunker_version=CHUNKER_VERSION,
        )
        self.assertTrue(
            self.catalog.needs_processing(
                "fastapi",
                FASTAPI_URL,
                extracted_sha256="cc" * 32,
                chunker_version="heading-v2",
            )
        )

    def test_missing_url_observations(self) -> None:
        run = self.catalog.create_run("fastapi")
        self.catalog.register_url("fastapi", FASTAPI_URL, run_id=run.id)
        missing = self.catalog.record_missing_observation(
            "fastapi", FASTAPI_URL, run_id=run.id
        )
        self.assertEqual(missing.consecutive_missing_success_runs, 1)
        self.assertTrue(missing.is_in_corpus)

    def test_soft_delete_threshold_behavior(self) -> None:
        run = self.catalog.create_run("fastapi")
        self.catalog.register_url("fastapi", FASTAPI_URL, run_id=run.id)
        self.catalog.record_missing_observation("fastapi", FASTAPI_URL)
        still = self.catalog.apply_soft_deletes("fastapi", threshold=SOFT_DELETE_THRESHOLD)
        self.assertEqual(still, [])
        self.assertTrue(self.catalog.get_url("fastapi", FASTAPI_URL).is_in_corpus)
        self.catalog.record_missing_observation("fastapi", FASTAPI_URL)
        deleted = self.catalog.apply_soft_deletes("fastapi")
        self.assertEqual(len(deleted), 1)
        self.assertFalse(deleted[0].is_in_corpus)
        rediscovered = self.catalog.register_url("fastapi", FASTAPI_URL, run_id="run-new")
        self.assertTrue(rediscovered.is_in_corpus)
        self.assertEqual(rediscovered.consecutive_missing_success_runs, 0)

    def test_transaction_rollback(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.catalog.transaction():
                self.catalog.register_url("fastapi", FASTAPI_URL)
                raise RuntimeError("boom")
        self.assertIsNone(self.catalog.get_url("fastapi", FASTAPI_URL))

    def test_persistence_after_reopening_the_database(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.record_extraction(
            "fastapi",
            FASTAPI_URL,
            extracted_sha256="ee" * 32,
            chunker_version=CHUNKER_VERSION,
        )
        self.catalog.close()
        reopened = IngestionCatalog(self.path)
        try:
            record = reopened.get_url("fastapi", FASTAPI_URL)
            self.assertIsNotNone(record)
            self.assertEqual(record.extracted_sha256, "ee" * 32)
            self.assertEqual(record.chunker_version, CHUNKER_VERSION)
        finally:
            reopened.close()

    def test_source_isolation(self) -> None:
        self.catalog.register_url("fastapi", FASTAPI_URL)
        self.catalog.register_url("python", PYTHON_URL)
        self.assertIsNone(self.catalog.get_url("python", FASTAPI_URL))
        self.assertIsNotNone(self.catalog.get_url("fastapi", FASTAPI_URL))
        self.assertIsNotNone(self.catalog.get_url("python", PYTHON_URL))

    def test_deterministic_timestamps_and_ids(self) -> None:
        run = self.catalog.create_run("fastapi")
        record = self.catalog.register_url("fastapi", FASTAPI_URL, run_id=run.id)
        self.assertEqual(run.id, "run-1")
        self.assertEqual(run.started_at, STAMP.isoformat())
        self.assertEqual(record.created_at, STAMP.isoformat())
        self.assertEqual(record.updated_at, STAMP.isoformat())

    def test_failed_run_does_not_soft_delete(self) -> None:
        started = self.catalog.create_run("fastapi")
        self.catalog.register_url("fastapi", FASTAPI_URL, run_id=started.id)
        self.catalog.finish_run(started.id, succeeded=False, error_type="failed")
        unseen = self.catalog.record_missing_for_unseen("fastapi", "run-other")
        self.assertEqual(unseen[0].consecutive_missing_success_runs, 1)
        self.assertTrue(unseen[0].is_in_corpus)


class IndexingCatalogTests(unittest.TestCase):
    """Tests for needs_indexing, record_indexing, list_document_ids, and v1->v2 migration."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "url_catalog.sqlite"
        self.ids = iter([f"run-{i}" for i in range(1, 50)])
        self.catalog = IngestionCatalog(
            self.path,
            now=lambda: STAMP,
            new_id=lambda: next(self.ids),
        )
        run = self.catalog.create_run("fastapi")
        self.run_id = run.id
        self.catalog.register_url("fastapi", FASTAPI_URL, run_id=self.run_id)
        self.catalog.mark_fetch_started("fastapi", FASTAPI_URL, run_id=self.run_id)
        self.catalog.mark_fetch_succeeded("fastapi", FASTAPI_URL, run_id=self.run_id, http_status=200)
        self.catalog.record_extraction(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-abc",
            chunker_version=CHUNKER_VERSION,
            run_id=self.run_id,
        )

    def tearDown(self) -> None:
        self.catalog.close()
        self._tmp.cleanup()

    # --- needs_indexing --------------------------------------------------

    def test_needs_indexing_true_initially(self) -> None:
        self.assertTrue(
            self.catalog.needs_indexing(
                "fastapi", FASTAPI_URL,
                extracted_sha256="sha-abc",
                chunker_version=CHUNKER_VERSION,
            )
        )

    def test_needs_indexing_false_after_record_indexing(self) -> None:
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-abc",
            chunker_version=CHUNKER_VERSION,
            run_id=self.run_id,
        )
        self.assertFalse(
            self.catalog.needs_indexing(
                "fastapi", FASTAPI_URL,
                extracted_sha256="sha-abc",
                chunker_version=CHUNKER_VERSION,
            )
        )

    def test_needs_indexing_true_after_sha256_change(self) -> None:
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-abc",
            chunker_version=CHUNKER_VERSION,
            run_id=self.run_id,
        )
        # New content sha256.
        self.assertTrue(
            self.catalog.needs_indexing(
                "fastapi", FASTAPI_URL,
                extracted_sha256="sha-xyz",
                chunker_version=CHUNKER_VERSION,
            )
        )

    def test_needs_indexing_true_after_chunker_version_change(self) -> None:
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-abc",
            chunker_version="heading-v1",
            run_id=self.run_id,
        )
        # New chunker version.
        self.assertTrue(
            self.catalog.needs_indexing(
                "fastapi", FASTAPI_URL,
                extracted_sha256="sha-abc",
                chunker_version="heading-v2",
            )
        )

    # --- record_indexing -------------------------------------------------

    def test_record_indexing_persists_fields(self) -> None:
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-abc",
            chunker_version=CHUNKER_VERSION,
            run_id=self.run_id,
        )
        rec = self.catalog.get_url("fastapi", FASTAPI_URL)
        self.assertEqual(rec.indexed_sha256, "sha-abc")
        self.assertEqual(rec.indexed_chunker_version, CHUNKER_VERSION)

    def test_record_indexing_updates_existing_values(self) -> None:
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-v1",
            chunker_version="heading-v1",
            run_id=self.run_id,
        )
        self.catalog.record_indexing(
            "fastapi", FASTAPI_URL,
            extracted_sha256="sha-v2",
            chunker_version="heading-v2",
            run_id=self.run_id,
        )
        rec = self.catalog.get_url("fastapi", FASTAPI_URL)
        self.assertEqual(rec.indexed_sha256, "sha-v2")
        self.assertEqual(rec.indexed_chunker_version, "heading-v2")

    # --- list_document_ids -----------------------------------------------

    def test_list_document_ids_returns_registered_ids(self) -> None:
        doc_id = make_document_id("fastapi", FASTAPI_CANONICAL)
        ids = self.catalog.list_document_ids("fastapi")
        self.assertIn(doc_id, ids)

    def test_list_document_ids_empty_for_unknown_source(self) -> None:
        ids = self.catalog.list_document_ids("unknown-source")
        self.assertEqual(ids, [])

    def test_list_document_ids_does_not_include_other_source(self) -> None:
        run2 = self.catalog.create_run("python")
        self.catalog.register_url("python", PYTHON_URL, run_id=run2.id)
        fastapi_ids = self.catalog.list_document_ids("fastapi")
        python_ids = self.catalog.list_document_ids("python")
        # Use the catalog's canonical form rather than computing from raw PYTHON_URL.
        python_rec = self.catalog.get_url("python", PYTHON_URL)
        python_doc_id = python_rec.document_id
        self.assertNotIn(python_doc_id, fastapi_ids)
        self.assertIn(python_doc_id, python_ids)

    # --- v1 -> v2 migration ----------------------------------------------

    def test_migration_from_v1_adds_indexing_columns(self) -> None:
        """Create a v1 schema DB manually, then open it via IngestionCatalog and verify migration."""
        import sqlite3 as _sqlite3

        v1_path = Path(self._tmp.name) / "v1_catalog.sqlite"
        # Build a minimal v1 schema without the new columns.
        conn = _sqlite3.connect(str(v1_path))
        conn.execute(
            """
            CREATE TABLE schema_meta (version INTEGER NOT NULL);
            """
        )
        conn.execute("INSERT INTO schema_meta (version) VALUES (1)")
        conn.execute(
            """
            CREATE TABLE catalog_urls (
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
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error_type TEXT,
                error_message TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        # Open with IngestionCatalog — migration should run automatically.
        cat = IngestionCatalog(v1_path)
        try:
            # Version should now be 2.
            row = cat._conn.execute("SELECT version FROM schema_meta").fetchone()
            self.assertEqual(row["version"], 2)
            # New columns must exist.
            cols = {
                r[1] for r in cat._conn.execute("PRAGMA table_info(catalog_urls)")
            }
            self.assertIn("indexed_sha256", cols)
            self.assertIn("indexed_chunker_version", cols)
        finally:
            cat.close()


if __name__ == "__main__":
    unittest.main()
