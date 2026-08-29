"""Tests for the embedding + Qdrant indexing layer.

All tests use FakeEmbedder and FakeVectorStore — no qdrant_client install or
model download is required.  The IngestionCatalog runs in SQLite :memory:.
"""
from __future__ import annotations

import hashlib
import unittest
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.ingestion.catalog import IngestionCatalog
from app.ingestion.catalog_models import UrlFetchStatus
from app.ingestion.document_models import DocumentChunk
from app.ingestion.embedder import EMBEDDING_DIM, EMBEDDING_MODEL
from app.ingestion.ids import CHUNKER_VERSION, make_chunk_id, make_document_id, make_point_id
from app.ingestion.indexer import (
    COLLECTION_NAME,
    EMBED_BATCH,
    IndexDocumentResult,
    PointData,
    QdrantIndexer,
)
from app.ingestion.registry import get_source


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic fake embedder.  Never loads sentence-transformers."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []  # record each batch for inspection

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        result = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            # Deterministic pseudo-random unit vector (not truly uniform but good enough).
            raw = [float((seed >> i) & 0xFF) / 255.0 - 0.5 for i in range(self.dim)]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            result.append([x / norm for x in raw])
        return result


class ErrEmbedder:
    """Embedder that always raises."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed failure injected")


class PartialUpsertStore:
    """Vector store that raises on the second upsert batch."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.points: dict[str, dict[str, PointData]] = {}
        self._upsert_calls = 0

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vector_size: int, distance: str) -> None:
        self.collections[name] = {"vector_size": vector_size, "distance": distance}
        self.points[name] = {}

    def collection_vector_size(self, name: str) -> int:
        return self.collections[name]["vector_size"]

    def upsert(self, name: str, points: list[PointData]) -> None:
        self._upsert_calls += 1
        if self._upsert_calls >= 2:
            raise RuntimeError("partial batch failure injected")
        for p in points:
            self.points[name][p.id] = p

    def scroll_document(self, name: str, document_id: str) -> list[str]:
        return [
            pid
            for pid, p in self.points.get(name, {}).items()
            if p.payload.get("document_id") == document_id
        ]

    def set_is_active(self, name: str, point_ids: list[str], is_active: bool) -> None:
        for pid in point_ids:
            if pid in self.points.get(name, {}):
                self.points[name][pid].payload["is_active"] = is_active


@dataclass
class FakeVectorStore:
    """In-memory VectorStoreClient.  No qdrant_client dependency."""

    collections: dict[str, dict] = field(default_factory=dict)
    points: dict[str, dict[str, PointData]] = field(default_factory=dict)

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vector_size: int, distance: str) -> None:
        self.collections[name] = {"vector_size": vector_size, "distance": distance}
        self.points[name] = {}

    def collection_vector_size(self, name: str) -> int:
        return self.collections[name]["vector_size"]

    def upsert(self, name: str, points: list[PointData]) -> None:
        for p in points:
            self.points[name][p.id] = p

    def scroll_document(self, name: str, document_id: str) -> list[str]:
        return [
            pid
            for pid, p in self.points.get(name, {}).items()
            if p.payload.get("document_id") == document_id
        ]

    def set_is_active(self, name: str, point_ids: list[str], is_active: bool) -> None:
        for pid in point_ids:
            if pid in self.points.get(name, {}):
                self.points[name][pid].payload["is_active"] = is_active


# ---------------------------------------------------------------------------
# Helpers for building DocumentChunk fixtures
# ---------------------------------------------------------------------------

FASTAPI_SOURCE_ID = "fastapi"
FASTAPI_URL = "https://fastapi.tiangolo.com/tutorial/first-steps"
PYTHON_SOURCE_ID = "python"
PYTHON_URL = "https://docs.python.org/3/tutorial/index.html"

_FASTAPI_DOC_ID = make_document_id(FASTAPI_SOURCE_ID, FASTAPI_URL)
_PYTHON_DOC_ID = make_document_id(PYTHON_SOURCE_ID, PYTHON_URL)


def _make_chunk(
    source_id: str = FASTAPI_SOURCE_ID,
    url: str = FASTAPI_URL,
    index: int = 0,
    text: str = "Hello world",
    extracted_sha256: str = "aabbcc",
    chunker_version: str = CHUNKER_VERSION,
) -> DocumentChunk:
    doc_id = make_document_id(source_id, url)
    chunk_id = make_chunk_id(source_id, url, index)
    ch = hashlib.sha256(text.encode()).hexdigest()
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_id=source_id,
        canonical_url=url,
        title="Test Page",
        headings=["Test Page"],
        breadcrumb="FastAPI > Tutorial > Test Page",
        text=text,
        primary_text=text,
        content_hash=ch,
        extracted_sha256=extracted_sha256,
        chunk_index=index,
        chunker_version=chunker_version,
        token_count=len(text.split()),
    )


def _make_catalog() -> IngestionCatalog:
    return IngestionCatalog(":memory:")


def _make_indexer(
    catalog: IngestionCatalog,
    *,
    embedder=None,
    store: FakeVectorStore | None = None,
    collection: str = COLLECTION_NAME,
) -> QdrantIndexer:
    return QdrantIndexer(
        store=store or FakeVectorStore(),
        embedder=embedder or FakeEmbedder(),
        catalog=catalog,
        collection_name=collection,
    )


def _register_and_fetch(catalog: IngestionCatalog, source_id: str, url: str) -> None:
    """Put the URL into the catalog with fetch_succeeded state so record_indexing can find it."""
    run = catalog.create_run(source_id)
    catalog.register_url(source_id, url, run_id=run.id)
    catalog.mark_fetch_started(source_id, url, run_id=run.id)
    catalog.mark_fetch_succeeded(source_id, url, run_id=run.id, http_status=200)
    catalog.record_extraction(
        source_id,
        url,
        extracted_sha256="aabbcc",
        chunker_version=CHUNKER_VERSION,
        run_id=run.id,
    )


# ---------------------------------------------------------------------------
# Collection management tests
# ---------------------------------------------------------------------------


class CollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_collection_created_on_ensure(self) -> None:
        self.assertFalse(self.store.collection_exists(COLLECTION_NAME))
        self.indexer.ensure_collection()
        self.assertTrue(self.store.collection_exists(COLLECTION_NAME))

    def test_correct_vector_size_on_create(self) -> None:
        self.indexer.ensure_collection()
        self.assertEqual(self.store.collection_vector_size(COLLECTION_NAME), EMBEDDING_DIM)
        self.assertEqual(self.store.collections[COLLECTION_NAME]["distance"], "COSINE")

    def test_correct_embedding_dimension_constant(self) -> None:
        self.assertEqual(EMBEDDING_DIM, 384)

    def test_ensure_collection_idempotent(self) -> None:
        """Calling ensure_collection twice must not raise."""
        self.indexer.ensure_collection()
        self.indexer.ensure_collection()  # should not raise
        self.assertEqual(len(self.store.collections), 1)

    def test_wrong_dim_raises_value_error(self) -> None:
        """If the collection already exists with the wrong dim, raise ValueError."""
        self.store.create_collection(COLLECTION_NAME, 768, "COSINE")
        with self.assertRaises(ValueError) as ctx:
            self.indexer.ensure_collection()
        self.assertIn("768", str(ctx.exception))
        self.assertIn(str(EMBEDDING_DIM), str(ctx.exception))


# ---------------------------------------------------------------------------
# Deterministic ID tests
# ---------------------------------------------------------------------------


class DeterministicIdTests(unittest.TestCase):
    def test_same_chunk_id_same_point_uuid(self) -> None:
        chunk = _make_chunk(index=0)
        p1 = make_point_id(chunk.chunk_id)
        p2 = make_point_id(chunk.chunk_id)
        self.assertEqual(p1, p2)

    def test_different_index_different_point_uuid(self) -> None:
        c0 = _make_chunk(index=0)
        c1 = _make_chunk(index=1)
        self.assertNotEqual(make_point_id(c0.chunk_id), make_point_id(c1.chunk_id))

    def test_point_uuid_is_valid_uuid(self) -> None:
        chunk = _make_chunk()
        parsed = UUID(make_point_id(chunk.chunk_id))
        self.assertEqual(parsed.version, 8)


# ---------------------------------------------------------------------------
# Successful upsert + payload metadata
# ---------------------------------------------------------------------------


class UpsertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_successful_upsert_stores_points(self) -> None:
        chunk = _make_chunk()
        res = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        self.assertTrue(res.ok)
        self.assertEqual(res.chunks_upserted, 1)
        point_id = make_point_id(chunk.chunk_id)
        self.assertIn(point_id, self.store.points[COLLECTION_NAME])

    def test_payload_contains_all_required_fields(self) -> None:
        chunk = _make_chunk(text="Documentation content here.")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-42")
        point_id = make_point_id(chunk.chunk_id)
        payload = self.store.points[COLLECTION_NAME][point_id].payload
        for key in (
            "source_id",
            "document_id",
            "chunk_id",
            "canonical_url",
            "title",
            "headings",
            "breadcrumb",
            "text",
            "content_hash",
            "chunker_version",
            "embedding_model",
            "ingestion_run_id",
            "is_active",
            "created_at",
        ):
            self.assertIn(key, payload, msg=f"Missing payload field: {key}")

    def test_payload_values_correct(self) -> None:
        chunk = _make_chunk(text="Some content.")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-7")
        pid = make_point_id(chunk.chunk_id)
        pl = self.store.points[COLLECTION_NAME][pid].payload
        self.assertEqual(pl["source_id"], FASTAPI_SOURCE_ID)
        self.assertEqual(pl["document_id"], _FASTAPI_DOC_ID)
        self.assertEqual(pl["chunk_id"], chunk.chunk_id)
        self.assertEqual(pl["canonical_url"], FASTAPI_URL)
        self.assertEqual(pl["is_active"], True)
        self.assertEqual(pl["embedding_model"], EMBEDDING_MODEL)
        self.assertEqual(pl["ingestion_run_id"], "run-7")

    def test_vector_has_correct_dimension(self) -> None:
        chunk = _make_chunk()
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        pid = make_point_id(chunk.chunk_id)
        vec = self.store.points[COLLECTION_NAME][pid].vector
        self.assertEqual(len(vec), EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


class BatchEmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.embedder = FakeEmbedder()
        self.indexer = _make_indexer(self.catalog, store=self.store, embedder=self.embedder)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_large_batch_produces_correct_point_count(self) -> None:
        n = EMBED_BATCH + 5  # crosses one batch boundary
        chunks = [_make_chunk(index=i, text=f"chunk text {i}") for i in range(n)]
        res = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, chunks, "run-1")
        self.assertTrue(res.ok)
        self.assertEqual(res.chunks_upserted, n)
        self.assertEqual(len(self.store.points[COLLECTION_NAME]), n)

    def test_embed_called_in_batches(self) -> None:
        n = EMBED_BATCH + 3
        chunks = [_make_chunk(index=i, text=f"t {i}") for i in range(n)]
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, chunks, "run-1")
        total_embedded = sum(len(b) for b in self.embedder.calls)
        self.assertEqual(total_embedded, n)
        # Two batches: one full (EMBED_BATCH) and one partial.
        self.assertEqual(len(self.embedder.calls), 2)


# ---------------------------------------------------------------------------
# Skip / re-index logic
# ---------------------------------------------------------------------------


class SkipReindexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.embedder = FakeEmbedder()
        self.indexer = _make_indexer(self.catalog, store=self.store, embedder=self.embedder)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_unchanged_chunks_skipped_on_second_call(self) -> None:
        chunk = _make_chunk()
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        calls_after_first = len(self.embedder.calls)
        # Second call: same sha256 + chunker_version → should skip.
        res2 = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-2")
        self.assertEqual(res2.chunks_skipped, 1)
        self.assertEqual(res2.chunks_upserted, 0)
        # Embedder must NOT have been called again.
        self.assertEqual(len(self.embedder.calls), calls_after_first)

    def test_changed_content_triggers_reindex(self) -> None:
        chunk1 = _make_chunk(text="original", extracted_sha256="sha-v1")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk1], "run-1")
        # Simulate content change: new sha256.
        chunk2 = _make_chunk(text="updated content", extracted_sha256="sha-v2")
        # Need to re-register with new extraction sha in catalog.
        self.catalog.record_extraction(
            FASTAPI_SOURCE_ID, FASTAPI_URL,
            extracted_sha256="sha-v2", chunker_version=CHUNKER_VERSION, run_id="run-2"
        )
        res2 = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk2], "run-2")
        self.assertTrue(res2.ok)
        self.assertEqual(res2.chunks_upserted, 1)

    def test_chunker_version_change_triggers_reindex(self) -> None:
        chunk_v1 = _make_chunk(chunker_version="heading-v1")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk_v1], "run-1")
        # Simulate chunker update.
        chunk_v2 = _make_chunk(chunker_version="heading-v2")
        res2 = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk_v2], "run-2")
        self.assertTrue(res2.ok)
        self.assertEqual(res2.chunks_upserted, 1)

    def test_no_duplicate_points_on_repeated_upsert(self) -> None:
        chunk = _make_chunk()
        # Index twice with intentionally different run_ids but same content.
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        # Force needs_indexing by clearing indexed state in catalog.
        self.catalog._conn.execute(
            "UPDATE catalog_urls SET indexed_sha256 = NULL WHERE source_id = ?",
            (FASTAPI_SOURCE_ID,),
        )
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-2")
        # Same chunk_id → same point UUID → still only one point.
        self.assertEqual(len(self.store.points[COLLECTION_NAME]), 1)

    def test_deterministic_repeated_indexing_idempotent(self) -> None:
        chunk = _make_chunk()
        res1 = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        pid = make_point_id(chunk.chunk_id)
        vec1 = list(self.store.points[COLLECTION_NAME][pid].vector)
        # Force re-index by clearing indexed state.
        self.catalog._conn.execute(
            "UPDATE catalog_urls SET indexed_sha256 = NULL WHERE source_id = ?",
            (FASTAPI_SOURCE_ID,),
        )
        res2 = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-2")
        vec2 = list(self.store.points[COLLECTION_NAME][pid].vector)
        self.assertEqual(vec1, vec2, "Same text must produce the same vector")
        self.assertEqual(len(self.store.points[COLLECTION_NAME]), 1)


# ---------------------------------------------------------------------------
# Stale point deactivation
# ---------------------------------------------------------------------------


class StalePointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_stale_points_deactivated_on_rechunk(self) -> None:
        # First version: two chunks.
        old_chunks = [
            _make_chunk(index=0, text="intro", extracted_sha256="sha-v1"),
            _make_chunk(index=1, text="body",  extracted_sha256="sha-v1"),
        ]
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, old_chunks, "run-1")
        old_pid = make_point_id(old_chunks[1].chunk_id)
        self.assertEqual(self.store.points[COLLECTION_NAME][old_pid].payload["is_active"], True)

        # Second version: one chunk (chunk[1] disappears).
        new_chunk = _make_chunk(index=0, text="intro", extracted_sha256="sha-v2")
        self.catalog.record_extraction(
            FASTAPI_SOURCE_ID, FASTAPI_URL,
            extracted_sha256="sha-v2", chunker_version=CHUNKER_VERSION, run_id="run-2"
        )
        res = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [new_chunk], "run-2")
        self.assertEqual(res.points_deactivated, 1)
        self.assertEqual(self.store.points[COLLECTION_NAME][old_pid].payload["is_active"], False)

    def test_deactivate_document_sets_all_is_active_false(self) -> None:
        chunks = [_make_chunk(index=i, text=f"c{i}") for i in range(3)]
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, chunks, "run-1")
        n = self.indexer.deactivate_document(_FASTAPI_DOC_ID)
        self.assertEqual(n, 3)
        for p in self.store.points[COLLECTION_NAME].values():
            self.assertFalse(p.payload["is_active"])

    def test_deactivate_document_returns_zero_for_unknown(self) -> None:
        n = self.indexer.deactivate_document("nonexistent-doc-id")
        self.assertEqual(n, 0)

    def test_empty_chunks_deactivates_existing_points(self) -> None:
        chunk = _make_chunk()
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        pid = make_point_id(chunk.chunk_id)
        self.assertTrue(self.store.points[COLLECTION_NAME][pid].payload["is_active"])
        # Now index with an empty list.
        res = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [], "run-2")
        self.assertEqual(res.points_deactivated, 1)
        self.assertFalse(self.store.points[COLLECTION_NAME][pid].payload["is_active"])


# ---------------------------------------------------------------------------
# Source isolation
# ---------------------------------------------------------------------------


class SourceIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)
        _register_and_fetch(self.catalog, PYTHON_SOURCE_ID, PYTHON_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_python_points_unaffected_by_fastapi_deactivation(self) -> None:
        fa_chunk = _make_chunk(source_id=FASTAPI_SOURCE_ID, url=FASTAPI_URL, index=0)
        py_chunk = _make_chunk(source_id=PYTHON_SOURCE_ID, url=PYTHON_URL, index=0, text="python content")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [fa_chunk], "run-1")
        self.indexer.index_document(PYTHON_SOURCE_ID, PYTHON_URL, [py_chunk], "run-1")
        # Deactivate the FastAPI document.
        self.indexer.deactivate_document(_FASTAPI_DOC_ID)
        py_pid = make_point_id(py_chunk.chunk_id)
        self.assertTrue(self.store.points[COLLECTION_NAME][py_pid].payload["is_active"])

    def test_deactivate_stale_for_source_only_affects_target_source(self) -> None:
        fa_chunk = _make_chunk(source_id=FASTAPI_SOURCE_ID, url=FASTAPI_URL, index=0)
        py_chunk = _make_chunk(source_id=PYTHON_SOURCE_ID, url=PYTHON_URL, index=0, text="python content")
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [fa_chunk], "run-1")
        self.indexer.index_document(PYTHON_SOURCE_ID, PYTHON_URL, [py_chunk], "run-1")
        # Stale for FastAPI (active set is empty → all FastAPI docs deactivated).
        self.indexer.deactivate_stale_for_source(FASTAPI_SOURCE_ID, active_document_ids=set())
        fa_pid = make_point_id(fa_chunk.chunk_id)
        py_pid = make_point_id(py_chunk.chunk_id)
        self.assertFalse(self.store.points[COLLECTION_NAME][fa_pid].payload["is_active"])
        self.assertTrue(self.store.points[COLLECTION_NAME][py_pid].payload["is_active"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_embed_failure_returns_error_result(self) -> None:
        err_indexer = QdrantIndexer(self.store, ErrEmbedder(), self.catalog, COLLECTION_NAME)
        chunk = _make_chunk()
        res = err_indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.error)
        self.assertIn("embed failure", res.error)

    def test_embed_failure_does_not_record_indexing(self) -> None:
        err_indexer = QdrantIndexer(self.store, ErrEmbedder(), self.catalog, COLLECTION_NAME)
        chunk = _make_chunk()
        err_indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        # needs_indexing must still return True (catalog not updated).
        self.assertTrue(
            self.catalog.needs_indexing(
                FASTAPI_SOURCE_ID, FASTAPI_URL,
                extracted_sha256=chunk.extracted_sha256,
                chunker_version=chunk.chunker_version,
            )
        )

    def test_partial_batch_upsert_failure_captured(self) -> None:
        """Upsert raises on second batch — result.ok=False, no crash."""
        partial_store = PartialUpsertStore()
        partial_store.create_collection(COLLECTION_NAME, EMBEDDING_DIM, "COSINE")
        indexer = QdrantIndexer(partial_store, FakeEmbedder(), self.catalog, COLLECTION_NAME)
        # Two batches: first succeeds, second raises.
        chunks = [_make_chunk(index=i, text=f"t{i}") for i in range(EMBED_BATCH + 1)]
        res = indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, chunks, "run-1")
        self.assertFalse(res.ok)
        self.assertIsNotNone(res.error)

    def test_one_document_failure_does_not_affect_sibling(self) -> None:
        """Index two separate documents; failure on first must not affect second."""
        _register_and_fetch(self.catalog, PYTHON_SOURCE_ID, PYTHON_URL)
        err_indexer = QdrantIndexer(self.store, ErrEmbedder(), self.catalog, COLLECTION_NAME)
        fa_chunk = _make_chunk(source_id=FASTAPI_SOURCE_ID, url=FASTAPI_URL)
        py_chunk = _make_chunk(source_id=PYTHON_SOURCE_ID, url=PYTHON_URL, text="python content")
        # Both will fail because ErrEmbedder, but they should not raise.
        res1 = err_indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [fa_chunk], "run-1")
        res2 = err_indexer.index_document(PYTHON_SOURCE_ID, PYTHON_URL, [py_chunk], "run-1")
        self.assertFalse(res1.ok)
        self.assertFalse(res2.ok)  # both fail gracefully; no exception propagated.


# ---------------------------------------------------------------------------
# SQLite / Qdrant state consistency
# ---------------------------------------------------------------------------


class StateConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeVectorStore()
        self.catalog = _make_catalog()
        self.indexer = _make_indexer(self.catalog, store=self.store)
        self.indexer.ensure_collection()
        _register_and_fetch(self.catalog, FASTAPI_SOURCE_ID, FASTAPI_URL)

    def tearDown(self) -> None:
        self.catalog.close()

    def test_record_indexing_called_after_successful_upsert(self) -> None:
        chunk = _make_chunk()
        self.assertTrue(
            self.catalog.needs_indexing(
                FASTAPI_SOURCE_ID, FASTAPI_URL,
                extracted_sha256=chunk.extracted_sha256,
                chunker_version=chunk.chunker_version,
            )
        )
        res = self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        self.assertTrue(res.ok)
        # After successful upsert, needs_indexing should be False.
        self.assertFalse(
            self.catalog.needs_indexing(
                FASTAPI_SOURCE_ID, FASTAPI_URL,
                extracted_sha256=chunk.extracted_sha256,
                chunker_version=chunk.chunker_version,
            )
        )

    def test_record_indexing_not_called_on_failure(self) -> None:
        err_indexer = QdrantIndexer(self.store, ErrEmbedder(), self.catalog, COLLECTION_NAME)
        chunk = _make_chunk()
        err_indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        # Catalog should still say indexing is needed.
        self.assertTrue(
            self.catalog.needs_indexing(
                FASTAPI_SOURCE_ID, FASTAPI_URL,
                extracted_sha256=chunk.extracted_sha256,
                chunker_version=chunk.chunker_version,
            )
        )

    def test_indexed_sha256_persisted_in_catalog(self) -> None:
        chunk = _make_chunk(extracted_sha256="deadbeef")
        self.catalog.record_extraction(
            FASTAPI_SOURCE_ID, FASTAPI_URL,
            extracted_sha256="deadbeef", chunker_version=CHUNKER_VERSION, run_id=None
        )
        self.indexer.index_document(FASTAPI_SOURCE_ID, FASTAPI_URL, [chunk], "run-1")
        rec = self.catalog.get_url(FASTAPI_SOURCE_ID, FASTAPI_URL)
        self.assertEqual(rec.indexed_sha256, "deadbeef")
        self.assertEqual(rec.indexed_chunker_version, CHUNKER_VERSION)


if __name__ == "__main__":
    unittest.main()
