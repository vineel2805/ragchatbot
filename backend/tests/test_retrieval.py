"""Unit tests for the retrieval and context assembly layer.

All tests use FakeEmbedder and FakeVectorStore — no live Qdrant, no network,
no HuggingFace downloads required.
"""
from __future__ import annotations

import hashlib
import unittest
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.embedder import EMBEDDING_DIM
from app.ingestion.ids import make_chunk_id, make_document_id
from app.ingestion.indexer import SearchHit
from app.retrieval.assembler import ContextAssembler
from app.retrieval.errors import (
    EmbedFailureError,
    InvalidQueryError,
    StoreFailureError,
)
from app.retrieval.models import RetrievalRequest, RetrievalResult
from app.retrieval.retriever import BGE_QUERY_PREFIX, Retriever


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic fake embedder — never loads sentence-transformers."""

    def __init__(self, dim: int = EMBEDDING_DIM, fail: bool = False) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.fail = fail

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embed failure injected")
        self.calls.append(list(texts))
        result = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            raw = [float((seed >> i) & 0xFF) / 255.0 - 0.5 for i in range(self.dim)]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            result.append([x / norm for x in raw])
        return result


class FakeVectorStore:
    """In-memory vector store with configurable search results."""

    def __init__(self, search_results: list[SearchHit] | None = None, fail: bool = False) -> None:
        self._results: list[SearchHit] = search_results or []
        self.fail = fail
        self.search_calls: list[dict] = []
        # Minimal stubs for the indexer protocol (not exercised in retrieval tests)
        self.collections: dict[str, dict] = {}
        self.points: dict[str, dict[str, Any]] = {}

    # --- VectorStoreClient required methods ---

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vector_size: int, distance: str) -> None:
        self.collections[name] = {"vector_size": vector_size}
        self.points[name] = {}

    def collection_vector_size(self, name: str) -> int:
        return self.collections[name]["vector_size"]

    def upsert(self, name: str, points) -> None:
        for p in points:
            self.points.setdefault(name, {})[p.id] = p

    def scroll_document(self, name: str, document_id: str) -> list[str]:
        return []

    def set_is_active(self, name: str, point_ids: list[str], is_active: bool) -> None:
        pass

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        source_id: str | None,
    ) -> list[SearchHit]:
        if self.fail:
            raise RuntimeError("store failure injected")
        self.search_calls.append(
            {"collection": collection_name, "top_k": top_k, "source_id": source_id}
        )
        # Apply source_id filter client-side (mirrors the adapter's server-side filter).
        filtered = [
            h for h in self._results
            if source_id is None or h.payload.get("source_id") == source_id
        ]
        # Apply is_active filter (active points only).
        filtered = [h for h in filtered if h.payload.get("is_active", True)]
        return filtered[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FASTAPI_SRC = "fastapi"
PYTHON_SRC = "python"
FASTAPI_URL = "https://fastapi.tiangolo.com/tutorial/first-steps"
PYTHON_URL = "https://docs.python.org/3/tutorial"


def _make_hit(
    score: float,
    source_id: str = FASTAPI_SRC,
    url: str = FASTAPI_URL,
    chunk_index: int = 0,
    text: str = "Some documentation text.",
    is_active: bool = True,
) -> SearchHit:
    doc_id = make_document_id(source_id, url)
    chunk_id = make_chunk_id(source_id, url, chunk_index)
    return SearchHit(
        score=score,
        point_id=f"point-{chunk_id[:8]}",
        payload={
            "source_id": source_id,
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "canonical_url": url,
            "title": "Test Page",
            "headings": ["Test Page", "Section"],
            "breadcrumb": f"{source_id} > Tutorial > Test",
            "text": text,
            "chunk_index": chunk_index,
            "is_active": is_active,
        },
    )


def _make_retriever(
    results: list[SearchHit] | None = None,
    *,
    embed_fail: bool = False,
    store_fail: bool = False,
    embedder_dim: int = EMBEDDING_DIM,
) -> Retriever:
    embedder = FakeEmbedder(dim=embedder_dim, fail=embed_fail)
    store = FakeVectorStore(results, fail=store_fail)
    return Retriever(store, embedder, collection_name="test_collection")


# ---------------------------------------------------------------------------
# 1. Request validation tests
# ---------------------------------------------------------------------------


class RequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = _make_retriever()

    def test_empty_query_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query=""))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_whitespace_only_query_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query="   \t\n  "))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_top_k_zero_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query="test", top_k=0))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_top_k_101_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query="test", top_k=101))
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_top_k_1_is_valid(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query="test", top_k=1))
        self.assertTrue(result.ok)

    def test_top_k_100_is_valid(self) -> None:
        result = self.retriever.retrieve(RetrievalRequest(query="test", top_k=100))
        self.assertTrue(result.ok)

    def test_negative_threshold_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(
            RetrievalRequest(query="test", score_threshold=-0.1)
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_threshold_above_one_returns_invalid_query_error(self) -> None:
        result = self.retriever.retrieve(
            RetrievalRequest(query="test", score_threshold=1.1)
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_threshold_zero_is_valid(self) -> None:
        result = self.retriever.retrieve(
            RetrievalRequest(query="test", score_threshold=0.0)
        )
        self.assertTrue(result.ok)

    def test_threshold_one_is_valid(self) -> None:
        result = self.retriever.retrieve(
            RetrievalRequest(query="test", score_threshold=1.0)
        )
        self.assertTrue(result.ok)

    def test_invalid_query_does_not_call_embedder(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        retriever = Retriever(store, embedder, "test_collection")
        retriever.retrieve(RetrievalRequest(query=""))
        self.assertEqual(embedder.calls, [])


# ---------------------------------------------------------------------------
# 2. Query embedding tests
# ---------------------------------------------------------------------------


class QueryEmbeddingTests(unittest.TestCase):
    def test_bge_query_prefix_prepended(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        retriever = Retriever(store, embedder, "test_collection")
        retriever.retrieve(RetrievalRequest(query="how do I use FastAPI?"))
        self.assertEqual(len(embedder.calls), 1)
        embedded_text = embedder.calls[0][0]
        self.assertTrue(embedded_text.startswith(BGE_QUERY_PREFIX))
        self.assertIn("how do I use FastAPI?", embedded_text)

    def test_bge_prefix_not_duplicated(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        retriever = Retriever(store, embedder, "test_collection")
        retriever.retrieve(RetrievalRequest(query="test query"))
        embedded = embedder.calls[0][0]
        self.assertEqual(embedded.count(BGE_QUERY_PREFIX), 1)

    def test_embed_called_exactly_once_per_retrieve(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore()
        retriever = Retriever(store, embedder, "test_collection")
        retriever.retrieve(RetrievalRequest(query="query one"))
        retriever.retrieve(RetrievalRequest(query="query two"))
        self.assertEqual(len(embedder.calls), 2)

    def test_embed_failure_returns_embed_failure_error_type(self) -> None:
        result = _make_retriever(embed_fail=True).retrieve(
            RetrievalRequest(query="valid query")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "EmbedFailure")

    def test_embed_failure_does_not_call_store(self) -> None:
        store = FakeVectorStore(fail=False)
        retriever = Retriever(store, FakeEmbedder(fail=True), "test_collection")
        retriever.retrieve(RetrievalRequest(query="valid query"))
        self.assertEqual(store.search_calls, [])

    def test_vector_dimension_passed_to_store(self) -> None:
        """The embedding vector passed to store.search must be EMBEDDING_DIM long."""
        embedder = FakeEmbedder(dim=EMBEDDING_DIM)
        received_vectors: list[list[float]] = []

        class CapturingStore(FakeVectorStore):
            def search(self, col, query_vector, top_k, source_id):
                received_vectors.append(query_vector)
                return []

        retriever = Retriever(CapturingStore(), embedder, "col")
        retriever.retrieve(RetrievalRequest(query="test"))
        self.assertEqual(len(received_vectors[0]), EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# 3. Search / filtering tests
# ---------------------------------------------------------------------------


class SearchTests(unittest.TestCase):
    def test_top_k_passed_to_store(self) -> None:
        store = FakeVectorStore([_make_hit(0.9)] * 20)
        retriever = Retriever(store, FakeEmbedder(), "col")
        retriever.retrieve(RetrievalRequest(query="q", top_k=5))
        self.assertEqual(store.search_calls[0]["top_k"], 5)

    def test_score_threshold_filters_low_scores(self) -> None:
        hits = [_make_hit(0.9), _make_hit(0.5), _make_hit(0.3)]
        result = _make_retriever(hits).retrieve(
            RetrievalRequest(query="q", score_threshold=0.6)
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(result.hits), 1)
        self.assertAlmostEqual(result.hits[0].score, 0.9)

    def test_active_only_filtering(self) -> None:
        hits = [_make_hit(0.9, is_active=True), _make_hit(0.8, is_active=False)]
        result = _make_retriever(hits).retrieve(RetrievalRequest(query="q"))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.hits), 1)
        self.assertAlmostEqual(result.hits[0].score, 0.9)

    def test_source_id_filter_passed_to_store(self) -> None:
        store = FakeVectorStore()
        retriever = Retriever(store, FakeEmbedder(), "col")
        retriever.retrieve(RetrievalRequest(query="q", source_id="fastapi"))
        self.assertEqual(store.search_calls[0]["source_id"], "fastapi")

    def test_source_id_none_passed_to_store_for_all_sources(self) -> None:
        store = FakeVectorStore()
        retriever = Retriever(store, FakeEmbedder(), "col")
        retriever.retrieve(RetrievalRequest(query="q", source_id=None))
        self.assertIsNone(store.search_calls[0]["source_id"])

    def test_source_id_filtering_excludes_other_sources(self) -> None:
        hits = [
            _make_hit(0.9, source_id="fastapi"),
            _make_hit(0.8, source_id="python", url=PYTHON_URL),
        ]
        result = _make_retriever(hits).retrieve(
            RetrievalRequest(query="q", source_id="fastapi")
        )
        for h in result.hits:
            self.assertEqual(h.source_id, "fastapi")

    def test_all_source_retrieval_returns_multiple_sources(self) -> None:
        hits = [
            _make_hit(0.9, source_id="fastapi"),
            _make_hit(0.8, source_id="python", url=PYTHON_URL),
        ]
        result = _make_retriever(hits).retrieve(
            RetrievalRequest(query="q", source_id=None)
        )
        self.assertTrue(result.ok)
        sources = {h.source_id for h in result.hits}
        self.assertIn("fastapi", sources)
        self.assertIn("python", sources)

    def test_no_results_returns_ok_with_empty_hits(self) -> None:
        result = _make_retriever([]).retrieve(RetrievalRequest(query="q"))
        self.assertTrue(result.ok)
        self.assertEqual(result.hits, [])

    def test_store_failure_returns_store_failure_error_type(self) -> None:
        result = _make_retriever(store_fail=True).retrieve(
            RetrievalRequest(query="valid query")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "StoreFailure")


# ---------------------------------------------------------------------------
# 4. Deterministic ranking tests
# ---------------------------------------------------------------------------


class RankingTests(unittest.TestCase):
    def test_results_sorted_by_score_descending(self) -> None:
        hits = [_make_hit(0.3), _make_hit(0.9), _make_hit(0.6)]
        result = _make_retriever(hits).retrieve(RetrievalRequest(query="q"))
        scores = [h.score for h in result.hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_equal_score_tie_broken_by_source_id(self) -> None:
        # Same score; python < fastapi lexicographically — python should come first.
        hit_fa = _make_hit(0.8, source_id="fastapi")
        hit_py = _make_hit(0.8, source_id="python", url=PYTHON_URL)
        result = _make_retriever([hit_fa, hit_py]).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(result.hits[0].source_id, "fastapi")
        self.assertEqual(result.hits[1].source_id, "python")

    def test_equal_score_source_equal_tie_broken_by_document_id(self) -> None:
        url_a = "https://fastapi.tiangolo.com/tutorial/a"
        url_b = "https://fastapi.tiangolo.com/tutorial/b"
        hit_b = _make_hit(0.8, url=url_b)
        hit_a = _make_hit(0.8, url=url_a)
        # Build actual doc IDs to know which comes first lexicographically.
        doc_a = make_document_id("fastapi", url_a)
        doc_b = make_document_id("fastapi", url_b)
        result = _make_retriever([hit_b, hit_a]).retrieve(RetrievalRequest(query="q"))
        hit_ids = [h.document_id for h in result.hits]
        expected = sorted([doc_a, doc_b])
        self.assertEqual(hit_ids, expected)

    def test_deterministic_across_multiple_calls(self) -> None:
        hits = [_make_hit(s, chunk_index=i) for i, s in enumerate([0.7, 0.9, 0.5, 0.8])]
        r1 = _make_retriever(list(hits)).retrieve(RetrievalRequest(query="q"))
        r2 = _make_retriever(list(hits)).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(
            [h.chunk_id for h in r1.hits],
            [h.chunk_id for h in r2.hits],
        )

    def test_score_descending_with_mixed_sources(self) -> None:
        hits = [
            _make_hit(0.5, source_id="fastapi"),
            _make_hit(0.9, source_id="python", url=PYTHON_URL),
            _make_hit(0.7, source_id="fastapi", chunk_index=1),
        ]
        result = _make_retriever(hits).retrieve(RetrievalRequest(query="q"))
        scores = [h.score for h in result.hits]
        self.assertGreaterEqual(scores[0], scores[1])
        self.assertGreaterEqual(scores[1], scores[2])


# ---------------------------------------------------------------------------
# 5. Deduplication tests
# ---------------------------------------------------------------------------


class DeduplicationTests(unittest.TestCase):
    def test_duplicate_chunk_id_kept_once(self) -> None:
        """Two SearchHits with the same chunk_id → only one RetrievalHit."""
        hit = _make_hit(0.9)
        duplicate = _make_hit(0.8)
        # Make duplicate share the same chunk_id.
        duplicate.payload["chunk_id"] = hit.payload["chunk_id"]
        result = _make_retriever([hit, duplicate]).retrieve(RetrievalRequest(query="q"))
        chunk_ids = [h.chunk_id for h in result.hits]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_duplicate_keeps_first_occurrence_by_sort_order(self) -> None:
        """After sort, the first occurrence (highest score) is kept."""
        hit_high = _make_hit(0.9)
        hit_low = _make_hit(0.5)
        hit_low.payload["chunk_id"] = hit_high.payload["chunk_id"]
        result = _make_retriever([hit_high, hit_low]).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(len(result.hits), 1)
        self.assertAlmostEqual(result.hits[0].score, 0.9)

    def test_distinct_chunks_not_deduplicated(self) -> None:
        hits = [_make_hit(0.9, chunk_index=i) for i in range(5)]
        result = _make_retriever(hits).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(len(result.hits), 5)

    def test_dedup_preserves_total_count_after_removal(self) -> None:
        """3 hits, 2 share a chunk_id → 2 unique results."""
        h0 = _make_hit(0.9, chunk_index=0)
        h1 = _make_hit(0.8, chunk_index=1)
        h2 = _make_hit(0.7, chunk_index=2)
        # h2 shares chunk_id with h0.
        h2.payload["chunk_id"] = h0.payload["chunk_id"]
        result = _make_retriever([h0, h1, h2]).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(len(result.hits), 2)


# ---------------------------------------------------------------------------
# 6. Metadata preservation tests
# ---------------------------------------------------------------------------


class MetadataTests(unittest.TestCase):
    def test_all_payload_fields_present_in_hit(self) -> None:
        hit = _make_hit(0.85, text="Detailed chunk content.")
        result = _make_retriever([hit]).retrieve(RetrievalRequest(query="q"))
        self.assertEqual(len(result.hits), 1)
        h = result.hits[0]
        self.assertAlmostEqual(h.score, 0.85)
        self.assertEqual(h.source_id, "fastapi")
        self.assertIsNotNone(h.document_id)
        self.assertIsNotNone(h.chunk_id)
        self.assertEqual(h.canonical_url, FASTAPI_URL)
        self.assertEqual(h.title, "Test Page")
        self.assertIsInstance(h.headings, list)
        self.assertIsInstance(h.breadcrumb, str)
        self.assertEqual(h.text, "Detailed chunk content.")
        self.assertEqual(h.chunk_index, 0)

    def test_missing_optional_payload_fields_handled_gracefully(self) -> None:
        """Payload missing title/headings/breadcrumb → defaults, no crash."""
        sparse_hit = SearchHit(
            score=0.7,
            point_id="sparse-point",
            payload={
                "source_id": "fastapi",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "canonical_url": FASTAPI_URL,
                "text": "Sparse payload.",
                "is_active": True,
            },
        )
        result = _make_retriever([sparse_hit]).retrieve(RetrievalRequest(query="q"))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.hits), 1)
        h = result.hits[0]
        self.assertEqual(h.title, "")
        self.assertEqual(h.headings, [])
        self.assertEqual(h.breadcrumb, "")

    def test_query_preserved_in_result(self) -> None:
        query = "How do I declare path parameters?"
        result = _make_retriever().retrieve(RetrievalRequest(query=query))
        self.assertEqual(result.query, query)

    def test_source_isolation_different_sources_separate(self) -> None:
        hits = [
            _make_hit(0.9, source_id="fastapi"),
            _make_hit(0.8, source_id="python", url=PYTHON_URL),
        ]
        result_fa = _make_retriever(hits).retrieve(
            RetrievalRequest(query="q", source_id="fastapi")
        )
        result_py = _make_retriever(hits).retrieve(
            RetrievalRequest(query="q", source_id="python")
        )
        for h in result_fa.hits:
            self.assertEqual(h.source_id, "fastapi")
        for h in result_py.hits:
            self.assertEqual(h.source_id, "python")


# ---------------------------------------------------------------------------
# 7. Error handling tests
# ---------------------------------------------------------------------------


class ErrorHandlingTests(unittest.TestCase):
    def test_embed_failure_error_type(self) -> None:
        result = _make_retriever(embed_fail=True).retrieve(
            RetrievalRequest(query="q")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "EmbedFailure")
        self.assertIsNotNone(result.error)

    def test_store_failure_error_type(self) -> None:
        result = _make_retriever(store_fail=True).retrieve(
            RetrievalRequest(query="q")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "StoreFailure")
        self.assertIsNotNone(result.error)

    def test_invalid_query_error_type_from_empty(self) -> None:
        result = _make_retriever().retrieve(RetrievalRequest(query=""))
        self.assertEqual(result.error_type, "InvalidQuery")

    def test_ok_false_hits_empty_on_error(self) -> None:
        result = _make_retriever(embed_fail=True).retrieve(
            RetrievalRequest(query="q")
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.hits, [])

    def test_no_results_is_not_an_error(self) -> None:
        result = _make_retriever([]).retrieve(RetrievalRequest(query="q"))
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)

    def test_error_does_not_propagate_as_exception(self) -> None:
        """Retriever must never propagate RetrievalError to the caller."""
        retriever = _make_retriever(embed_fail=True)
        try:
            result = retriever.retrieve(RetrievalRequest(query="q"))
        except Exception as exc:
            self.fail(f"retrieve() raised unexpectedly: {exc}")
        self.assertFalse(result.ok)


# ---------------------------------------------------------------------------
# 8. Context assembler tests
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Fake token counter: word count. No model download needed."""
    return len(text.split())


def _make_result(hits_data: list[tuple[float, str]]) -> RetrievalResult:
    """Build a RetrievalResult from (score, text) tuples."""
    from app.retrieval.models import RetrievalHit
    hits = []
    for i, (score, text) in enumerate(hits_data):
        chunk_id = make_chunk_id("fastapi", FASTAPI_URL, i)
        hits.append(
            RetrievalHit(
                score=score,
                source_id="fastapi",
                document_id=make_document_id("fastapi", FASTAPI_URL),
                chunk_id=chunk_id,
                canonical_url=FASTAPI_URL,
                title="Test",
                headings=["Test"],
                breadcrumb="FastAPI > Tutorial",
                text=text,
                chunk_index=i,
            )
        )
    return RetrievalResult(query="test query", hits=hits)


class AssemblerTests(unittest.TestCase):
    def _assembler(self) -> ContextAssembler:
        return ContextAssembler(count_tokens_fn=_word_count)

    def test_max_chunks_limits_output(self) -> None:
        result = _make_result([(0.9, f"chunk {i}") for i in range(10)])
        ctx = self._assembler().assemble(result, max_chunks=3, token_budget=10_000)
        self.assertEqual(len(ctx.chunks), 3)

    def test_token_budget_limits_output(self) -> None:
        # Each chunk is 3 words; budget = 7 → 2 chunks (6 tokens), not 3 (9 tokens).
        result = _make_result([(0.9, "one two three")] * 5)
        ctx = self._assembler().assemble(result, max_chunks=10, token_budget=7)
        self.assertEqual(len(ctx.chunks), 2)
        self.assertLessEqual(ctx.total_tokens, 7)

    def test_total_tokens_never_exceeds_budget(self) -> None:
        result = _make_result([(0.9, f"word{j} " * 20) for j in range(10)])
        ctx = self._assembler().assemble(result, max_chunks=10, token_budget=50)
        self.assertLessEqual(ctx.total_tokens, 50)

    def test_truncated_chunk_ids_populated_when_budget_exceeded(self) -> None:
        result = _make_result([(0.9, "a b c d e")] * 5)  # 5 tokens each
        ctx = self._assembler().assemble(result, max_chunks=10, token_budget=12)
        # 2 chunks fit (10 tokens), 3 skipped.
        self.assertEqual(len(ctx.truncated_chunk_ids), 3)

    def test_truncated_chunk_ids_empty_when_budget_not_exceeded(self) -> None:
        result = _make_result([(0.9, "short")] * 3)
        ctx = self._assembler().assemble(result, max_chunks=10, token_budget=1000)
        self.assertEqual(ctx.truncated_chunk_ids, [])

    def test_chunks_limited_by_max_chunks_records_remainder_as_truncated(self) -> None:
        result = _make_result([(0.9, "word")] * 5)
        ctx = self._assembler().assemble(result, max_chunks=2, token_budget=10_000)
        self.assertEqual(len(ctx.chunks), 2)
        self.assertEqual(len(ctx.truncated_chunk_ids), 3)

    def test_deterministic_context_order(self) -> None:
        """Same input → same chunk order across two calls."""
        result = _make_result([(0.9 - i * 0.1, f"chunk {i}") for i in range(5)])
        a = self._assembler().assemble(result, max_chunks=5, token_budget=10_000)
        b = self._assembler().assemble(result, max_chunks=5, token_budget=10_000)
        self.assertEqual(
            [c.chunk_id for c in a.chunks],
            [c.chunk_id for c in b.chunks],
        )

    def test_metadata_preserved_in_chunk_context(self) -> None:
        result = _make_result([(0.88, "preserved content here")])
        ctx = self._assembler().assemble(result, max_chunks=5, token_budget=10_000)
        self.assertEqual(len(ctx.chunks), 1)
        c = ctx.chunks[0]
        self.assertEqual(c.text, "preserved content here")
        self.assertEqual(c.source_id, "fastapi")
        self.assertEqual(c.canonical_url, FASTAPI_URL)
        self.assertAlmostEqual(c.score, 0.88)
        self.assertFalse(c.truncated)

    def test_empty_result_returns_empty_context(self) -> None:
        result = RetrievalResult(query="q", hits=[])
        ctx = self._assembler().assemble(result, max_chunks=5, token_budget=2000)
        self.assertEqual(ctx.chunks, [])
        self.assertEqual(ctx.total_tokens, 0)
        self.assertEqual(ctx.truncated_chunk_ids, [])

    def test_query_preserved_in_assembled_context(self) -> None:
        result = RetrievalResult(query="my special query", hits=[])
        ctx = self._assembler().assemble(result)
        self.assertEqual(ctx.query, "my special query")

    def test_token_budget_stored_in_context(self) -> None:
        result = RetrievalResult(query="q", hits=[])
        ctx = self._assembler().assemble(result, token_budget=1500)
        self.assertEqual(ctx.token_budget, 1500)

    def test_source_isolation_different_sources_in_context(self) -> None:
        """Chunks from different sources can coexist in the assembled context."""
        from app.retrieval.models import RetrievalHit
        hits = [
            RetrievalHit(
                score=0.9, source_id="fastapi",
                document_id=make_document_id("fastapi", FASTAPI_URL),
                chunk_id=make_chunk_id("fastapi", FASTAPI_URL, 0),
                canonical_url=FASTAPI_URL, title="FA", headings=[], breadcrumb="",
                text="FastAPI content", chunk_index=0,
            ),
            RetrievalHit(
                score=0.8, source_id="python",
                document_id=make_document_id("python", PYTHON_URL),
                chunk_id=make_chunk_id("python", PYTHON_URL, 0),
                canonical_url=PYTHON_URL, title="PY", headings=[], breadcrumb="",
                text="Python content", chunk_index=0,
            ),
        ]
        result = RetrievalResult(query="q", hits=hits)
        ctx = self._assembler().assemble(result, max_chunks=5, token_budget=10_000)
        sources = {c.source_id for c in ctx.chunks}
        self.assertIn("fastapi", sources)
        self.assertIn("python", sources)

    def test_assembler_lazy_tokenizer_not_called_in_tests(self) -> None:
        """ContextAssembler with injected count_tokens_fn never imports transformers."""
        assembler = ContextAssembler(count_tokens_fn=_word_count)
        result = _make_result([(0.9, "some text")])
        # If transformers were imported, test would fail in the env without it.
        ctx = assembler.assemble(result, max_chunks=5, token_budget=100)
        self.assertEqual(len(ctx.chunks), 1)


if __name__ == "__main__":
    unittest.main()
