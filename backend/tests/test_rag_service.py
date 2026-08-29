"""Unit tests for RAGService.

All tests use fake Retriever, Assembler, and Generator — no live Qdrant,
no network, no model downloads.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from app.generation.models import Citation, GenerationRequest, GenerationResult
from app.ingestion.ids import make_chunk_id, make_document_id
from app.rag.models import RAGRequest, RAGResponse
from app.rag.service import RAGService
from app.retrieval.assembler import ContextAssembler
from app.retrieval.models import (
    AssembledContext,
    ChunkContext,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_FASTAPI_URL = "https://fastapi.tiangolo.com/tutorial/first-steps"
_PYTHON_URL = "https://docs.python.org/3/library/json.html"


class FakeRetriever:
    """Returns a pre-configured RetrievalResult."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        return self._result


class FakeAssembler:
    """Wraps real ContextAssembler with a word-count token function."""

    def __init__(self, max_chunks_override: int | None = None) -> None:
        self._inner = ContextAssembler(count_tokens_fn=lambda t: len(t.split()))
        self._max_chunks_override = max_chunks_override
        self.calls: list[tuple] = []

    def assemble(
        self,
        result: RetrievalResult,
        *,
        max_chunks: int = 5,
        token_budget: int = 2000,
    ) -> AssembledContext:
        if self._max_chunks_override is not None:
            max_chunks = self._max_chunks_override
        self.calls.append((max_chunks, token_budget))
        return self._inner.assemble(result, max_chunks=max_chunks, token_budget=token_budget)


class FakeGenerator:
    """Returns a pre-configured GenerationResult."""

    def __init__(self, result: GenerationResult) -> None:
        self._result = result
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return self._result


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _hit(
    score: float = 0.9,
    source_id: str = "fastapi",
    url: str = _FASTAPI_URL,
    chunk_index: int = 0,
    text: str = "FastAPI is a web framework.",
) -> RetrievalHit:
    return RetrievalHit(
        score=score,
        source_id=source_id,
        document_id=make_document_id(source_id, url),
        chunk_id=make_chunk_id(source_id, url, chunk_index),
        canonical_url=url,
        title="Test",
        headings=["Test"],
        breadcrumb=f"{source_id} > Tutorial",
        text=text,
        chunk_index=chunk_index,
    )


def _ok_retrieval(*hits: RetrievalHit) -> RetrievalResult:
    return RetrievalResult(query="test query", hits=list(hits))


def _failed_retrieval(error: str = "embed failure", error_type: str = "EmbedFailure") -> RetrievalResult:
    return RetrievalResult(query="test query", ok=False, error=error, error_type=error_type)


def _ok_generation(answer: str = "Answer text.", citations=None) -> GenerationResult:
    return GenerationResult(
        query="test query",
        answer=answer,
        citations=citations or [Citation(url=_FASTAPI_URL, title="Test", source_id="fastapi")],
        ok=True,
    )


def _failed_generation(
    error: str = "provider error", error_type: str = "ProviderError"
) -> GenerationResult:
    return GenerationResult(query="test query", ok=False, error=error, error_type=error_type)


def _make_service(
    *,
    hits: list[RetrievalHit] | None = None,
    retrieval_fail: bool = False,
    generation_fail: bool = False,
    generation_answer: str = "Answer text.",
    max_chunks_override: int | None = None,
) -> tuple[RAGService, FakeRetriever, FakeAssembler, FakeGenerator]:
    if retrieval_fail:
        ret_result = _failed_retrieval()
    else:
        ret_result = _ok_retrieval(*hits) if hits is not None else _ok_retrieval(_hit())

    gen_result = (
        _failed_generation() if generation_fail else _ok_generation(answer=generation_answer)
    )

    retriever = FakeRetriever(ret_result)
    assembler = FakeAssembler(max_chunks_override=max_chunks_override)
    generator = FakeGenerator(gen_result)
    svc = RAGService(retriever, assembler, generator)
    return svc, retriever, assembler, generator


# ---------------------------------------------------------------------------
# 1. End-to-end happy path tests
# ---------------------------------------------------------------------------


class HappyPathTests(unittest.TestCase):
    def test_successful_pipeline_returns_ok(self) -> None:
        svc, *_ = _make_service()
        resp = svc.answer(RAGRequest(query="How do I use FastAPI?"))
        self.assertTrue(resp.ok)

    def test_answer_forwarded_to_response(self) -> None:
        svc, *_ = _make_service(generation_answer="Detailed answer here.")
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.answer, "Detailed answer here.")

    def test_query_preserved_in_response(self) -> None:
        svc, *_ = _make_service()
        resp = svc.answer(RAGRequest(query="What is dependency injection?"))
        self.assertEqual(resp.query, "What is dependency injection?")

    def test_citations_forwarded_to_response(self) -> None:
        svc, *_ = _make_service()
        resp = svc.answer(RAGRequest(query="q"))
        self.assertIsInstance(resp.citations, list)
        self.assertTrue(len(resp.citations) > 0)

    def test_chunks_retrieved_count_correct(self) -> None:
        hits = [_hit(chunk_index=i) for i in range(3)]
        svc, *_ = _make_service(hits=hits)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.chunks_retrieved, 3)

    def test_chunks_in_context_count_correct(self) -> None:
        hits = [_hit(chunk_index=i) for i in range(4)]
        svc, *_ = _make_service(hits=hits, max_chunks_override=2)
        resp = svc.answer(RAGRequest(query="q"))
        # Assembler limits to max_chunks=2.
        self.assertEqual(resp.chunks_in_context, 2)

    def test_fabricated_url_count_forwarded(self) -> None:
        gen_result = GenerationResult(
            query="q", ok=True, answer="ok", fabricated_url_count=2
        )
        retriever = FakeRetriever(_ok_retrieval(_hit()))
        assembler = FakeAssembler()
        generator = FakeGenerator(gen_result)
        svc = RAGService(retriever, assembler, generator)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.fabricated_url_count, 2)


# ---------------------------------------------------------------------------
# 2. Pipeline stage wiring tests
# ---------------------------------------------------------------------------


class PipelineWiringTests(unittest.TestCase):
    def test_retriever_called_once(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(len(retriever.calls), 1)

    def test_retriever_receives_correct_top_k(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q", top_k=7))
        self.assertEqual(retriever.calls[0].top_k, 7)

    def test_retriever_receives_correct_source_id(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q", source_id="python"))
        self.assertEqual(retriever.calls[0].source_id, "python")

    def test_retriever_receives_none_source_id_for_all_sources(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q", source_id=None))
        self.assertIsNone(retriever.calls[0].source_id)

    def test_retriever_receives_score_threshold(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q", score_threshold=0.7))
        self.assertAlmostEqual(retriever.calls[0].score_threshold, 0.7)

    def test_assembler_called_once(self) -> None:
        svc, _, assembler, _ = _make_service()
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(len(assembler.calls), 1)

    def test_assembler_receives_max_chunks(self) -> None:
        svc, _, assembler, _ = _make_service()
        svc.answer(RAGRequest(query="q", max_chunks=3))
        self.assertEqual(assembler.calls[0][0], 3)

    def test_assembler_receives_token_budget(self) -> None:
        svc, _, assembler, _ = _make_service()
        svc.answer(RAGRequest(query="q", token_budget=1500))
        self.assertEqual(assembler.calls[0][1], 1500)

    def test_generator_called_once(self) -> None:
        svc, _, _, generator = _make_service()
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(len(generator.calls), 1)

    def test_generator_receives_correct_query(self) -> None:
        svc, _, _, generator = _make_service()
        svc.answer(RAGRequest(query="specific question"))
        self.assertEqual(generator.calls[0].query, "specific question")


# ---------------------------------------------------------------------------
# 3. Retrieval failure tests
# ---------------------------------------------------------------------------


class RetrievalFailureTests(unittest.TestCase):
    def test_retrieval_failure_returns_ok_false(self) -> None:
        svc, *_ = _make_service(retrieval_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertFalse(resp.ok)

    def test_retrieval_failure_sets_error_stage(self) -> None:
        svc, *_ = _make_service(retrieval_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.error_stage, "retrieval")

    def test_retrieval_failure_error_message_forwarded(self) -> None:
        retriever = FakeRetriever(_failed_retrieval(error="embed failure injected"))
        svc = RAGService(retriever, FakeAssembler(), FakeGenerator(_ok_generation()))
        resp = svc.answer(RAGRequest(query="q"))
        self.assertIn("embed failure", resp.error)

    def test_retrieval_failure_does_not_call_generator(self) -> None:
        svc, _, _, generator = _make_service(retrieval_fail=True)
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(generator.calls, [])

    def test_retrieval_failure_does_not_call_assembler(self) -> None:
        svc, _, assembler, _ = _make_service(retrieval_fail=True)
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(assembler.calls, [])

    def test_retrieval_failure_answer_is_empty(self) -> None:
        svc, *_ = _make_service(retrieval_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.answer, "")


# ---------------------------------------------------------------------------
# 4. Empty context tests
# ---------------------------------------------------------------------------


class EmptyContextTests(unittest.TestCase):
    def test_zero_hits_returns_ok_false(self) -> None:
        svc, *_ = _make_service(hits=[])
        resp = svc.answer(RAGRequest(query="q"))
        self.assertFalse(resp.ok)

    def test_zero_hits_sets_context_was_empty(self) -> None:
        svc, *_ = _make_service(hits=[])
        resp = svc.answer(RAGRequest(query="q"))
        self.assertTrue(resp.context_was_empty)

    def test_zero_hits_does_not_call_generator(self) -> None:
        svc, _, _, generator = _make_service(hits=[])
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(generator.calls, [])

    def test_zero_hits_error_stage_is_retrieval(self) -> None:
        svc, *_ = _make_service(hits=[])
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.error_stage, "retrieval")

    def test_zero_hits_chunks_retrieved_is_zero(self) -> None:
        svc, *_ = _make_service(hits=[])
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.chunks_retrieved, 0)

    def test_all_chunks_filtered_by_budget_skips_generation(self) -> None:
        """If all hits are too large for the token budget, context is empty."""
        # Each hit has text = 1000 words; budget = 10 words → no chunk fits.
        big_text = " ".join(f"word{i}" for i in range(1000))
        hit = _hit(text=big_text)
        svc, _, _, generator = _make_service(hits=[hit])
        svc.answer(RAGRequest(query="q", token_budget=10))
        self.assertEqual(generator.calls, [])


# ---------------------------------------------------------------------------
# 5. Generation failure tests
# ---------------------------------------------------------------------------


class GenerationFailureTests(unittest.TestCase):
    def test_generation_failure_returns_ok_false(self) -> None:
        svc, *_ = _make_service(generation_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertFalse(resp.ok)

    def test_generation_failure_sets_error_stage(self) -> None:
        svc, *_ = _make_service(generation_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.error_stage, "generation")

    def test_generation_failure_error_forwarded(self) -> None:
        gen_result = _failed_generation(error="provider timeout")
        retriever = FakeRetriever(_ok_retrieval(_hit()))
        svc = RAGService(retriever, FakeAssembler(), FakeGenerator(gen_result))
        resp = svc.answer(RAGRequest(query="q"))
        self.assertIn("provider timeout", resp.error)

    def test_generation_failure_answer_is_empty(self) -> None:
        svc, *_ = _make_service(generation_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.answer, "")

    def test_generation_failure_chunks_retrieved_still_set(self) -> None:
        hits = [_hit(chunk_index=i) for i in range(2)]
        svc, *_ = _make_service(hits=hits, generation_fail=True)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertEqual(resp.chunks_retrieved, 2)


# ---------------------------------------------------------------------------
# 6. Source filtering tests
# ---------------------------------------------------------------------------


class SourceFilteringTests(unittest.TestCase):
    def test_source_id_forwarded_to_retriever(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q", source_id="docker"))
        self.assertEqual(retriever.calls[0].source_id, "docker")

    def test_all_sources_when_source_id_none(self) -> None:
        svc, retriever, *_ = _make_service()
        svc.answer(RAGRequest(query="q"))
        self.assertIsNone(retriever.calls[0].source_id)

    def test_multi_source_hits_all_forwarded_to_assembler(self) -> None:
        hits = [
            _hit(source_id="fastapi", url=_FASTAPI_URL, chunk_index=0),
            _hit(source_id="python", url=_PYTHON_URL, chunk_index=0),
        ]
        svc, _, assembler, _ = _make_service(hits=hits)
        svc.answer(RAGRequest(query="q"))
        self.assertEqual(len(assembler.calls), 1)


# ---------------------------------------------------------------------------
# 7. Never-raise contract tests
# ---------------------------------------------------------------------------


class NeverRaiseTests(unittest.TestCase):
    def test_retrieval_failure_does_not_propagate_exception(self) -> None:
        svc, *_ = _make_service(retrieval_fail=True)
        try:
            resp = svc.answer(RAGRequest(query="q"))
        except Exception as exc:
            self.fail(f"answer() raised unexpectedly: {exc}")
        self.assertFalse(resp.ok)

    def test_generation_failure_does_not_propagate_exception(self) -> None:
        svc, *_ = _make_service(generation_fail=True)
        try:
            resp = svc.answer(RAGRequest(query="q"))
        except Exception as exc:
            self.fail(f"answer() raised unexpectedly: {exc}")
        self.assertFalse(resp.ok)

    def test_empty_hits_does_not_propagate_exception(self) -> None:
        svc, *_ = _make_service(hits=[])
        try:
            resp = svc.answer(RAGRequest(query="q"))
        except Exception as exc:
            self.fail(f"answer() raised unexpectedly: {exc}")
        self.assertFalse(resp.ok)


# ---------------------------------------------------------------------------
# 8. Context truncation propagation tests
# ---------------------------------------------------------------------------


class TruncationTests(unittest.TestCase):
    def test_context_truncated_flag_true_when_assembler_drops_chunks(self) -> None:
        # 3 hits but max_chunks=1 → 2 are truncated.
        hits = [_hit(chunk_index=i) for i in range(3)]
        svc, *_ = _make_service(hits=hits, max_chunks_override=1)
        resp = svc.answer(RAGRequest(query="q"))
        self.assertTrue(resp.context_was_truncated)

    def test_context_truncated_flag_false_when_all_chunks_fit(self) -> None:
        hits = [_hit(chunk_index=0, text="short")]
        gen_result = GenerationResult(
            query="q", ok=True, answer="ok", context_was_truncated=False
        )
        retriever = FakeRetriever(_ok_retrieval(*hits))
        svc = RAGService(retriever, FakeAssembler(), FakeGenerator(gen_result))
        resp = svc.answer(RAGRequest(query="q"))
        self.assertFalse(resp.context_was_truncated)


if __name__ == "__main__":
    unittest.main()
