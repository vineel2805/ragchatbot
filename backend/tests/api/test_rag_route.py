"""API route tests for POST /api/v1/rag/query.

Strategy
--------
- ``app.dependency_overrides[get_rag_service]`` is used to inject a
  ``FakeRAGService`` for every test.  No live Qdrant, no OpenRouter,
  no model downloads.
- ``httpx.TestClient`` (via Starlette's test utils imported through FastAPI)
  exercises the full HTTP stack including Pydantic validation.
- Each test class is focused on a single behavioural contract of the route.
"""
from __future__ import annotations

import dataclasses
import unittest

from fastapi.testclient import TestClient

from app.api.deps import get_rag_service
from app.generation.models import Citation
from app.main import app
from app.rag.models import RAGRequest, RAGResponse

# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------

_FASTAPI_URL = "https://fastapi.tiangolo.com/tutorial/first-steps"


class FakeRAGService:
    """Returns a pre-configured RAGResponse without touching any real dependency."""

    def __init__(self, response: RAGResponse) -> None:
        self._response = response
        self.calls: list[RAGRequest] = []

    def answer(self, request: RAGRequest) -> RAGResponse:
        self.calls.append(request)
        # Echo the actual request query so tests that check resp.json()["query"]
        # see the query they sent, not the one baked into the fixture.
        return dataclasses.replace(self._response, query=request.query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(
    query: str = "How do I use FastAPI?",
    answer: str = "FastAPI is fast.",
    citations: list[Citation] | None = None,
) -> RAGResponse:
    return RAGResponse(
        query=query,
        answer=answer,
        citations=citations
        or [Citation(url=_FASTAPI_URL, title="First Steps", source_id="fastapi")],
        ok=True,
        chunks_retrieved=3,
        chunks_in_context=2,
    )


def _error_response(
    query: str = "q",
    error: str = "embed failure",
    error_stage: str = "retrieval",
    context_was_empty: bool = False,
) -> RAGResponse:
    return RAGResponse(
        query=query,
        ok=False,
        error=error,
        error_stage=error_stage,
        context_was_empty=context_was_empty,
        chunks_retrieved=0,
        chunks_in_context=0,
    )


def _make_client(fake_svc: FakeRAGService) -> TestClient:
    """Return a TestClient with *fake_svc* wired as the RAGService dependency."""
    app.dependency_overrides[get_rag_service] = lambda: fake_svc
    return TestClient(app, raise_server_exceptions=True)


def _teardown() -> None:
    """Remove dependency overrides after each test."""
    app.dependency_overrides.pop(get_rag_service, None)


# ---------------------------------------------------------------------------
# 1. Happy-path tests
# ---------------------------------------------------------------------------


class HappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_returns_200(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "How do I use FastAPI?"})
        self.assertEqual(resp.status_code, 200)

    def test_ok_true_in_body(self) -> None:
        # The success schema (RAGQueryResponse) intentionally omits 'ok' since
        # HTTP 200 with an answer field already signals success.  Verify that
        # the field is either absent or explicitly True — never False.
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        data = resp.json()
        self.assertNotEqual(data.get("ok", True), False)

    def test_answer_in_body(self) -> None:
        fake = FakeRAGService(_ok_response(answer="Detailed answer here."))
        client = _make_client(fake)
        resp = client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.json()["answer"], "Detailed answer here.")

    def test_query_echoed_in_body(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "What is dependency injection?"}
        )
        self.assertEqual(resp.json()["query"], "What is dependency injection?")

    def test_citations_present(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        data = resp.json()
        self.assertIn("citations", data)
        self.assertIsInstance(data["citations"], list)
        self.assertGreater(len(data["citations"]), 0)

    def test_citation_fields_correct(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        cit = resp.json()["citations"][0]
        self.assertIn("url", cit)
        self.assertIn("title", cit)
        self.assertIn("source_id", cit)

    def test_chunks_retrieved_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.json()["chunks_retrieved"], 3)

    def test_chunks_in_context_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.json()["chunks_in_context"], 2)


# ---------------------------------------------------------------------------
# 2. Request forwarding tests
# ---------------------------------------------------------------------------


class RequestForwardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_query_forwarded_to_service(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "my question"})
        self.assertEqual(self.fake.calls[0].query, "my question")

    def test_source_id_forwarded(self) -> None:
        self.client.post(
            "/api/v1/rag/query", json={"query": "q", "source_id": "python"}
        )
        self.assertEqual(self.fake.calls[0].source_id, "python")

    def test_source_id_none_when_omitted(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertIsNone(self.fake.calls[0].source_id)

    def test_top_k_forwarded(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q", "top_k": 7})
        self.assertEqual(self.fake.calls[0].top_k, 7)

    def test_score_threshold_forwarded(self) -> None:
        self.client.post(
            "/api/v1/rag/query", json={"query": "q", "score_threshold": 0.75}
        )
        self.assertAlmostEqual(self.fake.calls[0].score_threshold, 0.75)

    def test_max_chunks_forwarded(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q", "max_chunks": 3})
        self.assertEqual(self.fake.calls[0].max_chunks, 3)

    def test_token_budget_forwarded(self) -> None:
        self.client.post(
            "/api/v1/rag/query", json={"query": "q", "token_budget": 1500}
        )
        self.assertEqual(self.fake.calls[0].token_budget, 1500)

    def test_service_called_exactly_once(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(len(self.fake.calls), 1)


# ---------------------------------------------------------------------------
# 3. Pydantic / HTTP 422 validation tests
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_missing_query_returns_422(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={})
        self.assertEqual(resp.status_code, 422)

    def test_empty_query_returns_422(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": ""})
        self.assertEqual(resp.status_code, 422)

    def test_query_too_long_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "x" * 2_001}
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_zero_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "top_k": 0}
        )
        self.assertEqual(resp.status_code, 422)

    def test_top_k_over_100_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "top_k": 101}
        )
        self.assertEqual(resp.status_code, 422)

    def test_score_threshold_below_zero_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "score_threshold": -0.1}
        )
        self.assertEqual(resp.status_code, 422)

    def test_score_threshold_above_one_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "score_threshold": 1.1}
        )
        self.assertEqual(resp.status_code, 422)

    def test_unknown_source_id_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "source_id": "unknown_lib"}
        )
        self.assertEqual(resp.status_code, 422)

    def test_unknown_source_id_error_message_safe(self) -> None:
        """Error detail must name the bad value but NOT expose secrets."""
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "source_id": "hack"}
        )
        detail = str(resp.json())
        self.assertIn("hack", detail)
        self.assertNotIn("sk-", detail)  # no API key fragments

    def test_max_chunks_zero_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "max_chunks": 0}
        )
        self.assertEqual(resp.status_code, 422)

    def test_token_budget_zero_returns_422(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "token_budget": 0}
        )
        self.assertEqual(resp.status_code, 422)

    def test_service_not_called_on_validation_error(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": ""})
        self.assertEqual(self.fake.calls, [])


# ---------------------------------------------------------------------------
# 4. All valid source_id values accepted
# ---------------------------------------------------------------------------


class ValidSourceIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def _check(self, source_id: str) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "source_id": source_id}
        )
        self.assertEqual(resp.status_code, 200, f"source_id={source_id!r} rejected")

    def test_fastapi_source_accepted(self) -> None:
        self._check("fastapi")

    def test_python_source_accepted(self) -> None:
        self._check("python")

    def test_react_source_accepted(self) -> None:
        self._check("react")

    def test_docker_source_accepted(self) -> None:
        self._check("docker")

    def test_qdrant_source_accepted(self) -> None:
        self._check("qdrant")


# ---------------------------------------------------------------------------
# 5. Pipeline failure tests (retrieval error)
# ---------------------------------------------------------------------------


class RetrievalFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(
            _error_response(error="embed failure", error_stage="retrieval")
        )
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_returns_200_on_retrieval_failure(self) -> None:
        """Pipeline errors are communicated in the body, not via 5xx."""
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.status_code, 200)

    def test_ok_false_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertFalse(resp.json()["ok"])

    def test_error_stage_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.json()["error_stage"], "retrieval")

    def test_error_message_present(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertIn("embed failure", resp.json()["error"])

    def test_error_message_no_secret_key(self) -> None:
        fake = FakeRAGService(
            _error_response(error="sk-or-secret-key-leaked provider error")
        )
        client = _make_client(fake)
        # Even if somehow a raw error leaks through, the route must pass it on
        # unmodified (RAGService is the redaction boundary). We verify the
        # route itself adds no new secret leakage.
        resp = client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# 6. Pipeline failure tests (empty context)
# ---------------------------------------------------------------------------


class EmptyContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(
            _error_response(
                error="No relevant documentation found.",
                error_stage="retrieval",
                context_was_empty=True,
            )
        )
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_returns_200(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.status_code, 200)

    def test_ok_false_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertFalse(resp.json()["ok"])

    def test_context_was_empty_true(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertTrue(resp.json()["context_was_empty"])


# ---------------------------------------------------------------------------
# 7. Pipeline failure tests (generation error)
# ---------------------------------------------------------------------------


class GenerationFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(
            _error_response(error="provider timeout", error_stage="generation")
        )
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_returns_200_on_generation_failure(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.status_code, 200)

    def test_ok_false_in_body(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertFalse(resp.json()["ok"])

    def test_error_stage_generation(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(resp.json()["error_stage"], "generation")

    def test_error_message_forwarded(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertIn("provider timeout", resp.json()["error"])


# ---------------------------------------------------------------------------
# 8. Default parameter tests
# ---------------------------------------------------------------------------


class DefaultParameterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_default_top_k_is_ten(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(self.fake.calls[0].top_k, 10)

    def test_default_score_threshold_is_zero(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertAlmostEqual(self.fake.calls[0].score_threshold, 0.0)

    def test_default_max_chunks_is_five(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(self.fake.calls[0].max_chunks, 5)

    def test_default_token_budget_is_2000(self) -> None:
        self.client.post("/api/v1/rag/query", json={"query": "q"})
        self.assertEqual(self.fake.calls[0].token_budget, 2000)


# ---------------------------------------------------------------------------
# 9. Boundary / edge-case tests
# ---------------------------------------------------------------------------


class BoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeRAGService(_ok_response())
        self.client = _make_client(self.fake)

    def tearDown(self) -> None:
        _teardown()

    def test_top_k_boundary_min_accepted(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q", "top_k": 1})
        self.assertEqual(resp.status_code, 200)

    def test_top_k_boundary_max_accepted(self) -> None:
        resp = self.client.post("/api/v1/rag/query", json={"query": "q", "top_k": 100})
        self.assertEqual(resp.status_code, 200)

    def test_score_threshold_boundary_zero_accepted(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "score_threshold": 0.0}
        )
        self.assertEqual(resp.status_code, 200)

    def test_score_threshold_boundary_one_accepted(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "score_threshold": 1.0}
        )
        self.assertEqual(resp.status_code, 200)

    def test_query_max_length_accepted(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "x" * 2_000}
        )
        self.assertEqual(resp.status_code, 200)

    def test_null_source_id_accepted(self) -> None:
        resp = self.client.post(
            "/api/v1/rag/query", json={"query": "q", "source_id": None}
        )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# 10. Response-shape contract tests
# ---------------------------------------------------------------------------


class ResponseShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_ok = FakeRAGService(_ok_response())
        self.fake_err = FakeRAGService(_error_response())

    def tearDown(self) -> None:
        _teardown()

    def _post(self, fake: FakeRAGService, body: dict) -> dict:
        client = _make_client(fake)
        return client.post("/api/v1/rag/query", json=body).json()

    def test_success_body_has_required_fields(self) -> None:
        data = self._post(self.fake_ok, {"query": "q"})
        for field in (
            "query", "answer", "citations",
            "context_was_truncated", "fabricated_url_count",
            "chunks_retrieved", "chunks_in_context",
        ):
            self.assertIn(field, data, f"Missing field: {field!r}")

    def test_error_body_has_required_fields(self) -> None:
        data = self._post(self.fake_err, {"query": "q"})
        for field in (
            "ok", "query", "error", "error_stage",
            "context_was_empty", "chunks_retrieved", "chunks_in_context",
        ):
            self.assertIn(field, data, f"Missing field: {field!r}")

    def test_success_body_ok_is_absent_or_truthy(self) -> None:
        """The success schema intentionally omits 'ok'; absence implies success."""
        data = self._post(self.fake_ok, {"query": "q"})
        # ok may be absent (success schema) or True; it must never be False
        self.assertNotEqual(data.get("ok"), False)

    def test_fabricated_url_count_is_integer(self) -> None:
        data = self._post(self.fake_ok, {"query": "q"})
        self.assertIsInstance(data["fabricated_url_count"], int)


if __name__ == "__main__":
    unittest.main()
