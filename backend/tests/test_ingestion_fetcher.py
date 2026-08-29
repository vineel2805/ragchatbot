from __future__ import annotations

import logging
import unittest
from collections.abc import Callable

import httpx

from app.ingestion.fetcher import DocumentationFetcher
from app.ingestion.fetch_models import FetchStatus
from app.ingestion.registry import get_source
from app.ingestion.sanitize import sanitize_error_message
from app.ingestion.url_security import REASON_HTTPS_REQUIRED, REASON_INVALID_HOST, REASON_INVALID_PATH

DOC_URL = "https://fastapi.tiangolo.com/tutorial/first-steps/"
DOC_PATH = "/tutorial/first-steps"
HTML_BODY = b"<html><body><h1>First Steps</h1></body></html>"
USER_AGENT = "DevDocs API/0.1.0 (ingestion; fastapi)"
ALLOW_ROBOTS = "User-agent: *\nAllow: /\n"
DENY_ROBOTS = "User-agent: *\nDisallow: /\n"


class Site:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.robots_body = ALLOW_ROBOTS
        self.robots_status = 200
        self.doc_handler: Callable[[httpx.Request], httpx.Response] | None = None
        self.path_handlers: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.hit_counts: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        self.hit_counts[path] = self.hit_counts.get(path, 0) + 1
        if path == "/robots.txt":
            return httpx.Response(
                self.robots_status,
                text=self.robots_body,
                headers={"content-type": "text/plain"},
            )
        if path in self.path_handlers:
            return self.path_handlers[path](request)
        if self.doc_handler is not None:
            return self.doc_handler(request)
        return httpx.Response(404, text="missing")


def _html(status: int = 200, location: str | None = None, extra: dict[str, str] | None = None) -> httpx.Response:
    headers = {"content-type": "text/html; charset=utf-8"}
    if location is not None:
        headers["location"] = location
    if extra:
        headers.update(extra)
    return httpx.Response(status, content=HTML_BODY, headers=headers)


class FetcherTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.source = get_source("fastapi")
        self.site = Site()
        self.sleeps: list[float] = []

    def _fetcher(self, *, max_bytes: int = 1024 * 1024) -> DocumentationFetcher:
        return DocumentationFetcher(
            self.source,
            user_agent=USER_AGENT,
            transport=httpx.MockTransport(self.site),
            sleep=self.sleeps.append,
            max_bytes=max_bytes,
        )

    def _paths(self) -> list[str]:
        return [request.url.path for request in self.site.requests]

    def _user_agents(self) -> list[str]:
        return [request.headers.get("user-agent", "") for request in self.site.requests]


class SuccessAndRejectionTests(FetcherTestCase):
    def test_successful_html_response(self) -> None:
        self.site.doc_handler = lambda _request: _html()
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, FetchStatus.FETCHED)
        self.assertEqual(result.body, HTML_BODY)
        self.assertEqual(result.http_status, 200)
        self.assertIn("text/html", result.content_type or "")
        self.assertEqual(result.canonical_url, "https://fastapi.tiangolo.com/tutorial/first-steps")
        self.assertIn("/robots.txt", self._paths())
        self.assertIn(DOC_PATH, self._paths())

    def test_non_html_rejection(self) -> None:
        self.site.doc_handler = lambda _request: httpx.Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
        )
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.SKIPPED)
        self.assertEqual(result.reason, "non_html")
        self.assertIsNone(result.body)

    def test_404_handling(self) -> None:
        self.site.doc_handler = lambda _request: httpx.Response(404, text="not found")
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.GONE)
        self.assertEqual(result.http_status, 404)
        self.assertIsNone(result.body)

    def test_initial_disallowed_url(self) -> None:
        with self._fetcher() as fetcher:
            result = fetcher.fetch("https://fastapi.tiangolo.com/about/")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.REJECTED)
        self.assertEqual(result.reason, REASON_INVALID_PATH)
        self.assertEqual(self.site.requests, [])

    def test_response_size_limit(self) -> None:
        self.site.doc_handler = lambda _request: httpx.Response(
            200,
            content=b"x" * 50,
            headers={"content-type": "text/html", "content-length": "5000"},
        )
        with self._fetcher(max_bytes=100) as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.SKIPPED)
        self.assertEqual(result.reason, "size_exceeded")
        self.assertIsNone(result.body)

    def test_response_size_limit_from_body(self) -> None:
        self.site.doc_handler = lambda _request: httpx.Response(
            200,
            content=b"<html>" + b"n" * 200 + b"</html>",
            headers={"content-type": "text/html"},
        )
        with self._fetcher(max_bytes=80) as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertEqual(result.reason, "size_exceeded")
        self.assertIsNone(result.body)


class TransportFailureTests(FetcherTestCase):
    def test_timeout(self) -> None:
        def boom(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out")

        self.site.doc_handler = boom
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.FAILED)
        self.assertEqual(result.reason, "retry_exhausted")
        self.assertEqual(result.error_type, "timeout")
        self.assertEqual(result.attempts, 3)
        self.assertIsNone(result.body)
        self.assertEqual(self.site.hit_counts.get(DOC_PATH), 3)

    def test_connection_failure(self) -> None:
        def boom(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        self.site.doc_handler = boom
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertEqual(result.reason, "retry_exhausted")
        self.assertEqual(result.error_type, "connection_failure")
        self.assertEqual(result.attempts, 3)
        self.assertIsNone(result.body)


class RetryTests(FetcherTestCase):
    def test_429_retry(self) -> None:
        def flaky(_request: httpx.Request) -> httpx.Response:
            if self.site.hit_counts[DOC_PATH] == 1:
                return httpx.Response(429, headers={"retry-after": "1", "content-type": "text/plain"})
            return _html()

        self.site.doc_handler = flaky
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(self.site.hit_counts[DOC_PATH], 2)

    def test_5xx_retry(self) -> None:
        def flaky(_request: httpx.Request) -> httpx.Response:
            if self.site.hit_counts[DOC_PATH] == 1:
                return httpx.Response(503, text="unavailable")
            return _html()

        self.site.doc_handler = flaky
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertTrue(result.ok)
        self.assertEqual(result.http_status, 200)
        self.assertEqual(self.site.hit_counts[DOC_PATH], 2)

    def test_retry_exhaustion(self) -> None:
        self.site.doc_handler = lambda _request: httpx.Response(503, text="unavailable")
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "retry_exhausted")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(self.site.hit_counts[DOC_PATH], 3)
        self.assertIsNone(result.body)

    def test_retry_after_handling(self) -> None:
        def flaky(_request: httpx.Request) -> httpx.Response:
            if self.site.hit_counts[DOC_PATH] == 1:
                return httpx.Response(429, headers={"Retry-After": "9"})
            return _html()

        self.site.doc_handler = flaky
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertTrue(result.ok)
        self.assertEqual(self.sleeps, [9.0])


class RedirectTests(FetcherTestCase):
    def test_redirect_to_allowed_url(self) -> None:
        self.site.path_handlers["/tutorial"] = lambda _r: httpx.Response(
            302,
            headers={"location": "/tutorial/first-steps/"},
        )
        self.site.path_handlers[DOC_PATH] = lambda _r: _html()
        with self._fetcher() as fetcher:
            result = fetcher.fetch("https://fastapi.tiangolo.com/tutorial/")
        self.assertTrue(result.ok)
        self.assertEqual(result.final_url, "https://fastapi.tiangolo.com/tutorial/first-steps")
        self.assertEqual(result.body, HTML_BODY)

    def test_redirect_to_disallowed_url(self) -> None:
        self.site.doc_handler = lambda _r: httpx.Response(
            302,
            headers={"location": "https://evil.example/tutorial/first-steps/"},
        )
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.REJECTED)
        self.assertEqual(result.reason, "rejected_redirect")
        self.assertEqual(result.error_type, REASON_INVALID_HOST)
        self.assertIsNone(result.body)

    def test_http_redirect_rejection(self) -> None:
        self.site.doc_handler = lambda _r: httpx.Response(
            301,
            headers={"location": "http://fastapi.tiangolo.com/tutorial/first-steps/"},
        )
        with self._fetcher() as fetcher:
            result = fetcher.fetch("https://fastapi.tiangolo.com/tutorial/")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.REJECTED)
        self.assertEqual(result.error_type, REASON_HTTPS_REQUIRED)


class RobotsTests(FetcherTestCase):
    def test_robots_txt_allowed(self) -> None:
        self.site.robots_body = ALLOW_ROBOTS
        self.site.doc_handler = lambda _r: _html()
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertTrue(result.ok)
        self.assertEqual(self._paths()[0], "/robots.txt")

    def test_robots_txt_disallowed(self) -> None:
        self.site.robots_body = DENY_ROBOTS
        self.site.doc_handler = lambda _r: _html()
        with self._fetcher() as fetcher:
            result = fetcher.fetch(DOC_URL)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, FetchStatus.ROBOTS_DISALLOWED)
        self.assertEqual(self._paths(), ["/robots.txt"])
        self.assertNotIn(DOC_PATH, self._paths())
        self.assertIsNone(result.body)


class SafetyTests(FetcherTestCase):
    def test_user_agent_presence(self) -> None:
        self.site.doc_handler = lambda _r: _html()
        with self._fetcher() as fetcher:
            fetcher.fetch(DOC_URL)
        self.assertTrue(self.site.requests)
        for agent in self._user_agents():
            self.assertEqual(agent, USER_AGENT)
            self.assertIn("DevDocs API", agent)
            self.assertIn("0.1.0", agent)

    def test_no_sensitive_data_in_errors_or_logging(self) -> None:
        secret = "super-secret-token-xyz"
        records: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))
                records.append(str(record.__dict__))

        handler = Capture()
        log = logging.getLogger("app.ingestion.fetcher")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            def boom(_request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError(
                    f"connection refused Authorization: Bearer {secret} Cookie: session={secret}"
                )

            self.site.doc_handler = boom
            with self._fetcher() as fetcher:
                result = fetcher.fetch(DOC_URL)
        finally:
            log.removeHandler(handler)

        self.assertIsNotNone(result.error_message)
        blob = " ".join(records) + " " + (result.error_message or "")
        self.assertNotIn(secret, blob)
        self.assertNotIn("session=", result.error_message or "")
        sanitized = sanitize_error_message(
            f"Authorization: Bearer {secret} Cookie: session={secret}"
        )
        self.assertNotIn(secret, sanitized)
        self.assertIn("<redacted>", sanitized)
        self.assertIsNone(result.body)
        for request in self.site.requests:
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("cookie", request.headers)


if __name__ == "__main__":
    unittest.main()
