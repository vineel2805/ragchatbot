from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from app.ingestion.fetch_models import FetchResult, FetchStatus
from app.ingestion.http_client import ROBOTS_ACCEPT, HttpClient
from app.ingestion.retry import get_with_retries
from app.ingestion.sanitize import sanitize_error_message
from app.ingestion.sources.models import SourceDefinition
from app.ingestion.url_security import (
    REASON_CREDENTIALS,
    REASON_HTTPS_REQUIRED,
    REASON_INVALID_HOST,
    REASON_MALFORMED,
    REASON_UNSUPPORTED_SCHEME,
    CanonicalizationError,
    UrlValidationResult,
    canonicalize_url,
)

MAX_ROBOTS_BYTES = 256 * 1024
MAX_ROBOTS_REDIRECTS = 3
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


def validate_robots_txt_url(source: SourceDefinition, url: str) -> UrlValidationResult:
    """Allow only HTTPS robots.txt on an exact registered host. Not the docs allowlist."""
    raw = url.strip()
    if not raw:
        return UrlValidationResult(False, REASON_MALFORMED)
    parsed = urlparse(raw)
    if not parsed.scheme:
        return UrlValidationResult(False, REASON_MALFORMED)
    if parsed.scheme.lower() != "https":
        if parsed.scheme.lower() == "http":
            return UrlValidationResult(False, REASON_HTTPS_REQUIRED)
        return UrlValidationResult(False, REASON_UNSUPPORTED_SCHEME)
    if parsed.username is not None or parsed.password is not None:
        return UrlValidationResult(False, REASON_CREDENTIALS)
    try:
        canonical = canonicalize_url(raw)
    except CanonicalizationError:
        return UrlValidationResult(False, REASON_MALFORMED)
    host = urlparse(canonical).hostname
    if host is None or host not in source.allowed_hosts:
        return UrlValidationResult(False, REASON_INVALID_HOST, canonical)
    path = urlparse(canonical).path or "/"
    if path != "/robots.txt":
        return UrlValidationResult(False, "invalid_robots_path", canonical)
    return UrlValidationResult(True, "ok", canonical)


class RobotsChecker:
    """Fetches and evaluates robots.txt separately from documentation pages."""

    def __init__(
        self,
        client: HttpClient,
        source: SourceDefinition,
        *,
        sleep: Callable[[float], None],
        user_agent: str,
    ) -> None:
        self._client = client
        self._source = source
        self._sleep = sleep
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._failures: dict[str, FetchResult] = {}

    def allowed(self, document_url: str, requested_url: str) -> FetchResult | None:
        """Return a failed/disallowed FetchResult, or None if fetching the doc is allowed."""
        if not self._source.respect_robots_txt:
            return None
        host = urlparse(document_url).hostname
        if host is None:
            return FetchResult(
                ok=False,
                status=FetchStatus.REJECTED,
                reason=REASON_MALFORMED,
                requested_url=requested_url,
                error_type="malformed_url",
            )
        parser = self._parser_for_host(host, requested_url)
        if isinstance(parser, FetchResult):
            return parser
        if parser is None:
            return None
        if parser.can_fetch(self._user_agent, document_url):
            return None
        return FetchResult(
            ok=False,
            status=FetchStatus.ROBOTS_DISALLOWED,
            reason="robots_disallowed",
            requested_url=requested_url,
            canonical_url=document_url,
            final_url=document_url,
            error_type="robots_disallowed",
        )

    def _parser_for_host(self, host: str, requested_url: str) -> RobotFileParser | FetchResult | None:
        if host in self._failures:
            return self._failures[host]
        if host in self._parsers:
            return self._parsers[host]

        current = f"https://{host}/robots.txt"
        validation = validate_robots_txt_url(self._source, current)
        if not validation.allowed or validation.canonical_url is None:
            failure = FetchResult(
                ok=False,
                status=FetchStatus.REJECTED,
                reason=validation.reason,
                requested_url=requested_url,
                error_type=validation.reason,
            )
            self._failures[host] = failure
            return failure
        current = validation.canonical_url

        for _ in range(MAX_ROBOTS_REDIRECTS + 1):
            exchange, _attempts = get_with_retries(
                self._client,
                current,
                max_bytes=MAX_ROBOTS_BYTES,
                accept=ROBOTS_ACCEPT,
                sleep=self._sleep,
            )
            if exchange.size_exceeded or not exchange.completed:
                failure = FetchResult(
                    ok=False,
                    status=FetchStatus.FAILED,
                    reason="robots_fetch_failed",
                    requested_url=requested_url,
                    error_type=exchange.error_type or "robots_fetch_failed",
                    error_message=exchange.error_message
                    or sanitize_error_message("robots.txt fetch failed"),
                )
                self._failures[host] = failure
                return failure

            status = exchange.status_code or 0
            if status in _REDIRECT_STATUS:
                location = exchange.headers.get("location", "")
                hop = validate_robots_txt_url(self._source, urljoin(current, location))
                if not hop.allowed or hop.canonical_url is None:
                    failure = FetchResult(
                        ok=False,
                        status=FetchStatus.REJECTED,
                        reason="rejected_redirect",
                        requested_url=requested_url,
                        error_type="rejected_redirect",
                    )
                    self._failures[host] = failure
                    return failure
                current = hop.canonical_url
                continue

            if status == 404:
                self._parsers[host] = None
                return None
            if status >= 400:
                failure = FetchResult(
                    ok=False,
                    status=FetchStatus.FAILED,
                    reason="robots_fetch_failed",
                    requested_url=requested_url,
                    http_status=status,
                    error_type="robots_fetch_failed",
                )
                self._failures[host] = failure
                return failure

            text = (exchange.content or b"").decode("utf-8", errors="replace")
            parser = RobotFileParser()
            parser.parse(text.splitlines())
            self._parsers[host] = parser
            return parser

        failure = FetchResult(
            ok=False,
            status=FetchStatus.FAILED,
            reason="too_many_redirects",
            requested_url=requested_url,
            error_type="too_many_redirects",
        )
        self._failures[host] = failure
        return failure
