from __future__ import annotations

import logging
import time
from collections.abc import Callable
from urllib.parse import urljoin

import httpx

from app.core.config import get_settings
from app.ingestion.fetch_models import FetchResult, FetchStatus, HttpExchange
from app.ingestion.http_client import HTML_ACCEPT, XML_ACCEPT, HttpClient, default_timeout
from app.ingestion.retry import MAX_ATTEMPTS, get_with_retries
from app.ingestion.robots import RobotsChecker
from app.ingestion.sanitize import safe_log_fields, sanitize_error_message
from app.ingestion.sources.models import SourceDefinition
from app.ingestion.url_security import validate_redirect, validate_sitemap_url, validate_url

logger = logging.getLogger(__name__)

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_XML_TYPES = frozenset(
    {
        "application/xml",
        "text/xml",
        "application/atom+xml",
        "application/rss+xml",
        "text/plain",
    }
)


def build_user_agent(source: SourceDefinition) -> str:
    settings = get_settings()
    return f"{settings.app_name}/{settings.app_version} (ingestion; {source.source_id})"


def _media_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _is_html(content_type: str | None) -> bool:
    return _media_type(content_type) in _HTML_TYPES


class DocumentationFetcher:
    """Fetch a single allowlisted documentation URL. Does not crawl or parse HTML."""

    def __init__(
        self,
        source: SourceDefinition,
        *,
        user_agent: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
        sleep: Callable[[float], None] | None = None,
        max_bytes: int = MAX_DOCUMENT_BYTES,
        max_attempts: int = MAX_ATTEMPTS,
        client: HttpClient | None = None,
        robots: RobotsChecker | None = None,
    ) -> None:
        self.source = source
        self.user_agent = user_agent or build_user_agent(source)
        self._sleep = sleep or time.sleep
        self._max_bytes = max_bytes
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or HttpClient(
            user_agent=self.user_agent,
            timeout=timeout or default_timeout(),
            transport=transport,
        )
        self._robots = robots or RobotsChecker(
            self._client,
            source,
            sleep=self._sleep,
            user_agent=self.user_agent,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DocumentationFetcher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch(self, url: str) -> FetchResult:
        validation = validate_url(self.source, url)
        if not validation.allowed or validation.canonical_url is None:
            result = FetchResult(
                ok=False,
                status=FetchStatus.REJECTED,
                reason=validation.reason,
                requested_url=url,
                canonical_url=validation.canonical_url,
                error_type=validation.reason,
            )
            self._log(result)
            return result

        current = validation.canonical_url
        total_attempts = 0

        for _hop in range(MAX_REDIRECTS + 1):
            blocked = self._robots.allowed(current, url)
            if blocked is not None:
                self._log(blocked)
                return blocked

            exchange, attempts = get_with_retries(
                self._client,
                current,
                max_bytes=self._max_bytes,
                accept=HTML_ACCEPT,
                sleep=self._sleep,
                max_attempts=self._max_attempts,
            )
            total_attempts += attempts
            result = self._interpret(url, validation.canonical_url, current, exchange, total_attempts)
            if result is not None:
                self._log(result)
                return result

            location = exchange.headers.get("location", "")
            redirected = validate_redirect(self.source, current, location)
            if not redirected.allowed or redirected.canonical_url is None:
                failed = FetchResult(
                    ok=False,
                    status=FetchStatus.REJECTED,
                    reason="rejected_redirect",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=redirected.canonical_url,
                    http_status=exchange.status_code,
                    attempts=total_attempts,
                    error_type=redirected.reason or "rejected_redirect",
                )
                self._log(failed)
                return failed
            current = redirected.canonical_url

        failed = FetchResult(
            ok=False,
            status=FetchStatus.FAILED,
            reason="too_many_redirects",
            requested_url=url,
            canonical_url=validation.canonical_url,
            attempts=total_attempts,
            error_type="too_many_redirects",
        )
        self._log(failed)
        return failed

    def fetch_sitemap(self, url: str, *, allow_child: bool = False) -> FetchResult:
        """Fetch a sitemap over the same HTTP client, robots, and redirect rules.

        Sitemap paths are not documentation pages; they are validated with
        validate_sitemap_url instead of the docs allowlist.
        """
        validation = validate_sitemap_url(self.source, url, allow_child=allow_child)
        if not validation.allowed or validation.canonical_url is None:
            result = FetchResult(
                ok=False,
                status=FetchStatus.REJECTED,
                reason=validation.reason,
                requested_url=url,
                canonical_url=validation.canonical_url,
                error_type=validation.reason,
            )
            self._log(result)
            return result

        current = validation.canonical_url
        total_attempts = 0
        for _hop in range(MAX_REDIRECTS + 1):
            blocked = self._robots.allowed(current, url)
            if blocked is not None:
                self._log(blocked)
                return blocked
            exchange, attempts = get_with_retries(
                self._client,
                current,
                max_bytes=self._max_bytes,
                accept=XML_ACCEPT,
                sleep=self._sleep,
                max_attempts=self._max_attempts,
            )
            total_attempts += attempts
            if exchange.size_exceeded:
                result = FetchResult(
                    ok=False,
                    status=FetchStatus.SKIPPED,
                    reason="size_exceeded",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=current,
                    http_status=exchange.status_code,
                    attempts=total_attempts,
                    error_type="size_exceeded",
                )
                self._log(result)
                return result
            if not exchange.completed:
                result = FetchResult(
                    ok=False,
                    status=FetchStatus.FAILED,
                    reason="retry_exhausted",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=current,
                    attempts=total_attempts,
                    error_type=exchange.error_type or "failed",
                    error_message=exchange.error_message,
                )
                self._log(result)
                return result
            status = exchange.status_code or 0
            if status in _REDIRECT_STATUS:
                location = exchange.headers.get("location", "")
                resolved = urljoin(current, location)
                redirected = validate_sitemap_url(
                    self.source, resolved, allow_child=True
                )
                if not redirected.allowed or redirected.canonical_url is None:
                    failed = FetchResult(
                        ok=False,
                        status=FetchStatus.REJECTED,
                        reason="rejected_redirect",
                        requested_url=url,
                        canonical_url=validation.canonical_url,
                        http_status=status,
                        attempts=total_attempts,
                        error_type=redirected.reason or "rejected_redirect",
                    )
                    self._log(failed)
                    return failed
                current = redirected.canonical_url
                continue
            if status in {404, 410}:
                result = FetchResult(
                    ok=False,
                    status=FetchStatus.GONE,
                    reason="http_404" if status == 404 else "http_410",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=current,
                    http_status=status,
                    attempts=total_attempts,
                    error_type="gone",
                )
                self._log(result)
                return result
            if status >= 400:
                result = FetchResult(
                    ok=False,
                    status=FetchStatus.FAILED,
                    reason=f"http_{status}",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=current,
                    http_status=status,
                    attempts=total_attempts,
                    error_type="http_error",
                )
                self._log(result)
                return result
            body = exchange.content or b""
            media = _media_type(exchange.headers.get("content-type"))
            looks_xml = body.lstrip().startswith((b"<?xml", b"<urlset", b"<sitemapindex", b"<Urlset"))
            if media not in _XML_TYPES and media not in _HTML_TYPES and not looks_xml:
                result = FetchResult(
                    ok=False,
                    status=FetchStatus.SKIPPED,
                    reason="non_xml",
                    requested_url=url,
                    canonical_url=validation.canonical_url,
                    final_url=current,
                    http_status=status,
                    content_type=exchange.headers.get("content-type"),
                    attempts=total_attempts,
                    error_type="non_xml",
                )
                self._log(result)
                return result
            result = FetchResult(
                ok=True,
                status=FetchStatus.FETCHED,
                reason="ok",
                requested_url=url,
                canonical_url=validation.canonical_url,
                final_url=current,
                http_status=status,
                content_type=exchange.headers.get("content-type"),
                body=body,
                etag=exchange.headers.get("etag"),
                last_modified=exchange.headers.get("last-modified"),
                size_bytes=len(body),
                attempts=total_attempts,
            )
            self._log(result)
            return result
        failed = FetchResult(
            ok=False,
            status=FetchStatus.FAILED,
            reason="too_many_redirects",
            requested_url=url,
            canonical_url=validation.canonical_url,
            attempts=total_attempts,
            error_type="too_many_redirects",
        )
        self._log(failed)
        return failed

    def _interpret(
        self,
        requested_url: str,
        original_canonical: str,
        current_url: str,
        exchange: HttpExchange,
        attempts: int,
    ) -> FetchResult | None:
        if exchange.size_exceeded:
            return FetchResult(
                ok=False,
                status=FetchStatus.SKIPPED,
                reason="size_exceeded",
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                http_status=exchange.status_code,
                content_type=exchange.headers.get("content-type"),
                attempts=attempts,
                error_type="size_exceeded",
            )
        if not exchange.completed:
            reason = "retry_exhausted" if attempts >= self._max_attempts else (exchange.error_type or "failed")
            if exchange.error_type == "timeout":
                reason = "timeout" if attempts < self._max_attempts else "retry_exhausted"
            if exchange.error_type == "connection_failure":
                reason = (
                    "connection_failure" if attempts < self._max_attempts else "retry_exhausted"
                )
            if attempts >= self._max_attempts:
                reason = "retry_exhausted"
            return FetchResult(
                ok=False,
                status=FetchStatus.FAILED,
                reason=reason,
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                attempts=attempts,
                error_type=exchange.error_type or reason,
                error_message=exchange.error_message,
            )

        status = exchange.status_code or 0
        if status in _REDIRECT_STATUS:
            return None
        if status in {404, 410}:
            return FetchResult(
                ok=False,
                status=FetchStatus.GONE,
                reason="http_404" if status == 404 else "http_410",
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                http_status=status,
                attempts=attempts,
                error_type="gone",
            )
        if status == 429 or status >= 500:
            return FetchResult(
                ok=False,
                status=FetchStatus.FAILED,
                reason="retry_exhausted",
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                http_status=status,
                attempts=attempts,
                error_type="retry_exhausted",
                error_message=sanitize_error_message(f"HTTP {status}"),
            )
        if status >= 400:
            return FetchResult(
                ok=False,
                status=FetchStatus.FAILED,
                reason=f"http_{status}",
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                http_status=status,
                attempts=attempts,
                error_type="http_error",
            )
        if not _is_html(exchange.headers.get("content-type")):
            return FetchResult(
                ok=False,
                status=FetchStatus.SKIPPED,
                reason="non_html",
                requested_url=requested_url,
                canonical_url=original_canonical,
                final_url=current_url,
                http_status=status,
                content_type=exchange.headers.get("content-type"),
                attempts=attempts,
                error_type="non_html",
            )
        body = exchange.content or b""
        return FetchResult(
            ok=True,
            status=FetchStatus.FETCHED,
            reason="ok",
            requested_url=requested_url,
            canonical_url=original_canonical,
            final_url=current_url,
            http_status=status,
            content_type=exchange.headers.get("content-type"),
            body=body,
            etag=exchange.headers.get("etag"),
            last_modified=exchange.headers.get("last-modified"),
            size_bytes=len(body),
            attempts=attempts,
        )

    def _log(self, result: FetchResult) -> None:
        fields = safe_log_fields(
            source_id=self.source.source_id,
            requested_url=result.requested_url,
            canonical_url=result.canonical_url,
            final_url=result.final_url,
            status=result.status.value,
            reason=result.reason,
            http_status=result.http_status,
            attempts=result.attempts,
            error_type=result.error_type,
        )
        logger.info("ingestion_fetch %s", fields)
