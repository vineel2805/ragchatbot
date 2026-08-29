from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from app.ingestion.catalog import IngestionCatalog, SOFT_DELETE_THRESHOLD
from app.ingestion.catalog_models import IngestionRun, RunStatus, UrlFetchStatus
from app.ingestion.chunker import chunk_document
from app.ingestion.extract import extract_document
from app.ingestion.fetcher import DocumentationFetcher, build_user_agent
from app.ingestion.fetch_models import FetchResult, FetchStatus
from app.ingestion.ids import CHUNKER_VERSION
from app.ingestion.links import extract_html_links
from app.ingestion.rate_limit import RateLimiter
from app.ingestion.sitemap import parse_sitemap_xml
from app.ingestion.sources.models import SourceDefinition
from app.ingestion.url_security import validate_url

MAX_PAGES_PER_SOURCE = 10_000
MAX_CHILD_SITEMAPS = 20


@dataclass
class SourceIngestResult:
    source_id: str
    run: IngestionRun
    urls_registered: int = 0
    urls_fetched: int = 0
    url_failures: int = 0
    sitemaps_failed: int = 0
    errors: list[str] = field(default_factory=list)


def ingest_source(
    source: SourceDefinition,
    catalog: IngestionCatalog,
    *,
    fetcher: DocumentationFetcher | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    rate_limiter: RateLimiter | None = None,
    user_agent: str | None = None,
) -> SourceIngestResult:
    """Bounded discovery + fetch/extract/chunk for one allowlisted source."""
    sleeper = sleep or time.sleep
    clock = monotonic or time.monotonic
    limiter = rate_limiter or RateLimiter(
        source.rate_limit_rps, sleep=sleeper, monotonic=clock
    )
    owned_fetcher = fetcher is None
    active = fetcher or DocumentationFetcher(
        source,
        user_agent=user_agent or build_user_agent(source),
        transport=transport,
        sleep=sleeper,
    )
    run = catalog.create_run(source.source_id)
    result = SourceIngestResult(source_id=source.source_id, run=run)
    queued: set[str] = set()
    fetched: set[str] = set()
    queue: deque[str] = deque()
    seen_registered: set[str] = set()

    def enqueue(raw_url: str, discovered_from: str | None) -> None:
        check = validate_url(source, raw_url)
        if not check.allowed or check.canonical_url is None:
            return
        canonical = check.canonical_url
        catalog.register_url(
            source.source_id,
            canonical,
            run_id=run.id,
            discovered_from=discovered_from,
        )
        if canonical not in seen_registered:
            seen_registered.add(canonical)
            result.urls_registered += 1
        if canonical not in fetched and canonical not in queued:
            queued.add(canonical)
            queue.append(canonical)

    try:
        for seed in source.seed_urls:
            enqueue(seed, "seed")

        for sitemap_url in source.sitemap_urls:
            _ingest_sitemap(
                source,
                catalog,
                active,
                limiter,
                sitemap_url,
                enqueue,
                result,
                allow_child=False,
                remaining_children=[MAX_CHILD_SITEMAPS],
            )

        while queue and len(fetched) < MAX_PAGES_PER_SOURCE:
            url = queue.popleft()
            queued.discard(url)
            if url in fetched:
                continue
            fetched.add(url)
            _process_document(
                source,
                catalog,
                active,
                limiter,
                run.id,
                url,
                enqueue,
                result,
            )

        catalog.record_missing_for_unseen(source.source_id, run.id)
        catalog.apply_soft_deletes(source.source_id, threshold=SOFT_DELETE_THRESHOLD)
        result.run = catalog.finish_run(run.id, succeeded=True)
    except Exception as exc:
        result.errors.append(type(exc).__name__)
        try:
            if catalog.get_run(run.id) and catalog.get_run(run.id).status is RunStatus.STARTED:
                result.run = catalog.finish_run(
                    run.id,
                    succeeded=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        except Exception:
            pass
        raise
    finally:
        if owned_fetcher:
            active.close()
    return result


def _ingest_sitemap(
    source: SourceDefinition,
    catalog: IngestionCatalog,
    fetcher: DocumentationFetcher,
    limiter: RateLimiter,
    sitemap_url: str,
    enqueue,
    result: SourceIngestResult,
    *,
    allow_child: bool,
    remaining_children: list[int],
    seen_sitemaps: set[str] | None = None,
) -> None:
    seen = seen_sitemaps if seen_sitemaps is not None else set()
    if sitemap_url in seen:
        return
    seen.add(sitemap_url)
    try:
        limiter.wait()
        fetched = fetcher.fetch_sitemap(sitemap_url, allow_child=allow_child)
    except Exception:
        result.sitemaps_failed += 1
        return
    if not fetched.ok or fetched.body is None:
        result.sitemaps_failed += 1
        return
    parsed = parse_sitemap_xml(fetched.body)
    if not parsed.ok:
        result.sitemaps_failed += 1
        return
    if parsed.is_index:
        for child in parsed.urls:
            if remaining_children[0] <= 0:
                break
            remaining_children[0] -= 1
            _ingest_sitemap(
                source,
                catalog,
                fetcher,
                limiter,
                child,
                enqueue,
                result,
                allow_child=True,
                remaining_children=remaining_children,
                seen_sitemaps=seen,
            )
        return
    for loc in parsed.urls:
        enqueue(loc, sitemap_url)


def _process_document(
    source: SourceDefinition,
    catalog: IngestionCatalog,
    fetcher: DocumentationFetcher,
    limiter: RateLimiter,
    run_id: str,
    url: str,
    enqueue,
    result: SourceIngestResult,
) -> None:
    try:
        limiter.wait()
        catalog.mark_fetch_started(source.source_id, url, run_id=run_id)
        fetched = fetcher.fetch(url)
        result.urls_fetched += 1
        _record_fetch(catalog, source.source_id, url, run_id, fetched)
        if not fetched.ok or fetched.body is None:
            result.url_failures += 1
            return
        page_url = fetched.final_url or fetched.canonical_url or url
        for href in extract_html_links(fetched.body, page_url):
            enqueue(href, url)
        extracted = extract_document(source, fetched.body, page_url)
        if not extracted.ok:
            catalog.mark_fetch_failed(
                source.source_id,
                url,
                run_id=run_id,
                error_type=extracted.reason,
                error_message=extracted.reason,
                fetch_status=UrlFetchStatus.FETCHED,
            )
            result.url_failures += 1
            return
        if catalog.needs_processing(
            source.source_id,
            url,
            extracted_sha256=extracted.extracted_sha256,
            chunker_version=CHUNKER_VERSION,
        ):
            chunk_document(source, extracted)
            catalog.record_extraction(
                source.source_id,
                url,
                extracted_sha256=extracted.extracted_sha256,
                chunker_version=CHUNKER_VERSION,
                run_id=run_id,
            )
        else:
            catalog.record_extraction(
                source.source_id,
                url,
                extracted_sha256=extracted.extracted_sha256,
                chunker_version=CHUNKER_VERSION,
                run_id=run_id,
            )
    except Exception as exc:
        result.url_failures += 1
        try:
            catalog.mark_fetch_failed(
                source.source_id,
                url,
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception:
            pass


def _record_fetch(
    catalog: IngestionCatalog,
    source_id: str,
    url: str,
    run_id: str,
    fetched: FetchResult,
) -> None:
    if fetched.ok:
        body = fetched.body or b""
        catalog.mark_fetch_succeeded(
            source_id,
            url,
            run_id=run_id,
            http_status=fetched.http_status,
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            content_type=fetched.content_type,
            size_bytes=fetched.size_bytes,
            content_sha256=hashlib.sha256(body).hexdigest(),
        )
        return
    status = UrlFetchStatus(fetched.status.value)
    catalog.mark_fetch_failed(
        source_id,
        url,
        run_id=run_id,
        error_type=fetched.error_type or fetched.reason,
        error_message=fetched.error_message or fetched.reason,
        http_status=fetched.http_status,
        fetch_status=status,
    )
