from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from app.ingestion.fetch_models import HttpExchange
from app.ingestion.http_client import HttpClient

MAX_ATTEMPTS = 3
MAX_RETRY_AFTER_SECONDS = 60.0
BASE_BACKOFF_SECONDS = 0.5

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_ERRORS = frozenset({"timeout", "connection_failure", "http_error"})


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        return min(float(raw), MAX_RETRY_AFTER_SECONDS)
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        delay = (when - datetime.now(timezone.utc)).total_seconds()
        return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)
    except (TypeError, ValueError, OverflowError):
        return None


def backoff_seconds(attempt: int) -> float:
    return min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_AFTER_SECONDS)


def should_retry(exchange: HttpExchange, attempt: int, max_attempts: int = MAX_ATTEMPTS) -> RetryDecision:
    if attempt >= max_attempts:
        return RetryDecision(False, 0.0)
    if exchange.error_type in _RETRYABLE_ERRORS:
        return RetryDecision(True, backoff_seconds(attempt))
    if exchange.status_code in _RETRYABLE_STATUS:
        delay = parse_retry_after(exchange.headers.get("retry-after"))
        if delay is None:
            delay = backoff_seconds(attempt)
        return RetryDecision(True, delay)
    return RetryDecision(False, 0.0)


def get_with_retries(
    client: HttpClient,
    url: str,
    *,
    max_bytes: int,
    accept: str | None = None,
    sleep: Callable[[float], None],
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[HttpExchange, int]:
    last: HttpExchange | None = None
    for attempt in range(1, max_attempts + 1):
        last = client.get(url, max_bytes=max_bytes, accept=accept)
        decision = should_retry(last, attempt, max_attempts=max_attempts)
        if not decision.retry:
            return last, attempt
        if decision.delay_seconds > 0:
            sleep(decision.delay_seconds)
    assert last is not None
    return last, max_attempts
