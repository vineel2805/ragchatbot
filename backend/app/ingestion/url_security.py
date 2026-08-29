from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse

from app.ingestion.sources.models import SourceDefinition

_DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}

_PYTHON_LIBRARY_ROOT = "/3/library"

REASON_OK = "ok"
REASON_MALFORMED = "malformed_url"
REASON_HTTPS_REQUIRED = "https_required"
REASON_UNSUPPORTED_SCHEME = "unsupported_scheme"
REASON_CREDENTIALS = "credentials_rejected"
REASON_INVALID_HOST = "invalid_host"
REASON_INVALID_PATH = "invalid_path"


class CanonicalizationError(ValueError):
    """Raised when a URL cannot be parsed into a canonical form."""


@dataclass(frozen=True)
class UrlValidationResult:
    allowed: bool
    reason: str
    canonical_url: str | None = None


def canonicalize_url(url: str, *, keep_query_strings: bool = False) -> str:
    """Normalize a URL per the ingestion design (no fetch).

    - lowercase scheme and host
    - strip credentials from the authority (callers must still reject them)
    - drop default ports
    - collapse duplicate slashes in the path
    - drop fragment
    - drop query string unless keep_query_strings is True
    - drop trailing slash except for path '/'
    """
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.hostname:
        raise CanonicalizationError(f"URL is missing scheme or host: {url!r}")

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    netloc = host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"

    path = parsed.path if parsed.path else "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query = parsed.query if keep_query_strings else ""
    return urlunparse((scheme, netloc, path, "", query, ""))


def validate_url(source: SourceDefinition, url: str) -> UrlValidationResult:
    """Return whether url is allowed for this source (HTTPS, host, path, no credentials)."""
    raw = url.strip()
    if not raw:
        return UrlValidationResult(False, REASON_MALFORMED)

    parsed = urlparse(raw)
    if not parsed.scheme:
        return UrlValidationResult(False, REASON_MALFORMED)
    if parsed.scheme.lower() != "https":
        if parsed.scheme.lower() in {"http"}:
            return UrlValidationResult(False, REASON_HTTPS_REQUIRED)
        return UrlValidationResult(False, REASON_UNSUPPORTED_SCHEME)
    if parsed.username is not None or parsed.password is not None:
        return UrlValidationResult(False, REASON_CREDENTIALS)
    if not parsed.hostname:
        return UrlValidationResult(False, REASON_MALFORMED)

    try:
        canonical = canonicalize_url(raw, keep_query_strings=source.keep_query_strings)
    except CanonicalizationError:
        return UrlValidationResult(False, REASON_MALFORMED)

    canonical_parsed = urlparse(canonical)
    host = canonical_parsed.hostname
    if host is None or host not in source.allowed_hosts:
        return UrlValidationResult(False, REASON_INVALID_HOST, canonical)

    path = canonical_parsed.path or "/"
    if not _path_allowed(source, path):
        return UrlValidationResult(False, REASON_INVALID_PATH, canonical)

    return UrlValidationResult(True, REASON_OK, canonical)


def validate_sitemap_url(
    source: SourceDefinition,
    url: str,
    *,
    allow_child: bool = False,
) -> UrlValidationResult:
    """Allow registered sitemap URLs (and same-host .xml children of a sitemap index)."""
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
        canonical = canonicalize_url(raw, keep_query_strings=source.keep_query_strings)
    except CanonicalizationError:
        return UrlValidationResult(False, REASON_MALFORMED)
    host = urlparse(canonical).hostname
    if host is None or host not in source.allowed_hosts:
        return UrlValidationResult(False, REASON_INVALID_HOST, canonical)
    listed: set[str] = set()
    for sitemap in source.sitemap_urls:
        try:
            listed.add(canonicalize_url(sitemap, keep_query_strings=source.keep_query_strings))
        except CanonicalizationError:
            continue
    if canonical in listed:
        return UrlValidationResult(True, REASON_OK, canonical)
    path = urlparse(canonical).path or "/"
    if allow_child and (path.endswith(".xml") or "sitemap" in path.lower()):
        return UrlValidationResult(True, REASON_OK, canonical)
    return UrlValidationResult(False, REASON_INVALID_PATH, canonical)


def validate_redirect(
    source: SourceDefinition,
    from_url: str,
    location: str,
) -> UrlValidationResult:
    """Validate a redirect target against the same source allowlist.

    Resolves relative Location values against from_url. Does not perform HTTP.
    """
    if location is None or not str(location).strip():
        return UrlValidationResult(False, REASON_MALFORMED)
    resolved = urljoin(from_url, location.strip())
    return validate_url(source, resolved)


def _path_allowed(source: SourceDefinition, path: str) -> bool:
    if path in source.allowed_exact_paths:
        return True
    for prefix in source.allowed_path_prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    if source.library_allowlist and _python_library_path_allowed(path, source.library_allowlist):
        return True
    return False


def _python_library_path_allowed(path: str, stems: list[str]) -> bool:
    if path == _PYTHON_LIBRARY_ROOT or path == f"{_PYTHON_LIBRARY_ROOT}/index.html":
        return True
    prefix = _PYTHON_LIBRARY_ROOT + "/"
    if not path.startswith(prefix):
        return False
    rest = path.removeprefix(prefix)
    if not rest or "/" in rest:
        return False
    name = rest[:-5] if rest.endswith(".html") else rest
    if name in {"", "index"}:
        return True
    for stem in stems:
        if name == stem or name.startswith(stem + "-") or name.startswith(stem + "."):
            return True
    return False
