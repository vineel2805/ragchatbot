from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup

_SKIP_PREFIXES = ("mailto:", "javascript:", "data:", "tel:")
_SKIP_SUFFIXES = (
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ico",
)


def extract_html_links(html: str | bytes, base_url: str) -> list[str]:
    if isinstance(html, bytes):
        markup = html.decode("utf-8", errors="replace")
    else:
        markup = html
    soup = BeautifulSoup(markup, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for tag in soup.find_all("a", href=True):
        href = str(tag.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        lowered = href.lower()
        if lowered.startswith(_SKIP_PREFIXES):
            continue
        absolute = urljoin(base_url, href)
        path = absolute.split("?", 1)[0].split("#", 1)[0].lower()
        if path.endswith(_SKIP_SUFFIXES):
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links
