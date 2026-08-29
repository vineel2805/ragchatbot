from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

_LOC_TAG = "loc"


@dataclass(frozen=True)
class SitemapParseResult:
    ok: bool
    reason: str
    is_index: bool
    urls: tuple[str, ...]


def parse_sitemap_xml(payload: str | bytes) -> SitemapParseResult:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="replace")
    else:
        text = payload
    if not text.strip():
        return SitemapParseResult(False, "malformed_sitemap", False, ())
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return SitemapParseResult(False, "malformed_sitemap", False, ())

    tag = _local(root.tag).lower()
    urls = tuple(loc for loc in _find_locs(root) if loc)
    if tag == "sitemapindex":
        return SitemapParseResult(True, "ok", True, urls)
    if tag == "urlset":
        return SitemapParseResult(True, "ok", False, urls)
    if urls:
        return SitemapParseResult(True, "ok", False, urls)
    return SitemapParseResult(False, "malformed_sitemap", False, ())


def _local(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_locs(root: ET.Element) -> list[str]:
    found: list[str] = []
    for element in root.iter():
        if _local(element.tag).lower() == _LOC_TAG and element.text:
            loc = element.text.strip()
            if loc:
                found.append(loc)
    return found
