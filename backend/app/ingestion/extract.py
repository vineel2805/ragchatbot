from __future__ import annotations

import logging
import re
from dataclasses import replace
from html import unescape

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.document_models import ExtractResult
from app.ingestion.ids import make_document_id, sha256_text
from app.ingestion.normalize import normalize_extracted_text
from app.ingestion.sources.models import SourceDefinition
from app.ingestion.url_security import CanonicalizationError, canonicalize_url

logger = logging.getLogger(__name__)

MIN_EXTRACT_CHARS = 200
_ALWAYS_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "object",
    "embed",
    "form",
    "template",
    "svg",
)
_ALWAYS_STRIP_SELECTORS = (
    "button",
    ".copy",
    ".copied",
    ".headerlink",
    ".linenos",
    ".lineno",
    "[aria-hidden='true']",
)
_HEADINGS = {f"h{i}" for i in range(1, 7)}
_LANG_CLASS = re.compile(r"(?:language|highlight|hljs|lang)-([A-Za-z0-9_+-]+)")
_TITLE_SPLIT = re.compile(r"\s+[|\u2013\u2014-]\s+")


def extract_document(source: SourceDefinition, html: str | bytes, url: str) -> ExtractResult:
    canonical = _canonical(source, url)
    document_id = make_document_id(source.source_id, canonical)
    base = ExtractResult(
        ok=False,
        reason="extract_failed",
        document_id=document_id,
        canonical_url=canonical,
        source_id=source.source_id,
    )
    try:
        markup = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
        if not markup or not markup.strip():
            return replace(base, reason="empty")
        soup = BeautifulSoup(markup, "html.parser")
    except Exception:
        logger.info(
            "ingestion_extract %s",
            {"source_id": source.source_id, "reason": "extract_failed", "url": canonical},
        )
        return base

    _strip_non_document(soup, source.extraction.strip_selectors)
    main = _select_main(soup, source.extraction.content_selectors)
    if main is None:
        logger.info(
            "ingestion_extract %s",
            {"source_id": source.source_id, "reason": "extract_failed", "url": canonical},
        )
        return base

    page_title = _page_title(soup, source)
    markdown = _render(main)
    extracted = normalize_extracted_text(markdown)
    outline = _heading_outline(extracted)
    title = _document_title(extracted, page_title)

    if not extracted:
        return ExtractResult(
            ok=False,
            reason="empty",
            document_id=document_id,
            canonical_url=canonical,
            source_id=source.source_id,
            title=title,
        )
    if len(extracted) < MIN_EXTRACT_CHARS:
        return ExtractResult(
            ok=False,
            reason="too_short",
            document_id=document_id,
            canonical_url=canonical,
            source_id=source.source_id,
            title=title,
            extracted_text=extracted,
            extracted_sha256=sha256_text(extracted),
            heading_outline=outline,
        )

    return ExtractResult(
        ok=True,
        reason="ok",
        document_id=document_id,
        canonical_url=canonical,
        source_id=source.source_id,
        title=title,
        extracted_text=extracted,
        extracted_sha256=sha256_text(extracted),
        heading_outline=outline,
    )


def _canonical(source: SourceDefinition, url: str) -> str:
    try:
        return canonicalize_url(url, keep_query_strings=source.keep_query_strings)
    except CanonicalizationError:
        return url.strip()


def _strip_non_document(soup: BeautifulSoup, extra_selectors: list[str]) -> None:
    for tag_name in _ALWAYS_STRIP_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()
    for selector in (*_ALWAYS_STRIP_SELECTORS, *extra_selectors):
        try:
            for node in soup.select(selector):
                node.decompose()
        except Exception:
            continue


def _select_main(soup: BeautifulSoup, selectors: list[str]) -> Tag | None:
    for selector in selectors:
        try:
            node = soup.select_one(selector)
        except Exception:
            continue
        if node is None:
            continue
        if node.name in {"html", "body"}:
            continue
        return node
    return None


def _page_title(soup: BeautifulSoup, source: SourceDefinition) -> str:
    tag = soup.find("title")
    if tag is None:
        return ""
    raw = " ".join(tag.get_text().split())
    display = source.display_name
    lowered = raw.lower()
    suffix = display.lower()
    if lowered.endswith(suffix):
        trimmed = raw[: -len(display)].rstrip(" |-–—")
        return trimmed.strip() or raw
    parts = _TITLE_SPLIT.split(raw, maxsplit=1)
    if len(parts) == 2 and display.lower() in parts[1].lower():
        return parts[0].strip()
    return raw


def _document_title(markdown: str, page_title: str) -> str:
    for line in markdown.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("#"):
            continue
        if line.strip():
            break
    return page_title.strip()


def _heading_outline(markdown: str) -> list[str]:
    from app.ingestion.normalize import split_fence_segments

    outline: list[str] = []
    for is_fence, segment in split_fence_segments(markdown):
        if is_fence:
            continue
        for line in segment.split("\n"):
            if line.startswith("#") and not line.startswith("#" * 7):
                text = line.lstrip("#").strip()
                if text:
                    outline.append(text)
    return outline


def _render(node: Tag) -> str:
    return _normalize_blocks(_render_children(node))


def _normalize_blocks(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _render_children(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        rendered = _render_node(child)
        if rendered:
            parts.append(rendered)
    return "".join(parts)


def _render_node(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        parent = node.parent.name if node.parent is not None else ""
        if parent in {"pre", "code"}:
            return str(node)
        if not str(node).strip():
            return ""
        return unescape(str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name or ""
    if name in _HEADINGS:
        level = int(name[1])
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            return ""
        return f"{'#' * level} {text}\n\n"
    if name == "pre":
        return _render_pre(node)
    if name == "code":
        inner = unescape(node.get_text())
        return f"`{inner}`"
    if name == "p":
        text = _render_inline(node).strip()
        return f"{text}\n\n" if text else ""
    if name in {"ul", "ol"}:
        return _render_list(node, ordered=(name == "ol"))
    if name == "li":
        text = _render_inline(node).strip()
        return f"{text}\n"
    if name == "br":
        return "\n"
    if name == "hr":
        return "\n---\n\n"
    if name == "blockquote":
        inner = _render_children(node).strip()
        quoted = "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))
        return f"{quoted}\n\n"
    if name == "img":
        alt = (node.get("alt") or "").strip()
        return alt
    if name == "a":
        return _render_link(node)
    if name == "table":
        return _render_table(node)
    if name in {"thead", "tbody", "tfoot", "tr", "td", "th", "span", "div", "section", "article", "main"}:
        return _render_children(node)
    return _render_children(node)


def _render_inline(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(unescape(str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "code":
            parts.append(f"`{unescape(child.get_text())}`")
        elif child.name == "a":
            parts.append(_render_link(child))
        elif child.name == "br":
            parts.append("\n")
        elif child.name == "img":
            parts.append((child.get("alt") or "").strip())
        elif child.name == "pre":
            parts.append(_render_pre(child))
        else:
            parts.append(_render_inline(child))
    return re.sub(r"[ \t]+", " ", "".join(parts))


def _render_link(node: Tag) -> str:
    text = _render_inline(node).strip()
    href = (node.get("href") or "").strip()
    if text and href and not href.startswith("#"):
        return f"[{text}]({href})"
    return text


def _render_list(node: Tag, *, ordered: bool) -> str:
    lines: list[str] = []
    index = 1
    for child in node.children:
        if not isinstance(child, Tag) or child.name != "li":
            continue
        body = _render_inline(child).strip()
        if not body:
            continue
        prefix = f"{index}. " if ordered else "- "
        index += 1
        nested = ""
        for nested_list in child.find_all(["ul", "ol"], recursive=False):
            nested += _render_list(nested_list, ordered=(nested_list.name == "ol"))
        item = prefix + body
        if nested:
            indented = "\n".join("  " + line for line in nested.strip().split("\n"))
            item = f"{item}\n{indented}"
        lines.append(item)
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def _render_pre(node: Tag) -> str:
    work = BeautifulSoup(str(node), "html.parser")
    pre = work.find("pre") or work
    if isinstance(pre, Tag):
        for junk in pre.select("button, .copy, .copied, .linenos, .lineno, .headerlink"):
            junk.decompose()
    code = pre.find("code") if isinstance(pre, Tag) else None
    target = code if isinstance(code, Tag) else pre
    raw = unescape(target.get_text()) if isinstance(target, Tag) else unescape(str(node.get_text()))
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    lang = _code_language(node)
    if isinstance(code, Tag) and not lang:
        lang = _code_language(code)
    marker = "```"
    while marker in raw:
        marker += "`"
    return f"{marker}{lang}\n{raw}\n{marker}\n\n"


def _code_language(node: Tag) -> str:
    classes: list[str] = []
    class_attr = node.get("class") or []
    if isinstance(class_attr, str):
        classes = class_attr.split()
    else:
        classes = [str(item) for item in class_attr]
    for item in classes:
        match = _LANG_CLASS.search(item)
        if match:
            return match.group(1).lower()
    for item in classes:
        lowered = item.lower()
        if lowered in {"python", "py", "js", "javascript", "ts", "typescript", "bash", "sh", "json", "yaml", "go", "rust"}:
            return lowered
    return ""


def _render_table(node: Tag) -> str:
    rows: list[str] = []
    for row in node.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "\n".join(rows) + "\n\n"
