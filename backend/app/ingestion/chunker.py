from __future__ import annotations

import re

from app.ingestion.document_models import DocumentChunk, ExtractResult
from app.ingestion.ids import CHUNKER_VERSION, make_chunk_id, sha256_text
from app.ingestion.normalize import split_fence_segments
from app.ingestion.sources.models import SourceDefinition
from app.ingestion.tokenize import count_tokens, overlap_text

TARGET_TOKENS = 400
OVERLAP_TOKENS = 50

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(source: SourceDefinition, extracted: ExtractResult) -> list[DocumentChunk]:
    if not extracted.ok or not extracted.extracted_text:
        return []

    chunks: list[DocumentChunk] = []
    previous_primary = ""
    index = 0
    for section_headings, body in _iter_sections(extracted.extracted_text):
        headings = _effective_headings(extracted.title, section_headings)
        breadcrumb = make_breadcrumb(source.display_name, extracted.title, headings)
        for primary in _split_section(body):
            if not primary.strip():
                continue
            overlap = overlap_text(previous_primary, OVERLAP_TOKENS) if previous_primary else ""
            stored = _stored_text(breadcrumb, overlap, primary)
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source.source_id, extracted.canonical_url, index),
                    document_id=extracted.document_id,
                    source_id=source.source_id,
                    canonical_url=extracted.canonical_url,
                    title=extracted.title,
                    headings=list(headings),
                    breadcrumb=breadcrumb,
                    text=stored,
                    primary_text=primary,
                    content_hash=sha256_text(primary),
                    extracted_sha256=extracted.extracted_sha256,
                    chunk_index=index,
                    chunker_version=CHUNKER_VERSION,
                    token_count=count_tokens(stored),
                )
            )
            previous_primary = primary
            index += 1
    return chunks


def make_breadcrumb(display_name: str, title: str, headings: list[str]) -> str:
    parts: list[str] = []
    for item in (display_name, title, *headings):
        cleaned = item.strip()
        if cleaned and (not parts or cleaned != parts[-1]):
            parts.append(cleaned)
    return " > ".join(parts)


def _stored_text(breadcrumb: str, overlap: str, primary: str) -> str:
    if overlap:
        return f"{breadcrumb}\n{overlap}\n{primary}"
    return f"{breadcrumb}\n{primary}"


def _effective_headings(title: str, section_headings: list[str]) -> list[str]:
    if section_headings:
        return section_headings
    if title:
        return [title]
    return []


def _iter_sections(markdown: str) -> list[tuple[list[str], str]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], str]] = []
    buf: list[str] = []
    started = False

    def flush() -> None:
        nonlocal started
        body = "\n".join(buf).strip()
        if started or body:
            sections.append(([text for _, text in stack], body))
        buf.clear()
        started = True

    for is_fence, segment in split_fence_segments(markdown):
        if is_fence:
            buf.extend(segment.split("\n"))
            continue
        for line in segment.split("\n"):
            match = _HEADING_LINE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                buf.append(line)
                continue
            buf.append(line)
    flush()
    return [(path, body) for path, body in sections if body]


def _split_section(body: str) -> list[str]:
    expanded: list[str] = []
    for unit in _units(body):
        if count_tokens(unit) > TARGET_TOKENS and not _is_fence(unit):
            expanded.extend(_split_oversize_prose(unit))
        else:
            expanded.append(unit)

    packed: list[str] = []
    current: list[str] = []
    for unit in expanded:
        if _is_fence(unit) and count_tokens(unit) > TARGET_TOKENS:
            if current:
                packed.append("\n\n".join(current))
                current = []
            packed.append(unit)
            continue
        candidate = unit if not current else "\n\n".join([*current, unit])
        if current and count_tokens(candidate) > TARGET_TOKENS:
            packed.append("\n\n".join(current))
            current = [unit]
        elif not current and count_tokens(unit) > TARGET_TOKENS and not _is_fence(unit):
            packed.extend(_split_oversize_prose(unit))
        else:
            current.append(unit)
    if current:
        packed.append("\n\n".join(current))
    return packed


def _units(body: str) -> list[str]:
    units: list[str] = []
    pending_heading = ""
    for is_fence, segment in split_fence_segments(body):
        if is_fence:
            if pending_heading:
                units.append(pending_heading)
                pending_heading = ""
            units.append(segment.strip("\n"))
            continue
        for para in re.split(r"\n{2,}", segment):
            piece = para.strip()
            if not piece:
                continue
            if _HEADING_LINE.match(piece) and "\n" not in piece:
                pending_heading = piece if not pending_heading else f"{pending_heading}\n\n{piece}"
                continue
            if pending_heading:
                piece = f"{pending_heading}\n\n{piece}"
                pending_heading = ""
            units.append(piece)
    if pending_heading:
        units.append(pending_heading)
    return units


def _is_fence(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _split_oversize_prose(text: str) -> list[str]:
    if count_tokens(text) <= TARGET_TOKENS:
        return [text]
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _split_by_tokens(text)
    packed: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if count_tokens(sentence) > TARGET_TOKENS:
            if current:
                packed.append(" ".join(current))
                current = []
            packed.extend(_split_by_tokens(sentence))
            continue
        candidate = sentence if not current else " ".join([*current, sentence])
        if count_tokens(candidate) > TARGET_TOKENS and current:
            packed.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        packed.append(" ".join(current))
    return packed


def _split_by_tokens(text: str) -> list[str]:
    from app.ingestion.tokenize import bge_tokenizer

    tokenizer = bge_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= TARGET_TOKENS:
        return [text]
    pieces: list[str] = []
    start = 0
    total = len(ids)
    while start < total:
        size = min(TARGET_TOKENS, total - start)
        placed = False
        while size > 0:
            decoded = tokenizer.decode(
                ids[start : start + size],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if count_tokens(decoded) <= TARGET_TOKENS:
                pieces.append(decoded)
                start += size
                placed = True
                break
            size -= 1
        if not placed:
            pieces.append(
                tokenizer.decode(
                    ids[start : start + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            start += 1
    return [piece for piece in pieces if piece.strip()]
