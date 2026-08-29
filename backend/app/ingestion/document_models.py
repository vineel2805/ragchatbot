from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractResult:
    ok: bool
    reason: str
    document_id: str
    canonical_url: str
    source_id: str
    title: str = ""
    extracted_text: str = ""
    extracted_sha256: str = ""
    heading_outline: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_id: str
    canonical_url: str
    title: str
    headings: list[str]
    breadcrumb: str
    text: str
    primary_text: str
    content_hash: str
    extracted_sha256: str
    chunk_index: int
    chunker_version: str
    token_count: int
