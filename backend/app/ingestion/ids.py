from __future__ import annotations

import hashlib

CHUNKER_VERSION = "heading-v1"


def make_document_id(source_id: str, canonical_url: str) -> str:
    payload = f"{source_id}\n{canonical_url}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_chunk_id(source_id: str, canonical_url: str, chunk_index: int) -> str:
    payload = f"{source_id}{canonical_url}{CHUNKER_VERSION}{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
