from __future__ import annotations

import hashlib
from uuid import UUID

CHUNKER_VERSION = "heading-v1"


def make_document_id(source_id: str, canonical_url: str) -> str:
    payload = f"{source_id}\n{canonical_url}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_id_digest(source_id: str, canonical_url: str, chunk_index: int) -> bytes:
    payload = f"{source_id}{canonical_url}{CHUNKER_VERSION}{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def make_chunk_id(source_id: str, canonical_url: str, chunk_index: int) -> str:
    return chunk_id_digest(source_id, canonical_url, chunk_index).hex()


def make_point_id(chunk_id_hex: str) -> str:
    """Deterministic Qdrant point UUID from payload chunk_id (SHA-256 hex).

    First 16 bytes of the digest, with RFC 9562 version 8 and RFC 4122 variant
    bits set so the value is a valid UUID. Payload chunk_id remains the hex.
    """
    digest = bytes.fromhex(chunk_id_hex)
    if len(digest) != 32:
        raise ValueError("chunk_id must be 32-byte SHA-256 hex")
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x80
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(UUID(bytes=bytes(raw)))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
