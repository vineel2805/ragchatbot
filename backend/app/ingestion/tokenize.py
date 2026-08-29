from __future__ import annotations

from functools import lru_cache

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def bge_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    tokenizer.model_max_length = 100_000
    return tokenizer


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(bge_tokenizer().encode(text, add_special_tokens=False))


def overlap_text(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    tokenizer = bge_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""
    tail = ids[-overlap_tokens:]
    return tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False)
