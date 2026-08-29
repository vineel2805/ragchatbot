from __future__ import annotations

import re
import unicodedata

_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")


def split_fence_segments(text: str) -> list[tuple[bool, str]]:
    """Split markdown into (is_fence, segment) pairs. Fence segments include markers."""
    lines = text.split("\n")
    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    def flush(is_fence: bool) -> None:
        if buf:
            segments.append((is_fence, "\n".join(buf)))
            buf.clear()

    for line in lines:
        if in_fence:
            buf.append(line)
            stripped = line.rstrip()
            if (
                stripped.startswith(fence_char * fence_len)
                and set(stripped.strip() or fence_char) <= {fence_char}
                and len(stripped.strip()) >= fence_len
            ):
                flush(True)
                in_fence = False
            continue
        match = _FENCE_OPEN.match(line)
        if match:
            flush(False)
            in_fence = True
            fence_char = match.group(1)[0]
            fence_len = len(match.group(1))
            buf.append(line)
            continue
        buf.append(line)
    flush(in_fence)
    return segments


def normalize_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    pieces: list[str] = []
    for is_fence, segment in split_fence_segments(text):
        if is_fence:
            pieces.append(segment)
            continue
        lines = [line.rstrip() for line in segment.split("\n")]
        collapsed: list[str] = []
        blank_run = 0
        for line in lines:
            if line == "":
                blank_run += 1
                if blank_run <= 2:
                    collapsed.append(line)
            else:
                blank_run = 0
                collapsed.append(line)
        pieces.append("\n".join(collapsed))
    normalized = "\n".join(pieces)
    return normalized.strip()
