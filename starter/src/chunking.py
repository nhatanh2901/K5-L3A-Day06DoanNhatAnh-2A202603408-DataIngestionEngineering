"""Fixed-window and paragraph-aware chunking with source offsets."""

import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, end: int) -> Chunk | None:
    raw_text = page.text[start:end]
    leading = len(raw_text) - len(raw_text.lstrip())
    text = raw_text.strip()
    if not text:
        return None

    actual_start = start + leading
    actual_end = actual_start + len(text)
    identity = (
        f"{page.document_id}\0{page.source}\0{page.page}\0"
        f"{actual_start}\0{actual_end}\0{text}"
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "start_offset": actual_start,
            "end_offset": actual_end,
        },
    )


def _split_window(page: DocumentPage, start: int, end: int, size: int,
                  overlap: int) -> list[Chunk]:
    step = size - overlap
    chunks: list[Chunk] = []
    for window_start in range(start, end, step):
        chunk = _chunk(page, window_start, min(window_start + size, end))
        if chunk is not None:
            chunks.append(chunk)
        if window_start + size >= end:
            break
    return chunks


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")
    if not page.text.strip():
        return []
    return _split_window(page, 0, len(page.text), size, overlap)


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Greedily group source lines, splitting oversized lines at word boundaries."""
    if max_size <= 0:
        raise ValueError("Maximum chunk size must be greater than zero.")

    lines = [
        match for match in re.finditer(r"\S[^\n]*", page.text)
    ]
    if not lines:
        return []

    groups: list[tuple[int, int]] = []
    group_start = lines[0].start()
    group_end = lines[0].end()
    for line in lines[1:]:
        if line.end() - group_start <= max_size:
            group_end = line.end()
        else:
            groups.append((group_start, group_end))
            group_start = line.start()
            group_end = line.end()
    groups.append((group_start, group_end))

    chunks: list[Chunk] = []
    for start, end in groups:
        if end - start <= max_size:
            chunk = _chunk(page, start, end)
            if chunk is not None:
                chunks.append(chunk)
            continue

        # Prefer word boundaries for oversized lines, falling back to a hard split.
        cursor = start
        while cursor < end:
            boundary = min(cursor + max_size, end)
            if boundary < end:
                whitespace = page.text.rfind(" ", cursor, boundary)
                if whitespace > cursor:
                    boundary = whitespace
            chunk = _chunk(page, cursor, boundary)
            if chunk is not None:
                chunks.append(chunk)
            cursor = boundary
            while cursor < end and page.text[cursor].isspace():
                cursor += 1
    return chunks
