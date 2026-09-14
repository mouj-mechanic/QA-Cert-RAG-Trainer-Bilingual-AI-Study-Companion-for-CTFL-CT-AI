"""
Character-based chunking with configurable size and overlap.

Baseline (see src.config): CHUNK_SIZE=1000, CHUNK_OVERLAP=200.
These are starting points for measurement — not claimed optima.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.document_loader import LoadedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with retrieval metadata."""

    text: str
    metadata: dict[str, Any]

    def to_chroma(self) -> tuple[str, dict[str, Any]]:
        """Return (document, metadata) suitable for ChromaDB."""
        # Chroma metadata values must be str | int | float | bool
        clean: dict[str, Any] = {}
        for key, value in self.metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return self.text, clean


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, preferring paragraph/sentence breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer break at paragraph, then sentence, then space
            window = text[start:end]
            break_at = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind(" "),
            )
            if break_at > chunk_size * 0.4:
                end = start + break_at + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Create chunks from a loaded document, preserving page metadata when present."""
    results: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        if not page.text.strip():
            continue
        pieces = _split_text(page.text, chunk_size, overlap)
        for piece in pieces:
            metadata = {
                "certification": document.certification,
                "version": document.version,
                "language": document.language,
                "document": document.document,
                "chapter": page.chapter or "",
                "section": page.section or "",
                "page": page.page if page.page is not None else "",
                "source_file": document.source_file,
                "chunk_index": chunk_index,
            }
            results.append(Chunk(text=piece, metadata=metadata))
            chunk_index += 1

    logger.debug(
        "Chunked %s → %s chunks (size=%s, overlap=%s)",
        document.source_file,
        len(results),
        chunk_size,
        overlap,
    )
    return results


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a list of documents."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
