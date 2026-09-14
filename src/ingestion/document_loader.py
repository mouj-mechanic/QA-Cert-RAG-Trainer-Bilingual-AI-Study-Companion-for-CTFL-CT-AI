"""
Load PDF, TXT and Markdown files from /resources with certification metadata.

Never invent page numbers or section labels — only extract what is available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from src.config import CERTIFICATIONS, RESOURCES_DIR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Optional chapter/section heuristics from headings like "1.3 Risk-based testing"
_SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.{3,120})$",
    re.MULTILINE,
)


@dataclass
class DocumentPage:
    """One page (or logical page) of extracted text."""

    text: str
    page: int | None
    chapter: str | None = None
    section: str | None = None


@dataclass
class LoadedDocument:
    """A source document with path-derived metadata and page texts."""

    source_file: str
    certification: str
    version: str
    language: str
    document: str
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def _cert_from_folder(folder_name: str) -> tuple[str, str] | None:
    """Map resource folder name → (certification_id, version)."""
    for cert_id, meta in CERTIFICATIONS.items():
        if meta["resource_folder"] == folder_name:
            return cert_id, meta["version"]
    return None


def _document_name(path: Path) -> str:
    return path.stem


def _infer_section_from_text(text: str) -> tuple[str | None, str | None]:
    """Best-effort chapter/section from leading numbered heading. Never invents."""
    match = _SECTION_RE.search(text.strip()[:500])
    if not match:
        return None, None
    number = match.group(1)
    parts = number.split(".")
    chapter = parts[0]
    section = number if len(parts) > 1 else None
    return chapter, section


def _load_pdf(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 — surface and continue indexing
        logger.error("Failed to read PDF %s: %s", path, exc)
        return pages

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract page %s of %s: %s", i, path, exc)
            text = ""
        chapter, section = _infer_section_from_text(text)
        pages.append(
            DocumentPage(text=text.strip(), page=i, chapter=chapter, section=section)
        )
    return pages


def _load_text_file(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    chapter, section = _infer_section_from_text(text)
    # Plain text has no reliable page numbers — leave page as None.
    return [DocumentPage(text=text.strip(), page=None, chapter=chapter, section=section)]


def load_document(
    path: Path,
    certification: str,
    version: str,
    language: str,
    resources_root: Path | None = None,
) -> LoadedDocument:
    """Load a single supported file into a LoadedDocument."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        pages = _load_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    root = resources_root or RESOURCES_DIR
    try:
        source_file = str(path.relative_to(root))
    except ValueError:
        source_file = str(path)

    return LoadedDocument(
        source_file=source_file,
        certification=certification,
        version=version,
        language=language,
        document=_document_name(path),
        pages=pages,
    )


def iter_resource_files(resources_dir: Path | None = None) -> Iterator[tuple[Path, str, str, str]]:
    """
    Yield (path, certification, version, language) for every supported file.

    Expected layout:
      resources/<ctfl|ct_ai>/<en|fr>/<files>
    """
    root = resources_dir or RESOURCES_DIR
    if not root.exists():
        logger.warning("Resources directory does not exist: %s", root)
        return

    for cert_folder in sorted(root.iterdir()):
        if not cert_folder.is_dir() or cert_folder.name.startswith("."):
            continue
        cert_info = _cert_from_folder(cert_folder.name)
        if not cert_info:
            logger.info("Skipping unknown resource folder: %s", cert_folder.name)
            continue
        certification, version = cert_info

        for lang_folder in sorted(cert_folder.iterdir()):
            if not lang_folder.is_dir():
                continue
            language = lang_folder.name.lower()
            if language not in {"en", "fr"}:
                logger.info("Skipping non FR/EN folder: %s", lang_folder)
                continue

            for path in sorted(lang_folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield path, certification, version, language


def load_all_documents(resources_dir: Path | None = None) -> list[LoadedDocument]:
    """Scan resources and load every supported document."""
    root = resources_dir or RESOURCES_DIR
    documents: list[LoadedDocument] = []
    for path, certification, version, language in iter_resource_files(root):
        logger.info("Loading %s (%s/%s)", path.name, certification, language)
        doc = load_document(path, certification, version, language, resources_root=root)
        if not doc.full_text.strip():
            logger.warning("Empty text extracted from %s — skipped", path)
            continue
        documents.append(doc)
    return documents
