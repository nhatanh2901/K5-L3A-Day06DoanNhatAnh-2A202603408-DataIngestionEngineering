"""Read PDF pages while preserving page and section provenance."""

import re
import unicodedata
from pathlib import Path

from .models import DocumentPage


_CHAPTER_PATTERN = re.compile(
    r"CHƯƠNG\s+([0-9]+)\s*[.\-–:]?\s*([^\n]+)", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without joining separate paragraphs."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    lines = [re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _section_for_page(text: str, previous_section: str) -> str:
    chapter = _CHAPTER_PATTERN.search(text[:300])
    if chapter:
        title = re.split(r"\s+\d+\.\d+(?:\.\d+)?\b", chapter.group(2), maxsplit=1)[0]
        return f"Chương {chapter.group(1)} - {title.strip(' .:-').casefold()}"
    if re.search(r"\bTÀI LIỆU THAM KHẢO\b", text[:250], re.IGNORECASE):
        return "Tài liệu tham khảo"
    if re.search(r"\bKẾT LUẬN\b", text[:250], re.IGNORECASE):
        return "Kết luận"
    return previous_section or "Chưa phân loại"


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page, keeping its printed position and section."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing dependency is missing; install starter/requirements.txt."
        ) from exc

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    pages: list[DocumentPage] = []
    section = ""
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        section = _section_for_page(text, section)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "section": section,
                    "physical_page": page_number,
                },
            )
        )
    return pages


def is_content_page(page: DocumentPage, minimum_characters: int = 150) -> bool:
    """Exclude empty/short pages and bibliography-only pages from corpus counts."""
    if len(page.text) < minimum_characters:
        return False
    return page.metadata.get("section") != "Tài liệu tham khảo"
