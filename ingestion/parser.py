import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse(filepath: Path) -> list[dict]:
    """
    Parse a file into a list of {page_number: int | None, text: str}.

    PDF: one dict per page (1-indexed). Pages with no extractable text are skipped.
    TXT/MD: one dict with page_number=None.
    Never raises — returns [] and logs on any failure.
    Assumption: OCR is out of scope; scanned PDFs with no text layer return [].
    """
    ext = filepath.suffix.lower()
    try:
        if ext == ".pdf":
            return _parse_pdf(filepath)
        elif ext in (".txt", ".md"):
            return _parse_text(filepath)
        else:
            logger.warning("Unsupported extension '%s' for %s — skipping", ext, filepath)
            return []
    except Exception as exc:
        logger.error("Failed to parse %s: %s", filepath, exc)
        return []


def _parse_pdf(filepath: Path) -> list[dict]:
    try:
        import pypdf  # noqa: PLC0415 — lazy import to avoid hard dep at module level
    except ImportError:
        logger.error("pypdf not installed — cannot parse PDFs")
        return []

    pages: list[dict] = []
    try:
        reader = pypdf.PdfReader(str(filepath))
    except Exception as exc:
        logger.error("Could not open PDF %s: %s", filepath, exc)
        return []

    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not extract text from page %d of %s: %s", i + 1, filepath, exc)
            raw = ""

        text = raw.strip()
        if not text:
            # Assumption: empty page after strip means no text layer — skip silently.
            continue
        pages.append({"page_number": i + 1, "text": text})

    return pages


def _parse_text(filepath: Path) -> list[dict]:
    # errors="replace" prevents UnicodeDecodeError on files with encoding issues.
    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [{"page_number": None, "text": text}]
