import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Reason strings returned on failure — used by callers for user-facing messages.
_REASON_NO_TEXT = "no_text"
_REASON_ENCRYPTED = "encrypted"
_REASON_CORRUPTED = "corrupted"
_REASON_UNSUPPORTED = "unsupported"
_REASON_PARSE_ERROR = "parse_error"


def parse(filepath: Path) -> tuple[list[dict], str | None]:
    """
    Parse a file into a list of {page_number: int | None, text: str}.

    Returns (pages, reason) where reason is None on success or a short string
    describing the failure:
      "no_text"    — file opened but yielded no extractable text
      "encrypted"  — password-protected PDF
      "corrupted"  — malformed or unreadable PDF
      "unsupported"— file extension not handled
      "parse_error"— unexpected exception (details in logs)

    PDF: one dict per page (1-indexed). Pages with no extractable text are skipped.
    TXT/MD: one dict with page_number=None.
    Never raises — returns ([], reason) and logs on any failure.
    Assumption: OCR is out of scope; scanned PDFs with no text layer return ([], "no_text").
    """
    ext = filepath.suffix.lower()
    try:
        if ext == ".pdf":
            return _parse_pdf(filepath)
        elif ext in (".txt", ".md"):
            return _parse_text(filepath)
        else:
            logger.warning("Unsupported extension '%s' for %s — skipping", ext, filepath)
            return ([], _REASON_UNSUPPORTED)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", filepath, exc)
        return ([], _REASON_PARSE_ERROR)


def _parse_pdf(filepath: Path) -> tuple[list[dict], str | None]:
    try:
        import pypdf  # noqa: PLC0415
        import pypdf.errors
    except ImportError:
        logger.error("pypdf not installed — cannot parse PDFs")
        return ([], _REASON_PARSE_ERROR)

    try:
        reader = pypdf.PdfReader(str(filepath))
    except pypdf.errors.PdfReadError as exc:
        logger.error("Corrupted PDF %s: %s", filepath, exc)
        return ([], _REASON_CORRUPTED)
    except pypdf.errors.PdfStreamError as exc:
        logger.error("Corrupted PDF stream %s: %s", filepath, exc)
        return ([], _REASON_CORRUPTED)
    except Exception as exc:
        logger.error("Could not open PDF %s: %s", filepath, exc)
        return ([], _REASON_PARSE_ERROR)

    if reader.is_encrypted:
        logger.warning("Encrypted PDF %s — cannot extract text without password", filepath)
        return ([], _REASON_ENCRYPTED)

    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not extract text from page %d of %s: %s", i + 1, filepath, exc)
            raw = ""

        text = raw.strip()
        if not text:
            continue
        pages.append({"page_number": i + 1, "text": text})

    if not pages:
        return ([], _REASON_NO_TEXT)
    return (pages, None)


def _parse_text(filepath: Path) -> tuple[list[dict], str | None]:
    # errors="replace" prevents UnicodeDecodeError on files with encoding issues.
    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return ([], _REASON_NO_TEXT)
    return ([{"page_number": None, "text": text}], None)
