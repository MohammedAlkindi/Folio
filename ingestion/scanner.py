import hashlib
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_folder(folder: str, extensions: list[str]) -> list[Path]:
    """
    Recursively walk folder and return sorted paths matching extensions.
    Hidden files (name starts with '.') are skipped.
    """
    root = Path(folder)
    if not root.exists():
        logger.warning("Folder does not exist: %s", root)
        return []

    # Normalise extensions to lowercase with leading dot.
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    counts: defaultdict[str, int] = defaultdict(int)
    found: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in ext_set:
            found.append(path)
            counts[path.suffix.lower()] += 1

    found.sort()
    for ext, count in sorted(counts.items()):
        logger.info("Found %d %s file(s)", count, ext)
    logger.info("Total files found: %d", len(found))
    return found


def doc_id(filepath: Path) -> str:
    """Stable SHA-256 of the absolute path string — same file = same ID every run."""
    return hashlib.sha256(str(filepath.resolve()).encode()).hexdigest()
