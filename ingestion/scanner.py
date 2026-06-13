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


def content_hash(filepath: Path) -> str:
    """SHA-256 of file content bytes only — used to detect when a file has been modified."""
    return hashlib.sha256(filepath.resolve().read_bytes()).hexdigest()


def doc_id(filepath: Path) -> str:
    """
    SHA-256 of file content bytes combined with the absolute path.
    Changes when the file is modified. Unique per file even when two files share
    identical content (filepath is the tiebreaker).
    # Assumption: content + path combined hash satisfies both change-detection
    # and per-file uniqueness requirements.
    """
    resolved = filepath.resolve()
    data = resolved.read_bytes() + b"\x00" + str(resolved).encode()
    return hashlib.sha256(data).hexdigest()


def path_hash(filepath: Path) -> str:
    """SHA-256 of the absolute path string — legacy behaviour from before content hashing. Migration use only."""
    return hashlib.sha256(str(filepath.resolve()).encode()).hexdigest()
