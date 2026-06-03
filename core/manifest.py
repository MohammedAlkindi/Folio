import json
import logging
from datetime import datetime, timezone

from core.paths import MANIFEST_PATH, ensure_dirs

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(record: dict) -> None:
    ensure_dirs()
    try:
        with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        # Manifest write failure is non-fatal — log and continue.
        logger.warning("Could not write manifest entry: %s", exc)


def ingest_start(folder: str) -> None:
    _append({"event": "ingest_start", "folder": folder, "timestamp": _now()})


def file_ingested(path: str, chunks: int, status: str) -> None:
    _append(
        {
            "event": "file_ingested",
            "path": path,
            "chunks": chunks,
            "status": status,
            "timestamp": _now(),
        }
    )


def file_skipped(path: str, reason: str) -> None:
    _append(
        {
            "event": "file_skipped",
            "path": path,
            "reason": reason,
            "timestamp": _now(),
        }
    )


def ingest_complete(total_files: int, total_chunks: int) -> None:
    _append(
        {
            "event": "ingest_complete",
            "total_files": total_files,
            "total_chunks": total_chunks,
            "timestamp": _now(),
        }
    )


def query_logged(question: str, sources_used: list[str]) -> None:
    _append(
        {
            "event": "query",
            "question": question,
            "sources_used": sources_used,
            "timestamp": _now(),
        }
    )
