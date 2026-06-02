from pathlib import Path

# Assumption: this file lives at folio/core/paths.py, so ROOT is two levels up.
ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = ROOT / "data"
LOGS_DIR: Path = ROOT / "logs"
DB_PATH: Path = DATA_DIR / "folio.db"
CHROMA_DIR: Path = DATA_DIR / "chroma"
MANIFEST_PATH: Path = LOGS_DIR / "manifest.jsonl"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
