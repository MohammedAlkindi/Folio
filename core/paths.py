from pathlib import Path

# Assumption: this file lives at <project_root>/core/paths.py, so ROOT is one level up.
ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT / "data"
LOGS_DIR: Path = ROOT / "logs"
DB_PATH: Path = DATA_DIR / "folio.db"
CHROMA_DIR: Path = DATA_DIR / "chroma"
MANIFEST_PATH: Path = LOGS_DIR / "manifest.jsonl"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
