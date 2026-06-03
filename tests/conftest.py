"""Session-level fixtures and sys.path setup for the Folio test suite."""
import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to sys.path so that `from core.X`, `from store.X`, etc. resolve.
# Assumption: conftest.py lives at <project_root>/tests/conftest.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.fixtures.make_fixtures import generate_all  # noqa: E402 — after sys.path insert


@pytest.fixture(scope="session", autouse=True)
def _generate_fixtures():
    """Generate all fixture files once before any test runs."""
    generate_all()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a fresh, empty SQLite database in a temp directory.

    Tests MUST use this fixture rather than data/folio.db so the production
    database is never touched during a test run.
    """
    return tmp_path / "test_folio.db"


@pytest.fixture()
def cfg() -> Any:
    """Minimal Config object with sensible defaults for unit tests."""
    # Construct a Config without reading a YAML file to keep tests self-contained.
    from core.config import Config

    return Config(
        {
            "workspace": "test",
            "ingestion": {
                "folder": "docs/",
                "supported_extensions": [".pdf", ".txt", ".md"],
                "chunk_size": 50,
                "chunk_overlap": 10,
            },
            "embedding": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
            },
            "retrieval": {"top_k": 3},
            "qa": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 512,
            },
        }
    )
