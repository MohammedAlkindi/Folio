"""Session-level fixtures for the Folio test suite."""
# sys.path manipulation removed — project must be installed with pip install -e .
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.make_fixtures import generate_all


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
