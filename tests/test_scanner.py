"""Tests for ingestion/scanner.py — uses real fixture files, no mocking."""
from pathlib import Path

import pytest

from ingestion.scanner import doc_id, scan_folder

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_scan_finds_txt():
    results = scan_folder(str(FIXTURES_DIR), [".txt"])
    names = [p.name for p in results]
    assert "sample.txt" in names


def test_scan_skips_hidden(tmp_path: Path):
    hidden = tmp_path / ".hidden.txt"
    hidden.write_text("secret", encoding="utf-8")
    visible = tmp_path / "visible.txt"
    visible.write_text("visible", encoding="utf-8")

    results = scan_folder(str(tmp_path), [".txt"])
    names = [p.name for p in results]
    assert "visible.txt" in names
    assert ".hidden.txt" not in names


def test_doc_id_stable():
    path = FIXTURES_DIR / "sample.txt"
    assert doc_id(path) == doc_id(path)


def test_doc_id_different_paths():
    path_a = FIXTURES_DIR / "sample.txt"
    path_b = FIXTURES_DIR / "sample_multipage.txt"
    # sample_multipage.txt is generated at session start by make_fixtures.generate_all().
    assert doc_id(path_a) != doc_id(path_b)


def test_missing_folder():
    results = scan_folder("/nonexistent/path/that/does/not/exist", [".txt"])
    assert results == []


def test_doc_id_unique_across_paths_same_content(tmp_path: Path):
    """doc_id incorporates the filepath, so two files with identical content produce different IDs."""
    data = b"identical file content for testing"
    path_a = tmp_path / "file_a.txt"
    path_b = tmp_path / "file_b.txt"
    path_a.write_bytes(data)
    path_b.write_bytes(data)
    # Assumption: doc_id = SHA256(content + "\x00" + abs_path) ensures per-file uniqueness.
    assert doc_id(path_a) != doc_id(path_b)


def test_content_hash_detects_change(tmp_path: Path):
    """content_hash changes when file content changes, enabling automatic reindex detection."""
    from ingestion.scanner import content_hash
    f = tmp_path / "file.txt"
    f.write_bytes(b"version one")
    h1 = content_hash(f)
    f.write_bytes(b"version two")
    h2 = content_hash(f)
    assert h1 != h2


def test_content_hash_stable(tmp_path: Path):
    """content_hash is deterministic for unchanged content."""
    from ingestion.scanner import content_hash
    f = tmp_path / "stable.txt"
    f.write_bytes(b"stable content")
    assert content_hash(f) == content_hash(f)
