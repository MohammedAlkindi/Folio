"""Tests for ingestion/parser.py — uses real fixture files, no mocking."""
from pathlib import Path

import pytest

from ingestion.parser import parse

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_parse_txt():
    pages, reason = parse(FIXTURES_DIR / "sample.txt")
    assert isinstance(pages, list)
    assert len(pages) == 1
    assert pages[0]["page_number"] is None
    assert len(pages[0]["text"]) > 0
    assert reason is None


def test_parse_empty_txt(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    pages, reason = parse(empty)
    assert pages == []
    assert reason == "no_text"


def test_parse_unsupported_ext(tmp_path: Path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    pages, reason = parse(csv_file)
    assert pages == []
    assert reason == "unsupported"


def test_parse_does_not_raise(tmp_path: Path):
    # A file that might trigger a parse error must return [] rather than raising.
    # Assumption: a .txt file with binary-like content will not crash the parser.
    bad_file = tmp_path / "bad.txt"
    bad_file.write_bytes(b"\xff\xfe some text \x00\x01")
    pages, reason = parse(bad_file)
    # Either [] (empty after strip) or a list with text — must not raise.
    assert isinstance(pages, list)
