# Changelog

All notable changes to Folio are documented here.

## [Unreleased]

## [0.1.0] — 2026-06-07

### Added
- Local document ingestion pipeline: parse, chunk, embed, store
- Semantic Q&A with citations traceable to source document and page number
- Forced tool-use via Claude API for structured {answer, sources, confidence} output
- Majority-page attribution for chunks spanning page boundaries
- Context budget guard to prevent silent truncation on large corpora
- Live folder watching with `folio watch` via watchdog
- Multi-workspace support via named ChromaDB collections
- `folio remove` and `folio reindex` commands
- SQLite chunk preview column with `folio list --verbose`
- Specific failure reasons for encrypted and corrupted PDFs
- Exponential backoff retry on all external API calls
- pytest suite with temporary DB isolation — never touches production folio.db
- Installable via `pip install -e .` with `folio` CLI entry point
