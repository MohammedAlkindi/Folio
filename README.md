# Folio

Folio is a local document intelligence pipeline. It ingests a folder of PDFs and text files,
chunks and embeds them using a local model, and lets you query across the entire corpus with
answers that include traceable citations to the source document and page number. Nothing leaves
your machine — no cloud storage, no remote embeddings.

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/yourname/folio.git
cd folio

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the example config and adjust paths
cp config/folio_config.example.yaml config/folio_config.yaml
# Edit config/folio_config.yaml — at minimum set ingestion.folder

# 5. Set your Anthropic API key (only needed for the query command)
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Ingest a folder

```bash
python -m folio.cli.main ingest --config config/folio_config.yaml
```

Or, if you add a `[project.scripts]` entry in `pyproject.toml`:

```bash
folio ingest --config config/folio_config.yaml
```

Sample output:

```
Scanned:       42 files
Ingested:      38 new
Skipped:        4 (already indexed)
Failed:         0
Chunks:     1,847 total
```

Ingestion is **idempotent** — re-running on the same folder produces 0 new, N skipped.

---

## Query

```bash
python -m folio.cli.main query --config config/folio_config.yaml
```

You will enter an interactive prompt:

```
Folio — ask questions about your documents. Type 'quit' to exit.

> What were the key findings in the Q3 report?

The Q3 report highlights a 12% revenue increase driven by...
(Source: q3_report.pdf, p.4)

Sources: q3_report.pdf (p.4), executive_summary.pdf (p.1)
Confidence: high
```

Type `quit` or press `Ctrl+C` to exit.

---

## List ingested documents

```bash
python -m folio.cli.main list --config config/folio_config.yaml
```

```
Filename                                  Pages   Chunks  Ingested
---------------------------------------- ------  ------- ----------
q3_report.pdf                                24      187  2026-06-01
notes.md                                      —       14  2026-06-01
```

---

## Supported file types

| Extension | How it is parsed |
|-----------|-----------------|
| `.pdf`    | Page-by-page text extraction via pypdf. Pages with no text layer are skipped. |
| `.txt`    | Full file read as UTF-8 (invalid bytes replaced). |
| `.md`     | Same as `.txt` — Markdown syntax is preserved as-is in chunks. |

**OCR is out of scope.** Scanned PDFs with no embedded text layer will produce no chunks and be
logged as skipped.

---

## What is never stored

- Raw binary content or PDF byte streams — only extracted plain text.
- Embeddings are stored in ChromaDB; text is stored alongside them as document metadata.
- No data is sent to external services except your Anthropic API key at query time.

---

## How citations work

Every chunk carries the filename and 1-indexed page number from its source PDF page (or `None`
for plain-text files). Claude is instructed to cite every factual claim using
`(Source: filename, p.N)` inline, and also returns a structured `sources` list. The CLI prints
both. Page numbers match the PDF's own page numbering because pypdf returns pages in order
starting from page 1.

---

## Adding new files / re-ingesting

Drop new files into your configured `ingestion.folder` and run `folio ingest` again. Folio
computes a stable SHA-256 ID from each file's absolute path. Only files that are new (ID not
yet in the database) are processed. Already-indexed files are reported as skipped.

To force a full re-index of a file, delete its record from `data/folio.db` and its vectors
from `data/chroma/`, then run ingest again.

---

## Audit trail

Every ingestion and query is appended to `logs/manifest.jsonl` as a JSON line with a UTC
timestamp. This file is the human-readable record of what Folio has processed. It is never
truncated — append-only.

---

## Project layout

```
folio/
├── cli/            CLI entry point (ingest, query, list)
├── core/           Config, paths, retry, manifest
├── ingestion/      Scanner, parser, chunker, embedder
├── store/          SQLite (db.py) + ChromaDB (vector.py) + dataclasses (models.py)
├── retrieval/      Embed query → ChromaDB search
├── qa/             Claude tool-use Q&A with structured citations
config/
data/               gitignored — SQLite + ChromaDB live here
logs/               gitignored — manifest.jsonl lives here
```
