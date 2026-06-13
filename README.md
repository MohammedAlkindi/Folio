# Folio

**Search your own documents the way you search the web — with answers, not just keywords.**

---

## The problem

You have a folder of PDFs, research papers, contracts, or notes. You need to find
something specific. Ctrl+F only works if you know the exact words. Uploading to
ChatGPT means your documents leave your machine, get chunked by someone else's
pipeline, and disappear into a black box you can't audit or extend.

Folio runs entirely on your hardware. It builds a local semantic index, retrieves
the most relevant passages for any question you ask, and generates a cited answer
using Claude — every claim traceable to a specific document and page number.

---

## Try it now

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install -e . && folio demo
```

No config file. No folder setup. Folio ingests the included sample board minutes and
opens a Q&A session in under a minute. Re-run with `--reset` to clear the demo index
and start fresh.

---

## Quick demo

```
$ folio ingest --config config/folio_config.yaml
  Ingesting  q3_report.pdf...      OK (187 chunks)
  Ingesting  board_minutes.pdf...  OK (94 chunks)
  Ingesting  notes.md...           OK (14 chunks)

Workspace: default
Scanned:        3 files
Ingested:       3 new
Skipped:        0 (already indexed)
Failed:         0
Chunks:       295 total

$ folio query --config config/folio_config.yaml
Folio — ask questions about your documents. Type 'quit' to exit.

> What revenue targets were approved in the board meeting?

The board approved a Q4 revenue target of $2.4M, representing a 12% increase
over Q3 actuals. The motion passed unanimously. (Source: board_minutes.pdf, p.3)

Sources: board_minutes.pdf (p.3), q3_report.pdf (p.1)
Confidence: high

Show excerpts? [y/N] y

--- [1] board_minutes.pdf (p.3) ---
...the motion to approve Q4 targets of $2.4M was put to a vote. All five
board members voted in favour. The CFO noted this assumes the enterprise
pipeline closes on schedule...

--- [2] q3_report.pdf (p.1) ---
Q3 closed at $2.14M against a target of $2.1M. Management proposes a 12%
stretch target for Q4 based on current pipeline visibility...
```

---

## Why Folio

- **Nothing leaves your machine.** Embeddings are computed locally using
  sentence-transformers. Only your Claude API call touches the network, and only
  at query time — your documents never do.

- **Every answer is auditable.** Claude is forced via tool-use to return a
  structured `{answer, sources, confidence}` response. The CLI prints inline
  citations, a source list, a confidence label, and on request the raw chunk text
  that grounded the answer.

- **Honest about uncertainty.** When the corpus doesn't have strong evidence,
  Folio says so before answering, rather than generating a confident-sounding
  hallucination. Low-confidence answers come with a prompt to ingest more.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/MohammedAlkindi/Folio.git && cd Folio

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Folio as an editable package (registers the `folio` CLI)
pip install -e .

# 5. Config
cp config/folio_config.example.yaml config/folio_config.yaml
# Edit config/folio_config.yaml — set ingestion.folder and optionally workspace

# 6. API key (query command only — ingest works without it)
export ANTHROPIC_API_KEY="sk-ant-..."

# 7. Drop documents into your configured folder and ingest
folio ingest --config config/folio_config.yaml
```

---

## Commands

| Command | What it does |
|---------|-------------|
| `folio demo` | Ingest the built-in sample PDF and start Q&A — no config needed. |
| `folio demo --reset` | Clear the demo index and re-ingest from scratch. |
| `folio ingest` | Scan folder, chunk, embed, store. Idempotent. |
| `folio query` | Interactive Q&A loop with citations. |
| `folio list` | Print a table of all indexed documents. |
| `folio list --verbose` | Include chunk previews per document. |
| `folio remove <filename>` | Delete a document from SQLite and ChromaDB. |
| `folio reindex <filename>` | Remove and re-ingest a document in one step. |
| `folio watch` | Live folder ingestion — auto-ingest files on change. |
| `folio workspace list` | List all workspaces with chunk counts. |

All commands except `folio demo` accept `--config <path>` (default: `config/folio_config.yaml`).

---

## How citations work

Every chunk carries the filename and 1-indexed page number from its source PDF (or
`None` for plain-text files). When you run a query, Folio retrieves the top-k most
semantically similar chunks, then sends them to Claude with a strict instruction:
cite every claim using `(Source: filename, p.N)` inline, and return a structured
`sources` list alongside a `confidence` rating.

Claude is called with `tool_choice: forced` — it cannot respond without filling the
structured schema. If Claude rates confidence as `insufficient_data`, Folio warns you
before showing the answer. Type `y` at the excerpt prompt to see the raw passage each
source claim is drawn from.

Page numbers match the PDF's own numbering because pypdf returns pages in order
starting from 1.

---

## Workspaces

Keep separate indexes for separate projects without running multiple instances:

```yaml
# config/research.yaml
workspace: "research"

# config/work.yaml
workspace: "work"
```

Each workspace maps to its own ChromaDB collection (`folio_research`, `folio_work`).
Ingest and query against whichever config you pass. List all workspaces:

```
$ folio workspace list
Workspace                      Collection                          Chunks
------------------------------ ----------------------------------- -------
default                        folio_default                          295
research                       folio_research                        1847
work                           folio_work                             512
```

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing
```

Tests use temporary SQLite databases and real fixture files. No test touches
`data/folio.db`. No mocking of file I/O.

---

## Roadmap

- **PDF table extraction** — structured data in tables is currently treated as
  raw text; a dedicated table parser would improve answer quality on financial docs.
- **Incremental re-index on file change** — detect modified files by content hash
  and auto-reindex without a manual `folio reindex` call.
- **Multi-file semantic diff** — given two versions of a document, show what changed
  in meaning rather than just line diffs.
- **Local LLM support** — swap Claude for a locally-served Ollama model for fully
  air-gapped operation.
