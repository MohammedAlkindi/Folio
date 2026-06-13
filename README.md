# Folio

**Search your own documents the way you search the web — with cited answers, not keyword matches.**

Folio is a local-first document intelligence pipeline. Drop in PDFs, Markdown notes, or plain-text files. Ask questions in natural language. Get back a grounded answer with inline citations, page references, and a calibrated confidence rating — every claim traceable to a specific passage in your corpus.

Embeddings are computed locally. Your documents never leave your machine. Only your question and the retrieved excerpts touch the network, at query time, via the Claude API.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Folio Does](#what-folio-does)
3. [Architecture](#architecture)
4. [Pipeline Deep Dive](#pipeline-deep-dive)
   - [Ingestion](#ingestion)
   - [Chunking](#chunking)
   - [Embedding](#embedding)
   - [Storage](#storage)
   - [Retrieval](#retrieval)
   - [Answer Generation](#answer-generation)
5. [Configuration Reference](#configuration-reference)
6. [CLI Reference](#cli-reference)
7. [Web UI](#web-ui)
8. [Deployment](#deployment)
   - [Local Development](#local-development)
   - [Streamlit Web App](#streamlit-web-app)
9. [Retrieval Design](#retrieval-design)
   - [Semantic Search](#semantic-search)
   - [Hybrid Search — BM25 + RRF](#hybrid-search--bm25--rrf)
10. [Answer Quality and Confidence](#answer-quality-and-confidence)
11. [Citations](#citations)
12. [Workspaces](#workspaces)
13. [LLM Backends](#llm-backends)
14. [Running Tests](#running-tests)
15. [Design Decisions](#design-decisions)
16. [Roadmap](#roadmap)

---

## Quick Start

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install -e . && folio demo
```

No config file. No folder setup. Folio ingests the included sample board minutes PDF and opens an interactive Q&A session in under a minute. Re-run with `--reset` to clear the demo index and start fresh.

---

## What Folio Does

You have a folder of PDFs — research papers, contracts, board minutes, technical specs. You need to find something specific. `Ctrl+F` only works if you know the exact words. Uploading to a SaaS tool means your documents leave your machine, get chunked by someone else's pipeline, and disappear into a black box you can't audit or extend.

Folio solves this differently:

- **Ingests** PDF, TXT, and Markdown files into a local SQLite + ChromaDB index
- **Retrieves** the most relevant passages using sentence-transformer embeddings, with an optional BM25 keyword layer fused via Reciprocal Rank Fusion
- **Answers** using Claude with `tool_choice: forced` — the model cannot respond without filling a structured `{answer, sources, confidence}` schema
- **Cites** every claim with the originating filename and page number
- **Rates confidence** in four tiers: `high`, `medium`, `low`, `insufficient_data` — and tells you when it doesn't know

### Example session

```
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

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                           Folio Pipeline                           │
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────┐  │
│  │  Parser  │──▶│ Chunker  │──▶│ Embedder │──▶│    Storage    │  │
│  │  pypdf   │   │word-win  │   │sentence- │   │ ChromaDB (vec)│  │
│  │  txt/md  │   │+sentence │   │transform-│   │ SQLite  (meta)│  │
│  └──────────┘   │boundary  │   │ers local │   └───────────────┘  │
│                 └──────────┘   └──────────┘          │           │
│                                                       │           │
│  ┌─────────────────────────────────────────┐          │           │
│  │              Query Path                 │          │           │
│  │                                         │          │           │
│  │  Question ──▶ Embed ──▶ Semantic Search ◀──────────┘           │
│  │                              │                                 │
│  │                    (hybrid?) │                                 │
│  │                         ┌───▼────┐   ┌──────┐                 │
│  │                         │  BM25  │──▶│  RRF │                 │
│  │                         │ Index  │   │ Fuse │                 │
│  │                         └────────┘   └──┬───┘                 │
│  │                                         │                     │
│  │                              top-k chunks                     │
│  │                                         │                     │
│  │                                    ┌────▼─────────────────┐   │
│  │                                    │  Claude API          │   │
│  │                                    │  tool_choice: forced │   │
│  │                                    │  {answer, sources,   │   │
│  │                                    │   confidence}        │   │
│  │                                    └──────────────────────┘   │
│  └─────────────────────────────────────────────────────────────  │
└────────────────────────────────────────────────────────────────────┘
```

### Component map

| Module | Responsibility |
|--------|---------------|
| `ingestion/parser.py` | Parse PDF/TXT/MD into `{page_number, text}` dicts. Never raises. |
| `ingestion/chunker.py` | Sliding-window word chunker with sentence-boundary snap. |
| `ingestion/embedder.py` | Batch embedding via `sentence-transformers`. Converts numpy → `list[float]`. |
| `store/vector.py` | ChromaDB wrapper — upsert, query, get-all, delete by `doc_id`. |
| `store/db.py` | SQLite wrapper — WAL mode, documents + chunks tables, schema migrations. |
| `store/models.py` | `Document` and `Chunk` dataclasses. |
| `retrieval/retriever.py` | Orchestrates semantic search and optional hybrid fusion. |
| `retrieval/bm25.py` | `BM25Index` — builds `rank_bm25.BM25Okapi` from chunk texts; normalises scores to `[0,1]`. |
| `retrieval/fusion.py` | Reciprocal Rank Fusion — merges two ranked lists by `(filename, chunk_index)` key. |
| `qa/answerer.py` | Dispatches to configured LLM backend. |
| `qa/backends/anthropic_backend.py` | Claude API call with `tool_choice: forced`, context trimming, retry. |
| `qa/backends/ollama_backend.py` | Ollama local inference backend. |
| `core/config.py` | `Config` — typed accessor layer over YAML. |
| `core/retry.py` | `@with_retry` decorator — exponential backoff, configurable exceptions. |
| `core/manifest.py` | Append-only JSONL audit trail at `logs/manifest.jsonl`. |
| `cli/main.py` | Click CLI — `ingest`, `query`, `list`, `remove`, `reindex`, `watch`, `workspace`, `demo`. |
| `app.py` | Streamlit web UI — Query / Ingest / Documents pages. |

---

## Pipeline Deep Dive

### Ingestion

Ingestion is idempotent. Running `folio ingest` twice against the same folder produces the same index state as running it once.

**Entry point:** `cli/main.py ingest` or the Upload/Scan tabs in `app.py`.

```
File path
  └─▶ compute_doc_id(path)            # SHA-256(file_bytes + "\x00" + abs_path)
        └─▶ document_exists(doc_id)?  # SQLite lookup
              ├─ YES → skip (already indexed)
              └─ NO  → parse → chunk → embed → store
```

**`doc_id` derivation** combines the file's content bytes with its absolute path, separated by a null byte. This means the same content at a different path yields a different ID (two copies of the same report are treated as separate documents), and modifying the content while keeping the filename yields a new ID (triggering re-ingestion on the next `folio reindex` call).

**Graceful failure modes:**

| Failure | Behaviour |
|---------|-----------|
| Encrypted PDF | Logged, returns `([], "encrypted")`, skipped with warning |
| Corrupted PDF | Logged, returns `([], "corrupted")`, skipped with warning |
| Scanned PDF (no text layer) | Returns `([], "no_text")`, skipped with note |
| Unsupported extension | Returns `([], "unsupported")` |
| Unexpected exception | Logged at ERROR, returns `([], "parse_error")` |

No exception propagates past the parser. The ingest loop always continues to the next file.

---

### Chunking

**File:** `ingestion/chunker.py`

Folio uses a word-window sliding chunker with a sentence-boundary snap, not a naive fixed-length split.

**Algorithm:**

1. Flatten all pages into a single word list, tracking which page each word came from.
2. Start a window at position `start = 0` with target end at `start + chunk_size`.
3. Look ahead up to `_MAX_SENTENCE_LOOKAHEAD = 50` words past the target end for a word ending with `.`, `!`, `?`, `."`, `!"`, or `?"`. Extend the window to include that word.
4. Assign `page_number` to the page that contributes the most words to the chunk (ties go to the higher page number).
5. Advance: `next_start = end - overlap`. If this wouldn't move forward (very short documents), force `next_start = start + 1`.
6. Stop when `end >= total` — never generate tail overlap-only chunks.

**Configuration:**

```yaml
ingestion:
  chunk_size: 800    # target words per chunk
  chunk_overlap: 150 # words of overlap between adjacent chunks
```

**Why words, not characters or tokens?** Word count is a stable, model-agnostic approximation of token count for English text. It avoids a tokeniser dependency in the ingestion path while remaining within 15–20% of actual token counts for sentence-transformer inputs.

**Output schema:**

```python
{
    "text":        str,       # space-joined chunk words
    "page_number": int|None,  # majority-vote page assignment
    "page_start":  int|None,  # page of first word in chunk
    "page_end":    int|None,  # page of last word in chunk
    "chunk_index": int,       # 0-based sequential index within document
}
```

---

### Embedding

**File:** `ingestion/embedder.py`

Embeddings are computed locally using `sentence-transformers`. No external embedding API is called during ingestion.

**Model:** `all-MiniLM-L6-v2` (default) — 384-dimensional vectors, ~22M parameters, runs on CPU in under a second per batch.

**Batching:** Texts are encoded in batches of 64 (`_BATCH_SIZE`). Each batch is timed and logged at DEBUG level. NumPy arrays are converted to `list[float]` before storage.

**Caching:** The model is loaded once and cached:
- In the CLI: module-level cache in `retriever.py` keyed by model name.
- In the Streamlit app: `@st.cache_resource` keyed by model name.

The model is never re-loaded across hot reloads or re-runs within the same process.

**Changing the model:** Update `embedding.model` in your config YAML. Any change requires re-indexing all documents, since vectors from different models are not comparable. Run `folio reindex <filename>` for each document, or wipe `data/` and re-run `folio ingest`.

---

### Storage

Folio maintains two stores in parallel. They are not interchangeable: ChromaDB holds vectors + full text; SQLite holds metadata + chunk previews for display and audit.

#### ChromaDB (Vector Store)

**File:** `store/vector.py`

- `PersistentClient` stored at `data/chroma/`
- One collection per workspace: `folio_{workspace_name}`
- Distance metric: cosine (`"hnsw:space": "cosine"`) — more stable than L2 for sentence-transformer embeddings
- Each chunk is upserted with metadata: `{filename, page_number, chunk_index, doc_id}`
- `page_number=0` is the sentinel for "no page" — ChromaDB metadata values must be `str | int | float | bool`; `None` is rejected
- `collection.upsert()` is used throughout — safe to call on re-ingest without creating duplicates
- Delete by `doc_id` uses `collection.delete(where={"doc_id": doc_id})`

#### SQLite (Metadata Store)

**File:** `store/db.py`

- Stored at `data/folio.db`
- WAL journal mode — allows concurrent readers alongside one writer, appropriate for Streamlit's multi-thread model
- Two tables: `documents` and `chunks`

**`documents` schema:**

```sql
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,  -- SHA-256 doc_id
    filename     TEXT NOT NULL,
    filepath     TEXT NOT NULL,
    extension    TEXT NOT NULL,
    page_count   INTEGER,
    chunk_count  INTEGER DEFAULT 0,
    ingested_at  TEXT NOT NULL,     -- ISO 8601
    content_hash TEXT               -- SHA-256 of content bytes; for change detection
);
```

**`chunks` schema:**

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,  -- "{doc_id}:chunk:{index}"
    doc_id         TEXT NOT NULL REFERENCES documents(id),
    filename       TEXT NOT NULL,
    page_number    INTEGER,
    chunk_index    INTEGER NOT NULL,
    token_estimate INTEGER,           -- len(text.split()) — word count proxy
    preview        TEXT,              -- first 200 characters of chunk text
    ingested_at    TEXT NOT NULL
);
```

Chunk full text is **not** stored in SQLite to avoid duplication — it lives only in ChromaDB. SQLite chunk rows exist for auditing, listing, and the `--verbose` flag on `folio list`.

**Schema migrations** are applied at startup via `init_db()`:

```python
try:
    con.execute("ALTER TABLE chunks ADD COLUMN preview TEXT")
except sqlite3.OperationalError:
    pass  # column already exists
```

This pattern keeps the schema current without an external migration framework and is safe to run on startup on every process launch.

---

### Retrieval

**File:** `retrieval/retriever.py`

The retriever is the bridge between a natural-language question and the chunks that will ground the LLM answer. It always returns `list[dict]` and never raises — on any failure it returns `[]`, which the QA layer converts to an `insufficient_data` response.

**Semantic path (always active):**

```
question
  └─▶ embed_texts([question], model)   # same model as ingestion
        └─▶ vector_query(embedding, top_k, workspace)
              └─▶ ChromaDB cosine search → top-k {text, filename, page_number, chunk_index, distance}
```

**Hybrid path (when `retrieval.hybrid: true`):**

```
semantic_results  ──────────────────────────────┐
                                                 ▼
get_all_chunks(workspace)                   RRF fusion
  └─▶ BM25Index(all_chunks)                     │
        └─▶ bm25_index.search(question, top_k) ─┘
              └─▶ bm25_results               → fused[:top_k]
```

See [Retrieval Design](#retrieval-design) for the full algorithmic discussion.

---

### Answer Generation

**File:** `qa/answerer.py`, `qa/backends/anthropic_backend.py`

The QA layer dispatches to the configured backend (`anthropic` or `ollama`), passing the question and retrieved chunks. The Anthropic backend:

1. **Context trimming:** Estimates word count of all chunks. If `overhead + sum(chunk_words) > context_budget`, drops lowest-ranked chunks (those at the end of the list) until the budget is satisfied. At minimum one chunk is always sent.

2. **Message construction:** Formats chunks as numbered excerpts with filename and page label:
   ```
   Question: <question>

   Document excerpts:
   [1] report.pdf (page 3):
   <chunk text>

   [2] notes.md (no page):
   <chunk text>

   Answer with citations in the format: (Source: filename, p.N)
   ```

3. **API call with forced tool use:**
   ```python
   client.messages.create(
       model=cfg.qa_model(),
       tools=[_ANSWER_TOOL],
       tool_choice={"type": "tool", "name": "structured_answer"},
       messages=[{"role": "user", "content": user_message}],
   )
   ```
   `tool_choice: forced` means Claude **cannot** respond without calling `structured_answer`. There is no free-form fallback path at the API level.

4. **Retry with exponential backoff:** The API call is decorated with `@with_retry(max_attempts=3, base_delay=2.0)`. Transient network errors and rate limits are retried automatically.

5. **Structured output schema:**
   ```json
   {
     "answer": "The answer text with inline citations (Source: filename, p.N).",
     "sources": [
       {"filename": "report.pdf", "page_number": 3},
       {"filename": "notes.md", "page_number": null}
     ],
     "confidence": "high" | "medium" | "low" | "insufficient_data"
   }
   ```

---

## Configuration Reference

Copy the example config and adjust:

```bash
cp config/folio_config.example.yaml config/folio_config.yaml
```

Full annotated reference:

```yaml
# Named workspace — each workspace is an isolated ChromaDB collection.
# Use different values across config files to keep separate knowledge bases.
# Collection name: folio_{workspace}
workspace: "default"

ingestion:
  # Folder to scan for documents. Relative paths are resolved from the project root.
  folder: "docs/"

  # File extensions to include when scanning a folder.
  # Upload tab in the web UI accepts the same set.
  supported_extensions: [".pdf", ".txt", ".md"]

  # Target words per chunk. Word count is used as a proxy for token count.
  # Increase for longer, more context-rich chunks (better for dense technical docs).
  # Decrease for shorter, more precise chunks (better for Q&A over structured data).
  chunk_size: 800

  # Words of overlap between adjacent chunks.
  # Prevents context loss at chunk boundaries for multi-sentence answers.
  chunk_overlap: 150

embedding:
  # Only "sentence_transformers" is supported. Claude API is never used for embeddings.
  provider: "sentence_transformers"

  # Sentence-transformers model name. Must match the model used at ingest time.
  # Changing this requires re-indexing all documents.
  # all-MiniLM-L6-v2: fast (384-dim), good general-purpose baseline
  # all-mpnet-base-v2: slower (768-dim), higher quality
  model: "all-MiniLM-L6-v2"

retrieval:
  # Number of chunks to retrieve per query and pass to the LLM.
  # Higher values give the LLM more context but increase latency and API cost.
  top_k: 6

  # Enable hybrid search: fuses semantic (ChromaDB cosine) with BM25 keyword search
  # via Reciprocal Rank Fusion. Recommended for corpora with precise terminology
  # (legal, medical, financial) where exact keyword matches matter.
  hybrid: true

qa:
  # LLM backend. Options: "anthropic" (default), "ollama"
  provider: "anthropic"

  # Claude model ID. Defaults to claude-sonnet-4-6.
  model: "claude-sonnet-4-6"

  # Name of the environment variable holding the Anthropic API key.
  # Folio reads this at query time — ingest works without it.
  api_key_env: "ANTHROPIC_API_KEY"

  # Max tokens in the Claude response.
  max_tokens: 2048

  # Max words of chunk text to send as context. Chunks are trimmed from the bottom
  # (lowest-ranked first) until the total fits within this budget.
  context_budget: 3500

  # Ollama settings — only used when provider is "ollama"
  ollama_model: "llama3"
  ollama_host: "http://localhost:11434"
```

### Multiple config files

Use separate config files for separate projects. All CLI commands accept `--config <path>`:

```bash
folio ingest --config config/research.yaml
folio query  --config config/work.yaml
```

---

## CLI Reference

Install the `folio` command:

```bash
pip install -e .
```

All commands except `folio demo` accept `--config <path>` (default: `config/folio_config.yaml`).

### `folio demo`

```bash
folio demo [--reset]
```

Zero-config entry point. Ingests the bundled sample PDF (`docs/sample/board_minutes.pdf`) into a `demo` workspace and starts an interactive Q&A session. Pass `--reset` to wipe the demo index and re-ingest from scratch.

---

### `folio ingest`

```bash
folio ingest [--config PATH]
```

Scan the configured `ingestion.folder`, parse all supported files, chunk, embed, and store. Already-indexed files are detected by `doc_id` and skipped. Prints a summary table on completion:

```
Workspace: default
Scanned:        3 files
Ingested:       2 new
Skipped:        1 (already indexed)
Failed:         0
Chunks:       281 total
```

---

### `folio query`

```bash
folio query [--config PATH]
```

Interactive Q&A loop. Embeds each question, retrieves top-k chunks, calls the configured LLM backend, and prints the structured answer with sources and confidence. Type `y` at the excerpt prompt to display the raw chunk text behind each source.

Confidence gating:
- `insufficient_data` — prints a warning before showing the answer
- `low` — prints a soft note
- `medium`, `high` — answer shown directly

---

### `folio list`

```bash
folio list [--verbose] [--config PATH]
```

Print a table of all indexed documents:

```
ID          Filename             Ext    Pages  Chunks  Ingested
a3f2...     board_minutes.pdf   .pdf    12     94      2025-06-01 14:22:03
8c1d...     notes.md            .md     —      14      2025-06-01 14:22:09
```

`--verbose` adds a preview of each chunk below each document row.

---

### `folio remove`

```bash
folio remove <filename> [--config PATH]
```

Delete a document from both SQLite and ChromaDB by filename. Cascades to chunk rows. Looks up the document by filename; if multiple documents share the same name (different paths), removes the first match and warns.

---

### `folio reindex`

```bash
folio reindex <filename> [--config PATH]
```

Remove and re-ingest a document in one step. Use this after modifying a file to update its index without manually calling `remove` then `ingest`.

---

### `folio watch`

```bash
folio watch [--config PATH]
```

Live folder ingestion. Uses `watchdog` to monitor the configured `ingestion.folder` for new or modified files and auto-ingests them on change. Useful for keeping the index current while working in a folder.

---

### `folio workspace list`

```bash
folio workspace list [--config PATH]
```

List all Folio workspaces (ChromaDB collections prefixed with `folio_`) with their chunk counts:

```
Workspace                      Collection                          Chunks
------------------------------ ----------------------------------- -------
default                        folio_default                          295
research                       folio_research                        1847
work                           folio_work                             512
```

---

## Web UI

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app has three pages, navigated from the sidebar radio:

### Query page

The primary interface. A `st.chat_message` conversation with full history. Each assistant message displays:

- The answer text with inline citation references
- A colour-coded confidence badge (`High` / `Medium` / `Low` / `Insufficient Data`)
- Source chips showing `filename · p.N` for each cited document

The **Hybrid search** checkbox in the sidebar toggles BM25+semantic fusion for the current session without modifying the config file.

The **Clear conversation** button appears when history is non-empty and resets the session.

### Ingest page

Two tabs:

**Upload files** — browser-based file picker (PDF, TXT, MD). Supports multiple files. Progress bar updates per file. Each file reports success or the skip/failure reason.

**Scan folder** — enter a folder path and scan it server-side. Intended for local deployments where the Folio process has direct access to the filesystem. In cloud deployments, use the Upload tab instead.

### Documents page

Library view of all indexed documents. Columns: Name, Type, Pages, Chunks, Remove. Each row shows the ingestion timestamp. Remove permanently deletes the document from both stores.

---

## Deployment

### Local Development

```bash
# 1. Clone
git clone https://github.com/MohammedAlkindi/Folio.git && cd Folio

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Folio as an editable package (registers the folio CLI)
pip install -e .

# 5. Config
cp config/folio_config.example.yaml config/folio_config.yaml
# Edit config/folio_config.yaml — set ingestion.folder and optionally workspace

# 6. API key (query and web UI only — ingest works without it)
export ANTHROPIC_API_KEY="sk-ant-..."

# 7. Ingest
folio ingest --config config/folio_config.yaml

# 8. Query via CLI
folio query --config config/folio_config.yaml

# 9. Query via web UI
streamlit run app.py
```

### Streamlit Web App

Folio is designed primarily as a local tool. When running on Streamlit Community Cloud or any shared deployment:

- **File uploads** work correctly. Use the Upload tab in the Ingest page.
- **Folder scan** resolves paths on the server, not the client's machine. It is only useful if the server has direct filesystem access to the documents folder.
- **API key** must be set as a Streamlit secret (`ANTHROPIC_API_KEY`) in the app's Settings → Secrets panel on Streamlit Cloud, or as an environment variable in any other deployment environment.
- **Shared state:** all visitors to a single deployed instance share the same SQLite database and ChromaDB collection. For a personal knowledge base this is expected behaviour. For a multi-user deployment, extend the workspace system to derive a per-session workspace name.

To run the web app with explicit port and dark theme, create `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501
maxUploadSize = 50

[theme]
base = "dark"
```

---

## Retrieval Design

### Semantic Search

Each document chunk is embedded at ingest time using `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB with cosine distance. At query time, the question is embedded with the same model and ChromaDB returns the `top_k` closest chunks by cosine similarity.

**Why cosine over L2?** Cosine distance measures the angle between vectors, making it invariant to vector magnitude. Sentence-transformer embeddings vary in magnitude across sentences of different lengths; cosine similarity normalises this out and produces more stable rankings.

**Why local embeddings?** Privacy and cost. A 300-page PDF generates roughly 300–400 chunks. Embedding those chunks via an API call at ingest time — and re-embedding every query — adds latency, cost, and a hard dependency on network availability. `all-MiniLM-L6-v2` runs on a CPU laptop in milliseconds per batch with no API key required.

### Hybrid Search — BM25 + RRF

Enable hybrid search in config (`retrieval.hybrid: true`) or toggle it in the sidebar. When active, the retriever runs two parallel searches and fuses the results.

**BM25 (Best Match 25)** is a probabilistic keyword ranking function. It scores chunks by term frequency weighted by inverse document frequency, with length normalisation. BM25 excels where semantic search struggles: exact product codes, proper nouns, version numbers, and other terms whose meaning is carried by the exact string rather than semantic context.

**Implementation** (`retrieval/bm25.py`):

```python
tokenized = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)
scores = bm25.get_scores(query.lower().split())
```

BM25Okapi can return negative scores (when a term appears in all documents, IDF goes negative). Scores are normalised to `[0, 1]` via min-max before distance conversion: `distance = 1.0 - normalised_score`.

**Reciprocal Rank Fusion** (`retrieval/fusion.py`) merges the two ranked lists without requiring score calibration between them:

```
score(d) = Σ_lists  1 / (k + rank(d, list))
```

where `k = 60` (the standard constant that dampens the influence of very high ranks). Documents are deduplicated by `(filename, chunk_index)`. The fused list is sorted by descending RRF score and truncated to `top_k`.

**When to use hybrid search:**

| Document type | Recommendation |
|---------------|---------------|
| Research papers, narratives, reports | Semantic only — meaning matters more than keywords |
| Legal contracts, financial filings | Hybrid — clause numbers, defined terms, exact amounts |
| Technical specs, API docs, code comments | Hybrid — function names, version strings, identifiers |
| Mixed corpora | Hybrid — safe default |

---

## Answer Quality and Confidence

Folio uses Claude's `tool_choice: forced` feature to guarantee that every response conforms to the `structured_answer` schema. The model cannot produce a free-form text response; it must call the tool.

This eliminates an entire class of failure mode: the model cannot hedge with "I think..." and fail to populate `sources`, or give a high-confidence answer with an empty `sources` list. The schema enforces co-presence of answer, sources, and confidence.

**Confidence tiers and their meaning:**

| Level | Meaning | Folio behaviour |
|-------|---------|----------------|
| `high` | Strong evidence in multiple retrieved chunks | Answer shown directly |
| `medium` | Good evidence but some gaps or ambiguity | Answer shown directly |
| `low` | Weak or indirect evidence; answer may be inferential | Soft warning printed before answer |
| `insufficient_data` | Corpus lacks the information needed to answer | Warning shown; answer (often "I don't know") shown after |

The `insufficient_data` path is intentional. Rather than generating a confident-sounding hallucination, Folio surfaces the knowledge gap and tells the user to ingest more relevant documents.

---

## Citations

Every chunk stores its originating `filename` and 1-indexed `page_number`. Page numbers come from pypdf's page enumeration (1-indexed, matching the PDF's physical page order). Plain-text and Markdown files have `page_number = None` (stored as `0` in ChromaDB, converted back to `None` on retrieval).

The LLM prompt instructs Claude to cite every factual claim using `(Source: filename, p.N)` inline, and to populate the `sources` array with the full list of cited documents.

In the CLI, sources are printed as a list below the answer. In the web UI, they appear as inline chips beside each assistant message. The `Show excerpts? [y/N]` prompt in the CLI (or the chunk text in the UI) lets you verify the raw passage that grounded each claim.

Page numbers match the PDF's own numbering because pypdf returns pages in order starting from 1, with no re-mapping. What you see in Folio is what you'd see if you opened the PDF to that page.

---

## Workspaces

A workspace is an isolated ChromaDB collection. Documents ingested into `workspace: "research"` are never retrieved when querying `workspace: "work"`.

```yaml
# config/research.yaml
workspace: "research"

# config/work.yaml
workspace: "work"
```

Each workspace maps to a collection named `folio_{workspace}`. The SQLite metadata store is shared across workspaces (the `documents` and `chunks` tables hold rows from all workspaces), but queries are always scoped to a single workspace at the ChromaDB level.

**Use cases:**

- Separate projects with overlapping terminology (two clients with similar contracts)
- Different security classifications (public docs vs. internal docs)
- Experimentation (test a new chunking config in a scratch workspace)

List all workspaces and their sizes:

```bash
folio workspace list
```

---

## LLM Backends

Folio supports two answer-generation backends, selectable per config file.

### Anthropic (Claude)

```yaml
qa:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  api_key_env: "ANTHROPIC_API_KEY"
  max_tokens: 2048
  context_budget: 3500
```

Uses the Claude Messages API with `tool_choice: forced` to guarantee structured output. Retried up to 3 times with exponential backoff (2s, 4s) on any exception. The `context_budget` setting limits the total words of chunk text sent to the model — lowest-ranked chunks are dropped first if the budget is exceeded.

### Ollama (Local Inference)

```yaml
qa:
  provider: "ollama"
  ollama_model: "llama3"
  ollama_host: "http://localhost:11434"
```

Sends the question and retrieved chunks to a locally-running Ollama instance. Fully air-gapped — no API key, no network call beyond `localhost`. Suitable for sensitive documents that must not touch external services.

Start Ollama before querying:

```bash
ollama serve
ollama pull llama3
folio query --config config/ollama.yaml
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific module
pytest tests/test_chunker.py -v
```

**Test philosophy:**

- All tests use the `tmp_db` fixture (defined in `tests/conftest.py`) which monkeypatches `DB_PATH` to a temporary file. No test ever touches `data/folio.db`.
- File I/O is tested against real fixture files in `tests/fixtures/`. No mocking of file operations.
- The chunker, scanner, parser, and database layer each have dedicated test modules covering boundary conditions.

**Running without an API key:** The test suite does not call the Claude or Ollama APIs. All 23 tests pass without `ANTHROPIC_API_KEY` set.

---

## Design Decisions

This section explains the non-obvious choices in Folio's implementation — the kind of decisions that have clear reasons which aren't visible from the code alone.

### `tool_choice: forced` for structured QA output

Most RAG implementations prompt the LLM to "respond in JSON format" and then parse the response. This fails silently: the model may produce valid JSON with an empty `sources` array, or respond in prose when it decides JSON isn't appropriate.

Folio uses Claude's `tool_choice: {"type": "tool", "name": "structured_answer"}` parameter. This is a hard enforcement mechanism at the API level — the model cannot respond without invoking the named tool. The schema is validated by the API before the response is returned. There is no code path where a free-form response reaches the UI.

### SHA-256 content-addressed `doc_id`

The `doc_id` is derived from both the file's content bytes and its absolute path. This makes ingestion idempotent across process restarts (same file → same `doc_id` → `document_exists` returns `True` → skip), while ensuring that two physically different files with the same name are treated as separate documents.

The `content_hash` column (SHA-256 of content bytes only) is stored separately for change detection: if a file's content changes but its path stays the same, `doc_id` changes (triggering re-ingest on `folio reindex`) and `content_hash` provides a secondary signal for detecting modifications without recomputing `doc_id`.

### ChromaDB `page_number=0` as `None` sentinel

ChromaDB metadata values are restricted to `str`, `int`, `float`, and `bool`. Storing `None` raises a validation error at upsert time. Plain-text files have no page numbers. Rather than using a string sentinel (which complicates comparison logic) or omitting the field (which breaks metadata filtering), Folio uses `0` as the sentinel value and converts it back to `None` on retrieval. This is an explicit assumption documented at both the upsert and query sites.

### WAL-mode SQLite

Streamlit reruns the script on every user interaction, potentially from multiple browser tabs simultaneously. WAL (Write-Ahead Logging) mode allows any number of concurrent readers alongside a single writer. Without WAL, two simultaneous reads around an ingest operation could block each other. WAL is set on every connection, not just at initialisation, so new connections immediately benefit.

### No text storage in SQLite chunks table

Storing chunk text in both ChromaDB and SQLite would create a synchronisation problem: any update to a chunk would need to be reflected in two stores. Folio treats ChromaDB as the authoritative text store and SQLite as the metadata store. SQLite chunk rows store a 200-character preview for display in `folio list --verbose` and the Documents page, but the full text lives only in ChromaDB.

### Sentence-boundary snap in the chunker

A naive fixed-size word window frequently splits sentences across chunk boundaries. When a chunk ends mid-sentence, the LLM receives incomplete context that can distort the meaning of the passage. The `_MAX_SENTENCE_LOOKAHEAD` extension scans up to 50 words past the target boundary for a sentence-terminal token, then snaps the chunk end to that position. This produces slightly variable chunk sizes in exchange for semantically cleaner boundaries.

### Exponential backoff on Claude API calls

The Claude API is subject to transient rate limits and network errors. Rather than surfacing these as immediate failures (which would disrupt an interactive Q&A session), Folio retries up to 3 times with a 2s base delay and 2× backoff factor. The `@with_retry` decorator is generic and can be applied to any function; the label parameter appears in logs for tracing.

### Reciprocal Rank Fusion over score normalisation

Fusing a semantic score (cosine distance in `[0, 1]`) with a BM25 score (magnitude depends on corpus size and term frequency distribution) requires either normalising both to the same scale or using a rank-based fusion method. Score normalisation is fragile: the normalisation factors change as the corpus grows, making historical comparisons unreliable.

RRF uses only the rank order from each system, not the raw scores. The formula `1 / (k + rank)` assigns diminishing returns to lower ranks regardless of what the raw scores look like. The constant `k = 60` was established empirically in the original RRF paper and remains a robust default across diverse retrieval tasks.

---

## Roadmap

| Feature | Status | Notes |
|---------|--------|-------|
| PDF parsing (text layer) | ✅ Done | `pypdf` — scanned PDFs (no text layer) return `no_text` |
| Plain-text and Markdown ingestion | ✅ Done | UTF-8 with `errors="replace"` |
| Semantic search (ChromaDB + sentence-transformers) | ✅ Done | Cosine distance, all-MiniLM-L6-v2 default |
| Hybrid search (BM25 + RRF) | ✅ Done | `rank-bm25`, Reciprocal Rank Fusion |
| Structured Q&A with citations | ✅ Done | Claude `tool_choice: forced`, `{answer, sources, confidence}` |
| Confidence-gated output | ✅ Done | Four tiers; `insufficient_data` surfaces knowledge gaps |
| Idempotent ingestion | ✅ Done | SHA-256 `doc_id`, skip on re-run |
| Workspace isolation | ✅ Done | Per-workspace ChromaDB collections |
| Streamlit web UI | ✅ Done | Query / Ingest / Documents pages |
| Local LLM support (Ollama) | ✅ Done | `provider: "ollama"` in config |
| Live folder watch (`folio watch`) | ✅ Done | `watchdog` integration |
| Append-only audit trail | ✅ Done | `logs/manifest.jsonl` |
| PDF table extraction | Planned | Tables are currently treated as raw text; structured table parsing would improve answer quality on financial and data-heavy documents |
| Incremental re-index on content change | Planned | `content_hash` column is in place; auto-detection of modified files without manual `folio reindex` |
| Multi-file semantic diff | Planned | Given two document versions, surface meaning changes rather than line diffs |
| Workspace switcher in web UI | Planned | Currently locked to config value; sidebar dropdown over `list_workspaces()` |
| Per-session workspaces in web UI | Planned | Multi-user cloud deployments need workspace isolation per browser session |

---

## Project Context

Folio is part of a series of local-first personal agents built to the same engineering standards:

- **Shell** → local shell automation agent
- **Pulse** → Gmail / GitHub / Notion integration agent
- **Folio** → document intelligence agent (this repo)

Shared principles across all three: graceful degradation (no exceptions escape module boundaries), idempotent operations (safe to re-run), auditable outputs (append-only logs), no mandatory cloud storage.
