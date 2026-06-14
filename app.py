from __future__ import annotations

import datetime
from pathlib import Path

import streamlit as st

from core.config import Config, load_config
from core.paths import ROOT, ensure_dirs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts, load_model
from ingestion.parser import parse
from ingestion.scanner import doc_id as compute_doc_id, scan_folder
from qa.answerer import answer
from retrieval.retriever import retrieve
from store.db import (
    delete_document,
    document_exists,
    init_db,
    insert_chunk,
    insert_document,
    list_documents,
)
from store.models import Chunk, Document
from store.vector import delete_by_doc_id, upsert_chunks

_config_yaml = ROOT / "config" / "folio_config.yaml"
_config_example = ROOT / "config" / "folio_config.example.yaml"
CONFIG_PATH = _config_yaml if _config_yaml.exists() else _config_example

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Folio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}
.badge-high              { background:#14532d; color:#86efac; }
.badge-medium            { background:#78350f; color:#fcd34d; }
.badge-low               { background:#7c2d12; color:#fdba74; }
.badge-insufficient_data { background:#1e1b4b; color:#a5b4fc; }

.chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 1px 7px;
    margin: 2px 2px 0 0;
    font-size: 0.78em;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ── Startup ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _startup():
    ensure_dirs()
    init_db()
    return load_config(CONFIG_PATH)


@st.cache_resource(show_spinner="Loading embedding model…")
def _get_model(model_name: str):
    return load_model(model_name)


cfg = _startup()
embed_model = _get_model(cfg.embedding_model())

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "ingest_summary" not in st.session_state:
    st.session_state.ingest_summary: dict | None = None

# ── Ingest helper ─────────────────────────────────────────────────────────────

def _ingest(filepath: Path) -> tuple[bool, str]:
    did = compute_doc_id(filepath)
    if document_exists(did):
        return False, f"`{filepath.name}` already ingested — skipped."

    pages, reason = parse(filepath)
    if not pages:
        return False, f"`{filepath.name}` could not be parsed: `{reason}`"

    chunks_data = chunk_pages(pages, cfg.chunk_size(), cfg.chunk_overlap())
    if not chunks_data:
        return False, f"`{filepath.name}` produced no chunks."

    texts = [c["text"] for c in chunks_data]
    embeddings = embed_texts(texts, embed_model)

    pnums = [p["page_number"] for p in pages if p.get("page_number") is not None]
    doc = Document(
        id=did,
        filename=filepath.name,
        filepath=str(filepath.resolve()),
        extension=filepath.suffix.lower(),
        page_count=max(pnums) if pnums else None,
        chunk_count=len(chunks_data),
        ingested_at=datetime.datetime.now().isoformat(),
    )
    insert_document(doc)

    chunk_objs: list[Chunk] = []
    for i, cd in enumerate(chunks_data):
        chunk = Chunk(
            id=f"{did}:chunk:{i}",
            doc_id=did,
            filename=filepath.name,
            page_number=cd.get("page_number"),
            chunk_index=i,
            text=cd["text"],
            token_estimate=len(cd["text"].split()),
            preview=cd["text"][:200],
        )
        insert_chunk(chunk)
        chunk_objs.append(chunk)

    upsert_chunks(chunk_objs, embeddings, workspace=cfg.workspace())
    return True, f"`{filepath.name}` — {len(chunks_data)} chunks ingested."


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Folio")
    st.caption(f"Workspace: **{cfg.workspace()}**")
    st.divider()
    page = st.radio(
        "Navigation",
        ["💬 Query", "⬆ Ingest", "🗂 Documents"],
        label_visibility="collapsed",
        key="nav",
    )
    st.divider()
    st.metric("Documents", len(list_documents()))
    if st.button("View all →", key="nav_docs", use_container_width=True):
        st.session_state.nav = "🗂 Documents"
        st.rerun()

# ── Query ─────────────────────────────────────────────────────────────────────

if page == "💬 Query":
    col_hdr, col_toggle = st.columns([5, 2])
    with col_hdr:
        st.header("Ask your documents")
    with col_toggle:
        st.write("")  # vertical alignment spacer
        use_hybrid = st.toggle(
            "Hybrid search",
            value=cfg.retrieval_hybrid(),
            help="Combines BM25 keyword search with semantic search for better recall.",
        )

    if not list_documents():
        st.write("")
        _, card_col, _ = st.columns([1, 2, 1])
        with card_col:
            st.markdown(
                "<div style='text-align:center; padding:3rem 2rem 1.5rem;"
                " border:1px solid #334155; border-radius:12px;"
                " background:#0f172a; margin-top:1rem;'>"
                "<div style='font-size:3rem; line-height:1; margin-bottom:1rem;'>📄</div>"
                "<h3 style='margin:0 0 0.5rem; color:#f1f5f9;'>No documents yet</h3>"
                "<p style='color:#94a3b8; margin:0;'>Ingest a PDF, TXT, or Markdown file"
                "<br>to start asking questions.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                if st.button("Go to Ingest →", type="primary", use_container_width=True):
                    st.session_state.nav = "⬆ Ingest"
                    st.rerun()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    conf = msg.get("confidence", "")
                    sources = msg.get("sources", [])
                    if conf:
                        label = conf.replace("_", " ").title()
                        st.markdown(
                            f'<span class="badge badge-{conf}">{label}</span>',
                            unsafe_allow_html=True,
                        )
                    if sources:
                        chips = "".join(
                            f'<span class="chip">{s["filename"]}'
                            + (f" · p.{s['page_number']}" if s.get("page_number") else "")
                            + "</span>"
                            for s in sources
                        )
                        st.markdown(chips, unsafe_allow_html=True)

        if question := st.chat_input("Ask a question about your documents…"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving…"):
                    _retrieval_cfg = Config({
                        **cfg._data,
                        "retrieval": {**cfg._data.get("retrieval", {}), "hybrid": use_hybrid},
                    })
                    chunks = retrieve(question, _retrieval_cfg)

                if not chunks:
                    reply = "No relevant content found for that question."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    with st.spinner("Generating answer…"):
                        try:
                            result = answer(question, chunks, cfg)
                        except EnvironmentError as e:
                            st.error(str(e))
                            st.stop()

                    answer_text = result["answer"]
                    confidence = result.get("confidence", "")
                    sources = result.get("sources", [])

                    st.markdown(answer_text)
                    if confidence:
                        label = confidence.replace("_", " ").title()
                        st.markdown(
                            f'<span class="badge badge-{confidence}">{label}</span>',
                            unsafe_allow_html=True,
                        )
                    if sources:
                        chips = "".join(
                            f'<span class="chip">{s["filename"]}'
                            + (f" · p.{s['page_number']}" if s.get("page_number") else "")
                            + "</span>"
                            for s in sources
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "confidence": confidence,
                        "sources": sources,
                    })

        if st.session_state.messages:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.messages = []
                st.rerun()

# ── Ingest ────────────────────────────────────────────────────────────────────

elif page == "⬆ Ingest":
    st.header("Ingest Documents")

    if st.session_state.ingest_summary:
        s = st.session_state.ingest_summary
        parts = [
            f"**{s['ingested']} ingested**",
            f"{s['skipped']} skipped",
            f"{s['failed']} failed",
        ]
        summary_line = " · ".join(parts) + f" — {s['total']} file(s) processed"
        if s["failed"] > 0:
            st.warning(summary_line)
        elif s["skipped"] == s["total"]:
            st.info(summary_line)
        else:
            st.success(summary_line)

    tab_upload, tab_folder = st.tabs(["Upload files", "Scan folder"])

    with tab_upload:
        st.write("Upload PDF, TXT, or Markdown files to add them to your knowledge base.")
        uploaded = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Ingest", type="primary", key="btn_upload"):
            docs_dir = Path(cfg.ingestion_folder())
            if not docs_dir.is_absolute():
                docs_dir = ROOT / docs_dir
            docs_dir.mkdir(parents=True, exist_ok=True)

            ingested = skipped = failed = 0
            bar = st.progress(0, text="Ingesting…")
            for i, uf in enumerate(uploaded):
                dest = docs_dir / uf.name
                dest.write_bytes(uf.getvalue())
                ok, msg = _ingest(dest)
                if ok:
                    ingested += 1
                elif "skipped" in msg:
                    skipped += 1
                else:
                    failed += 1
                bar.progress((i + 1) / len(uploaded))
            bar.empty()
            st.session_state.ingest_summary = {
                "ingested": ingested, "skipped": skipped,
                "failed": failed, "total": len(uploaded),
            }
            st.rerun()

    with tab_folder:
        st.write("Scan a local folder and ingest all supported files.")
        folder_input = st.text_input(
            "Folder path",
            value=str((ROOT / cfg.ingestion_folder()).resolve()),
        )
        if st.button("Scan and ingest", type="primary", key="btn_folder"):
            folder = Path(folder_input)
            if not folder.exists():
                st.error(f"Folder not found: `{folder_input}`")
            else:
                files = scan_folder(str(folder), cfg.supported_extensions())
                if not files:
                    st.info("No supported files found in that folder.")
                else:
                    ingested = skipped = failed = 0
                    bar = st.progress(0, text=f"Found {len(files)} file(s)…")
                    for i, fp in enumerate(files):
                        ok, msg = _ingest(fp)
                        if ok:
                            ingested += 1
                        elif "skipped" in msg:
                            skipped += 1
                        else:
                            failed += 1
                        bar.progress((i + 1) / len(files))
                    bar.empty()
                    st.session_state.ingest_summary = {
                        "ingested": ingested, "skipped": skipped,
                        "failed": failed, "total": len(files),
                    }
                    st.rerun()

# ── Documents ─────────────────────────────────────────────────────────────────

elif page == "🗂 Documents":
    st.header("Documents")

    docs = list_documents()
    if not docs:
        st.info("No documents ingested yet. Go to **⬆ Ingest** to add some.")
    else:
        hcols = st.columns([4, 1, 1, 1, 2, 1])
        for col, label in zip(hcols, ["Name", "Type", "Pages", "Chunks", "Ingested", ""]):
            col.markdown(f"**{label}**")
        st.divider()

        for doc in docs:
            c1, c2, c3, c4, c5, c6 = st.columns([4, 1, 1, 1, 2, 1])
            c1.write(doc.filename)
            c2.write(doc.extension)
            c3.write(str(doc.page_count) if doc.page_count else "—")
            c4.write(str(doc.chunk_count))
            c5.caption(doc.ingested_at[:19].replace("T", " "))
            with c6.popover("Remove"):
                st.markdown(f"Remove **{doc.filename}**?")
                if st.button("Confirm", key=f"confirm_{doc.id}", type="primary"):
                    delete_document(doc.id)
                    delete_by_doc_id(doc.id, workspace=cfg.workspace())
                    st.rerun()
