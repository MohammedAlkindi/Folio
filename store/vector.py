import logging

import chromadb

from folio.core.paths import CHROMA_DIR, ensure_dirs
from folio.store.models import Chunk

logger = logging.getLogger(__name__)

# Module-level client cache — one client per process.
_client: chromadb.Client | None = None


def _get_client() -> chromadb.Client:
    global _client
    if _client is None:
        ensure_dirs()
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(name: str = "folio") -> chromadb.Collection:
    client = _get_client()
    # get_or_create is idempotent — safe to call on every startup.
    return client.get_or_create_collection(
        name=name,
        # cosine distance is more stable than L2 for sentence-transformer vectors.
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection_name: str = "folio",
) -> None:
    if not chunks:
        return
    collection = get_collection(collection_name)
    collection.upsert(
        documents=[c.text for c in chunks],
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "filename": c.filename,
                # ChromaDB metadata values must be str/int/float/bool — None is not allowed.
                # Assumption: page_number=0 means "not applicable" (plain-text files).
                "page_number": c.page_number if c.page_number is not None else 0,
                "chunk_index": c.chunk_index,
                "doc_id": c.doc_id,
            }
            for c in chunks
        ],
    )
    logger.debug("Upserted %d chunks into collection '%s'", len(chunks), collection_name)


def query(
    embedding: list[float],
    top_k: int,
    collection_name: str = "folio",
) -> list[dict]:
    """Return top_k results as list of {text, filename, page_number, chunk_index, distance}."""
    collection = get_collection(collection_name)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, distances):
        page = meta.get("page_number", 0)
        output.append(
            {
                "text": text,
                "filename": meta.get("filename", ""),
                # Convert sentinel 0 back to None for plain-text files.
                "page_number": page if page != 0 else None,
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist,
            }
        )
    return output


def delete_by_doc_id(doc_id: str, collection_name: str = "folio") -> None:
    collection = get_collection(collection_name)
    collection.delete(where={"doc_id": doc_id})
    logger.debug("Deleted vectors for doc_id=%s", doc_id)
