"""Generate test fixture files. Called once per pytest session by conftest.py."""
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

_SAMPLE_TXT = FIXTURES_DIR / "sample.txt"

_MULTIPAGE_CONTENT = """\
Section One: Introduction to Document Processing

Document processing involves extracting structured information from unstructured text.
Modern pipelines combine parsing, chunking, embedding, and retrieval to enable semantic
search across large corpora. The first section establishes the foundational concepts
that will be expanded upon in subsequent sections of this document.
\f
Section Two: Embedding and Vector Stores

Embeddings transform text into dense numerical vectors that capture semantic meaning.
Similar documents cluster together in the high-dimensional embedding space. Vector
databases like ChromaDB store these embeddings and support approximate nearest neighbor
queries. The second section covers the technical details of how vectors are created
and stored efficiently for fast retrieval at query time.
\f
Section Three: Retrieval and Question Answering

Retrieval-augmented generation combines a retrieval step with a generative model.
The retrieval step finds the most semantically relevant chunks from the corpus.
The generative model synthesizes an answer grounded in those chunks. Citations are
produced by tracking which source document and page each chunk came from. This section
concludes the overview of the end-to-end pipeline architecture.
"""


def generate_all() -> None:
    """Idempotent — safe to call multiple times; will not overwrite sample.txt."""
    # sample.txt is committed to the repo; create it only if missing.
    if not _SAMPLE_TXT.exists():
        _SAMPLE_TXT.write_text(
            "This is a fallback sample fixture. The committed version should be present.",
            encoding="utf-8",
        )

    multipage = FIXTURES_DIR / "sample_multipage.txt"
    multipage.write_text(_MULTIPAGE_CONTENT, encoding="utf-8")
