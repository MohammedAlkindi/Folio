from dataclasses import dataclass, field


@dataclass
class Document:
    id: str          # sha256 of absolute path string
    filename: str
    filepath: str
    extension: str
    page_count: int | None
    chunk_count: int
    ingested_at: str  # ISO 8601


@dataclass
class Chunk:
    id: str           # f"{doc_id}:chunk:{index}"
    doc_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    text: str
    token_estimate: int  # len(text.split()) — word count as proxy for tokens
    preview: str = field(default="")  # first 200 chars, stored in SQLite for auditing
