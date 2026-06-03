import logging
from collections import Counter

logger = logging.getLogger(__name__)

_SENTENCE_ENDINGS = {".", "!", "?", '."', '!"', '?"'}
_MAX_SENTENCE_LOOKAHEAD = 50  # words to scan forward for a sentence boundary


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Split page texts into overlapping word-window chunks.
    Returns list of {text, page_number, page_start, page_end, chunk_index}.
    chunk_size and overlap are in words (approximate token count).
    page_number is the page containing the majority of words in the chunk;
    ties are broken by taking the higher page number.
    page_start/page_end are the pages of the first/last word (debug metadata).
    chunk_index is 0-based sequential across the document.
    """
    if not pages:
        return []

    # Build a flat word list and a parallel page-number array.
    words: list[str] = []
    word_pages: list[int | None] = []

    for page in pages:
        page_words = page["text"].split()
        words.extend(page_words)
        word_pages.extend([page["page_number"]] * len(page_words))

    if not words:
        return []

    chunks: list[dict] = []
    start = 0
    total = len(words)

    while start < total:
        end = min(start + chunk_size, total)

        # Try to extend end to the next sentence boundary (within lookahead).
        if end < total:
            for offset in range(_MAX_SENTENCE_LOOKAHEAD):
                probe = end + offset
                if probe >= total:
                    end = total
                    break
                # Check if the word ends with sentence-terminal punctuation.
                w = words[probe]
                if any(w.endswith(p) for p in _SENTENCE_ENDINGS):
                    end = probe + 1
                    break

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        page_slice = word_pages[start:end]

        counts = Counter(page_slice)
        max_count = max(counts.values())
        # Among pages tied for most words, pick the highest page number.
        page_number = max(p for p, c in counts.items() if c == max_count)

        chunks.append(
            {
                "text": chunk_text,
                "page_number": page_number,
                "page_start": word_pages[start],
                "page_end": word_pages[end - 1],
                "chunk_index": len(chunks),
            }
        )

        # All words consumed — stop before generating tail overlap-only chunks.
        if end >= total:
            break

        next_start = end - overlap
        # Guarantee forward progress — avoid infinite loops on very short documents.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    logger.debug("Produced %d chunks from %d words", len(chunks), total)
    return chunks
