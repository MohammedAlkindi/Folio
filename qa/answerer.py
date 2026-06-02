import json
import logging
import os

from folio.core.config import Config
from folio.core.retry import with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise document analyst. Answer the question "
    "using only the provided document excerpts. Every factual "
    "claim must be traceable to a specific excerpt. If the "
    "excerpts do not contain enough information to answer "
    "confidently, say so explicitly. Never fabricate information."
)

_ANSWER_TOOL = {
    "name": "structured_answer",
    "description": "Return a structured answer with citations and confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question, with inline citations (Source: filename, p.N).",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "page_number": {"type": ["integer", "null"]},
                    },
                    "required": ["filename", "page_number"],
                },
                "description": "List of source documents cited in the answer.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "insufficient_data"],
                "description": "Confidence in the answer based on the quality and completeness of the excerpts.",
            },
        },
        "required": ["answer", "sources", "confidence"],
    },
}


def _build_user_message(question: str, chunks: list[dict]) -> str:
    lines = [f"Question: {question}", "", "Document excerpts:"]
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number")
        page_label = f"page {page}" if page else "no page"
        lines.append(f"[{i}] {chunk['filename']} ({page_label}):")
        lines.append(chunk["text"])
        lines.append("")
    lines.append("Answer with citations in the format: (Source: filename, p.N)")
    return "\n".join(lines)


def answer(
    question: str,
    chunks: list[dict],
    cfg: Config,
) -> dict:
    """
    Send question + retrieved chunks to Claude via tool use.
    Returns {answer, sources, confidence}.
    Falls back to {answer: error message, sources: [], confidence: "insufficient_data"} on failure.
    """
    if not chunks:
        return {
            "answer": "No relevant document excerpts found. Please ingest documents first.",
            "sources": [],
            "confidence": "insufficient_data",
        }

    api_key = cfg.qa_api_key()
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Export it before running queries."
        )

    try:
        import anthropic  # noqa: PLC0415 — lazy import; only needed for query command
    except ImportError:
        raise ImportError("anthropic package is required: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    @with_retry(max_attempts=3, base_delay=2.0, label="Claude QA")
    def _call() -> dict:
        user_message = _build_user_message(question, chunks)
        response = client.messages.create(
            model=cfg.qa_model(),
            max_tokens=cfg.qa_max_tokens(),
            system=_SYSTEM_PROMPT,
            tools=[_ANSWER_TOOL],
            # Force the model to use our structured tool — no free-form fallback.
            tool_choice={"type": "tool", "name": "structured_answer"},
            messages=[{"role": "user", "content": user_message}],
        )
        # Extract tool-use block from response.
        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_answer":
                return block.input  # type: ignore[return-value]
        # Assumption: if tool_choice is forced, this branch should not be reached.
        logger.warning("Claude response contained no tool_use block — returning raw text")
        text = " ".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        return {
            "answer": text,
            "sources": [],
            "confidence": "low",
        }

    try:
        return _call()
    except Exception as exc:
        logger.error("QA call failed: %s", exc)
        return {
            "answer": f"Error calling Claude: {exc}",
            "sources": [],
            "confidence": "insufficient_data",
        }
