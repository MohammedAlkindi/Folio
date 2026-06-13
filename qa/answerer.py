import logging

from core.config import Config

logger = logging.getLogger(__name__)


def answer(
    question: str,
    chunks: list[dict],
    cfg: Config,
) -> dict:
    """
    Dispatch question + retrieved chunks to the configured LLM backend.
    Returns {answer, sources, confidence}.
    Falls back to {answer: error message, sources: [], confidence: "insufficient_data"} on failure.
    """
    if not chunks:
        return {
            "answer": "No relevant document excerpts found. Please ingest documents first.",
            "sources": [],
            "confidence": "insufficient_data",
        }

    if cfg.qa_provider() == "ollama":
        from qa.backends.ollama_backend import OllamaBackend  # noqa: PLC0415
        backend: object = OllamaBackend()
    else:
        from qa.backends.anthropic_backend import AnthropicBackend  # noqa: PLC0415
        backend = AnthropicBackend()

    return backend.answer(question, chunks, cfg)  # type: ignore[union-attr]
