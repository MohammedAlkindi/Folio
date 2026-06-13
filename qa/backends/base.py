from typing import Protocol, runtime_checkable

from core.config import Config


@runtime_checkable
class LLMBackend(Protocol):
    def answer(self, question: str, chunks: list[dict], cfg: Config) -> dict:
        """Return {answer, sources, confidence}."""
        ...
