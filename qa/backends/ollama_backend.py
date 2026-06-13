import json
import logging

from core.config import Config
from qa.backends.anthropic_backend import _SYSTEM_PROMPT, _build_user_message

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object, no markdown fences, no other text:\n"
    '{"answer": "...", '
    '"sources": [{"filename": "...", "page_number": <integer or null>}], '
    '"confidence": "high|medium|low|insufficient_data"}'
)


class OllamaBackend:
    def answer(self, question: str, chunks: list[dict], cfg: Config) -> dict:
        """Send question + chunks to a local Ollama model. Returns {answer, sources, confidence}."""
        try:
            import ollama  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "ollama package is required for the Ollama backend: pip install ollama"
            )

        user_message = _build_user_message(question, chunks)
        prompt = f"{_SYSTEM_PROMPT}\n\n{user_message}{_JSON_INSTRUCTION}"

        try:
            client = ollama.Client(host=cfg.qa_ollama_host())
            response = client.chat(
                model=cfg.qa_ollama_model(),
                messages=[{"role": "user", "content": prompt}],
            )
            # Handle both attribute-style and dict-style response objects.
            if hasattr(response, "message"):
                raw: str = response.message.content
            else:
                raw = response["message"]["content"]
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return {
                "answer": f"Error calling Ollama: {exc}",
                "sources": [],
                "confidence": "insufficient_data",
            }

        try:
            text = raw.strip()
            # Strip markdown code fences if the model wraps output in them.
            if text.startswith("```"):
                parts = text.split("```", 2)
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                text = inner.rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Ollama JSON parse failed; returning raw response as answer")
            return {"answer": raw, "sources": [], "confidence": "low"}
