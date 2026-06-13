import os
from pathlib import Path
from typing import Any
import yaml


class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # ── ingestion ────────────────────────────────────────────────────────────

    def ingestion_folder(self) -> str:
        return self._data["ingestion"]["folder"]

    def supported_extensions(self) -> list[str]:
        return self._data["ingestion"]["supported_extensions"]

    def chunk_size(self) -> int:
        return int(self._data["ingestion"].get("chunk_size", 800))

    def chunk_overlap(self) -> int:
        return int(self._data["ingestion"].get("chunk_overlap", 150))

    # ── embedding ────────────────────────────────────────────────────────────

    def embedding_provider(self) -> str:
        return self._data["embedding"].get("provider", "sentence_transformers")

    def embedding_model(self) -> str:
        return self._data["embedding"].get("model", "all-MiniLM-L6-v2")

    # ── retrieval ────────────────────────────────────────────────────────────

    def retrieval_top_k(self) -> int:
        return int(self._data["retrieval"].get("top_k", 6))

    def retrieval_hybrid(self) -> bool:
        return bool(self._data["retrieval"].get("hybrid", True))

    # ── qa ───────────────────────────────────────────────────────────────────

    def qa_provider(self) -> str:
        return str(self._data["qa"].get("provider", "anthropic"))

    def qa_model(self) -> str:
        return self._data["qa"].get("model", "claude-sonnet-4-6")

    def qa_api_key(self) -> str:
        env_var = self._data["qa"].get("api_key_env", "ANTHROPIC_API_KEY")
        key = os.environ.get(env_var, "")
        # Defer the missing-key error to call time so ingest works without a key.
        return key

    def qa_max_tokens(self) -> int:
        return int(self._data["qa"].get("max_tokens", 2048))

    def qa_context_budget(self) -> int:
        return int(self._data["qa"].get("context_budget", 3500))

    def qa_ollama_model(self) -> str:
        return str(self._data["qa"].get("ollama_model", "llama3"))

    def qa_ollama_host(self) -> str:
        return str(self._data["qa"].get("ollama_host", "http://localhost:11434"))

    # ── workspace ────────────────────────────────────────────────────────────

    def workspace(self) -> str:
        # Assumption: top-level "workspace" key in config; falls back to "default".
        return str(self._data.get("workspace", "default"))


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy config/folio_config.example.yaml and adjust it."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
