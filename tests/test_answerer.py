"""Tests for qa.answerer dispatch and qa.backends.anthropic_backend logic."""
from unittest.mock import MagicMock, patch

import pytest

from core.config import Config


def _chunk(text: str = "some text", filename: str = "doc.pdf", page: int = 1) -> dict:
    return {"text": text, "filename": filename, "page_number": page, "chunk_index": 0}


def _cfg(budget: int = 3500, provider: str = "anthropic") -> Config:
    return Config({
        "workspace": "test",
        "ingestion": {
            "folder": "docs/",
            "supported_extensions": [".pdf"],
            "chunk_size": 50,
            "chunk_overlap": 10,
        },
        "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "retrieval": {"top_k": 3},
        "qa": {
            "provider": provider,
            "model": "claude-sonnet-4-6",
            "api_key_env": "ANTHROPIC_API_KEY",
            "max_tokens": 512,
            "context_budget": budget,
        },
    })


# ── dispatch ─────────────────────────────────────────────────────────────────

def test_empty_chunks_returns_no_content_response():
    from qa.answerer import answer
    result = answer("What is this?", [], _cfg())
    assert result["confidence"] == "insufficient_data"
    assert result["sources"] == []
    assert "No relevant" in result["answer"]


def test_dispatches_to_anthropic_backend_by_default():
    expected = {"answer": "test answer", "sources": [], "confidence": "high"}
    with patch("qa.backends.anthropic_backend.AnthropicBackend.answer", return_value=expected):
        from qa.answerer import answer
        result = answer("q", [_chunk()], _cfg(provider="anthropic"))
    assert result == expected


def test_dispatches_to_ollama_backend_when_configured():
    cfg_ollama = Config({
        "workspace": "test",
        "ingestion": {"folder": "docs/", "supported_extensions": [".pdf"], "chunk_size": 50, "chunk_overlap": 10},
        "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
        "retrieval": {"top_k": 3},
        "qa": {
            "provider": "ollama",
            "ollama_model": "llama3",
            "ollama_host": "http://localhost:11434",
            "max_tokens": 512,
        },
    })
    expected = {"answer": "ollama answer", "sources": [], "confidence": "medium"}
    with patch("qa.backends.ollama_backend.OllamaBackend.answer", return_value=expected):
        from qa.answerer import answer
        result = answer("q", [_chunk()], cfg_ollama)
    assert result == expected


# ── context trimming ─────────────────────────────────────────────────────────

def test_context_trimming_drops_chunks_when_over_budget():
    """Chunks exceeding the word budget are dropped from the tail."""
    # budget=10: overhead = len("q".split()) + 100 = 101 > 10,
    # so while loop keeps popping until only 1 chunk remains.
    cfg = _cfg(budget=10)
    chunks = [_chunk("word " * 20, filename=f"doc{i}.pdf") for i in range(3)]

    captured: dict = {}

    def fake_invoke(_client, _model, _max_tokens, user_message: str) -> dict:
        captured["msg"] = user_message
        return {"answer": "ok", "sources": [], "confidence": "high"}

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
         patch("qa.backends.anthropic_backend._invoke_claude", side_effect=fake_invoke):
        from qa.backends.anthropic_backend import AnthropicBackend
        AnthropicBackend().answer("q", chunks, cfg)

    msg = captured["msg"]
    assert "doc0.pdf" in msg
    assert "doc1.pdf" not in msg
    assert "doc2.pdf" not in msg


def test_context_trimming_keeps_all_chunks_when_within_budget():
    """No chunks are dropped when total word count fits the budget."""
    cfg = _cfg(budget=100_000)
    chunks = [_chunk("short text", filename=f"doc{i}.pdf") for i in range(3)]

    captured: dict = {}

    def fake_invoke(_client, _model, _max_tokens, user_message: str) -> dict:
        captured["msg"] = user_message
        return {"answer": "ok", "sources": [], "confidence": "high"}

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
         patch("qa.backends.anthropic_backend._invoke_claude", side_effect=fake_invoke):
        from qa.backends.anthropic_backend import AnthropicBackend
        AnthropicBackend().answer("q", chunks, cfg)

    msg = captured["msg"]
    for i in range(3):
        assert f"doc{i}.pdf" in msg


def test_missing_api_key_raises_environment_error():
    from qa.backends.anthropic_backend import AnthropicBackend
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            AnthropicBackend().answer("q", [_chunk()], _cfg())


# ── tool-use parsing ─────────────────────────────────────────────────────────

def test_invoke_claude_parses_tool_use_block():
    """_invoke_claude returns block.input from the structured_answer tool_use block."""
    import anthropic
    from qa.backends.anthropic_backend import _invoke_claude

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "structured_answer"
    tool_block.input = {"answer": "42", "sources": [], "confidence": "high"}

    mock_response = MagicMock()
    mock_response.content = [tool_block]

    client = anthropic.Anthropic(api_key="sk-test")
    with patch.object(client.messages, "create", return_value=mock_response):
        result = _invoke_claude(client, "claude-sonnet-4-6", 512, "test message")

    assert result == {"answer": "42", "sources": [], "confidence": "high"}


def test_invoke_claude_falls_back_to_text_when_no_tool_use_block():
    """When the response has no tool_use block, text content is returned with low confidence."""
    import anthropic
    from qa.backends.anthropic_backend import _invoke_claude

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "plain text response"

    mock_response = MagicMock()
    mock_response.content = [text_block]

    client = anthropic.Anthropic(api_key="sk-test")
    with patch.object(client.messages, "create", return_value=mock_response):
        result = _invoke_claude(client, "claude-sonnet-4-6", 512, "test message")

    assert result["confidence"] == "low"
    assert "plain text response" in result["answer"]
    assert result["sources"] == []
