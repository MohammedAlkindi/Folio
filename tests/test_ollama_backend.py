"""Tests for qa.backends.ollama_backend — JSON fence stripping and response shapes."""
import json
from unittest.mock import MagicMock, patch

from core.config import Config
from qa.backends.ollama_backend import OllamaBackend


def _chunk(text: str = "some context text") -> dict:
    return {"text": text, "filename": "doc.pdf", "page_number": 1, "chunk_index": 0}


def _cfg() -> Config:
    return Config({
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


def _attr_response(content: str) -> MagicMock:
    """Attribute-style response: response.message.content."""
    msg = MagicMock()
    msg.content = content
    response = MagicMock()
    response.message = msg
    return response


def _dict_response(content: str) -> dict:
    """Dict-style response: response["message"]["content"]."""
    return {"message": {"content": content}}


def _mock_ollama(return_value) -> MagicMock:
    mock = MagicMock()
    mock.Client.return_value.chat.return_value = return_value
    return mock


# ── JSON fence stripping ─────────────────────────────────────────────────────

def test_strips_json_fenced_response():
    """Output wrapped in ```json ... ``` is parsed correctly."""
    payload = {"answer": "42", "sources": [], "confidence": "high"}
    raw = f"```json\n{json.dumps(payload)}\n```"

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_attr_response(raw))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result == payload


def test_strips_plain_fenced_response():
    """Output wrapped in ``` (no language tag) is parsed correctly."""
    payload = {"answer": "hello", "sources": [], "confidence": "medium"}
    raw = f"```\n{json.dumps(payload)}\n```"

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_attr_response(raw))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result == payload


def test_plain_json_response_no_fences():
    """Plain JSON with no fences is parsed directly."""
    payload = {"answer": "direct", "sources": [], "confidence": "low"}

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_attr_response(json.dumps(payload)))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result == payload


# ── response shape handling ───────────────────────────────────────────────────

def test_attribute_style_response_is_handled():
    """Object with .message.content attribute is read correctly."""
    payload = {"answer": "attr style", "sources": [], "confidence": "high"}

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_attr_response(json.dumps(payload)))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result == payload


def test_dict_style_response_is_handled():
    """Dict response["message"]["content"] is read correctly."""
    payload = {"answer": "dict style", "sources": [], "confidence": "high"}

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_dict_response(json.dumps(payload)))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result == payload


# ── error handling ────────────────────────────────────────────────────────────

def test_ollama_request_failure_returns_error_dict():
    """When the Ollama client raises, the backend returns an error result."""
    mock = MagicMock()
    mock.Client.return_value.chat.side_effect = ConnectionError("connection refused")

    with patch.dict("sys.modules", {"ollama": mock}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result["confidence"] == "insufficient_data"
    assert "connection refused" in result["answer"]
    assert result["sources"] == []


def test_json_parse_failure_returns_raw_as_answer():
    """Non-JSON output is returned as the answer text with low confidence."""
    raw = "I cannot answer that question."

    with patch.dict("sys.modules", {"ollama": _mock_ollama(_attr_response(raw))}):
        result = OllamaBackend().answer("q", [_chunk()], _cfg())

    assert result["confidence"] == "low"
    assert result["answer"] == raw
    assert result["sources"] == []
