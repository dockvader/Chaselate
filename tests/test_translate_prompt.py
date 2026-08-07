"""Tests for chaselate.translate: pure prompt/message building and the Ollama HTTP client.

No real network calls -- requests.Session is mocked wherever HTTP would happen.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from chaselate.translate import (
    ContextPair,
    OllamaClient,
    OllamaModel,
    OllamaModelMissing,
    OllamaUnavailable,
    build_instruction,
    build_messages,
)


# -- build_instruction ---------------------------------------------------------

def test_build_instruction_includes_language_names_and_codes():
    text = build_instruction("en", "zh-TW", "hello")
    assert "English" in text
    assert "en" in text
    assert "Traditional Chinese" in text
    assert "zh-TW" in text


def test_build_instruction_has_two_blank_lines_before_payload():
    text = build_instruction("en", "zh-TW", "hello world")
    assert "\n\n\nhello world" in text


def test_build_instruction_includes_extra_instructions():
    text = build_instruction("en", "ja", "hi", extra_instructions="Use a formal tone.")
    assert "Use a formal tone." in text


def test_build_instruction_unknown_source_lang_uses_fallback_not_none():
    text = build_instruction(None, "ja", "hi")
    assert "None" not in text
    assert "the source language" in text


def test_build_instruction_unrecognized_source_code_also_falls_back():
    text = build_instruction("not-a-real-lang", "ja", "hi")
    assert "None" not in text


# -- build_messages ---------------------------------------------------------

def test_build_messages_no_context_yields_single_user_message():
    messages = build_messages("en", "ja", "hello")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_build_messages_with_context_produces_alternating_roles():
    context = [
        ContextPair(source="hi", translation="こんにちは"),
        ContextPair(source="bye", translation="さようなら"),
    ]
    messages = build_messages("en", "ja", "how are you", context)
    assert len(messages) == 2 * len(context) + 1
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["role"] == "user"


def test_build_messages_skips_context_pairs_with_empty_source_or_translation():
    context = [
        ContextPair(source="", translation="empty source"),
        ContextPair(source="valid", translation=""),
        ContextPair(source="ok", translation="good"),
    ]
    messages = build_messages("en", "ja", "final text", context)
    # Only the one valid pair should turn into 2 messages, plus the final user message.
    assert len(messages) == 3


# -- OllamaModel ---------------------------------------------------------

def test_ollama_model_label_and_specialist_flag():
    specialist = OllamaModel(name="translategemma:latest", size=int(4.5e9))
    generic = OllamaModel(name="llama3:8b", size=int(4.7e9))
    assert specialist.is_translation_specialist is True
    assert generic.is_translation_specialist is False
    assert "translategemma:latest" in specialist.label


def test_ollama_model_size_gb_computed_from_bytes():
    model = OllamaModel(name="x", size=2_000_000_000)
    assert model.size_gb == pytest.approx(2.0)


def test_ollama_model_size_gb_zero_when_size_unknown():
    model = OllamaModel(name="x")
    assert model.size_gb == 0.0


# -- OllamaClient.base_url ---------------------------------------------------------

def test_base_url_adds_missing_scheme():
    from chaselate.config import TranslateConfig

    client = OllamaClient(TranslateConfig(base_url="localhost:11434"))
    assert client.base_url == "http://localhost:11434/"


def test_base_url_handles_trailing_slash():
    from chaselate.config import TranslateConfig

    client = OllamaClient(TranslateConfig(base_url="http://127.0.0.1:11434/"))
    assert client.base_url == "http://127.0.0.1:11434/"


def test_url_builds_correct_endpoint():
    from chaselate.config import TranslateConfig

    client = OllamaClient(TranslateConfig(base_url="http://127.0.0.1:11434"))
    assert client._url("api/tags") == "http://127.0.0.1:11434/api/tags"


# -- list_models (mocked HTTP) ---------------------------------------------------------

def _client_with_mock_session():
    client = OllamaClient()
    client._session = MagicMock(spec=requests.Session)
    return client


def test_list_models_parses_normal_payload():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "models": [
            {
                "name": "translategemma:latest",
                "size": 5_000_000_000,
                "details": {"family": "gemma", "parameter_size": "4B", "quantization_level": "Q4_0"},
                "modified_at": "2024-01-01",
            },
            {"name": "llama3:8b", "size": 4_700_000_000, "details": {}},
        ]
    }
    client._session.get.return_value = resp

    models = client.list_models()
    assert len(models) == 2
    # Translation specialist sorts first.
    assert models[0].name == "translategemma:latest"
    assert models[1].name == "llama3:8b"


def test_list_models_handles_empty_payload():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {}
    client._session.get.return_value = resp

    assert client.list_models() == []


def test_list_models_skips_entries_missing_name():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"size": 123}, {"name": "ok-model"}]}
    client._session.get.return_value = resp

    models = client.list_models()
    assert len(models) == 1
    assert models[0].name == "ok-model"


def test_request_404_raises_model_missing_with_pull_hint():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.status_code = 404
    client._session.post.return_value = resp

    with pytest.raises(OllamaModelMissing) as excinfo:
        client._request({"model": "missing-model"}, stream=False)
    assert "ollama pull" in str(excinfo.value)
    assert "missing-model" in str(excinfo.value)


def test_list_models_connection_failure_raises_unavailable():
    client = _client_with_mock_session()
    client._session.get.side_effect = requests.exceptions.ConnectionError("refused")
    with pytest.raises(OllamaUnavailable):
        client.list_models()


def test_health_returns_false_and_message_on_failure_without_raising():
    client = _client_with_mock_session()
    client._session.get.side_effect = requests.exceptions.ConnectionError("refused")
    ok, message = client.health()
    assert ok is False
    assert isinstance(message, str)
    assert message  # non-empty


def test_health_returns_true_on_success():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"version": "0.5.1"}
    client._session.get.return_value = resp
    ok, message = client.health()
    assert ok is True
    assert "0.5.1" in message


# -- _stream_chat parsing ---------------------------------------------------------

def _lines(*jsons):
    import json as _json

    return [_json.dumps(j).encode("utf-8") for j in jsons]


def test_stream_chat_accumulates_deltas_and_calls_on_delta():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = _lines(
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True},
    )
    client._session.post.return_value = resp

    seen = []
    result = client._stream_chat({"model": "x"}, on_delta=seen.append, cancel=None)
    assert result == "Hello world"
    assert seen[-1] == "Hello world"


def test_stream_chat_raises_on_error_event():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = _lines({"error": "model exploded"})
    client._session.post.return_value = resp

    with pytest.raises(Exception) as excinfo:
        client._stream_chat({"model": "x"}, on_delta=None, cancel=None)
    assert "model exploded" in str(excinfo.value)


def test_stream_chat_respects_cancel_event():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = _lines(
        {"message": {"content": "partial"}, "done": False},
        {"message": {"content": "more"}, "done": False},
    )
    client._session.post.return_value = resp

    cancel = threading.Event()
    cancel.set()
    from chaselate.translate import TranslationCancelled

    with pytest.raises(TranslationCancelled):
        client._stream_chat({"model": "x"}, on_delta=None, cancel=cancel)


def test_stream_chat_skips_unparsable_lines():
    client = _client_with_mock_session()
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = [
        b"not json at all",
        b'{"message": {"content": "ok"}, "done": true}',
    ]
    client._session.post.return_value = resp

    result = client._stream_chat({"model": "x"}, on_delta=None, cancel=None)
    assert result == "ok"
