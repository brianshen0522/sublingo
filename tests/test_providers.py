"""Tests for provider abstraction and JSON parsing."""

import json

import pytest

from sublingo.providers.base import BaseLLMProvider, extract_json_array


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__(model="test", temperature=0.3)
        self.responses = responses or []
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
            self._call_count += 1
            return resp
        raise RuntimeError("No more mock responses")


class TestExtractJsonArray:
    def test_plain_json(self):
        text = '[{"index": 0, "text": "Hello"}]'
        result = extract_json_array(text)
        assert result == [{"index": 0, "text": "Hello"}]

    def test_markdown_fenced(self):
        text = '```json\n[{"index": 0, "text": "Hello"}]\n```'
        result = extract_json_array(text)
        assert result == [{"index": 0, "text": "Hello"}]

    def test_with_surrounding_text(self):
        text = 'Here is the translation:\n[{"index": 0, "text": "Hola"}]\nDone.'
        result = extract_json_array(text)
        assert result == [{"index": 0, "text": "Hola"}]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            extract_json_array("this is not json at all")


class TestMockProvider:
    def test_translate_success(self):
        response = json.dumps([
            {"index": 0, "text": "Hola, ¿cómo estás?"},
            {"index": 1, "text": "Estoy bien, gracias."},
        ])
        provider = MockProvider(responses=[response])
        texts = [
            {"index": 0, "text": "Hello, how are you?"},
            {"index": 1, "text": "I'm doing well, thank you."},
        ]
        result = provider.translate(texts, "English", "Spanish")
        assert len(result) == 2
        assert result[0]["text"] == "Hola, ¿cómo estás?"

    def test_translate_retry_on_bad_json(self):
        good = json.dumps([{"index": 0, "text": "Hola"}])
        provider = MockProvider(responses=["not json", good])
        texts = [{"index": 0, "text": "Hello"}]
        result = provider.translate(texts, "English", "Spanish")
        assert result[0]["text"] == "Hola"

    def test_translate_all_retries_fail(self):
        provider = MockProvider(responses=["bad", "bad", "bad"])
        texts = [{"index": 0, "text": "Hello"}]
        with pytest.raises(RuntimeError, match="Failed to get valid translation"):
            provider.translate(texts, "English", "Spanish")
