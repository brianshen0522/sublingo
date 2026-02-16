"""Tests for the translation service."""

import json
from pathlib import Path
from unittest.mock import patch

from sublingo.services.translation_service import PROVIDERS, get_provider, translate_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_get_provider_openai():
    provider = get_provider({"provider": "openai", "api_key": "test"})
    assert provider.name == "openai"


def test_get_provider_ollama():
    provider = get_provider({"provider": "ollama"})
    assert provider.name == "ollama"


def test_get_provider_vllm():
    provider = get_provider({"provider": "vllm"})
    assert provider.name == "vllm"


def test_get_provider_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider({"provider": "nonexistent"})


def test_providers_dict():
    assert "openai" in PROVIDERS
    assert "ollama" in PROVIDERS
    assert "vllm" in PROVIDERS


def test_translate_file_e2e(tmp_path):
    """End-to-end test with mocked provider."""
    # Create mock responses for language detection + 1 batch
    lang_response = json.dumps({"language": "English", "code": "en"})
    translation_response = json.dumps([
        {"index": 0, "text": "こんにちは、お元気ですか？"},
        {"index": 1, "text": "元気です、ありがとう。"},
        {"index": 2, "text": "今日はなんて美しい日でしょう。"},
        {"index": 3, "text": "散歩に行きませんか？"},
        {"index": 4, "text": "それはいい考えですね！"},
    ])

    with patch("sublingo.services.translation_service.get_provider") as mock_get:
        mock_provider = mock_get.return_value
        mock_provider.name = "mock"
        mock_provider.model = "test"
        mock_provider._call_api_interruptible.return_value = lang_response
        mock_provider.translate.return_value = [
            {"index": i, "text": t} for i, t in enumerate([
                "こんにちは、お元気ですか？",
                "元気です、ありがとう。",
                "今日はなんて美しい日でしょう。",
                "散歩に行きませんか？",
                "それはいい考えですね！",
            ])
        ]

        config = {
            "provider": "openai",
            "target_language": "ja",
            "source_language": "auto",
            "batch_size": 20,
            "bilingual": False,
            "output_format": None,
            "temperature": 0.3,
        }

        output = tmp_path / "output.srt"
        result = translate_file(FIXTURES / "sample.srt", config, output_path=output)

        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "こんにちは" in content
