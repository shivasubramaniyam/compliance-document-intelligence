"""
Unit tests for LLM Extractor with Ollama and OpenRouter support.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from src.extraction.llm_extractor import (
    _call_ollama,
    _parse_llm_response,
    _validate_and_normalize,
    get_default_provider,
    llm_extract_fields,
    ExtractionMetrics,
)
from src.ingestion.loader import IngestedDocument


class TestLLMExtractor(unittest.TestCase):

    def test_parse_llm_response_json(self):
        """Test parsing raw JSON string."""
        raw = '{"doc_type": "sop", "title": "Batch Review SOP", "version": "1.0"}'
        parsed = _parse_llm_response(raw)
        self.assertEqual(parsed, {"doc_type": "sop", "title": "Batch Review SOP", "version": "1.0"})

    def test_parse_llm_response_markdown(self):
        """Test parsing JSON inside markdown code block."""
        raw = "Here is the extracted data:\n```json\n{\"doc_type\": \"adverse_event\", \"severity\": \"high\"}\n```"
        parsed = _parse_llm_response(raw)
        self.assertEqual(parsed, {"doc_type": "adverse_event", "severity": "high"})

    def test_validate_and_normalize(self):
        """Test data normalization against schema."""
        raw_data = {
            "docType": "adverse_event",
            "event_id": "AE-2024-001",
            "severity": "HIGH",
            "follow_up_status": "in-progress",
            "date_reported": "2024-06-20",
        }
        normalized = _validate_and_normalize(raw_data, "adverse_event")
        self.assertEqual(normalized["severity"], "high")
        self.assertEqual(normalized["follow_up_status"], "in_progress")
        self.assertEqual(normalized["date_reported"], "2024-06-20")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_default_provider_without_env(self):
        """Default provider should be 'ollama' when no env vars are set."""
        self.assertEqual(get_default_provider(), "ollama")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"})
    def test_get_default_provider_with_openrouter_key(self):
        """Default provider should be 'openrouter' when OPENROUTER_API_KEY is present."""
        self.assertEqual(get_default_provider(), "openrouter")

    @patch.dict("os.environ", {"LLM_PROVIDER": "ollama", "OPENROUTER_API_KEY": "sk-or-test"})
    def test_get_default_provider_explicit_override(self):
        """Explicit LLM_PROVIDER should override OPENROUTER_API_KEY."""
        self.assertEqual(get_default_provider(), "ollama")

    @patch("httpx.Client")
    def test_call_ollama_success(self, mock_client_cls):
        """Test successful call to Ollama."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '{"title": "Test SOP", "version": "1.0"}'},
            "prompt_eval_count": 50,
            "eval_count": 20,
        }
        mock_client.post.return_value = mock_response

        messages = [{"role": "user", "content": "Extract fields"}]
        text, tokens_in, tokens_out, latency = _call_ollama(messages, model="llama3.2:3b")

        self.assertIn("Test SOP", text)
        self.assertEqual(tokens_in, 50)
        self.assertEqual(tokens_out, 20)
        self.assertGreaterEqual(latency, 0.0)

    @patch("src.extraction.llm_extractor._call_ollama")
    def test_llm_extract_fields_with_ollama(self, mock_call_ollama):
        """Test end-to-end extraction with Ollama provider."""
        mock_call_ollama.return_value = (
            '{"title": "Standard Operating Procedure", "version": "1.0", "effective_date": "2024-01-01"}',
            30,
            15,
            120.0,
        )

        doc = IngestedDocument(
            doc_id="sop_001",
            file_path="dummy.pdf",
            file_format="pdf",
            raw_text="Title: Standard Operating Procedure\nVersion: 1.0",
            doc_type_guess="sop",
        )

        result = llm_extract_fields(doc, provider="ollama", model="llama3.2:3b")
        self.assertEqual(result["doc_type"], "sop")
        self.assertEqual(result["title"], "Standard Operating Procedure")
        self.assertEqual(result["version"], "1.0")


if __name__ == "__main__":
    unittest.main()
