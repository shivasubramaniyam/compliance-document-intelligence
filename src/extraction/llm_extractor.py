"""
LLM-Based Field Extraction — Phase 2

Uses OpenRouter API with structured output to extract fields from
compliance documents. Includes retry/validation logic, cost tracking, and
few-shot prompt templates per document type.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from ..ingestion.loader import IngestedDocument
from ..schemas import (
    DocumentType,
    FollowUpStatus,
    FindingStatus,
    SeverityLevel,
    get_schema,
)

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when OpenRouter returns 429 Too Many Requests."""
    def __init__(self, message: str, retry_after: float = 10.0):
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Load .env file if present
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env file from project root."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)


_load_env()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default model — free tier on OpenRouter (updated 2026-09)
# minimax-m3 is fast, returns clean JSON without thinking overhead
DEFAULT_MODEL = "minimax/minimax-m3:free"

# Backup free models (tried in order if default fails)
BACKUP_FREE_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "thinkingmachines/inkling:free",
    "minimax/minimax-m3:free",
]


# ---------------------------------------------------------------------------
# Cost & latency tracking
# ---------------------------------------------------------------------------

@dataclass
class ExtractionMetrics:
    """Tracks cost and latency for LLM extraction calls."""
    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    errors: int = 0
    retries: int = 0

    def log_call(self, tokens_in: int, tokens_out: int, latency_ms: float, cost_usd: float) -> None:
        self.total_calls += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd

    def summary(self) -> dict[str, Any]:
        avg_latency = self.total_latency_ms / self.total_calls if self.total_calls else 0
        return {
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_1k_docs": round(self.total_cost_usd / max(self.total_calls, 1) * 1000, 4),
            "errors": self.errors,
            "retries": self.retries,
        }


# Global metrics tracker
metrics = ExtractionMetrics()


# ---------------------------------------------------------------------------
# Prompt templates per document type
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a compliance document field extractor. Your ONLY job is to output a JSON object with the extracted fields.

CRITICAL RULES:
- Output ONLY a valid JSON object. No thinking, no reasoning, no explanation, no markdown.
- Start your response directly with { and end with }
- Extract ONLY fields present in the document text
- Use exact values — do not infer or fabricate
- Dates: YYYY-MM-DD format
- Enums (severity, status): lowercase
- Omit fields not present in the text (no null)
"""

FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "sop": [
        {
            "input": """Standard Operating Procedure: Batch Record Review

Version: 2.1
Effective Date: 2024-03-15
Owner: Dr. Sarah Chen
Department: Quality Assurance
Approval Date: 2024-03-10
Review Cycle: Annual

Scope: This procedure applies to all batch record review activities for manufactured pharmaceutical products.

References:
• ICH Q10
• 21 CFR Part 211
• EU GMP Annex 15""",
            "output": """{
  "doc_type": "sop",
  "title": "Standard Operating Procedure: Batch Record Review",
  "version": "2.1",
  "effective_date": "2024-03-15",
  "owner": "Dr. Sarah Chen",
  "department": "Quality Assurance",
  "approval_date": "2024-03-10",
  "review_cycle": "Annual",
  "scope": "This procedure applies to all batch record review activities for manufactured pharmaceutical products.",
  "references": ["ICH Q10", "21 CFR Part 211", "EU GMP Annex 15"]
}""",
        }
    ],
    "adverse_event": [
        {
            "input": """Adverse Event Report

Event ID: AE-2024-00123
Date Reported: 2024-06-20
Product ID: PROD-A1234
Batch ID: BATCH-123456
Severity: HIGH
Follow-Up Status: IN PROGRESS

Reporter: Dr. Michael Torres

Description of Event: Patient experienced nausea within 24 hours of administration of product batch.""",
            "output": """{
  "doc_type": "adverse_event",
  "event_id": "AE-2024-00123",
  "date_reported": "2024-06-20",
  "product_id": "PROD-A1234",
  "batch_id": "BATCH-123456",
  "severity": "high",
  "description": "Patient experienced nausea within 24 hours of administration of product batch.",
  "reporter": "Dr. Michael Torres",
  "follow_up_status": "in_progress"
}""",
        }
    ],
    "audit_finding": [
        {
            "input": """Audit Finding Report

Audit ID: AUD-2024-00045
Audit Date: 2024-07-10
Finding Category: Documentation
Severity: MEDIUM

Finding Description: Records reviewed during the audit showed that batch records were not completed within the required timeframe.

Corrective Action Required: Update documentation procedures and conduct a follow-up audit within 90 days.

Due Date: 2024-10-08
Responsible Party: James Wilson""",
            "output": """{
  "doc_type": "audit_finding",
  "audit_id": "AUD-2024-00045",
  "audit_date": "2024-07-10",
  "finding_category": "Documentation",
  "severity": "medium",
  "description": "Records reviewed during the audit showed that batch records were not completed within the required timeframe.",
  "corrective_action": "Update documentation procedures and conduct a follow-up audit within 90 days.",
  "due_date": "2024-10-08",
  "responsible_party": "James Wilson",
  "status": "open"
}""",
        }
    ],
}


def _build_prompt(doc_type: str, raw_text: str) -> list[dict[str, str]]:
    """Build the full prompt with system, few-shot examples, and user input."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    examples = FEW_SHOT_EXAMPLES.get(doc_type, [])
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})

    # Add the actual document
    messages.append({
        "role": "user",
        "content": f"Extract structured fields from this {doc_type} document:\n\n{raw_text}",
    })

    return messages# ---------------------------------------------------------------------------
# OpenRouter API client
# ---------------------------------------------------------------------------

def _discover_working_model(api_key: str) -> str:
    """Query OpenRouter for available free models and return the first one that works."""
    import httpx

    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        r.raise_for_status()
        models = r.json().get("data", [])
        free_ids = {m["id"] for m in models if ":free" in m.get("id", "")}

        # Try default first, then backups, then any free model
        candidates = [DEFAULT_MODEL] + [m for m in BACKUP_FREE_MODELS if m in free_ids]
        candidates += [m for m in sorted(free_ids) if m not in candidates]

        for model_id in candidates:
            logger.info(f"Probing model: {model_id}")
            try:
                probe_resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Say OK"}],
                        "max_tokens": 5,
                    },
                    timeout=30.0,
                )
                if probe_resp.status_code == 200:
                    logger.info(f"Using model: {model_id}")
                    return model_id
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Model discovery failed: {e}")

    return DEFAULT_MODEL


# Cached working model to avoid re-probing every call
_working_model: Optional[str] = None


def _get_model(api_key: str) -> str:
    """Get the model to use: env override, cached discovery, or default."""
    global _working_model

    # Explicit env override takes priority
    env_model = os.environ.get("OPENROUTER_MODEL")
    if env_model:
        return env_model

    # Use cached working model
    if _working_model:
        return _working_model

    # Discover one that works
    _working_model = _discover_working_model(api_key)
    return _working_model


def _call_openrouter(messages: list[dict[str, str]], api_key: str, max_tokens: int = 1024) -> tuple[str, int, int, float]:
    """
    Call OpenRouter API (OpenAI-compatible format).
    
    Returns: (response_text, tokens_in, tokens_out, latency_ms)
    """
    import httpx

    model = _get_model(api_key)

    start = time.perf_counter()
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/compliance-doc-intel",
                "X-Title": "Compliance Document Intelligence",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
    latency_ms = (time.perf_counter() - start) * 1000

    # On 429 rate limit, extract Retry-After and raise with delay info
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else 10.0
        raise RateLimitError(f"Rate limited — retry after {delay}s", retry_after=delay)

    # On 404 (model not found), invalidate cache so next call re-discovers
    if response.status_code == 404:
        global _working_model
        _working_model = None
        response.raise_for_status()

    response.raise_for_status()
    data = response.json()

    # Handle OpenRouter error responses
    if "error" in data:
        raise ValueError(f"OpenRouter API error: {data['error']}")

    text = data["choices"][0]["message"]["content"]
    tokens_in = data.get("usage", {}).get("prompt_tokens", 0)
    tokens_out = data.get("usage", {}).get("completion_tokens", 0)

    return text, tokens_in, tokens_out, latency_ms


# ---------------------------------------------------------------------------
# JSON parsing & validation
# ---------------------------------------------------------------------------

def _parse_llm_response(response_text: str) -> Optional[dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks and thinking text."""
    text = response_text.strip()

    # Strategy 1: Strip markdown code block if present
    if "```" in text:
        import re
        code_blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if code_blocks:
            # Use the last code block (most likely the final answer)
            for block in reversed(code_blocks):
                try:
                    return json.loads(block.strip())
                except json.JSONDecodeError:
                    continue

    # Strategy 2: Try parsing the whole thing as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find a JSON object in the text (handles thinking/reasoning output)
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try to find first { to last } and parse
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse LLM response as JSON: {text[:300]}...")
    return None


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _validate_and_normalize(data: dict[str, Any], doc_type: str) -> dict[str, Any]:
    """
    Validate extracted data against Pydantic schema and normalize types.
    Returns cleaned dict.
    """
    from datetime import datetime

    # Convert camelCase keys to snake_case
    data = {_camel_to_snake(k): v for k, v in data.items()}

    schema = get_schema(DocumentType(doc_type))

    # Normalize date fields to strings (YYYY-MM-DD)
    date_fields = []
    for field_name, field_info in schema.model_fields.items():
        if field_info.annotation == date or (hasattr(field_info.annotation, "__origin__") and field_info.annotation == Optional[date]):
            date_fields.append(field_name)

    for df in date_fields:
        if df in data and data[df] is not None:
            val = data[df]
            if isinstance(val, str):
                # Try parsing common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                    try:
                        data[df] = datetime.strptime(val, fmt).date().isoformat()
                        break
                    except ValueError:
                        continue
            elif hasattr(val, "isoformat"):
                data[df] = val.isoformat()

    # Normalize enum fields
    enum_map = {
        "severity": SeverityLevel,
        "follow_up_status": FollowUpStatus,
        "status": FindingStatus,
        "doc_type": DocumentType,
    }

    for field_name, enum_cls in enum_map.items():
        if field_name in data and data[field_name] is not None:
            val = str(data[field_name]).lower().strip().replace(" ", "_")
            try:
                data[field_name] = enum_cls(val).value
            except ValueError:
                # Try common mappings
                if field_name == "follow_up_status" and val == "pending":
                    data[field_name] = FollowUpStatus.PENDING_REVIEW.value
                elif field_name == "severity":
                    sev_map = {"life_threatening": "critical", "severe": "high", "mild": "low"}
                    data[field_name] = sev_map.get(val, val)

    # Remove None values
    return {k: v for k, v in data.items() if v is not None}


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def llm_extract_fields(
    doc: IngestedDocument,
    max_retries: int = 5,
    base_delay_ms: float = 2000.0,
) -> dict[str, Any]:
    """
    Extract structured fields from a document using OpenRouter LLM.

    Args:
        doc: Ingested document with raw text and type guess
        max_retries: Maximum retry attempts on validation failure
        retry_delay_ms: Delay between retries (ms)

    Returns:
        Extracted fields as a dictionary

    Raises:
        ValueError: If extraction fails after all retries
    """
    import random
    global metrics

    doc_type = doc.doc_type_guess
    if doc_type == "unknown":
        return {"doc_type": "unknown", "error": "Could not classify document type"}

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Get a free key at https://openrouter.ai/keys "
            "then set it in .env file or environment variable."
        )

    messages = _build_prompt(doc_type, doc.raw_text)

    logger.info(f"Extracting fields from {doc.doc_id}")

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Call OpenRouter API
            response_text, tokens_in, tokens_out, latency_ms = _call_openrouter(
                messages, api_key
            )

            # Free models have $0 cost
            cost_usd = 0.0
            metrics.log_call(tokens_in, tokens_out, latency_ms, cost_usd)

            # Parse response
            data = _parse_llm_response(response_text)
            if data is None:
                raise ValueError("Failed to parse LLM response as JSON")

            # Validate and normalize
            cleaned = _validate_and_normalize(data, doc_type)

            # Ensure doc_type is set
            if "doc_type" not in cleaned:
                cleaned["doc_type"] = doc_type

            logger.info(f"Extracted {len(cleaned)} fields from {doc.doc_id} (attempt {attempt + 1})")
            return cleaned

        except RateLimitError as e:
            last_error = e
            metrics.errors += 1
            # Respect Retry-After header from OpenRouter
            delay_s = max(e.retry_after, base_delay_ms / 1000 * (2 ** attempt))
            logger.warning(f"Rate limited on {doc.doc_id} (attempt {attempt + 1}). Waiting {delay_s:.0f}s...")
            metrics.retries += 1
            if attempt < max_retries:
                time.sleep(delay_s)

        except Exception as e:
            last_error = e
            metrics.errors += 1
            logger.warning(f"Extraction attempt {attempt + 1} failed for {doc.doc_id}: {e}")

            if attempt < max_retries:
                metrics.retries += 1
                # Exponential backoff with jitter: 2s, 4s, 8s, 16s, 32s
                delay_s = (base_delay_ms / 1000) * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"  Retrying in {delay_s:.1f}s...")
                time.sleep(delay_s)

    raise ValueError(f"Extraction failed after {max_retries + 1} attempts: {last_error}")


def get_extraction_metrics() -> ExtractionMetrics:
    """Get the current extraction metrics."""
    return metrics


def reset_metrics() -> None:
    """Reset metrics for a new extraction run."""
    global metrics
    metrics = ExtractionMetrics()
