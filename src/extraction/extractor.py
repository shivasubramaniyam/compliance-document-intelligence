"""
Baseline Field Extraction — Rule-Based

Phase 1 baseline uses regex + positional heuristics to extract structured fields
from raw document text. This is intentionally simple; Phase 2 upgrades to
fine-tuned NER or LLM-based extraction.
"""


from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Optional

from ..ingestion.loader import IngestedDocument
from ..schemas import (
    AdverseEventReport,
    AuditFindingReport,
    DocumentType,
    FollowUpStatus,
    FindingStatus,
    SeverityLevel,
    SOPDocument,
)


# ---------------------------------------------------------------------------
# Utility extraction helpers
# ---------------------------------------------------------------------------

def _find_field(text: str, field_label: str, pattern: str | None = None) -> Optional[str]:
    """
    Extract a field value from text using a label + regex pattern.

    Looks for lines like "Field Label: value" or "Field Label | value".
    """
    escaped_label = re.escape(field_label)
    if pattern:
        regex = rf"{escaped_label}\s*[:|\-]\s*{pattern}"
    else:
        regex = rf"{escaped_label}\s*[:|\-]\s*(.+?)(?:\n|$)"
    match = re.search(regex, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _parse_date(value: str) -> Optional[date]:
    """Try to parse a date string in common formats."""
    if not value:
        return None
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%B %d, %Y",
        "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _parse_severity(value: str) -> Optional[SeverityLevel]:
    """Parse severity string to enum."""
    if not value:
        return None
    value_lower = value.strip().lower()
    mapping = {
        "critical": SeverityLevel.CRITICAL,
        "high": SeverityLevel.HIGH,
        "medium": SeverityLevel.MEDIUM,
        "low": SeverityLevel.LOW,
        "info": SeverityLevel.INFO,
    }
    return mapping.get(value_lower)


def _parse_follow_up_status(value: str) -> Optional[FollowUpStatus]:
    if not value:
        return None
    value_lower = value.strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "open": FollowUpStatus.OPEN,
        "in_progress": FollowUpStatus.IN_PROGRESS,
        "closed": FollowUpStatus.CLOSED,
        "pending_review": FollowUpStatus.PENDING_REVIEW,
        "pending": FollowUpStatus.PENDING_REVIEW,
    }
    return mapping.get(value_lower)


def _parse_finding_status(value: str) -> Optional[FindingStatus]:
    if not value:
        return None
    value_lower = value.strip().lower().replace(" ", "_")
    mapping = {
        "open": FindingStatus.OPEN,
        "in_progress": FindingStatus.IN_PROGRESS,
        "closed": FindingStatus.CLOSED,
        "overdue": FindingStatus.OVERDUE,
    }
    return mapping.get(value_lower)


# ---------------------------------------------------------------------------
# SOP extractor
# ---------------------------------------------------------------------------

def extract_sop(text: str) -> dict[str, Any]:
    """Extract SOP fields from raw text using regex heuristics."""
    result: dict[str, Any] = {"doc_type": DocumentType.SOP}

    # Title: usually the first heading or largest text
    title_match = re.search(r"^(.+?)(?:\n|$)", text)
    if title_match:
        result["title"] = title_match.group(1).strip()

    # Standard fields
    result["version"] = _find_field(text, "Version") or _find_field(text, "Rev")
    result["effective_date"] = _find_field(text, "Effective Date")
    result["owner"] = _find_field(text, "Owner")
    result["department"] = _find_field(text, "Department")
    result["approval_date"] = _find_field(text, "Approval Date")
    result["review_cycle"] = _find_field(text, "Review Cycle")

    # Parse dates
    if result.get("effective_date"):
        parsed = _parse_date(result["effective_date"])
        if parsed:
            result["effective_date"] = parsed
    if result.get("approval_date"):
        parsed = _parse_date(result["approval_date"])
        if parsed:
            result["approval_date"] = parsed

    # Optional fields
    scope_match = re.search(r"Scope\s*[:\n]\s*(.+?)(?:\n\n|References|Procedure)", text, re.DOTALL | re.IGNORECASE)
    if scope_match:
        result["scope"] = scope_match.group(1).strip()

    # References: look for bullet-pointed items under "References"
    refs_section = re.search(r"References\s*[:\n]\s*(.+?)(?:\n\n|\nProcedure|\nScope)", text, re.DOTALL | re.IGNORECASE)
    if refs_section:
        refs = re.findall(r"[•\-\*]\s*(.+)", refs_section.group(1))
        if refs:
            result["references"] = [r.strip() for r in refs]

    return result


# ---------------------------------------------------------------------------
# Adverse Event extractor
# ---------------------------------------------------------------------------

def extract_adverse_event(text: str) -> dict[str, Any]:
    """Extract Adverse Event Report fields from raw text."""
    result: dict[str, Any] = {"doc_type": DocumentType.ADVERSE_EVENT}

    result["event_id"] = _find_field(text, "Event ID") or _find_field(text, "Event Id")
    result["product_id"] = _find_field(text, "Product ID") or _find_field(text, "Product Id")
    result["batch_id"] = _find_field(text, "Batch ID") or _find_field(text, "Batch Id")
    result["reporter"] = _find_field(text, "Reporter")
    result["follow_up_status"] = _find_field(text, "Follow-Up Status") or _find_field(text, "Follow up status")

    # Date
    date_str = _find_field(text, "Date Reported")
    if date_str:
        parsed = _parse_date(date_str)
        result["date_reported"] = parsed if parsed else date_str  # keep raw if parse fails

    # Severity
    sev_str = _find_field(text, "Severity")
    if sev_str:
        parsed = _parse_severity(sev_str)
        result["severity"] = parsed.value if parsed else sev_str

    # Description: look for "Description" section
    desc_match = re.search(
        r"Description\s*(?:of\s*Event)?\s*[:\n]\s*(.+?)(?:\n\n|\nNarrative|\nFollow|\nSeverity)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Audit Finding extractor
# ---------------------------------------------------------------------------

def extract_audit_finding(text: str) -> dict[str, Any]:
    """Extract Audit Finding Report fields from raw text."""
    result: dict[str, Any] = {"doc_type": DocumentType.AUDIT_FINDING}

    result["audit_id"] = _find_field(text, "Audit ID") or _find_field(text, "Audit Id")
    result["finding_category"] = _find_field(text, "Finding Category")
    result["responsible_party"] = _find_field(text, "Responsible Party")

    # Dates
    audit_date_str = _find_field(text, "Audit Date")
    if audit_date_str:
        parsed = _parse_date(audit_date_str)
        result["audit_date"] = parsed if parsed else audit_date_str

    due_date_str = _find_field(text, "Due Date")
    if due_date_str:
        parsed = _parse_date(due_date_str)
        result["due_date"] = parsed if parsed else due_date_str

    # Severity
    sev_str = _find_field(text, "Severity")
    if sev_str:
        parsed = _parse_severity(sev_str)
        result["severity"] = parsed.value if parsed else sev_str

    # Status
    status_str = _find_field(text, "Status")
    if status_str:
        parsed = _parse_finding_status(status_str)
        result["status"] = parsed.value if parsed else status_str

    # Description
    desc_match = re.search(
        r"Finding\s*Description\s*[:\n]\s*(.+?)(?:\n\n|\nCorrective|\nEvidence)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if desc_match:
        result["description"] = desc_match.group(1).strip()

    # Corrective action
    ca_match = re.search(
        r"Corrective\s*Action\s*(?:Required)?\s*[:\n]\s*(.+?)(?:\n\n|\nEvidence|\nRoot)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if ca_match:
        result["corrective_action"] = ca_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Main extraction dispatcher
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "sop": extract_sop,
    "adverse_event": extract_adverse_event,
    "audit_finding": extract_audit_finding,
}


def extract_fields(doc: IngestedDocument) -> dict[str, Any]:
    """
    Extract structured fields from an ingested document.

    Uses the doc_type_guess to select the appropriate extractor.
    Returns a dict of extracted fields (not yet validated against schema).
    """
    extractor = EXTRACTORS.get(doc.doc_type_guess)
    if extractor is None:
        return {"doc_type": "unknown", "error": "Could not classify document type"}

    return extractor(doc.raw_text)


def extract_and_validate(doc: IngestedDocument) -> dict[str, Any]:
    """
    Extract fields and return them as a validated dictionary.

    Returns the extracted fields dict regardless of schema validation errors.
    The evaluation harness will compare these against ground truth.
    """
    fields = extract_fields(doc)

    # Clean up None values
    cleaned = {k: v for k, v in fields.items() if v is not None}

    return cleaned
