"""
Document Ingestion Pipeline

Detects file type, extracts raw text, normalizes whitespace, and returns
a standardized internal representation. Also includes a heuristic document-type classifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class IngestedDocument:
    """Standardized internal representation after ingestion."""
    doc_id: str
    doc_type_guess: str  # "sop", "adverse_event", "audit_finding", "unknown"
    raw_text: str
    metadata: dict = field(default_factory=dict)
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Text extraction backends
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX using python-docx."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                paragraphs.append(row_text)

    return "\n".join(paragraphs)


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Normalize whitespace and remove page-break artifacts."""
    # Replace form feeds and other control characters
    text = re.sub(r"[\x0c\x0b]", " ", raw)
    # Collapse multiple whitespace (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Heuristic document-type classifier
# ---------------------------------------------------------------------------

DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "sop": [
        "standard operating procedure",
        "sop",
        "review cycle",
        "effective date",
        "scope",
        "procedure",
        "gmp",
    ],
    "adverse_event": [
        "adverse event",
        "event id",
        "ae-",
        "batch id",
        "product id",
        "severity",
        "follow-up",
        "reporter",
        "pharmacovigilance",
    ],
    "audit_finding": [
        "audit finding",
        "audit id",
        "aud-",
        "finding category",
        "corrective action",
        "due date",
        "responsible party",
        "non-conformance",
        "root cause",
    ],
}


def classify_document(text: str) -> str:
    """
    Heuristic keyword-based document-type classifier.

    Returns: "sop", "adverse_event", "audit_finding", or "unknown"
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
        scores[doc_type] = score

    if not scores or max(scores.values()) == 0:
        return "unknown"

    best_type = max(scores, key=scores.get)  # type: ignore
    # Require at least 2 keyword matches to be confident
    if scores[best_type] < 2:
        return "unknown"
    return best_type


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_document(path: Path) -> IngestedDocument:
    """
    Load and ingest a document.

    Steps:
        1. Detect file type by extension
        2. Extract raw text using the appropriate backend
        3. Clean/normalize the text
        4. Heuristically classify the document type
        5. Return standardized IngestedDocument
    """
    suffix = path.suffix.lower()
    if suffix not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(EXTRACTORS.keys())}")

    raw_text = EXTRACTORS[suffix](path)
    cleaned = _clean_text(raw_text)
    doc_type = classify_document(cleaned)

    # Derive doc_id from filename (strip extension)
    doc_id = path.stem
    # Strip common suffixes like _ground_truth, _meta
    for suffix_str in ["_ground_truth", "_meta"]:
        if doc_id.endswith(suffix_str):
            doc_id = doc_id[: -len(suffix_str)]

    return IngestedDocument(
        doc_id=doc_id,
        doc_type_guess=doc_type,
        raw_text=cleaned,
        metadata={
            "file_name": path.name,
            "file_extension": suffix,
            "text_length": len(cleaned),
        },
        source_path=str(path),
    )


def load_directory(
    directory: Path,
    recursive: bool = False,
) -> list[IngestedDocument]:
    """
    Load all supported documents from a directory.

    Filters: only .pdf and .docx files; skips *_ground_truth.json and *_meta.json.
    """
    pattern = "**/*" if recursive else "*"
    supported = {".pdf", ".docx"}
    docs: list[IngestedDocument] = []

    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.suffix.lower() in supported:
            try:
                doc = load_document(p)
                docs.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")

    return docs
