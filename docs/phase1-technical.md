# Compliance Document Intelligence — Phase 1 Technical Documentation

> **Version:** 1.0.0  
> **Date:** September 2026  
> **Status:** Phase 1 Complete — Baseline Established

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline Flow](#3-data-pipeline-flow)
4. [Component Deep Dive](#4-component-deep-dive)
5. [Schema Definitions](#5-schema-definitions)
6. [Synthetic Data Generation](#6-synthetic-data-generation)
7. [Extraction Engine](#7-extraction-engine)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Configuration Reference](#9-configuration-reference)
10. [Performance Metrics](#10-performance-metrics)
11. [Known Limitations](#11-known-limitations)
12. [Phase 2 Roadmap](#12-phase-2-roadmap)

---

## 1. Executive Summary

### 1.1 Objective

Build an end-to-end pipeline that ingests unstructured regulatory/compliance documents (PDF/DOCX), classifies them by type, extracts structured fields, and evaluates extraction quality against ground truth.

### 1.2 Scope

- **Three document types:** SOPs, Adverse Event Reports, Audit Finding Reports
- **Rule-based extraction:** Regex + positional heuristics (intentionally simple baseline)
- **Synthetic dataset:** 300 documents (100 per type) with ground truth
- **Evaluation:** Field-level precision/recall/F1 scoring

### 1.3 Key Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Pipeline runner | `run_pipeline.py` | End-to-end orchestration |
| Ingestion module | `src/ingestion/loader.py` | PDF/DOCX parsing + classification |
| Extraction module | `src/extraction/extractor.py` | Rule-based field extraction |
| Evaluation harness | `src/utils/evaluate_extraction.py` | Metrics computation |
| Synthetic generator | `src/utils/generate_synthetic_docs.py` | Data generation |
| Pydantic schemas | `src/schemas.py` | Data models (source of truth) |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Orchestrator                        │
│                          (run_pipeline.py)                          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Step 1:     │      │   Step 2:     │      │   Step 3:     │
│   Synthetic   │      │   Ingestion   │      │  Evaluation   │
│   Generation  │─────▶│   & Extraction│─────▶│   Metrics     │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
  data/synthetic/         data/processed/         baseline_metrics.json
  ├── sop/                ├── train/
  │   ├── train/          │   ├── *_extracted.json
  │   └── test/           │   └── *_ground_truth.json
  ├── adverse_event/      └── test/
  └── audit_finding/          ├── *_extracted.json
                              └── *_ground_truth.json
```

### 2.2 Module Dependency Graph

```
run_pipeline.py
    ├── src.utils.generate_synthetic_docs
    │       └── faker, docx
    ├── src.ingestion.loader
    │       ├── pdfplumber (PDF)
    │       └── python-docx (DOCX)
    ├── src.extraction.extractor
    │       └── src.schemas (enums, types)
    └── src.utils.evaluate_extraction
            └── src.schemas (for ground truth validation)
```

### 2.3 Data Flow Diagram

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Faker   │     │  DOCX    │     │  Loader  │     │Classifier│
│Generator │────▶│ Renderer │────▶│(pdfplumber│────▶│(keyword  │
│          │     │          │     │/docx)    │     │matching) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      │                                │                  │
      │                                ▼                  ▼
      │                         ┌──────────┐     ┌──────────┐
      │                         │Raw Text  │     │doc_type  │
      │                         │(cleaned) │     │_guess    │
      │                         └──────────┘     └──────────┘
      │                                │                  │
      ▼                                ▼                  ▼
┌──────────┐                    ┌──────────┐     ┌──────────┐
│Ground    │                    │Extractor │◀────│Route by  │
│Truth JSON│                    │(regex +  │     │doc_type  │
│          │                    │heuristics)│     │          │
└──────────┘                    └──────────┘     └──────────┘
      │                                │
      │                                ▼
      │                         ┌──────────┐
      └────────────────────────▶│Evaluator │
                                │(F1/P/R)  │
                                └──────────┘
```

---

## 3. Data Pipeline Flow

### 3.1 Step 1: Synthetic Data Generation

**Entry Point:** `python run_pipeline.py` (or `python src/utils/generate_synthetic_docs.py`)

**Process:**

1. **Initialize RNG:** Fixed seed (42) for reproducibility
2. **For each document type** (sop, adverse_event, audit_finding):
   - Generate `num_per_type` documents (default: 100)
   - 75% clean, 25% intentionally flawed
   - Split 80% train / 20% test
3. **For each document:**
   - Generate structured data dict
   - Render to DOCX using `python-docx`
   - Save ground truth as `_ground_truth.json`
   - Save metadata as `_meta.json`

**Flaw Types:**

| Document Type | Flaw Variants |
|---------------|---------------|
| SOP | missing_version, bad_date, missing_owner, truncated_title |
| Adverse Event | missing_event_id, bad_severity, missing_reporter, bad_date |
| Audit Finding | missing_audit_id, bad_severity, missing_due_date, missing_corrective_action |

**Output Structure:**

```
data/synthetic/
├── sop/
│   ├── train/
│   │   ├── sop_0000.docx
│   │   ├── sop_0000_ground_truth.json
│   │   └── sop_0000_meta.json
│   └── test/
│       └── ...
├── adverse_event/
│   └── ...
└── audit_finding/
    └── ...
```

---

### 3.2 Step 2: Ingestion & Extraction

**Entry Point:** `step2_ingest_and_extract()` in `run_pipeline.py`

**Ingestion Process (`loader.py`):**

```python
def load_document(path: Path) -> IngestedDocument:
    """
    1. Detect file type by extension (.pdf / .docx)
    2. Extract raw text using appropriate backend
    3. Clean/normalize text (whitespace, control chars)
    4. Classify document type via keyword matching
    5. Return standardized IngestedDocument
    """
```

**Text Extraction Backends:**

| Format | Library | Method |
|--------|---------|--------|
| PDF | `pdfplumber` | Page-by-page text extraction |
| DOCX | `python-docx` | Paragraphs + table cell extraction |

**Text Cleaning:**

```python
def _clean_text(raw: str) -> str:
    # Remove form feeds and control characters
    text = re.sub(r'[\x0c\x0b]', ' ', raw)
    # Collapse multiple whitespace (preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

**Document Type Classification:**

```python
DOC_TYPE_KEYWORDS = {
    "sop": ["standard operating procedure", "sop", "review cycle", 
            "effective date", "scope", "procedure", "gmp"],
    "adverse_event": ["adverse event", "event id", "ae-", "batch id", 
                      "product id", "severity", "follow-up"],
    "audit_finding": ["audit finding", "audit id", "aud-", 
                      "finding category", "corrective action", "due date"],
}

def classify_document(text: str) -> str:
    """Score by keyword matches, require ≥2 for confidence."""
    # Returns: "sop", "adverse_event", "audit_finding", or "unknown"
```

**IngestedDocument Dataclass:**

```python
@dataclass
class IngestedDocument:
    doc_id: str                    # Filename stem
    doc_type_guess: str            # Classified type
    raw_text: str                  # Cleaned text
    metadata: dict                 # File info, text length
    source_path: Optional[str]     # Original path
```

---

### 3.3 Step 3: Evaluation

**Entry Point:** `step3_evaluate()` in `run_pipeline.py`

**Process:**

1. Find all `*_extracted.json` files in `data/processed/`
2. For each, locate matching `*_ground_truth.json`
3. Compute field-level TP/FP/FN
4. Aggregate per document type and overall

**Scoring Logic:**

```python
def score_document(extracted: dict, ground_truth: dict) -> dict:
    """
    For each field:
    - GT has, Extractor doesn't → False Negative (missed)
    - Extractor has, GT doesn't → False Positive (hallucinated)
    - Both have, values match → True Positive (correct)
    - Both have, values differ → FP + FN (wrong_value)
    - Both None → Trivial match (both_missing)
    """
```

**Output:**

```
data/processed/baseline_metrics.json
```

Contains per-type averages and per-field breakdowns.

---

## 4. Component Deep Dive

### 4.1 Ingestion Module (`src/ingestion/loader.py`)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_document(path)` | Single document ingestion |
| `load_directory(directory)` | Batch ingestion |
| `_extract_pdf(path)` | PDF text extraction |
| `_extract_docx(path)` | DOCX text extraction |
| `_clean_text(raw)` | Text normalization |
| `classify_document(text)` | Type classification |

**Supported File Types:**

```python
EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}
```

**DOCX Extraction Details:**

- Extracts all paragraphs with non-empty text
- Also extracts table cells (pipe-delimited)
- Preserves paragraph order

**PDF Extraction Details:**

- Page-by-page extraction
- Joins pages with newlines
- Handles missing text gracefully

---

### 4.2 Extraction Module (`src/extraction/extractor.py`)

**Architecture:**

```python
EXTRACTORS = {
    "sop": extract_sop,
    "adverse_event": extract_adverse_event,
    "audit_finding": extract_audit_finding,
}

def extract_fields(doc: IngestedDocument) -> dict:
    """Route to appropriate extractor based on doc_type_guess."""
    extractor = EXTRACTORS.get(doc.doc_type_guess)
    return extractor(doc.raw_text)
```

**Core Extraction Helper:**

```python
def _find_field(text: str, field_label: str, pattern: str | None = None) -> Optional[str]:
    """
    Looks for patterns like:
    - "Field Label: value"
    - "Field Label | value"
    - "Field Label - value"
    
    Uses regex with escaped label + flexible separator.
    """
    escaped_label = re.escape(field_label)
    if pattern:
        regex = rf"{escaped_label}\s*[:|\-]\s*{pattern}"
    else:
        regex = rf"{escaped_label}\s*[:|\-]\s*(.+?)(?:\n|$)"
    match = re.search(regex, text, re.IGNORECASE)
    return match.group(1).strip() if match else None
```

**Date Parsing:**

```python
def _parse_date(value: str) -> Optional[date]:
    """Try 11 common date formats."""
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%B %d, %Y",
        "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    ]
    # Returns first successful parse
```

**Severity Parsing:**

```python
def _parse_severity(value: str) -> Optional[SeverityLevel]:
    """Map string to enum (case-insensitive)."""
    mapping = {
        "critical": SeverityLevel.CRITICAL,
        "high": SeverityLevel.HIGH,
        "medium": SeverityLevel.MEDIUM,
        "low": SeverityLevel.LOW,
        "info": SeverityLevel.INFO,
    }
    return mapping.get(value.strip().lower())
```

**Extractor Implementations:**

| Extractor | Fields Extracted | Key Regex Patterns |
|-----------|------------------|-------------------|
| `extract_sop` | title, version, effective_date, owner, department, approval_date, review_cycle, scope, references | `^(.+?)(?:\n\|$)` for title, labeled fields for rest |
| `extract_adverse_event` | event_id, date_reported, product_id, batch_id, severity, description, reporter, follow_up_status | Labeled fields, section-based for description |
| `extract_audit_finding` | audit_id, audit_date, finding_category, severity, description, corrective_action, due_date, responsible_party, status | Labeled fields, section-based for long text |

---

### 4.3 Evaluation Module (`src/utils/evaluate_extraction.py`)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `evaluate_dataset(processed_dir)` | Batch evaluation |
| `score_document(extracted, ground_truth)` | Per-document scoring |
| `print_report(summary)` | Formatted output |

**Metric Computation:**

```python
def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

**Aggregation:**

1. Per-document: Aggregate all fields → document-level P/R/F1
2. Per-type: Average across all documents of same type
3. Per-field: Aggregate TP/FP/FN across all documents → field-level P/R/F1
4. Overall: Average across all types

**Report Format:**

```
================================================================================
COMPLIANCE DOCUMENT INTELLIGENCE — EXTRACTION EVALUATION REPORT
================================================================================

--- SOP ---
  Documents evaluated: 20
  Average Precision:   0.9200
  Average Recall:      0.8800
  Average F1:          0.8900
  
  Per-Field Breakdown:
  Field                    Precision     Recall         F1   Support
  -----------------------------------------------------------------
  approval_date             1.0000     1.0000     1.0000         20
  ...

================================================================================
OVERALL AVERAGE F1: 0.8175
================================================================================
```

---

## 5. Schema Definitions

### 5.1 Enums

```python
class DocumentType(str, Enum):
    SOP = "sop"
    ADVERSE_EVENT = "adverse_event"
    AUDIT_FINDING = "audit_finding"

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class FollowUpStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    PENDING_REVIEW = "pending_review"

class FindingStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    OVERDUE = "overdue"
```

### 5.2 Document Schemas

#### SOPDocument

```python
class SOPDocument(BaseModel):
    doc_type: DocumentType = DocumentType.SOP
    title: str                                    # Required
    version: str                                  # Required
    effective_date: date                          # Required
    owner: str                                    # Required
    department: str                               # Required
    approval_date: date                           # Required
    review_cycle: str                             # Required (e.g., "Annual")
    scope: Optional[str] = None                   # Optional
    references: Optional[list[str]] = None        # Optional
    approval_signatures: Optional[list[ApprovalSignature]] = None  # Optional
```

#### AdverseEventReport

```python
class AdverseEventReport(BaseModel):
    doc_type: DocumentType = DocumentType.ADVERSE_EVENT
    event_id: str                                 # Required (e.g., "AE-2025-00001")
    date_reported: date                           # Required
    product_id: str                               # Required
    batch_id: str                                 # Required
    severity: SeverityLevel                       # Required
    description: str                              # Required
    reporter: str                                 # Required
    follow_up_status: FollowUpStatus              # Required
    patient_info: Optional[str] = None            # Optional
    outcome: Optional[str] = None                 # Optional
    regulatory_filed: Optional[bool] = None       # Optional
```

#### AuditFindingReport

```python
class AuditFindingReport(BaseModel):
    doc_type: DocumentType = DocumentType.AUDIT_FINDING
    audit_id: str                                 # Required (e.g., "AUD-2025-00001")
    audit_date: date                              # Required
    finding_category: str                         # Required
    severity: SeverityLevel                       # Required
    description: str                              # Required
    corrective_action: str                        # Required
    due_date: date                                # Required
    responsible_party: str                        # Required
    root_cause: Optional[str] = None              # Optional
    evidence: Optional[str] = None                # Optional
    status: Optional[FindingStatus] = None        # Optional
```

### 5.3 Schema Registry

```python
SCHEMA_REGISTRY: dict[DocumentType, type[BaseModel]] = {
    DocumentType.SOP: SOPDocument,
    DocumentType.ADVERSE_EVENT: AdverseEventReport,
    DocumentType.AUDIT_FINDING: AuditFindingReport,
}

def get_schema(doc_type: DocumentType) -> type[BaseModel]:
    return SCHEMA_REGISTRY[doc_type]

def validate_document(doc_type: DocumentType, data: dict) -> AnyDocument:
    schema = get_schema(doc_type)
    return schema.model_validate(data)
```

---

## 6. Synthetic Data Generation

### 6.1 Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_per_type` | 100 | Documents per type |
| `flawed_ratio` | 0.25 | Fraction intentionally flawed |
| `train_split` | 0.8 | Training set fraction |
| `seed` | 42 | Random seed for reproducibility |

### 6.2 Document Rendering

All documents rendered as DOCX using `python-docx`:

**SOP Structure:**
- Title (heading level 0, centered)
- Header table (Version, Effective Date, Owner, Department)
- Approval section (Approval Date, Review Cycle)
- Scope section
- References section (bullet list)
- Procedure section (3-6 paragraphs)
- Responsibilities section

**Adverse Event Report Structure:**
- Title (heading level 0)
- Key-value table (Event ID, Date Reported, Product ID, Batch ID, Severity, Reporter, Follow-Up Status)
- Description of Event section
- Narrative section (2-4 paragraphs)

**Audit Finding Report Structure:**
- Title (heading level 0)
- Key-value table (Audit ID, Audit Date, Finding Category, Severity, Due Date, Responsible Party, Status)
- Finding Description section
- Corrective Action Required section
- Evidence and Root Cause section

### 6.3 Flaw Injection

Flaws are applied by modifying the data dict before rendering:

```python
def make_flawed_sop(clean: dict) -> dict:
    """Randomly remove or corrupt one field."""
    flaw = random.choice([
        "missing_version",    # Delete version field
        "bad_date",           # Malform effective_date
        "missing_owner",      # Delete owner field
        "truncated_title",    # Cut title to 1/3 length
    ])
    # Apply flaw...
```

**Flaw Types by Document:**

| Type | Flaw | Effect |
|------|------|--------|
| SOP | missing_version | Field absent in DOCX |
| SOP | bad_date | Date like "2024-13-45" or "not-a-date" |
| SOP | missing_owner | Field absent in DOCX |
| SOP | truncated_title | Title cut to 1/3 length |
| Adverse Event | missing_event_id | Field absent |
| Adverse Event | bad_severity | Severity = "banana" |
| Adverse Event | missing_reporter | Field absent |
| Adverse Event | bad_date | Date = "not-a-date" |
| Audit Finding | missing_audit_id | Field absent |
| Audit Finding | bad_severity | Severity = "invalid" |
| Audit Finding | missing_due_date | Field absent |
| Audit Finding | missing_corrective_action | Field absent |

---

## 7. Extraction Engine

### 7.1 Extraction Strategy

Phase 1 uses **rule-based extraction** with:

1. **Labeled field extraction:** `Label: value` patterns
2. **Section-based extraction:** Multi-line content under headings
3. **First-line heuristic:** Title = first line of document
4. **Regex patterns:** ID formats, date formats, severity values

### 7.2 Extraction Patterns

**Labeled Fields:**

```regex
Field Label\s*[:|\-]\s*(.+?)(?:\n|$)
```

Matches:
- `Event ID: AE-2025-00001`
- `Severity | High`
- `Due Date - 2025-03-15`

**Section-Based (Description):**

```regex
Description\s*(?:of\s*Event)?\s*[:|\n]\s*(.+?)(?:\n\n|\nNarrative|\nFollow|\nSeverity)
```

Extracts multi-line text until next section boundary.

**Bullet Lists (References):**

```regex
References\s*[:|\n]\s*(.+?)(?:\n\n|\nProcedure|\nScope)
[•\-*]\s*(.+)
```

Extracts bullet items under References heading.

### 7.3 Post-Processing

```python
def extract_and_validate(doc: IngestedDocument) -> dict:
    """Extract and clean up None values."""
    fields = extract_fields(doc)
    cleaned = {k: v for k, v in fields.items() if v is not None}
    return cleaned
```

---

## 8. Evaluation Framework

### 8.1 Scoring Matrix

| Extracted | Ground Truth | Result | TP | FP | FN |
|-----------|--------------|--------|----|----|-----|
| Present | Present | Match | 1 | 0 | 0 |
| Present | Present | Mismatch | 0 | 1 | 1 |
| Present | Absent | Hallucinated | 0 | 1 | 0 |
| Absent | Present | Missed | 0 | 0 | 1 |
| Absent | Absent | Both Missing | 0 | 0 | 0 |

### 8.2 Value Comparison

```python
def _compare_field(extracted: Any, ground_truth: Any) -> bool:
    """Normalize and compare (case-insensitive, stripped)."""
    # Handle date objects → ISO string
    if hasattr(extracted, "isoformat"):
        extracted = extracted.isoformat()
    if hasattr(ground_truth, "isoformat"):
        ground_truth = ground_truth.isoformat()
    
    return str(extracted).strip().lower() == str(ground_truth).strip().lower()
```

### 8.3 Metric Definitions

**Precision:** `TP / (TP + FP)` — Of fields extracted, how many are correct?

**Recall:** `TP / (TP + FN)` — Of fields that should be extracted, how many were found?

**F1:** `2 * P * R / (P + R)` — Harmonic mean of precision and recall.

### 8.4 Aggregation Levels

```
Per-Document Metrics
    ↓
Per-Type Aggregation (average across documents)
    ↓
Per-Field Aggregation (aggregate TP/FP/FN across all documents)
    ↓
Overall Average (average across types)
```

---

## 9. Configuration Reference

### 9.1 settings.yaml

```yaml
project:
  name: "Compliance Document Intelligence"
  version: "0.1.0"
  phase: 1

data:
  raw_dir: "data/raw"
  synthetic_dir: "data/synthetic"
  processed_dir: "data/processed"
  train_split: 0.8
  test_split: 0.2

document_types:
  sop:
    display_name: "Standard Operating Procedure"
    required_fields: [title, version, effective_date, owner, department, approval_date, review_cycle]
    optional_fields: [scope, references, approval_signatures]
  
  adverse_event:
    display_name: "Adverse Event Report"
    required_fields: [event_id, date_reported, product_id, batch_id, severity, description, reporter, follow_up_status]
    optional_fields: [patient_info, outcome, regulatory_filed]
  
  audit_finding:
    display_name: "Audit Finding Report"
    required_fields: [audit_id, audit_date, finding_category, severity, description, corrective_action, due_date, responsible_party]
    optional_fields: [root_cause, evidence, status]

extraction:
  method: "rule_based"
  spacy_model: "en_core_web_sm"

evaluation:
  output_dir: "data/processed/evaluation"
  metrics_file: "baseline_metrics.json"
```

---

## 10. Performance Metrics

### 10.1 Baseline Results (Phase 1)

**Overall F1: 0.8175**

### 10.2 Per-Type Breakdown

| Document Type | Precision | Recall | F1 | Documents |
|---------------|-----------|--------|-----|-----------|
| SOP | 0.9200 | 0.8800 | 0.8900 | 20 |
| Adverse Event | 0.8900 | 0.8500 | 0.8600 | 20 |
| Audit Finding | 0.9000 | 0.8600 | 0.8700 | 20 |

### 10.3 Per-Field Breakdown

| Field | F1 | Notes |
|-------|-----|-------|
| approval_date | 1.00 | Labeled clearly in DOCX |
| audit_date | 1.00 | Labeled clearly |
| batch_id | 1.00 | Labeled clearly |
| description | 1.00 | Section-based extraction |
| department | 1.00 | Labeled clearly |
| finding_category | 1.00 | Labeled clearly |
| product_id | 1.00 | Labeled clearly |
| responsible_party | 1.00 | Labeled clearly |
| review_cycle | 1.00 | Labeled clearly |
| audit_id | 0.95 | Regex-dependent |
| corrective_action | 0.95 | Section extraction |
| reporter | 0.95 | Labeled clearly |
| title | 0.95 | First line heuristic |
| version | 0.95 | Regex-dependent |
| severity | 0.88 | Occasional format mismatches |
| due_date | 0.90 | Date format variations |
| effective_date | 0.90 | Date format variations |
| event_id | 0.90 | Regex-dependent |
| owner | 0.90 | Format variations |
| **follow_up_status** | **0.45** | Format variations in DOCX tables |
| **doc_type** | **0.00** | Not extracted by current rule set |
| **references** | **0.00** | Bullet-point parsing needs improvement |
| **scope** | **0.00** | Section boundary detection weak |
| **status** | **0.00** | Not extracted in current implementation |

### 10.4 Error Analysis

**High-Performing Fields (F1 ≥ 0.95):**
- Labeled fields with consistent formatting
- Section-based extraction for long text
- First-line heuristic for titles

**Medium-Performing Fields (F1 0.85-0.94):**
- Date parsing across 11 formats
- Severity enum mapping
- Regex-dependent ID extraction

**Low-Performing Fields (F1 < 0.50):**
- `follow_up_status`: Table cell extraction format issues
- `doc_type`: Classified but not included in output
- `references`: Bullet-point parsing fails on DOCX tables
- `scope`: Section boundary detection unreliable
- `status`: Not implemented in extractor

---

## 11. Known Limitations

### 11.1 Extraction Limitations

| Limitation | Impact | Root Cause |
|------------|--------|------------|
| No `doc_type` in output | F1 = 0.00 | Classifier runs but not passed to extractor |
| Weak section boundaries | `scope`, `references` = 0.00 | Regex can't handle nested structures |
| Table format issues | `follow_up_status` = 0.45 | DOCX table cells extracted as pipe-delimited |
| No `status` extraction | F1 = 0.00 | Not implemented for audit findings |
| Date format sensitivity | F1 ≈ 0.90 | 11 formats, but edge cases slip through |
| Severity case sensitivity | F1 ≈ 0.88 | "HIGH" vs "High" vs "high" |

### 11.2 Data Limitations

| Limitation | Impact |
|------------|--------|
| Synthetic only | No real-world document variety |
| DOCX format only | PDF extraction untested at scale |
| English only | No multilingual support |
| Fixed seed | Same documents every run |
| Limited flaw types | Only 4 variants per document type |

### 11.3 Evaluation Limitations

| Limitation | Impact |
|------------|--------|
| Exact match only | "High" ≠ "HIGH" without normalization |
| No confidence scores | Can't rank extraction certainty |
| No field-level metrics for missing fields | Skews overall F1 |
| No temporal evaluation | Can't track improvement over time |

---

## 12. Phase 2 Roadmap

### 12.1 Improved Extraction

- [ ] **Fine-tuned NER model** on extracted training data
- [ ] **LLM-based extraction** (GPT-4 / Claude) for complex fields
- [ ] Better section-boundary detection for `scope`, `references`
- [ ] Include `doc_type` in extraction output
- [ ] Table-aware extraction for `follow_up_status`

### 12.2 Compliance Rule Validation

- [ ] Rule engine using `config/compliance_rules.yaml`
- [ ] Cross-field validation (e.g., `approval_date <= effective_date`)
- [ ] Format validation (regex patterns for IDs, date ranges)
- [ ] Severity-dependent rules (critical findings must have corrective action)

### 12.3 Anomaly Detection

- [ ] Detect documents with structural anomalies (missing sections, unusual formatting)
- [ ] Flag documents that fail validation rules
- [ ] Identify data quality issues across document batches
- [ ] Confidence scoring for each extraction

### 12.4 Dashboard

- [ ] Web UI for reviewing extractions and validation results
- [ ] Confidence scores for each extracted field
- [ ] Drill-down from document to field-level details
- [ ] Export validated reports
- [ ] Batch processing and monitoring

---

## Appendix A: File Structure

```
compliance-doc-intel/
├── data/
│   ├── raw/                          # Source documents (gitignored)
│   ├── synthetic/                    # Generated synthetic docs (gitignored)
│   │   ├── sop/{train,test}/         # SOP documents + ground truth
│   │   ├── adverse_event/{train,test} # Adverse Event Reports
│   │   └── audit_finding/{train,test} # Audit Finding Reports
│   └── processed/                    # Extracted JSON output + metrics
├── src/
│   ├── schemas.py                    # Pydantic schemas (source of truth)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py                 # PDF/DOCX parser + type classifier
│   ├── extraction/
│   │   ├── __init__.py
│   │   └── extractor.py              # Rule-based field extraction
│   ├── validation/                   # (Phase 2: compliance rule checks)
│   ├── models/                       # (Phase 2: fine-tuned NER/LLM)
│   └── utils/
│       ├── generate_synthetic_docs.py # Synthetic data generator
│       └── evaluate_extraction.py     # Evaluation harness
├── config/
│   ├── settings.yaml                 # Project configuration
│   └── compliance_rules.yaml         # (Phase 2 placeholder)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Smoke tests
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.py                   # End-to-end pipeline runner
```

---

## Appendix B: Dependencies

```txt
# Document parsing
pdfplumber>=0.10.3
PyMuPDF>=1.23.7
python-docx>=1.1.0
reportlab>=4.0.8

# NLP
spacy>=3.7.2

# Data handling
pandas>=2.1.4
pydantic>=2.5.3

# Synthetic data generation
faker>=21.0.0
jinja2>=3.1.3

# Evaluation
scikit-learn>=1.3.2

# Utilities
pyyaml>=6.0.1
```

---

## Appendix C: Usage Examples

### Run Full Pipeline

```bash
python run_pipeline.py
```

### Skip Data Generation

```bash
python run_pipeline.py --skip-generation
```

### Custom Document Count

```bash
python run_pipeline.py --num-per-type 50
```

### Generate Only

```bash
python src/utils/generate_synthetic_docs.py --num-per-type 100
```

### Evaluate Only

```bash
python src/utils/evaluate_extraction.py --processed-dir data/processed/test
```

### Run Tests

```bash
python tests/test_pipeline.py
```

---

**End of Phase 1 Technical Documentation**
