# Compliance Document Intelligence

AI-powered system for extracting structured information from unstructured regulatory/compliance documents, validating extractions against compliance rules, and flagging gaps or inconsistencies.

## Phase 1: Data Foundation & Extraction Baseline

Phase 1 establishes the end-to-end pipeline: raw document -> text extraction -> document type detection -> baseline field extraction -> structured JSON output. Extraction is intentionally rule-based (regex + heuristics) to provide an honest baseline for Phase 2 improvements.

---

## Project Structure

```
compliance-doc-intel/
├── data/
│   ├── raw/                          # Source documents
│   ├── synthetic/                    # Generated synthetic docs
│   │   ├── sop/{train,test}/         # SOP documents + ground truth
│   │   ├── adverse_event/{train,test} # Adverse Event Reports
│   │   └── audit_finding/{train,test} # Audit Finding Reports
│   └── processed/                    # Extracted JSON output + metrics
├── src/
│   ├── schemas.py                    # Pydantic schemas (source of truth)
│   ├── ingestion/
│   │   └── loader.py                 # PDF/DOCX parser + type classifier
│   ├── extraction/
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
│   └── test_pipeline.py              # Smoke tests
├── run_pipeline.py                   # End-to-end pipeline runner
└── requirements.txt
```

## Quick Start

```bash
cd compliance-doc-intel
pip install -r requirements.txt

# Run the full pipeline (generate -> extract -> evaluate)
python run_pipeline.py

# Or skip generation if data already exists
python run_pipeline.py --skip-generation

# Run smoke tests
python tests/test_pipeline.py
```

## Document Types (Phase 1)

| Type | Required Fields | Optional Fields |
|------|----------------|-----------------|
| **SOP** | title, version, effective_date, owner, department, approval_date, review_cycle | scope, references, approval_signatures |
| **Adverse Event Report** | event_id, date_reported, product_id, batch_id, severity, description, reporter, follow_up_status | patient_info, outcome, regulatory_filed |
| **Audit Finding Report** | audit_id, audit_date, finding_category, severity, description, corrective_action, due_date, responsible_party | root_cause, evidence, status |

## Synthetic Dataset

- **300 documents** (100 per type) split 80/20 train/test
- Each document rendered as a real DOCX file
- Paired with `_ground_truth.json` for evaluation
- Includes **intentionally flawed** variants (~25%) with:
  - Missing required fields
  - Malformed dates
  - Invalid severity values
  - Truncated titles

## Pipeline Architecture

```
[DOCX/PDF] -> [Loader] -> [Type Classifier] -> [Extractor] -> [Structured JSON]
     |              |              |                  |
     |              |              +-- keyword/heuristic rules
     |              +-- pdfplumber / python-docx
     |
     +-- Each doc paired with _ground_truth.json for evaluation
```

### Components

1. **Ingestion** (`src/ingestion/loader.py`)
   - Detects file type by extension
   - Extracts text via `pdfplumber` (PDF) or `python-docx` (DOCX)
   - Normalizes whitespace, removes page-break artifacts
   - Heuristic classifier using keyword matching

2. **Extraction** (`src/extraction/extractor.py`)
   - Rule-based per document type (regex + positional heuristics)
   - Extracts fields matching the Pydantic schema
   - Handles date parsing, severity normalization, status mapping

3. **Evaluation** (`src/utils/evaluate_extraction.py`)
   - Field-level precision/recall/F1 computation
   - Per-document-type and per-field breakdown
   - Summary metrics report

## Phase 1 Baseline Metrics

**Overall F1: 0.8175**

### Per-Field F1 (Test Set)

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

### Known Weaknesses (Phase 2 Targets)

- `doc_type` is classified but not included in extraction output
- `references` and `scope` require better section-boundary detection
- `follow_up_status` suffers from table-cell extraction format issues
- `status` field is not currently extracted for audit findings

---

## Phase 2 Roadmap

### 2.1: Improved Extraction
- **Fine-tuned NER model** on extracted training data
- **LLM-based extraction** as alternative to rule-based (GPT-4 / Claude for complex fields)
- Better section-boundary detection for `scope`, `references`
- Include `doc_type` in extraction output

### 2.2: Compliance Rule Validation
- Rule engine using `config/compliance_rules.yaml`
- Cross-field validation (e.g., approval_date <= effective_date)
- Format validation (regex patterns for IDs, date ranges)
- Severity-dependent rules (critical findings must have corrective action)

### 2.3: Anomaly Detection
- Detect documents with structural anomalies (missing sections, unusual formatting)
- Flag documents that fail validation rules
- Identify data quality issues across document batches

### 2.4: Dashboard
- Web UI for reviewing extractions and validation results
- Confidence scores for each extracted field
- Drill-down from document to field-level details
- Export validated reports

## Configuration

All settings in `config/settings.yaml`:
- Document type definitions with required/optional fields
- Extraction method (rule_based for Phase 2)
- Train/test split ratio
- Evaluation output paths

Compliance rules in `config/compliance_rules.yaml` (Phase 2 placeholder):
- Per-document-type validation rules
- Field format requirements
- Cross-field consistency checks
