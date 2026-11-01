# Compliance Document Intelligence

AI-powered system for extracting structured information from unstructured regulatory/compliance documents, validating extractions against compliance rules, and flagging gaps or inconsistencies.

---

## Phase 2: AI-Powered Extraction & Compliance Validation

Phase 2 replaces the rule-based extraction baseline with LLM-powered extraction and adds a compliance rule validation engine that flags missing fields, inconsistencies, and gaps.

### Key Improvements over Phase 1

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Extraction Method | Regex + heuristics | LLM (Claude/GPT) with structured output |
| Validation | None | Configurable compliance rules |
| Cross-Document Checks | None | Batch consistency validation |
| Reporting | Basic metrics | Full compliance report (JSON + Markdown) |
| Expected F1 | ~0.82 | ~0.95+ (LLM-based) |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Rule-based extraction (no API key needed)
python run_pipeline.py --method rule_based --skip-generation

# Phase 2: LLM-based extraction (requires API key)
export ANTHROPIC_API_KEY=your-key-here
python run_pipeline.py --method llm --validate --skip-generation

# Run tests
pytest tests/ -v
```

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
│   │   ├── extractor.py              # Phase 1: Rule-based extraction
│   │   └── llm_extractor.py          # Phase 2: LLM-based extraction
│   ├── validation/                   # Phase 2: Compliance validation
│   │   ├── rule_engine.py            # Configurable rule validation
│   │   ├── cross_document.py         # Cross-document consistency checks
│   │   └── report_generator.py       # Compliance report generation
│   └── utils/
│       ├── generate_synthetic_docs.py # Synthetic data generator
│       └── evaluate_extraction.py     # Evaluation harness
├── config/
│   ├── settings.yaml                 # Project configuration
│   └── compliance_rules.yaml         # Phase 2: Compliance rules
├── tests/
│   ├── test_pipeline.py              # Smoke tests
│   └── test_rule_engine.py           # Phase 2: Rule engine tests
├── docs/
│   └── phase1-technical.md           # Phase 1 technical documentation
├── run_pipeline.py                   # End-to-end pipeline runner
└── requirements.txt
```

---

## Document Types

| Type | Required Fields | Optional Fields |
|------|----------------|-----------------|
| **SOP** | title, version, effective_date, owner, department, approval_date, review_cycle | scope, references, approval_signatures |
| **Adverse Event Report** | event_id, date_reported, product_id, batch_id, severity, description, reporter, follow_up_status | patient_info, outcome, regulatory_filed |
| **Audit Finding Report** | audit_id, audit_date, finding_category, severity, description, corrective_action, due_date, responsible_party | root_cause, evidence, status |

---

## Phase 2 Features

### 1. LLM-Based Extraction

Uses Claude or GPT-4 with structured output to extract fields from documents.

**Features:**
- Few-shot prompt templates per document type
- Retry/validation logic with error correction
- Cost and latency tracking
- Automatic schema validation

**Usage:**
```bash
# With Anthropic Claude
export ANTHROPIC_API_KEY=your-key
python run_pipeline.py --method llm --skip-generation

# With OpenAI GPT
export OPENAI_API_KEY=your-key
python run_pipeline.py --method llm --skip-generation
```

### 2. Compliance Rule Validation Engine

Validates extracted data against configurable rules in `config/compliance_rules.yaml`.

**Rule Types:**
- **Required fields:** Missing fields flagged as critical
- **Format validation:** Regex patterns for IDs, dates
- **Allowed values:** Enum validation for severity, status
- **Date logic:** Future dates, date ordering
- **Length checks:** Min/max field lengths

**Usage:**
```bash
python run_pipeline.py --validate --skip-generation
```

### 3. Cross-Document Consistency

Validates that references between documents are consistent:
- Batch IDs exist in corpus
- Product IDs are referenced
- Reporters are known entities
- Departments are consistent

### 4. Compliance Report Generation

Produces both machine-readable and human-readable reports:

**JSON Report:** `data/processed/compliance_report.json`
- Full violation details
- Per-document breakdown
- Extraction metrics

**Markdown Report:** `data/processed/compliance_report.md`
- Executive summary
- Violation counts by severity
- Document-type breakdown
- Detailed violation list

---

## Configuration

### compliance_rules.yaml

Defines validation rules per document type:

```yaml
adverse_event:
  required_fields:
    - event_id
    - date_reported
    - severity
  field_rules:
    event_id:
      pattern: "^AE-[0-9]{4}-[0-9]{5}$"
    severity:
      allowed_values: [critical, high, medium, low]
    date_reported:
      must_not_be_future: true
```

### settings.yaml

Project configuration including:
- Document type definitions
- Extraction method
- Train/test split ratios
- Output paths

---

## Evaluation

### Phase 1 Baseline (Rule-Based)

**Overall F1: 0.82**

| Field | F1 | Notes |
|-------|-----|-------|
| approval_date | 1.00 | Labeled clearly |
| batch_id | 1.00 | Labeled clearly |
| description | 1.00 | Section-based |
| severity | 0.88 | Format variations |
| follow_up_status | 0.45 | Table extraction issues |
| scope | 0.00 | Section boundary weak |

### Phase 2 Target (LLM-Based)

**Target F1: 0.95+**

Expected improvements:
- `scope`, `references`: LLM handles complex sections
- `follow_up_status`: Better table understanding
- `doc_type`: Included in extraction output
- Overall: ~15% F1 improvement

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run rule engine tests specifically
pytest tests/test_rule_engine.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## API Key Setup

### Anthropic Claude (Recommended)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI GPT

```bash
export OPENAI_API_KEY=sk-...
```

**Cost Estimate:**
- Claude Sonnet: ~$0.003/1K input tokens, ~$0.015/1K output tokens
- Per document: ~$0.01-0.02
- Full dataset (300 docs): ~$3-6

---

## Phase 3 Roadmap

- [ ] Web dashboard for reviewing extractions
- [ ] Confidence scores for each field
- [ ] Batch processing with progress tracking
- [ ] Export to PDF compliance reports
- [ ] Real-time document processing API
- [ ] Integration with document management systems

---

## Technical Documentation

See `docs/phase1-technical.md` for detailed Phase 1 architecture and implementation details.

---

## License

Internal use only.
