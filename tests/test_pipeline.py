"""
Smoke tests for Phase 1 pipeline components.
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import (
    AdverseEventReport,
    AuditFindingReport,
    DocumentType,
    SOPDocument,
    SeverityLevel,
    validate_document,
)
from src.ingestion.loader import classify_document, _clean_text
from src.extraction.extractor import (
    extract_sop,
    extract_adverse_event,
    extract_audit_finding,
    _parse_date,
    _parse_severity,
)
from src.utils.evaluate_extraction import score_document, _normalize_value


def test_schemas_valid():
    """Test that clean data validates against schemas."""
    sop = SOPDocument(
        title="Test SOP",
        version="1.0",
        effective_date="2025-01-01",
        owner="John Doe",
        department="QA",
        approval_date="2024-12-15",
        review_cycle="Annual",
    )
    assert sop.doc_type == DocumentType.SOP
    assert sop.title == "Test SOP"

    ae = AdverseEventReport(
        event_id="AE-2025-00001",
        date_reported="2025-03-15",
        product_id="PROD-A1234",
        batch_id="BATCH-123456",
        severity=SeverityLevel.HIGH,
        description="Test event",
        reporter="Jane Smith",
        follow_up_status="open",
    )
    assert ae.event_id == "AE-2025-00001"

    audit = AuditFindingReport(
        audit_id="AUD-2025-00001",
        audit_date="2025-06-01",
        finding_category="Documentation",
        severity=SeverityLevel.MEDIUM,
        description="Test finding",
        corrective_action="Update procedures",
        due_date="2025-09-01",
        responsible_party="Bob Jones",
    )
    assert audit.audit_id == "AUD-2025-00001"
    print("  [OK] Schemas validate correctly")


def test_document_classifier():
    """Test heuristic document type classifier."""
    sop_text = "Standard Operating Procedure\nReview Cycle: Annual\nEffective Date: 2025-01-01\nScope: This procedure covers GMP compliance."
    assert classify_document(sop_text) == "sop"

    ae_text = "Adverse Event Report\nEvent ID: AE-2025-00001\nBatch ID: BATCH-123456\nSeverity: HIGH\nFollow-Up Status: Open"
    assert classify_document(ae_text) == "adverse_event"

    audit_text = "Audit Finding Report\nAudit ID: AUD-2025-00001\nFinding Category: Documentation\nCorrective Action: Update procedures"
    assert classify_document(audit_text) == "audit_finding"

    unknown_text = "This is just some random document with no compliance keywords."
    assert classify_document(unknown_text) == "unknown"
    print("  [OK] Document classifier works correctly")


def test_date_parsing():
    """Test date parsing with various formats."""
    assert _parse_date("2025-01-15") is not None
    assert _parse_date("01/15/2025") is not None
    assert _parse_date("not-a-date") is None
    assert _parse_date("") is None
    assert _parse_date("2025/13/45") is None
    print("  [OK] Date parsing works correctly")


def test_severity_parsing():
    """Test severity parsing."""
    assert _parse_severity("critical") == SeverityLevel.CRITICAL
    assert _parse_severity("HIGH") == SeverityLevel.HIGH
    assert _parse_severity("medium") == SeverityLevel.MEDIUM
    assert _parse_severity("banana") is None
    print("  [OK] Severity parsing works correctly")


def test_extraction_sop():
    """Test SOP field extraction from text."""
    text = """Test SOP v1.0
Version: 2.1
Effective Date: 2025-03-01
Owner: Dr. Smith
Department: Quality Assurance
Approval Date: 2025-02-15
Review Cycle: Annual

Scope
This procedure covers batch record review activities.

References
• ICH Q10
• 21 CFR Part 211

Procedure
This is the body of the procedure.
"""
    result = extract_sop(text)
    assert result.get("title") == "Test SOP v1.0"
    assert result.get("version") == "2.1"
    assert result.get("owner") == "Dr. Smith"
    assert result.get("department") == "Quality Assurance"
    print("  [OK] SOP extraction works correctly")


def test_extraction_adverse_event():
    """Test Adverse Event Report field extraction."""
    text = """Adverse Event Report

Event ID: AE-2025-00042
Date Reported: 2025-06-15
Product ID: PROD-A1234
Batch ID: BATCH-654321
Severity: HIGH
Reporter: Dr. Johnson
Follow-Up Status: In Progress

Description of Event
Patient experienced nausea within 24 hours of administration.
"""
    result = extract_adverse_event(text)
    assert result.get("event_id") == "AE-2025-00042"
    assert result.get("product_id") == "PROD-A1234"
    assert result.get("severity") == "high"
    print("  [OK] Adverse Event extraction works correctly")


def test_extraction_audit_finding():
    """Test Audit Finding Report field extraction."""
    text = """Audit Finding Report

Audit ID: AUD-2025-00015
Audit Date: 2025-04-10
Finding Category: Documentation
Severity: MEDIUM
Due Date: 2025-07-10
Responsible Party: Jane Doe

Finding Description
Records reviewed showed incomplete batch documentation.

Corrective Action Required
Implement revised documentation procedures and retrain personnel.
"""
    result = extract_audit_finding(text)
    assert result.get("audit_id") == "AUD-2025-00015"
    assert result.get("finding_category") == "Documentation"
    assert result.get("responsible_party") == "Jane Doe"
    print("  [OK] Audit Finding extraction works correctly")


def test_evaluation_scoring():
    """Test the evaluation scoring logic."""
    extracted = {
        "event_id": "AE-2025-00001",
        "severity": "high",
        "reporter": "John",
        "extra_field": "unexpected",
    }
    ground_truth = {
        "event_id": "AE-2025-00001",
        "severity": "high",
        "reporter": "Jane",  # wrong value
        "missing_field": "should_be_here",
    }

    result = score_document(extracted, ground_truth)

    fr = result["field_results"]
    assert fr["event_id"]["status"] == "correct"
    assert fr["severity"]["status"] == "correct"
    assert fr["reporter"]["status"] == "wrong_value"
    assert fr["extra_field"]["status"] == "hallucinated"
    assert fr["missing_field"]["status"] == "missed"

    agg = result["aggregate"]
    assert agg["tp"] == 2  # event_id + severity
    assert agg["fp"] == 2  # wrong reporter + hallucinated extra_field
    assert agg["fn"] == 2  # missing reporter match + missing_field
    print("  [OK] Evaluation scoring works correctly")


def test_generate_and_load():
    """Integration test: generate a small dataset, load, and extract."""
    from src.utils.generate_synthetic_docs import generate_dataset

    test_dir = Path("data/synthetic_test_run")
    try:
        generate_dataset(test_dir, num_per_type=3, flawed_ratio=0.33)

        # Check files were created
        for doc_type in ["sop", "adverse_event", "audit_finding"]:
            for split in ["train", "test"]:
                split_dir = test_dir / doc_type / split
                docs = list(split_dir.glob("*.docx"))
                gts = list(split_dir.glob("*_ground_truth.json"))
                assert len(docs) > 0, f"No docs in {doc_type}/{split}"
                assert len(gts) > 0, f"No ground truths in {doc_type}/{split}"
                assert len(docs) == len(gts), f"Mismatched count in {doc_type}/{split}"

                # Load and extract from first doc
                from src.ingestion.loader import load_document
                from src.extraction.extractor import extract_and_validate

                doc = load_document(docs[0])
                assert doc.raw_text
                assert doc.doc_type_guess in ("sop", "adverse_event", "audit_finding", "unknown")

                extracted = extract_and_validate(doc)
                assert isinstance(extracted, dict)
                assert len(extracted) > 0

        print("  [OK] Integration test passed (generate -> load -> extract)")
    finally:
        # Clean up
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


def main():
    print("\nRunning smoke tests...\n")
    test_schemas_valid()
    test_document_classifier()
    test_date_parsing()
    test_severity_parsing()
    test_extraction_sop()
    test_extraction_adverse_event()
    test_extraction_audit_finding()
    test_evaluation_scoring()
    test_generate_and_load()
    print("\n[SUCCESS] All smoke tests passed!\n")


if __name__ == "__main__":
    main()
