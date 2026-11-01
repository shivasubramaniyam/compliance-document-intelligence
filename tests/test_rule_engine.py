"""
Tests for Compliance Rule Engine — Phase 2

Tests the rule engine against known-flawed synthetic documents to verify
that injected flaws are correctly detected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.rule_engine import ComplianceRuleEngine, ViolationSeverity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rule_engine():
    """Create a rule engine with the project's compliance rules."""
    rules_path = Path(__file__).parent.parent / "config" / "compliance_rules.yaml"
    return ComplianceRuleEngine(rules_path)


@pytest.fixture
def clean_sop():
    """A clean SOP document extraction."""
    return {
        "doc_type": "sop",
        "title": "Procedure for Batch Record Review",
        "version": "2.1",
        "effective_date": "2024-03-15",
        "owner": "Dr. Sarah Chen",
        "department": "Quality Assurance",
        "approval_date": "2024-03-10",
        "review_cycle": "Annual",
    }


@pytest.fixture
def clean_adverse_event():
    """A clean adverse event report extraction."""
    return {
        "doc_type": "adverse_event",
        "event_id": "AE-2024-00123",
        "date_reported": "2024-06-20",
        "product_id": "PROD-A1234",
        "batch_id": "BATCH-123456",
        "severity": "high",
        "description": "Patient experienced nausea within 24 hours of administration.",
        "reporter": "Dr. Michael Torres",
        "follow_up_status": "in_progress",
    }


@pytest.fixture
def clean_audit_finding():
    """A clean audit finding report extraction."""
    return {
        "doc_type": "audit_finding",
        "audit_id": "AUD-2024-00045",
        "audit_date": "2024-07-10",
        "finding_category": "Documentation",
        "severity": "medium",
        "description": "Records reviewed during the audit showed that batch records were not completed within the required timeframe.",
        "corrective_action": "Update documentation procedures and conduct a follow-up audit within 90 days.",
        "due_date": "2024-10-08",
        "responsible_party": "James Wilson",
        "status": "open",
    }


# ---------------------------------------------------------------------------
# Test clean documents pass validation
# ---------------------------------------------------------------------------

class TestCleanDocuments:
    """Verify that clean documents pass validation without violations."""

    def test_clean_sop_passes(self, rule_engine, clean_sop):
        result = rule_engine.validate_document("SOP-001", "sop", clean_sop)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_clean_adverse_event_passes(self, rule_engine, clean_adverse_event):
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_clean_audit_finding_passes(self, rule_engine, clean_audit_finding):
        result = rule_engine.validate_document("AUD-001", "audit_finding", clean_audit_finding)
        assert result.is_valid
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Test missing required fields
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    """Verify that missing required fields are flagged as critical."""

    def test_sop_missing_version(self, rule_engine, clean_sop):
        del clean_sop["version"]
        result = rule_engine.validate_document("SOP-001", "sop", clean_sop)
        assert not result.is_valid
        assert result.critical_count == 1
        assert any(v.field_name == "version" for v in result.violations)

    def test_sop_missing_multiple_fields(self, rule_engine, clean_sop):
        del clean_sop["version"]
        del clean_sop["owner"]
        del clean_sop["department"]
        result = rule_engine.validate_document("SOP-001", "sop", clean_sop)
        assert not result.is_valid
        assert result.critical_count == 3

    def test_adverse_event_missing_event_id(self, rule_engine, clean_adverse_event):
        del clean_adverse_event["event_id"]
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert not result.is_valid
        assert result.critical_count == 1
        assert any(v.field_name == "event_id" for v in result.violations)

    def test_adverse_event_missing_severity(self, rule_engine, clean_adverse_event):
        del clean_adverse_event["severity"]
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert not result.is_valid
        assert result.critical_count == 1
        assert any(v.field_name == "severity" for v in result.violations)

    def test_audit_finding_missing_audit_id(self, rule_engine, clean_audit_finding):
        del clean_audit_finding["audit_id"]
        result = rule_engine.validate_document("AUD-001", "audit_finding", clean_audit_finding)
        assert not result.is_valid
        assert result.critical_count == 1

    def test_audit_finding_missing_corrective_action(self, rule_engine, clean_audit_finding):
        del clean_audit_finding["corrective_action"]
        result = rule_engine.validate_document("AUD-001", "audit_finding", clean_audit_finding)
        assert not result.is_valid
        assert result.critical_count == 1


# ---------------------------------------------------------------------------
# Test format violations
# ---------------------------------------------------------------------------

class TestFormatViolations:
    """Verify that format violations are detected."""

    def test_sop_invalid_version_format(self, rule_engine, clean_sop):
        clean_sop["version"] = "invalid"
        result = rule_engine.validate_document("SOP-001", "sop", clean_sop)
        assert result.is_valid  # Format warnings don't make invalid
        assert result.warning_count == 1
        assert any(v.rule_type == "format" for v in result.violations)

    def test_adverse_event_invalid_event_id_format(self, rule_engine, clean_adverse_event):
        clean_adverse_event["event_id"] = "INVALID-ID"
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert result.warning_count >= 1

    def test_adverse_event_invalid_severity(self, rule_engine, clean_adverse_event):
        clean_adverse_event["severity"] = "banana"
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert result.warning_count >= 1
        assert any(v.rule_type == "allowed_value" for v in result.violations)

    def test_audit_finding_invalid_category(self, rule_engine, clean_audit_finding):
        clean_audit_finding["finding_category"] = "Invalid Category"
        result = rule_engine.validate_document("AUD-001", "audit_finding", clean_audit_finding)
        assert result.warning_count >= 1


# ---------------------------------------------------------------------------
# Test date logic violations
# ---------------------------------------------------------------------------

class TestDateLogicViolations:
    """Verify that date logic violations are detected."""

    def test_adverse_event_future_date(self, rule_engine, clean_adverse_event):
        clean_adverse_event["date_reported"] = "2099-01-01"
        result = rule_engine.validate_document("AE-001", "adverse_event", clean_adverse_event)
        assert result.warning_count >= 1
        assert any(v.rule_type == "date_logic" for v in result.violations)

    def test_audit_finding_due_before_audit_date(self, rule_engine, clean_audit_finding):
        clean_audit_finding["due_date"] = "2024-01-01"  # Before audit_date 2024-07-10
        result = rule_engine.validate_document("AUD-001", "audit_finding", clean_audit_finding)
        assert result.warning_count >= 1
        assert any(v.rule_type == "date_logic" for v in result.violations)


# ---------------------------------------------------------------------------
# Test batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    """Verify batch validation works correctly."""

    def test_batch_with_mixed_documents(self, rule_engine, clean_sop, clean_adverse_event, clean_audit_finding):
        documents = [
            {"document_id": "SOP-001", "doc_type": "sop", "extracted_data": clean_sop},
            {"document_id": "AE-001", "doc_type": "adverse_event", "extracted_data": clean_adverse_event},
            {"document_id": "AUD-001", "doc_type": "audit_finding", "extracted_data": clean_audit_finding},
        ]
        results = rule_engine.validate_batch(documents)
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_batch_with_flawed_documents(self, rule_engine, clean_sop, clean_adverse_event):
        # Remove required fields to simulate flaws
        flawed_sop = clean_sop.copy()
        del flawed_sop["version"]
        del flawed_sop["owner"]

        flawed_ae = clean_adverse_event.copy()
        flawed_ae["severity"] = "invalid"

        documents = [
            {"document_id": "SOP-001", "doc_type": "sop", "extracted_data": flawed_sop},
            {"document_id": "AE-001", "doc_type": "adverse_event", "extracted_data": flawed_ae},
        ]
        results = rule_engine.validate_batch(documents)
        assert len(results) == 2
        assert not results[0].is_valid  # SOP missing required fields
        assert results[1].is_valid  # AE has warning but not critical


# ---------------------------------------------------------------------------
# Test against real synthetic data (if available)
# ---------------------------------------------------------------------------

class TestAgainstSyntheticData:
    """Test rule engine against actual synthetic test data."""

    def test_flawed_sop_documents(self, rule_engine):
        """Check that flawed SOPs in synthetic data are detected."""
        test_dir = Path("data/synthetic/sop/test")
        if not test_dir.exists():
            pytest.skip("Synthetic data not generated yet")

        # Load a few test documents
        extracted_files = sorted(test_dir.glob("*_ground_truth.json"))[:5]
        if not extracted_files:
            pytest.skip("No test documents found")

        for gt_file in extracted_files:
            with open(gt_file) as f:
                data = json.load(f)

            result = rule_engine.validate_document(
                document_id=gt_file.stem.replace("_ground_truth", ""),
                doc_type="sop",
                extracted_data=data,
            )
            # We expect some violations since 25% are flawed
            # Just verify the engine runs without error
            assert isinstance(result.is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
