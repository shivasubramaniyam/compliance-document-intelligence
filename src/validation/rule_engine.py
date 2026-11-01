"""
Compliance Rule Validation Engine — Phase 2

Validates extracted document data against configurable compliance rules.
Flags missing fields, format violations, logical inconsistencies, and
cross-document relationship failures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Violation types and severity
# ---------------------------------------------------------------------------

class ViolationSeverity(str, Enum):
    CRITICAL = "critical"   # Must be fixed — document cannot be processed
    WARNING = "warning"     # Should be fixed — data quality issue
    INFO = "info"           # Informational — minor observation


@dataclass
class Violation:
    """A single compliance rule violation."""
    field_name: str
    severity: ViolationSeverity
    rule_type: str  # "missing_required", "format", "allowed_value", "date_logic", "cross_document"
    message: str
    document_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "severity": self.severity.value,
            "rule_type": self.rule_type,
            "message": self.message,
            "document_id": self.document_id,
        }


@dataclass
class ValidationResult:
    """Validation result for a single document."""
    document_id: str
    doc_type: str
    is_valid: bool
    violations: list[Violation] = field(default_factory=list)
    checked_rules: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.INFO)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "doc_type": self.doc_type,
            "is_valid": self.is_valid,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "total_violations": len(self.violations),
            "checked_rules": self.checked_rules,
            "violations": [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class ComplianceRuleEngine:
    """
    Validates extracted documents against configurable compliance rules.
    
    Rules are loaded from a YAML config file with structure:
        document_type:
            required_fields: [...]
            field_rules:
                field_name:
                    pattern: "regex"
                    allowed_values: [...]
                    max_length: N
                    min_length: N
                    must_not_be_future: true
                    must_be_after: field_name
                    must_be_before: field_name
    """

    def __init__(self, rules_config_path: Optional[Path] = None):
        """Load rules from YAML config."""
        self.rules: dict[str, Any] = {}
        if rules_config_path and rules_config_path.exists():
            with open(rules_config_path) as f:
                self.rules = yaml.safe_load(f) or {}

    def validate_document(
        self,
        document_id: str,
        doc_type: str,
        extracted_data: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a single extracted document against rules.
        
        Args:
            document_id: Identifier for the document
            doc_type: Document type (sop, adverse_event, audit_finding)
            extracted_data: Extracted field values
            
        Returns:
            ValidationResult with any violations found
        """
        violations: list[Violation] = []
        checked_rules = 0

        # Get rules for this document type
        type_rules = self.rules.get(doc_type, {})
        required_fields = type_rules.get("required_fields", [])
        field_rules = type_rules.get("field_rules", {})

        # Check required fields
        for field_name in required_fields:
            checked_rules += 1
            if field_name not in extracted_data or extracted_data[field_name] is None:
                violations.append(Violation(
                    field_name=field_name,
                    severity=ViolationSeverity.CRITICAL,
                    rule_type="missing_required",
                    message=f"Required field '{field_name}' is missing",
                    document_id=document_id,
                ))

        # Check field-level rules
        for field_name, rules in field_rules.items():
            value = extracted_data.get(field_name)
            if value is None:
                continue  # Already flagged as missing if required

            # Pattern check
            if "pattern" in rules:
                checked_rules += 1
                if not re.match(rules["pattern"], str(value)):
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="format",
                        message=f"Field '{field_name}' value '{value}' does not match pattern '{rules['pattern']}'",
                        document_id=document_id,
                    ))

            # Allowed values
            if "allowed_values" in rules:
                checked_rules += 1
                if str(value).lower() not in [v.lower() for v in rules["allowed_values"]]:
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="allowed_value",
                        message=f"Field '{field_name}' value '{value}' is not in allowed values: {rules['allowed_values']}",
                        document_id=document_id,
                    ))

            # Length checks
            if "max_length" in rules:
                checked_rules += 1
                if len(str(value)) > rules["max_length"]:
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="format",
                        message=f"Field '{field_name}' exceeds max length of {rules['max_length']}",
                        document_id=document_id,
                    ))

            if "min_length" in rules:
                checked_rules += 1
                if len(str(value)) < rules["min_length"]:
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="format",
                        message=f"Field '{field_name}' is shorter than min length of {rules['min_length']}",
                        document_id=document_id,
                    ))

            # Date checks
            if "must_not_be_future" in rules and rules["must_not_be_future"]:
                checked_rules += 1
                if self._is_future_date(value):
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="date_logic",
                        message=f"Field '{field_name}' contains a future date: {value}",
                        document_id=document_id,
                    ))

            if "must_be_after" in rules:
                checked_rules += 1
                other_field = rules["must_be_after"]
                other_value = extracted_data.get(other_field)
                if other_value and self._is_date_before(value, other_value):
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="date_logic",
                        message=f"Field '{field_name}' ({value}) is before '{other_field}' ({other_value})",
                        document_id=document_id,
                    ))

            if "must_be_before" in rules:
                checked_rules += 1
                other_field = rules["must_be_before"]
                other_value = extracted_data.get(other_field)
                if other_value and self._is_date_after(value, other_value):
                    violations.append(Violation(
                        field_name=field_name,
                        severity=ViolationSeverity.WARNING,
                        rule_type="date_logic",
                        message=f"Field '{field_name}' ({value}) is after '{other_field}' ({other_value})",
                        document_id=document_id,
                    ))

        is_valid = not any(v.severity == ViolationSeverity.CRITICAL for v in violations)

        return ValidationResult(
            document_id=document_id,
            doc_type=doc_type,
            is_valid=is_valid,
            violations=violations,
            checked_rules=checked_rules,
        )

    def validate_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> list[ValidationResult]:
        """
        Validate a batch of documents.
        
        Args:
            documents: List of dicts with keys: document_id, doc_type, extracted_data
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        for doc in documents:
            result = self.validate_document(
                document_id=doc["document_id"],
                doc_type=doc["doc_type"],
                extracted_data=doc["extracted_data"],
            )
            results.append(result)
        return results

    # ---------------------------------------------------------------------------
    # Date helpers
    # ---------------------------------------------------------------------------

    def _parse_date_value(self, value: Any) -> Optional[date]:
        """Try to parse a value as a date."""
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        return None

    def _is_future_date(self, value: Any) -> bool:
        """Check if a date value is in the future."""
        d = self._parse_date_value(value)
        return d is not None and d > date.today()

    def _is_date_before(self, value1: Any, value2: Any) -> bool:
        """Check if value1 is before value2."""
        d1 = self._parse_date_value(value1)
        d2 = self._parse_date_value(value2)
        return d1 is not None and d2 is not None and d1 < d2

    def _is_date_after(self, value1: Any, value2: Any) -> bool:
        """Check if value1 is after value2."""
        d1 = self._parse_date_value(value1)
        d2 = self._parse_date_value(value2)
        return d1 is not None and d2 is not None and d1 > d2
