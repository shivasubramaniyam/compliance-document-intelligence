"""
Cross-Document Consistency Checker — Phase 2

Validates that references between documents are consistent.
Checks that referenced IDs (batch IDs, product IDs, reporters) exist
in the document corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .rule_engine import Violation, ViolationSeverity


@dataclass
class DocumentRegistry:
    """
    In-memory registry of known entities across all processed documents.
    Used for cross-document consistency checks.
    """
    batch_ids: set[str] = field(default_factory=set)
    product_ids: set[str] = field(default_factory=set)
    reporters: set[str] = field(default_factory=set)
    audit_ids: set[str] = field(default_factory=set)
    event_ids: set[str] = field(default_factory=set)
    departments: set[str] = field(default_factory=set)
    owners: set[str] = field(default_factory=set)

    def add_document(self, doc_type: str, data: dict[str, Any]) -> None:
        """Add entities from a document to the registry."""
        if doc_type == "sop":
            if "department" in data:
                self.departments.add(data["department"])
            if "owner" in data:
                self.owners.add(data["owner"])

        elif doc_type == "adverse_event":
            if "batch_id" in data:
                self.batch_ids.add(data["batch_id"])
            if "product_id" in data:
                self.product_ids.add(data["product_id"])
            if "reporter" in data:
                self.reporters.add(data["reporter"])
            if "event_id" in data:
                self.event_ids.add(data["event_id"])

        elif doc_type == "audit_finding":
            if "audit_id" in data:
                self.audit_ids.add(data["audit_id"])
            if "responsible_party" in data:
                self.reporters.add(data["responsible_party"])  # Reporters can also be responsible parties

    def build_from_documents(self, documents: list[dict[str, Any]]) -> None:
        """Build registry from a list of document dicts."""
        for doc in documents:
            self.add_document(doc.get("doc_type", ""), doc.get("extracted_data", {}))


class CrossDocumentChecker:
    """
    Validates cross-document consistency rules.
    
    Rules:
    1. Adverse event batch_id should exist in other documents
    2. Adverse event product_id should exist in other documents
    3. Audit finding responsible_party should be a known reporter/person
    """

    def __init__(self, registry: Optional[DocumentRegistry] = None):
        self.registry = registry or DocumentRegistry()

    def check_document(
        self,
        document_id: str,
        doc_type: str,
        extracted_data: dict[str, Any],
    ) -> list[Violation]:
        """
        Check cross-document consistency for a single document.
        
        Returns list of violations (empty if all checks pass).
        """
        violations: list[Violation] = []

        if doc_type == "adverse_event":
            # Check batch_id exists in registry
            batch_id = extracted_data.get("batch_id")
            if batch_id and batch_id not in self.registry.batch_ids:
                violations.append(Violation(
                    field_name="batch_id",
                    severity=ViolationSeverity.WARNING,
                    rule_type="cross_document",
                    message=f"Batch ID '{batch_id}' not found in any other document in the corpus",
                    document_id=document_id,
                ))

            # Check product_id exists in registry
            product_id = extracted_data.get("product_id")
            if product_id and product_id not in self.registry.product_ids:
                violations.append(Violation(
                    field_name="product_id",
                    severity=ViolationSeverity.WARNING,
                    rule_type="cross_document",
                    message=f"Product ID '{product_id}' not found in any other document in the corpus",
                    document_id=document_id,
                ))

            # Check reporter is known
            reporter = extracted_data.get("reporter")
            if reporter and reporter not in self.registry.reporters:
                violations.append(Violation(
                    field_name="reporter",
                    severity=ViolationSeverity.INFO,
                    rule_type="cross_document",
                    message=f"Reporter '{reporter}' is not a known person in the corpus",
                    document_id=document_id,
                ))

        elif doc_type == "audit_finding":
            # Check responsible_party is known
            responsible = extracted_data.get("responsible_party")
            if responsible and responsible not in self.registry.reporters:
                violations.append(Violation(
                    field_name="responsible_party",
                    severity=ViolationSeverity.INFO,
                    rule_type="cross_document",
                    message=f"Responsible party '{responsible}' is not a known person in the corpus",
                    document_id=document_id,
                ))

            # Check if audit references a known event
            # (This would require more complex cross-referencing in a real system)

        elif doc_type == "sop":
            # Check department is consistent
            department = extracted_data.get("department")
            if department and department not in self.registry.departments:
                violations.append(Violation(
                    field_name="department",
                    severity=ViolationSeverity.INFO,
                    rule_type="cross_document",
                    message=f"Department '{department}' is not referenced in any other document",
                    document_id=document_id,
                ))

        return violations

    def check_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> list[list[Violation]]:
        """
        Check cross-document consistency for a batch of documents.
        
        First builds the registry, then checks each document.
        """
        # Build registry from all documents
        self.registry.build_from_documents(documents)

        # Check each document
        all_violations = []
        for doc in documents:
            violations = self.check_document(
                document_id=doc.get("document_id", "unknown"),
                doc_type=doc.get("doc_type", "unknown"),
                extracted_data=doc.get("extracted_data", {}),
            )
            all_violations.append(violations)

        return all_violations
