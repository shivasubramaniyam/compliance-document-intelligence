"""
Pydantic schemas for compliance document structured data.
These schemas define the "ground truth" and extraction output format.
Both ground-truth and extracted data must validate against these schemas.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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


class ApprovalSignature(BaseModel):
    name: str
    title: str
    date: date


class SOPDocument(BaseModel):
    """Standard Operating Procedure schema."""
    doc_type: DocumentType = DocumentType.SOP
    title: str
    version: str
    effective_date: date
    owner: str
    department: str
    approval_date: date
    review_cycle: str  # e.g., "Annual", "Semi-annual"
    scope: Optional[str] = None
    references: Optional[list[str]] = None
    approval_signatures: Optional[list[ApprovalSignature]] = None


class AdverseEventReport(BaseModel):
    """Adverse Event Report schema."""
    doc_type: DocumentType = DocumentType.ADVERSE_EVENT
    event_id: str  # e.g., "AE-2025-00001"
    date_reported: date
    product_id: str
    batch_id: str
    severity: SeverityLevel
    description: str
    reporter: str
    follow_up_status: FollowUpStatus
    patient_info: Optional[str] = None
    outcome: Optional[str] = None
    regulatory_filed: Optional[bool] = None


class AuditFindingReport(BaseModel):
    """Audit Finding Report schema."""
    doc_type: DocumentType = DocumentType.AUDIT_FINDING
    audit_id: str  # e.g., "AUD-2025-00001"
    audit_date: date
    finding_category: str
    severity: SeverityLevel
    description: str
    corrective_action: str
    due_date: date
    responsible_party: str
    root_cause: Optional[str] = None
    evidence: Optional[str] = None
    status: Optional[FindingStatus] = None


# Union type for any document
AnyDocument = SOPDocument | AdverseEventReport | AuditFindingReport

# Schema registry for dynamic lookup
SCHEMA_REGISTRY: dict[DocumentType, type[BaseModel]] = {
    DocumentType.SOP: SOPDocument,
    DocumentType.ADVERSE_EVENT: AdverseEventReport,
    DocumentType.AUDIT_FINDING: AuditFindingReport,
}


def get_schema(doc_type: DocumentType) -> type[BaseModel]:
    """Get the Pydantic schema for a given document type."""
    return SCHEMA_REGISTRY[doc_type]


def validate_document(doc_type: DocumentType, data: dict) -> AnyDocument:
    """Validate raw dict against the appropriate schema."""
    schema = get_schema(doc_type)
    return schema.model_validate(data)
