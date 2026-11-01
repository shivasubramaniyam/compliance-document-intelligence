"""
Compliance Report Generator — Phase 2

Generates structured compliance reports from extraction + validation results.
Outputs both machine-readable JSON and human-readable Markdown.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .rule_engine import ValidationResult, ViolationSeverity


@dataclass
class ComplianceReport:
    """Complete compliance report for a batch of documents."""
    report_id: str
    generated_at: str
    total_documents: int
    valid_documents: int
    documents_with_violations: int
    
    # Aggregated violation counts
    critical_violations: int
    warning_violations: int
    info_violations: int
    
    # Per-type breakdown
    by_type: dict[str, dict[str, Any]]
    
    # Violation type breakdown
    violation_types: dict[str, int]
    
    # Per-document details
    documents: list[dict[str, Any]]
    
    # Extraction metrics (if available)
    extraction_metrics: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "summary": {
                "total_documents": self.total_documents,
                "valid_documents": self.valid_documents,
                "documents_with_violations": self.documents_with_violations,
                "compliance_rate": round(
                    self.valid_documents / max(self.total_documents, 1) * 100, 2
                ),
            },
            "violations": {
                "critical": self.critical_violations,
                "warning": self.warning_violations,
                "info": self.info_violations,
                "total": self.critical_violations + self.warning_violations + self.info_violations,
            },
            "by_type": self.by_type,
            "violation_types": self.violation_types,
            "documents": self.documents,
            "extraction_metrics": self.extraction_metrics,
        }


def generate_compliance_report(
    validation_results: list[ValidationResult],
    extraction_metrics: Optional[dict[str, Any]] = None,
    report_id: Optional[str] = None,
) -> ComplianceReport:
    """
    Generate a compliance report from validation results.
    
    Args:
        validation_results: List of ValidationResult objects
        extraction_metrics: Optional extraction metrics from LLM extractor
        report_id: Optional report identifier
        
    Returns:
        ComplianceReport object
    """
    if report_id is None:
        report_id = f"CR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    total = len(validation_results)
    valid = sum(1 for r in validation_results if r.is_valid)
    with_violations = total - valid

    # Count violations by severity
    critical = sum(r.critical_count for r in validation_results)
    warning = sum(r.warning_count for r in validation_results)
    info = sum(r.info_count for r in validation_results)

    # Per-type breakdown
    by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0, "valid": 0, "critical": 0, "warning": 0, "info": 0
    })
    for result in validation_results:
        t = result.doc_type
        by_type[t]["count"] += 1
        if result.is_valid:
            by_type[t]["valid"] += 1
        by_type[t]["critical"] += result.critical_count
        by_type[t]["warning"] += result.warning_count
        by_type[t]["info"] += result.info_count

    # Violation type breakdown
    violation_types: dict[str, int] = defaultdict(int)
    for result in validation_results:
        for v in result.violations:
            violation_types[v.rule_type] += 1

    # Per-document details
    documents = [r.to_dict() for r in validation_results]

    return ComplianceReport(
        report_id=report_id,
        generated_at=datetime.now().isoformat(),
        total_documents=total,
        valid_documents=valid,
        documents_with_violations=with_violations,
        critical_violations=critical,
        warning_violations=warning,
        info_violations=info,
        by_type=dict(by_type),
        violation_types=dict(violation_types),
        documents=documents,
        extraction_metrics=extraction_metrics,
    )


def save_report_json(report: ComplianceReport, output_path: Path) -> None:
    """Save report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)


def generate_report_markdown(report: ComplianceReport) -> str:
    """Generate human-readable Markdown report."""
    lines = [
        f"# Compliance Report: {report.report_id}",
        f"",
        f"**Generated:** {report.generated_at}",
        f"",
        "---",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Documents | {report.total_documents} |",
        f"| Valid Documents | {report.valid_documents} |",
        f"| Documents with Violations | {report.documents_with_violations} |",
        f"| Compliance Rate | {report.valid_documents / max(report.total_documents, 1) * 100:.1f}% |",
        f"",
        "## Violation Summary",
        "",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| 🔴 Critical | {report.critical_violations} |",
        f"| 🟡 Warning | {report.warning_violations} |",
        f"| 🔵 Info | {report.info_violations} |",
        f"| **Total** | **{report.critical_violations + report.warning_violations + report.info_violations}** |",
        f"",
    ]

    # By type
    if report.by_type:
        lines.extend([
            "## Breakdown by Document Type",
            "",
            "| Type | Count | Valid | Critical | Warning | Info |",
            "|------|-------|-------|----------|---------|------|",
        ])
        for doc_type, stats in sorted(report.by_type.items()):
            lines.append(
                f"| {doc_type} | {stats['count']} | {stats['valid']} | "
                f"{stats['critical']} | {stats['warning']} | {stats['info']} |"
            )
        lines.append("")

    # Violation types
    if report.violation_types:
        lines.extend([
            "## Violation Types",
            "",
            "| Type | Count |",
            "|------|-------|",
        ])
        for vtype, count in sorted(report.violation_types.items()):
            lines.append(f"| {vtype} | {count} |")
        lines.append("")

    # Extraction metrics
    if report.extraction_metrics:
        m = report.extraction_metrics
        lines.extend([
            "## Extraction Metrics",
            "",
            f"- **API Calls:** {m.get('total_calls', 0)}",
            f"- **Total Tokens (In):** {m.get('total_tokens_in', 0):,}",
            f"- **Total Tokens (Out):** {m.get('total_tokens_out', 0):,}",
            f"- **Total Latency:** {m.get('total_latency_ms', 0):,.0f}ms",
            f"- **Average Latency:** {m.get('avg_latency_ms', 0):,.0f}ms",
            f"- **Total Cost:** ${m.get('total_cost_usd', 0):.4f}",
            f"- **Cost per 1K Docs:** ${m.get('cost_per_1k_docs', 0):.4f}",
            f"- **Errors:** {m.get('errors', 0)}",
            f"- **Retries:** {m.get('retries', 0)}",
            "",
        ])

    # Document details (only those with violations)
    docs_with_violations = [d for d in report.documents if d.get("total_violations", 0) > 0]
    if docs_with_violations:
        lines.extend([
            "## Documents with Violations",
            "",
        ])
        for doc in docs_with_violations[:20]:  # Limit to first 20
            status = "❌" if doc.get("critical_count", 0) > 0 else "⚠️"
            lines.extend([
                f"### {status} {doc['document_id']} ({doc['doc_type']})",
                "",
                f"- Critical: {doc.get('critical_count', 0)}",
                f"- Warning: {doc.get('warning_count', 0)}",
                f"- Info: {doc.get('info_count', 0)}",
                "",
            ])
            for v in doc.get("violations", [])[:5]:  # Limit violations shown
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(v["severity"], "")
                lines.append(f"  - {severity_icon} **{v['field']}**: {v['message']}")
            if len(doc.get("violations", [])) > 5:
                lines.append(f"  - ... and {len(doc['violations']) - 5} more")
            lines.append("")

    lines.extend([
        "---",
        "",
        f"*Report generated by Compliance Document Intelligence — Phase 2*",
    ])

    return "\n".join(lines)


def save_report_markdown(report: ComplianceReport, output_path: Path) -> None:
    """Save report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_report_markdown(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
