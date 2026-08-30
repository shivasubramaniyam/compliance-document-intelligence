"""
Evaluation Harness

Compares extracted JSON against ground-truth JSON per document.
Computes field-level precision/recall/F1 and outputs a summary report.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..schemas import DocumentType, get_schema


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _normalize_value(val: Any) -> str:
    """Normalize a value for comparison (lowercase string, strip whitespace)."""
    if val is None:
        return ""
    return str(val).strip().lower()


def _compare_field(extracted: Any, ground_truth: Any) -> bool:
    """Check if an extracted field value matches ground truth."""
    # Handle date comparisons — normalize to ISO string
    if hasattr(extracted, "isoformat"):
        extracted = extracted.isoformat()
    if hasattr(ground_truth, "isoformat"):
        ground_truth = ground_truth.isoformat()

    return _normalize_value(extracted) == _normalize_value(ground_truth)


# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

def score_document(
    extracted: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """
    Score a single document's extraction against ground truth.

    Returns per-field TP/FP/FN counts and accuracy.
    """
    gt_keys = set(ground_truth.keys())
    ext_keys = set(extracted.keys())
    all_keys = gt_keys | ext_keys

    field_results: dict[str, dict[str, Any]] = {}

    for field in sorted(all_keys):
        gt_val = ground_truth.get(field)
        ext_val = extracted.get(field)

        # Ground truth has field, extraction doesn't → False Negative
        if gt_val is not None and ext_val is None:
            field_results[field] = {
                "status": "missed",
                "gt": gt_val,
                "ext": None,
                "tp": 0, "fp": 0, "fn": 1,
            }
        # Extraction has field but ground truth doesn't → False Positive
        elif gt_val is None and ext_val is not None:
            field_results[field] = {
                "status": "hallucinated",
                "gt": None,
                "ext": ext_val,
                "tp": 0, "fp": 1, "fn": 0,
            }
        # Both have field → check value match
        elif gt_val is not None and ext_val is not None:
            if _compare_field(ext_val, gt_val):
                field_results[field] = {
                    "status": "correct",
                    "gt": gt_val,
                    "ext": ext_val,
                    "tp": 1, "fp": 0, "fn": 0,
                }
            else:
                field_results[field] = {
                    "status": "wrong_value",
                    "gt": gt_val,
                    "ext": ext_val,
                    "tp": 0, "fp": 1, "fn": 1,
                }
        else:
            # Both None — trivial match
            field_results[field] = {
                "status": "both_missing",
                "gt": None,
                "ext": None,
                "tp": 0, "fp": 0, "fn": 0,
            }

    # Aggregate
    total_tp = sum(f["tp"] for f in field_results.values())
    total_fp = sum(f["fp"] for f in field_results.values())
    total_fn = sum(f["fn"] for f in field_results.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "field_results": field_results,
        "aggregate": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_fields": len(all_keys),
        },
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    processed_dir: Path,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate all documents in a processed directory.

    Expected file layout:
        processed_dir/<doc_type>/<split>/
            <doc_id>_extracted.json
            <doc_id>_ground_truth.json

    Or flat:
        processed_dir/
            <doc_id>_extracted.json
            <doc_id>_ground_truth.json
    """
    # Find all extracted files
    extracted_files = sorted(processed_dir.rglob("*_extracted.json"))
    if not extracted_files:
        print(f"No extracted files found in {processed_dir}")
        return {}

    # Group by document type
    type_scores: dict[str, list[dict]] = defaultdict(list)

    for ext_path in extracted_files:
        # Find corresponding ground truth
        gt_path = ext_path.parent / ext_path.name.replace("_extracted.json", "_ground_truth.json")
        if not gt_path.exists():
            print(f"Warning: No ground truth for {ext_path.name}")
            continue

        with open(ext_path) as f:
            extracted = json.load(f)
        with open(gt_path) as f:
            ground_truth = json.load(f)

        # Determine doc type
        dtype = ground_truth.get("doc_type", ext_path.parent.name)
        if doc_type and dtype != doc_type:
            continue

        score = score_document(extracted, ground_truth)
        score["doc_id"] = ext_path.stem.replace("_extracted", "")
        score["doc_type"] = dtype
        type_scores[dtype].append(score)

    # Aggregate per document type
    summary: dict[str, Any] = {}
    for dtype, scores in type_scores.items():
        if not scores:
            continue

        # Average precision/recall/f1
        avg_precision = sum(s["aggregate"]["precision"] for s in scores) / len(scores)
        avg_recall = sum(s["aggregate"]["recall"] for s in scores) / len(scores)
        avg_f1 = sum(s["aggregate"]["f1"] for s in scores) / len(scores)

        # Per-field aggregates
        field_agg: dict[str, dict[str, float]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
        for s in scores:
            for field, result in s["field_results"].items():
                field_agg[field]["tp"] += result["tp"]
                field_agg[field]["fp"] += result["fp"]
                field_agg[field]["fn"] += result["fn"]
                field_agg[field]["count"] += 1

        field_metrics: dict[str, dict[str, float]] = {}
        for field, counts in field_agg.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            field_metrics[field] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": counts["count"],
            }

        summary[dtype] = {
            "num_documents": len(scores),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1": round(avg_f1, 4),
            "per_field": field_metrics,
        }

    return summary


def print_report(summary: dict[str, Any]) -> str:
    """Print and return a formatted evaluation report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("COMPLIANCE DOCUMENT INTELLIGENCE — EXTRACTION EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    for dtype, metrics in summary.items():
        lines.append(f"--- {dtype.upper()} ---")
        lines.append(f"  Documents evaluated: {metrics['num_documents']}")
        lines.append(f"  Average Precision:   {metrics['avg_precision']:.4f}")
        lines.append(f"  Average Recall:      {metrics['avg_recall']:.4f}")
        lines.append(f"  Average F1:          {metrics['avg_f1']:.4f}")
        lines.append("")
        lines.append("  Per-Field Breakdown:")
        lines.append(f"  {'Field':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        lines.append("  " + "-" * 65)
        for field, fm in sorted(metrics["per_field"].items()):
            lines.append(
                f"  {field:<25} {fm['precision']:>10.4f} {fm['recall']:>10.4f} {fm['f1']:>10.4f} {fm['support']:>8d}"
            )
        lines.append("")

    # Overall
    if summary:
        all_f1 = [m["avg_f1"] for m in summary.values()]
        overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        lines.append("=" * 80)
        lines.append(f"OVERALL AVERAGE F1: {overall_f1:.4f}")
        lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate extraction against ground truth")
    parser.add_argument("--processed-dir", type=str, default="data/processed/test")
    parser.add_argument("--doc-type", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    summary = evaluate_dataset(Path(args.processed_dir), doc_type=args.doc_type)
    if summary:
        report = print_report(summary)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nReport saved to {out_path}")
