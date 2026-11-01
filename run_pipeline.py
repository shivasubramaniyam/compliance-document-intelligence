"""
Compliance Document Intelligence — End-to-End Pipeline

Supports both Phase 1 (rule-based) and Phase 2 (LLM-based) extraction.

Usage:
    # Phase 1: Rule-based extraction
    python run_pipeline.py --method rule_based
    
    # Phase 2: LLM-based extraction (requires API key)
    python run_pipeline.py --method llm --skip-generation
    
    # Phase 2: With validation
    python run_pipeline.py --method llm --validate --skip-generation
    
    # Skip generation if data already exists
    python run_pipeline.py --skip-generation
    
    # Customize document count
    python run_pipeline.py --num-per-type 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import time

from src.ingestion.loader import load_document
from src.extraction.extractor import extract_and_validate as rule_based_extract
from src.utils.generate_synthetic_docs import generate_dataset
from src.utils.evaluate_extraction import evaluate_dataset, print_report


def step1_generate_data(base_dir: Path, num_per_type: int = 100) -> None:
    """Step 1: Generate synthetic compliance documents."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Documents")
    print("=" * 60)
    generate_dataset(base_dir, num_per_type=num_per_type, flawed_ratio=0.25)
    print("[OK] Synthetic documents generated")


def step2_ingest_and_extract(
    synthetic_dir: Path,
    output_dir: Path,
    method: str = "rule_based",
    validate: bool = False,
) -> dict:
    """Step 2: Ingest documents and run extraction."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Ingestion & Extraction ({method})")
    print("=" * 60)

    # Import LLM extractor if needed
    llm_metrics = None
    if method == "llm":
        from src.extraction.llm_extractor import llm_extract_fields, get_extraction_metrics, reset_metrics
        reset_metrics()

    doc_types = ["sop", "adverse_event", "audit_finding"]
    total_docs = 0
    total_errors = 0

    for doc_type in doc_types:
        for split in ["train", "test"]:
            split_dir = synthetic_dir / doc_type / split
            if not split_dir.exists():
                print(f"  Skipping {doc_type}/{split} — directory not found")
                continue

            # Create output directory
            out_split_dir = output_dir / split
            out_split_dir.mkdir(parents=True, exist_ok=True)

            # Find all document files
            doc_files = sorted(
                p for p in split_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".pdf", ".docx")
            )

            for doc_file in doc_files:
                try:
                    ext_dest = out_split_dir / f"{doc_file.stem}_extracted.json"

                    # Skip if already extracted (resume support)
                    if ext_dest.exists() and method == "llm":
                        total_docs += 1
                        continue

                    # Ingest
                    doc = load_document(doc_file)

                    # Extract
                    if method == "llm":
                        extracted = llm_extract_fields(doc)
                        # Rate-limit delay between LLM calls (free tier)
                        time.sleep(1.5)
                    else:
                        extracted = rule_based_extract(doc)

                    # Copy ground truth to processed dir
                    gt_source = split_dir / f"{doc_file.stem}_ground_truth.json"
                    if gt_source.exists():
                        gt_dest = out_split_dir / f"{doc_file.stem}_ground_truth.json"
                        shutil.copy2(gt_source, gt_dest)

                    # Save extracted data
                    with open(ext_dest, "w") as f:
                        json.dump(extracted, f, indent=2, default=str)

                    total_docs += 1
                    if total_docs % 10 == 0:
                        print(f"  ... processed {total_docs} docs so far", flush=True)
                except Exception as e:
                    print(f"  Error processing {doc_file.name}: {e}")
                    total_errors += 1

            count = len(doc_files)
            skipped = count - sum(1 for p in doc_files if not (out_split_dir / f'{p.stem}_extracted.json').exists())
            print(f"  {doc_type}/{split}: {count} total, {skipped} skipped (already done), {count - skipped} extracted")

    print(f"\n[OK] Total: {total_docs} documents processed, {total_errors} errors")

    # Return LLM metrics if available
    if method == "llm":
        from src.extraction.llm_extractor import get_extraction_metrics
        metrics = get_extraction_metrics()
        llm_metrics = metrics.summary()
        print(f"\n[LLM Metrics]")
        print(f"  API Calls: {llm_metrics['total_calls']}")
        print(f"  Total Cost: ${llm_metrics['total_cost_usd']:.4f}")
        print(f"  Avg Latency: {llm_metrics['avg_latency_ms']:.0f}ms")

    return {"llm_metrics": llm_metrics}


def step3_evaluate(processed_dir: Path) -> dict:
    """Step 3: Evaluate extraction quality."""
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)

    summary = evaluate_dataset(processed_dir)
    if summary:
        report = print_report(summary)

        # Save summary
        summary_path = processed_dir / "baseline_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n[OK] Metrics saved to {summary_path}")
    else:
        print("  No evaluation results found")

    return summary


def step4_validate(
    processed_dir: Path,
    rules_config: Path,
) -> dict:
    """Step 4: Run compliance rule validation."""
    print("\n" + "=" * 60)
    print("STEP 4: Compliance Rule Validation")
    print("=" * 60)

    from src.validation.rule_engine import ComplianceRuleEngine
    from src.validation.cross_document import CrossDocumentChecker
    from src.validation.report_generator import (
        generate_compliance_report,
        save_report_json,
        save_report_markdown,
    )

    # Load rule engine
    engine = ComplianceRuleEngine(rules_config)

    # Load all extracted documents
    documents = []
    for extracted_file in sorted(processed_dir.rglob("*_extracted.json")):
        with open(extracted_file) as f:
            extracted_data = json.load(f)

        # Get doc_type from extracted data or ground truth
        doc_type = extracted_data.get("doc_type", "unknown")
        if doc_type == "unknown":
            # Try to get from ground truth
            gt_file = extracted_file.parent / extracted_file.name.replace("_extracted.json", "_ground_truth.json")
            if gt_file.exists():
                with open(gt_file) as f:
                    gt_data = json.load(f)
                doc_type = gt_data.get("doc_type", "unknown")

        doc_id = extracted_file.stem.replace("_extracted", "")
        documents.append({
            "document_id": doc_id,
            "doc_type": doc_type,
            "extracted_data": extracted_data,
        })

    print(f"  Loaded {len(documents)} documents for validation")

    # Run rule engine validation
    validation_results = []
    for doc in documents:
        result = engine.validate_document(
            document_id=doc["document_id"],
            doc_type=doc["doc_type"],
            extracted_data=doc["extracted_data"],
        )
        validation_results.append(result)

    # Run cross-document checks
    checker = CrossDocumentChecker()
    cross_violations = checker.check_batch(documents)

    # Merge cross-document violations into results
    for result, cross_viol in zip(validation_results, cross_violations):
        result.violations.extend(cross_viol)

    # Generate report
    report = generate_compliance_report(validation_results)

    # Save reports
    json_path = processed_dir / "compliance_report.json"
    md_path = processed_dir / "compliance_report.md"

    save_report_json(report, json_path)
    save_report_markdown(report, md_path)

    print(f"\n[OK] Validation complete")
    print(f"  Valid documents: {report.valid_documents}/{report.total_documents}")
    print(f"  Compliance rate: {report.valid_documents / max(report.total_documents, 1) * 100:.1f}%")
    print(f"  Critical violations: {report.critical_violations}")
    print(f"  Warning violations: {report.warning_violations}")
    print(f"  Info violations: {report.info_violations}")
    print(f"\n[OK] Reports saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    return report.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Compliance Document Intelligence Pipeline")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation")
    parser.add_argument("--num-per-type", type=int, default=100, help="Documents per type")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Synthetic data dir")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Output dir")
    parser.add_argument("--method", choices=["rule_based", "llm"], default="rule_based", help="Extraction method")
    parser.add_argument("--validate", action="store_true", help="Run compliance validation (Phase 2)")
    args = parser.parse_args()

    base = Path(".")
    synthetic_dir = base / args.synthetic_dir
    processed_dir = base / args.processed_dir
    rules_config = base / "config" / "compliance_rules.yaml"

    phase = "Phase 2" if args.method == "llm" or args.validate else "Phase 1"
    print("+=" * 30)
    print(f"|  Compliance Document Intelligence — {phase} Pipeline  |")
    print("+=" * 30)

    # Step 1: Generate
    if not args.skip_generation:
        step1_generate_data(synthetic_dir, num_per_type=args.num_per_type)
    else:
        print("\n[Skipping data generation]")

    # Step 2: Ingest + Extract
    extraction_info = step2_ingest_and_extract(
        synthetic_dir,
        processed_dir,
        method=args.method,
        validate=args.validate,
    )

    # Step 3: Evaluate
    summary = step3_evaluate(processed_dir)

    # Step 4: Validate (Phase 2 only)
    if args.validate:
        validation_report = step4_validate(processed_dir, rules_config)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    if summary:
        overall_f1 = sum(m["avg_f1"] for m in summary.values()) / len(summary) if summary else 0
        print(f"Overall F1: {overall_f1:.4f}")

        if args.method == "llm":
            print("\nPhase 2 LLM-based extraction complete.")
            print("Compare this F1 to the Phase 1 rule-based baseline.")
        else:
            print("\nPhase 1 rule-based baseline complete.")
            print("Use --method llm --validate for Phase 2.")


if __name__ == "__main__":
    main()
