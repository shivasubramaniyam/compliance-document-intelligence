"""
Compliance Document Intelligence — End-to-End Pipeline (Phase 1)

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-generation  # Skip data generation if already exists
    python run_pipeline.py --num-per-type 50  # Customize document count
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.ingestion.loader import load_document
from src.extraction.extractor import extract_and_validate
from src.utils.generate_synthetic_docs import generate_dataset
from src.utils.evaluate_extraction import evaluate_dataset, print_report


def step1_generate_data(base_dir: Path, num_per_type: int = 100) -> None:
    """Step 1: Generate synthetic compliance documents."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Documents")
    print("=" * 60)
    generate_dataset(base_dir, num_per_type=num_per_type, flawed_ratio=0.25)
    print("[OK] Synthetic documents generated")


def step2_ingest_and_extract(synthetic_dir: Path, output_dir: Path) -> None:
    """Step 2: Ingest documents and run baseline extraction."""
    print("\n" + "=" * 60)
    print("STEP 2: Ingestion & Extraction")
    print("=" * 60)

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
                    # Ingest
                    doc = load_document(doc_file)

                    # Extract
                    extracted = extract_and_validate(doc)

                    # Copy ground truth to processed dir
                    gt_source = split_dir / f"{doc_file.stem}_ground_truth.json"
                    if gt_source.exists():
                        gt_dest = out_split_dir / f"{doc_file.stem}_ground_truth.json"
                        shutil.copy2(gt_source, gt_dest)

                    # Save extracted data
                    ext_dest = out_split_dir / f"{doc_file.stem}_extracted.json"
                    with open(ext_dest, "w") as f:
                        json.dump(extracted, f, indent=2, default=str)

                    total_docs += 1
                except Exception as e:
                    print(f"  Error processing {doc_file.name}: {e}")
                    total_errors += 1

            count = len(doc_files)
            print(f"  {doc_type}/{split}: {count} documents processed")

    print(f"\n[OK] Total: {total_docs} documents processed, {total_errors} errors")


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


def main():
    parser = argparse.ArgumentParser(description="Compliance Document Intelligence Pipeline")
    parser.add_argument("--skip-generation", action="store_true", help="Skip data generation")
    parser.add_argument("--num-per-type", type=int, default=100, help="Documents per type")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Synthetic data dir")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Output dir")
    args = parser.parse_args()

    base = Path(".")
    synthetic_dir = base / args.synthetic_dir
    processed_dir = base / args.processed_dir

    print("+=" * 30)
    print("|  Compliance Document Intelligence -- Phase 1 Pipeline  |")
    print("+=" * 30)

    # Step 1: Generate
    if not args.skip_generation:
        step1_generate_data(synthetic_dir, num_per_type=args.num_per_type)
    else:
        print("\n[Skipping data generation]")

    # Step 2: Ingest + Extract
    step2_ingest_and_extract(synthetic_dir, processed_dir)

    # Step 3: Evaluate
    summary = step3_evaluate(processed_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if summary:
        overall_f1 = sum(m["avg_f1"] for m in summary.values()) / len(summary) if summary else 0
        print(f"Overall baseline F1: {overall_f1:.4f}")
        print("\nThis is the Phase 1 baseline. Phase 2 will improve this with:")
        print("  - Fine-tuned NER / LLM-based extraction")
        print("  - Compliance rule validation")
        print("  - Anomaly detection for flawed documents")


if __name__ == "__main__":
    main()
