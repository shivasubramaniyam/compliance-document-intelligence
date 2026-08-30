"""
Synthetic Compliance Document Generator

Generates realistic compliance documents (SOP, Adverse Event Reports, Audit Finding Reports)
as PDF and DOCX files with paired ground-truth JSON files.

Produces both "clean" documents (all required fields present) and "intentionally flawed"
documents (missing fields, malformed dates, inconsistent IDs) for downstream evaluation.
"""
from __future__ import annotations

import json
import os
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _random_date(start_year: int = 2020, end_year: int = 2025) -> date:
    import random as _rng
    year = _rng.randint(start_year, end_year)
    month = _rng.randint(1, 12)
    day = _rng.randint(1, 28)  # safe day for any month
    return date(year, month, day)


def _random_id(prefix: str, year: int | None = None) -> str:
    if year is None:
        year = random.randint(2023, 2025)
    seq = random.randint(1, 9999)
    return f"{prefix}-{year}-{seq:05d}"


def _random_severity() -> str:
    return random.choice(["critical", "high", "medium", "low"])


def _random_review_cycle() -> str:
    return random.choice(["Annual", "Semi-annual", "Quarterly", "Biennial"])


def _malform_date(d: date) -> str:
    """Return a deliberately malformed date string."""
    fmt = random.choice([
        lambda d: f"{d.month}/{d.day}/{d.year}",       # M/D/YYYY
        lambda d: f"{d.year}-{d.month:02d}",            # YYYY-MM (missing day)
        lambda d: f"{d.day}.{d.month}.{d.year}",        # DD.MM.YYYY
        lambda d: "not-a-date",                         # completely invalid
        lambda d: f"{d.year}/13/45",                     # impossible values
    ])
    return fmt(d)


# ---------------------------------------------------------------------------
# SOP generator
# ---------------------------------------------------------------------------

def generate_sop_data() -> dict[str, Any]:
    dept = random.choice([
        "Quality Assurance", "Regulatory Affairs", "Clinical Operations",
        "Manufacturing", "Pharmacovigilance", "Research & Development",
    ])
    owner = fake.name()
    eff = _random_date(2023, 2025)
    return {
        "title": random.choice([
            f"Procedure for {random.choice(['Batch Record Review', 'Change Control', 'Deviation Management', 'CAPA Handling', 'Supplier Qualification', 'Document Control'])}",
            f"SOP-{random.randint(100,999)}: {random.choice(['Incoming Material Inspection', 'Out-of-Specification Investigation', 'Stability Testing', 'Cleaning Validation', 'Environmental Monitoring'])}",
        ]),
        "version": f"{random.randint(1,5)}.{random.randint(0,9)}",
        "effective_date": eff.isoformat(),
        "owner": owner,
        "department": dept,
        "approval_date": (eff + timedelta(days=random.randint(1, 30))).isoformat(),
        "review_cycle": _random_review_cycle(),
        "scope": f"This procedure applies to all {dept.lower()} activities related to GMP compliance.",
        "references": random.sample([
            "ICH Q10", "21 CFR Part 211", "ISO 13485:2016",
            "EU GMP Annex 15", "FDA Guidance for Industry",
        ], k=random.randint(1, 3)),
    }


def render_sop_docx(data: dict[str, Any], output_path: Path) -> None:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Title
    title_para = doc.add_heading(data["title"], level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Header table
    table = doc.add_table(rows=4, cols=2, style="Table Grid")
    headers = [
        ("Version", data.get("version", "N/A")),
        ("Effective Date", data.get("effective_date", "N/A")),
        ("Owner", data.get("owner", "N/A")),
        ("Department", data.get("department", "N/A")),
    ]
    for i, (label, value) in enumerate(headers):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = value
        for cell in table.rows[i].cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)

    doc.add_paragraph("")

    # Approval info
    doc.add_heading("Approval", level=1)
    doc.add_paragraph(f"Approval Date: {data.get('approval_date', 'N/A')}")
    doc.add_paragraph(f"Review Cycle: {data.get('review_cycle', 'N/A')}")

    # Scope
    if data.get("scope"):
        doc.add_heading("Scope", level=1)
        doc.add_paragraph(data.get('scope', ''))

    # References
    if data.get("references"):
        doc.add_heading("References", level=1)
        for ref in data.get('references', []):
            doc.add_paragraph(ref, style="List Bullet")

    # Body content (template text)
    doc.add_heading("Procedure", level=1)
    for _ in range(random.randint(3, 6)):
        doc.add_paragraph(fake.paragraph(nb_sentences=random.randint(3, 8)))

    doc.add_heading("Responsibilities", level=1)
    dept = data.get('department', 'the department')
    owner = data.get('owner', 'the department head')
    doc.add_paragraph(
        f"The {dept} department head ({owner}) is responsible "
        f"for ensuring adherence to this SOP."
    )

    doc.save(str(output_path))


# ---------------------------------------------------------------------------
# Adverse Event Report generator
# ---------------------------------------------------------------------------

def generate_adverse_event_data() -> dict[str, Any]:
    severity = random.choice(["critical", "high", "medium", "low"])
    return {
        "event_id": _random_id("AE"),
        "date_reported": _random_date(2023, 2025).isoformat(),
        "product_id": f"PROD-{random.choice(string.ascii_uppercase)}{random.randint(1000,9999)}",
        "batch_id": f"BATCH-{random.randint(100000,999999)}",
        "severity": severity,
        "description": random.choice([
            f"Patient experienced {random.choice(['nausea', 'dizziness', 'rash', 'headache', 'fatigue', 'shortness of breath'])} "
            f"within {random.randint(1, 72)} hours of administration of product batch.",
            f"Manufacturing anomaly detected: {random.choice(['foreign particle', 'temperature excursion', 'fill volume deviation', 'packaging defect'])} "
            f"in {random.choice(['visual inspection', 'stability testing', 'routine QC'])}.",
            f"Unexpected {random.choice(['interaction', 'contraindication', 'adverse reaction'])} reported during "
            f"{random.choice(['Phase III clinical trial', 'post-market surveillance', 'compassionate use program'])}.",
        ]),
        "reporter": fake.name(),
        "follow_up_status": random.choice(["open", "in_progress", "closed", "pending_review"]),
    }


def render_adverse_event_docx(data: dict[str, Any], output_path: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()

    doc.add_heading("Adverse Event Report", level=0)

    # Key-value pairs as a table
    table = doc.add_table(rows=8, cols=2, style="Table Grid")
    sev = str(data.get('severity', 'N/A')).upper()
    fus = str(data.get('follow_up_status', 'N/A')).replace('_', ' ').title()
    fields = [
        ("Event ID", data.get('event_id', 'N/A')),
        ("Date Reported", data.get('date_reported', 'N/A')),
        ("Product ID", data.get('product_id', 'N/A')),
        ("Batch ID", data.get('batch_id', 'N/A')),
        ("Severity", sev),
        ("Reporter", data.get('reporter', 'N/A')),
        ("Follow-Up Status", fus),
    ]
    for i, (label, value) in enumerate(fields):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = value
        for cell in table.rows[i].cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    table.rows[7].cells[0].text = "Description"
    table.rows[7].cells[1].text = ""

    doc.add_paragraph("")

    doc.add_heading("Description of Event", level=1)
    doc.add_paragraph(data.get('description', 'No description provided.'))

    doc.add_heading("Narrative", level=1)
    for _ in range(random.randint(2, 4)):
        doc.add_paragraph(fake.paragraph(nb_sentences=random.randint(2, 5)))

    doc.save(str(output_path))


# ---------------------------------------------------------------------------
# Audit Finding Report generator
# ---------------------------------------------------------------------------

def generate_audit_finding_data() -> dict[str, Any]:
    severity = random.choice(["critical", "high", "medium", "low"])
    audit_date = _random_date(2023, 2025)
    due = audit_date + timedelta(days=random.randint(15, 180))
    category = random.choice([
        "Documentation", "Facility", "Process", "Personnel Training",
        "Equipment Maintenance", "Data Integrity", "Supplier Management",
        "Change Control", "CAPA Effectiveness",
    ])
    return {
        "audit_id": _random_id("AUD"),
        "audit_date": audit_date.isoformat(),
        "finding_category": category,
        "severity": severity,
        "description": random.choice([
            f"{category} deficiency: Records reviewed during the audit showed that {fake.sentence()}",
            f"Non-conformance in {category.lower()} area: {fake.sentence()}",
            f"Regulatory gap identified in {category.lower()}: {fake.sentence()}",
        ]),
        "corrective_action": random.choice([
            f"Implement revised {category.lower()} procedures and retrain all affected personnel.",
            f"Update {category.lower()} documentation and conduct a follow-up audit within 90 days.",
            f"Deploy corrective measures including {fake.sentence().lower()}",
        ]),
        "due_date": due.isoformat(),
        "responsible_party": fake.name(),
    }


def render_audit_finding_docx(data: dict[str, Any], output_path: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()

    doc.add_heading("Audit Finding Report", level=0)

    table = doc.add_table(rows=7, cols=2, style="Table Grid")
    sev = str(data.get('severity', 'N/A')).upper()
    fields = [
        ("Audit ID", data.get('audit_id', 'N/A')),
        ("Audit Date", data.get('audit_date', 'N/A')),
        ("Finding Category", data.get('finding_category', 'N/A')),
        ("Severity", sev),
        ("Due Date", data.get('due_date', 'N/A')),
        ("Responsible Party", data.get('responsible_party', 'N/A')),
        ("Status", "Open"),
    ]
    for i, (label, value) in enumerate(fields):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = value
        for cell in table.rows[i].cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)

    doc.add_paragraph("")

    doc.add_heading("Finding Description", level=1)
    doc.add_paragraph(data.get('description', 'No description provided.'))

    doc.add_heading("Corrective Action Required", level=1)
    doc.add_paragraph(data.get('corrective_action', 'No corrective action specified.'))

    doc.add_heading("Evidence and Root Cause", level=1)
    doc.add_paragraph(fake.paragraph(nb_sentences=random.randint(2, 4)))

    doc.save(str(output_path))


# ---------------------------------------------------------------------------
# Intentionally flawed variants
# ---------------------------------------------------------------------------

def make_flawed_sop(clean: dict[str, Any]) -> dict[str, Any]:
    """Create a flawed version of an SOP by randomly corrupting fields."""
    flawed = clean.copy()
    flaw = random.choice([
        "missing_version", "bad_date", "missing_owner", "truncated_title",
    ])
    if flaw == "missing_version":
        del flawed["version"]
    elif flaw == "bad_date":
        flawed["effective_date"] = _malform_date(date.fromisoformat(clean["effective_date"]))
    elif flaw == "missing_owner":
        del flawed["owner"]
    elif flaw == "truncated_title":
        flawed["title"] = flawed["title"][: len(flawed["title"]) // 3]
    return flawed


def make_flawed_adverse_event(clean: dict[str, Any]) -> dict[str, Any]:
    flawed = clean.copy()
    flaw = random.choice([
        "missing_event_id", "bad_severity", "missing_reporter", "bad_date",
    ])
    if flaw == "missing_event_id":
        del flawed["event_id"]
    elif flaw == "bad_severity":
        flawed["severity"] = "banana"
    elif flaw == "missing_reporter":
        del flawed["reporter"]
    elif flaw == "bad_date":
        flawed["date_reported"] = "not-a-date"
    return flawed


def make_flawed_audit_finding(clean: dict[str, Any]) -> dict[str, Any]:
    flawed = clean.copy()
    flaw = random.choice([
        "missing_audit_id", "bad_severity", "missing_due_date", "missing_corrective_action",
    ])
    if flaw == "missing_audit_id":
        del flawed["audit_id"]
    elif flaw == "bad_severity":
        flawed["severity"] = "invalid"
    elif flaw == "missing_due_date":
        del flawed["due_date"]
    elif flaw == "missing_corrective_action":
        del flawed["corrective_action"]
    return flawed


FLAWED_MAKERS = {
    "sop": make_flawed_sop,
    "adverse_event": make_flawed_adverse_event,
    "audit_finding": make_flawed_audit_finding,
}

RENDERERS = {
    "sop": (generate_sop_data, render_sop_docx),
    "adverse_event": (generate_adverse_event_data, render_adverse_event_docx),
    "audit_finding": (generate_audit_finding_data, render_audit_finding_docx),
}


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_dataset(
    base_dir: Path,
    num_per_type: int = 100,
    flawed_ratio: float = 0.25,
    train_split: float = 0.8,
) -> None:
    """
    Generate the full synthetic dataset.

    Args:
        base_dir: Root of data/synthetic/
        num_per_type: Number of documents per type
        flawed_ratio: Fraction that are intentionally flawed
        train_split: Fraction going to train/
    """
    rng = random.Random(42)

    for doc_type, (gen_fn, render_fn) in RENDERERS.items():
        type_dir = base_dir / doc_type
        train_dir = type_dir / "train"
        test_dir = type_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        num_flawed = int(num_per_type * flawed_ratio)
        num_clean = num_per_type - num_flawed
        all_records: list[dict[str, Any]] = []

        # --- Clean documents ---
        for i in range(num_clean):
            doc_id = f"{doc_type}_{i:04d}"
            data = gen_fn()
            # Save ground truth
            all_records.append({
                "doc_id": doc_id,
                "is_flawed": False,
                "data": data,
            })

        # --- Flawed documents ---
        for i in range(num_flawed):
            doc_id = f"{doc_type}_{num_clean + i:04d}"
            clean_data = gen_fn()
            flawed_data = FLAWED_MAKERS[doc_type](clean_data)
            all_records.append({
                "doc_id": doc_id,
                "is_flawed": True,
                "data": flawed_data,
                "clean_data": clean_data,  # kept for reference
            })

        # Shuffle and split
        rng.shuffle(all_records)
        split_idx = int(len(all_records) * train_split)
        splits = {
            "train": all_records[:split_idx],
            "test": all_records[split_idx:],
        }

        for split_name, records in splits.items():
            out_dir = train_dir if split_name == "train" else test_dir
            for rec in records:
                doc_id = rec["doc_id"]
                data = rec["data"]

                # Write ground truth JSON (only the clean data fields)
                gt = rec.get("clean_data", data)
                gt_path = out_dir / f"{doc_id}_ground_truth.json"
                with open(gt_path, "w") as f:
                    json.dump(gt, f, indent=2, default=str)

                # Render document
                fmt = random.choice(["docx"])  # Use docx for fast generation
                if fmt == "docx":
                    doc_path = out_dir / f"{doc_id}.docx"
                    render_fn(data, doc_path)
                else:
                    doc_path = out_dir / f"{doc_id}.pdf"
                    render_fn(data, doc_path)  # for now, render as docx still

                # Write metadata
                meta = {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "is_flawed": rec["is_flawed"],
                    "format": fmt,
                    "file_path": str(doc_path.name),
                }
                meta_path = out_dir / f"{doc_id}_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

            print(f"  Generated {len(records)} {split_name} docs for {doc_type}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic compliance documents")
    parser.add_argument("--num-per-type", type=int, default=100, help="Documents per type")
    parser.add_argument("--flawed-ratio", type=float, default=0.25, help="Fraction flawed")
    parser.add_argument("--base-dir", type=str, default="data/synthetic", help="Output base dir")
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic documents in {base}/")
    generate_dataset(base, num_per_type=args.num_per_type, flawed_ratio=args.flawed_ratio)
    print("Done!")
