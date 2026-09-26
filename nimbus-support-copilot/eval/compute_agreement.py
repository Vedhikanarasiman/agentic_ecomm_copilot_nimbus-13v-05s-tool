"""
Compares your filled-in human labels against the judge's labels from the
same run, and reports exact agreement — this is the number to actually
say in an interview ("judge/human agreement was X% on N samples"), not
just "we used an LLM judge."

Usage:
    python eval/compute_agreement.py eval/results/run_<timestamp>.json \\
        eval/results/run_<timestamp>_human_review.csv
"""

import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python eval/compute_agreement.py <run.json> <human_review.csv>")
        sys.exit(1)

    run_path, csv_path = Path(sys.argv[1]), Path(sys.argv[2])

    with open(run_path) as f:
        results = {r["id"]: r for r in json.load(f)}

    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    missing_labels = [r["id"] for r in rows if not r["human_label"].strip()]
    if missing_labels:
        print(f"Warning: {len(missing_labels)} rows have no human_label filled in yet: {missing_labels}")
        print("Fill those in before computing agreement, or the number will be misleading.")

    labeled_rows = [r for r in rows if r["human_label"].strip()]
    if not labeled_rows:
        print("No labeled rows found — nothing to compute.")
        sys.exit(1)

    agree = 0
    disagreements = []
    for row in labeled_rows:
        case_id = row["id"]
        judge_label = results[case_id]["groundedness"]["label"]
        human_label = row["human_label"].strip()
        if judge_label == human_label:
            agree += 1
        else:
            disagreements.append((case_id, judge_label, human_label))

    total = len(labeled_rows)
    agreement_pct = agree / total * 100

    print(f"\nJudge/human agreement: {agreement_pct:.1f}% ({agree}/{total})\n")
    if disagreements:
        print("Disagreements:")
        for case_id, judge_label, human_label in disagreements:
            print(f"  {case_id}: judge said '{judge_label}', human said '{human_label}'")
    else:
        print("No disagreements on this sample.")


if __name__ == "__main__":
    main()
