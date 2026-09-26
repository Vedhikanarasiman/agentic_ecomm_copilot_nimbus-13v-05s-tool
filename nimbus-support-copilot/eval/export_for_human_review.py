"""
Samples a subset of RAG-path results from a completed eval run and writes
them to a CSV for manual labeling — deliberately WITHOUT the judge's label
visible, so your human label isn't anchored by what the judge already said.

Usage:
    python eval/export_for_human_review.py eval/results/run_<timestamp>.json
"""

import csv
import json
import random
import sys
from pathlib import Path

SAMPLE_SIZE = 15


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval/export_for_human_review.py <path-to-run-results.json>")
        sys.exit(1)

    run_path = Path(sys.argv[1])
    with open(run_path) as f:
        results = json.load(f)

    candidates = [r for r in results if r.get("groundedness")]
    random.seed(42)  # reproducible sample
    sample = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    out_path = run_path.parent / f"{run_path.stem}_human_review.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "query", "answer", "sources", "human_label"])
        for r in sample:
            writer.writerow(
                [
                    r["id"],
                    r["query"],
                    r["answer"],
                    "; ".join(r.get("sources", [])),
                    "",  # fill with: grounded / partially_grounded / not_grounded
                ]
            )

    print(f"Wrote {len(sample)} cases to {out_path}")
    print("Open it, read each answer against its sources, and fill in human_label")
    print("with exactly one of: grounded / partially_grounded / not_grounded")
    print("Do this BEFORE looking at the judge's labels in the run JSON — that's")
    print("what makes the agreement number meaningful instead of circular.")


if __name__ == "__main__":
    main()
