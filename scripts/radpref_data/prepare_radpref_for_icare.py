#!/usr/bin/env python3
"""
Convert RadPref preference_data.json into an ICARE-ready CSV.

Each case has two candidate reports (C1, C2); we produce one row per
candidate (200 rows total: 100 C1 rows followed by 100 C2 rows).

Output columns:
  case_id             – original RadPref case identifier
  candidate           – "C1" or "C2"
  candidate_source    – model that generated the candidate
  generated_report    – candidate report text   (ICARE input)
  ground_truth_report – reference report text   (ICARE input)
  PatientAge, PatientSex, Indication, Comparison, StudyDescription
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RadPref JSON for ICARE execution."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to preference_data.json",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to write ICARE-ready CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)

    with open(input_path) as f:
        data = json.load(f)

    # C1 rows first (indices 0..N-1), then C2 rows (indices N..2N-1).
    # ICARE's mcqa_evaluation.py uses the integer row index as Report_ID,
    # so keeping candidates separated makes post-hoc splitting trivial.
    rows = []
    for cand in ("C1", "C2"):
        for sample in data:
            rows.append(
                {
                    "case_id": sample["id"],
                    "candidate": cand,
                    "candidate_source": sample[f"{cand}_source"],
                    "generated_report": sample[f"{cand}_report"],
                    "ground_truth_report": sample["ground_truth_report"],
                    "PatientAge": sample.get("PatientAge", ""),
                    "PatientSex": sample.get("PatientSex", ""),
                    "Indication": sample.get("Indication", ""),
                    "Comparison": sample.get("Comparison", ""),
                    "StudyDescription": sample.get("StudyDescription", ""),
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Input cases:  {len(data)}")
    print(f"Output rows:  {len(df)}  (C1 rows 0–{len(data)-1}, C2 rows {len(data)}–{len(df)-1})")
    print(f"Saved to:     {output_path}")

    print("\nCandidate / source breakdown:")
    print(df.groupby(["candidate", "candidate_source"]).size().to_string())


if __name__ == "__main__":
    main()
