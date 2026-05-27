#!/usr/bin/env python3
"""
Normalize RexVal CSV schema for ICARE and optionally validate label join keys.

This script keeps RexVal join keys (`study_number`, `origin`) intact while
renaming report columns to ICARE-compatible names.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_INPUT_COLUMNS = [
    "study_number",
    "origin",
    "generated reports",
    "ground truth reports",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RexVal 200-pair CSV for ICARE execution."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to RexVal_test.csv (long-format, 200 pairs).",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to write ICARE-ready normalized CSV.",
    )
    parser.add_argument(
        "--key_counts_csv",
        required=False,
        default=None,
        help=(
            "Optional output path for per-key counts over "
            "(study_number, origin)."
        ),
    )
    parser.add_argument(
        "--labels_csv",
        required=False,
        default=None,
        help=(
            "Optional PhysioNet labels CSV "
            "(6_valid_raters_per_rater_error_categories.csv) for key validation."
        ),
    )
    parser.add_argument(
        "--validation_output_dir",
        required=False,
        default=None,
        help="Optional output directory for key validation artifacts.",
    )
    return parser.parse_args()


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def run_optional_key_validation(
    normalized: pd.DataFrame, labels_csv: str, validation_output_dir: str
) -> None:
    labels_df = pd.read_csv(labels_csv)
    required_label_columns = {"study_number", "candidate_type"}
    missing_label_cols = required_label_columns - set(labels_df.columns)
    if missing_label_cols:
        raise ValueError(
            f"Missing required label columns: {sorted(missing_label_cols)}. "
            f"Found columns: {list(labels_df.columns)}"
        )

    icare_keys = (
        normalized[["study_number", "origin"]]
        .drop_duplicates()
        .rename(columns={"origin": "candidate_type"})
        .sort_values(["study_number", "candidate_type"])
        .reset_index(drop=True)
    )
    label_keys = (
        labels_df[["study_number", "candidate_type"]]
        .drop_duplicates()
        .sort_values(["study_number", "candidate_type"])
        .reset_index(drop=True)
    )

    only_in_icare = icare_keys.merge(
        label_keys, on=["study_number", "candidate_type"], how="left", indicator=True
    )
    only_in_icare = only_in_icare[only_in_icare["_merge"] == "left_only"].drop(
        columns="_merge"
    )

    only_in_labels = label_keys.merge(
        icare_keys, on=["study_number", "candidate_type"], how="left", indicator=True
    )
    only_in_labels = only_in_labels[only_in_labels["_merge"] == "left_only"].drop(
        columns="_merge"
    )

    key_counts = (
        normalized.groupby(["study_number", "origin"], dropna=False)
        .size()
        .reset_index(name="icare_row_count")
        .sort_values(["study_number", "origin"])
    )
    summary = pd.DataFrame(
        [
            {
                "icare_unique_keys": len(icare_keys),
                "labels_unique_keys": len(label_keys),
                "keys_only_in_icare": len(only_in_icare),
                "keys_only_in_labels": len(only_in_labels),
                "exact_key_match": len(only_in_icare) == 0
                and len(only_in_labels) == 0,
            }
        ]
    )

    out_dir = Path(validation_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_counts.to_csv(out_dir / "icare_input_key_counts.csv", index=False)
    icare_keys.to_csv(out_dir / "icare_unique_keys.csv", index=False)
    label_keys.to_csv(out_dir / "labels_unique_keys.csv", index=False)
    only_in_icare.to_csv(out_dir / "keys_only_in_icare.csv", index=False)
    only_in_labels.to_csv(out_dir / "keys_only_in_labels.csv", index=False)
    summary.to_csv(out_dir / "key_validation_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved key validation artifacts to: {out_dir}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    key_counts_path = Path(args.key_counts_csv) if args.key_counts_csv else None

    df = pd.read_csv(input_path)
    validate_input_columns(df)

    # Rename only the report text columns expected by ICARE.
    normalized = df.rename(
        columns={
            "generated reports": "generated_report",
            "ground truth reports": "ground_truth_report",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(normalized)}")
    print(f"Saved normalized CSV to: {output_path}")

    if key_counts_path is not None:
        key_counts = (
            normalized.groupby(["study_number", "origin"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["study_number", "origin"])
        )
        key_counts_path.parent.mkdir(parents=True, exist_ok=True)
        key_counts.to_csv(key_counts_path, index=False)
        print(f"Saved key counts CSV to: {key_counts_path}")

    if args.labels_csv:
        validation_out_dir = (
            args.validation_output_dir
            if args.validation_output_dir
            else str(output_path.parent / "key_validation")
        )
        run_optional_key_validation(
            normalized=normalized,
            labels_csv=args.labels_csv,
            validation_output_dir=validation_out_dir,
        )


if __name__ == "__main__":
    main()
