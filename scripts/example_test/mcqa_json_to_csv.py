"""
Convert mcqa_data.json to all_questions.csv for MCQ evaluation.

Same columns used by mcqa_evaluation.predict_answers_for_mcq_data as
filtered_questions_shuffled.csv (without filtering-specific columns).

Usage:
    python scripts/example_test/mcqa_json_to_csv.py \
        --input-json path/to/mcqa_data.json \
        --output-csv path/to/mcqa_eval_input/all_questions.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def json_to_csv(input_json, output_csv):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    idx = 0
    for report_id, report_data in enumerate(data["mcq_data"]):
        for question in report_data.get("questions", []):
            options = question["options"]
            rows.append({
                "Index": idx,
                "Report_ID": report_id,
                "Question_ID": question["question_id"],
                "Question_Text": question["question_text"],
                "Options": str(options),
                "Correct_Answer": question["correct_answer"],
            })
            idx += 1

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} questions to {out}")


def main():
    parser = argparse.ArgumentParser(description="Convert mcqa_data.json to all_questions.csv")
    parser.add_argument("--input-json", required=True, help="Path to mcqa_data.json")
    parser.add_argument("--output-csv", required=True, help="Path to output all_questions.csv")
    args = parser.parse_args()
    json_to_csv(args.input_json, args.output_csv)


if __name__ == "__main__":
    main()
