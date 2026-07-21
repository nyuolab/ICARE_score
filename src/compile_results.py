"""
Compile per-sample results from the ICARE pipeline into a single JSON and summary CSV.

Reads outputs under shuffled_ans_choices_data/ (same layout as run_eval_final_without_orig.sh).

For each sample (report), the JSON contains:
  - gt_report / gen_report text
  - All GT-reference and Gen-reference questions (pre-filtering)
  - Filtered GT and Gen questions (report-dependent only), with counts
  - Eval predictions for each filtered question
  - Agreeing / disagreeing question lists (GT ref and Gen ref separately)
  - Disagreement scores: omission (GT-ref) and hallucination (Gen-ref)

The summary CSV contains one row per sample:
  sample_id, study_id, gt_agreement_pct, gen_agreement_pct,
  total_gt_questions, total_gen_questions

Usage:
    python src/compile_results.py \\
        --base_dir  <pipeline output dir>  \\
        --input_csv <original reports CSV> \\
        --output    <path to write JSON> \\
        --summary_csv <path to write summary CSV> \\
        --timing_file <optional pipeline_timing.json>

    # Example:
    python src/compile_results.py \\
        --base_dir  test_data/output \\
        --input_csv test_data/sample_iuxray_reports.csv \\
        --output    test_data/output/icare_results.json \\
        --summary_csv test_data/output/icare_results_summary.csv \\
        --timing_file test_data/output/pipeline_timing.json
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mcqa_json(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)["mcq_data"]


def _load_filtered(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Options column may be stored as a string repr of a dict
    if "Options" in df.columns:
        df["Options"] = df["Options"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df


def _load_eval_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Options" in df.columns:
        df["Options"] = df["Options"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df


def _question_row_to_dict(row: pd.Series, extra_keys: Optional[List[str]] = None) -> dict:
    base = {
        "question_id":   int(row["Question_ID"]),
        "question_text": row["Question_Text"],
        "options":       row["Options"],
        "correct_answer": row["Correct_Answer"],
    }
    if extra_keys:
        for k in extra_keys:
            if k in row:
                base[k] = row[k]
    return base


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_sample_entry(
    sample_idx: int,
    gt_report: str,
    gen_report: str,
    gt_all_questions: List[dict],
    gen_all_questions: List[dict],
    gt_filtered_df: pd.DataFrame,
    gen_filtered_df: pd.DataFrame,
    gt_eval_df: pd.DataFrame,
    gen_eval_df: pd.DataFrame,
) -> dict:

    # ---- filtered questions ------------------------------------------------
    gt_filt = gt_filtered_df[gt_filtered_df["Report_ID"] == sample_idx]
    gen_filt = gen_filtered_df[gen_filtered_df["Report_ID"] == sample_idx]

    gt_filtered_list = [
        _question_row_to_dict(row, ["Predicted_Answer_with_report", "Predicted_Answer_without_report"])
        for _, row in gt_filt.iterrows()
    ]
    gen_filtered_list = [
        _question_row_to_dict(row, ["Predicted_Answer_with_report", "Predicted_Answer_without_report"])
        for _, row in gen_filt.iterrows()
    ]

    # ---- eval predictions (agreeing / disagreeing) -------------------------
    gt_eval = gt_eval_df[gt_eval_df["Report_ID"] == sample_idx].copy()
    gen_eval = gen_eval_df[gen_eval_df["Report_ID"] == sample_idx].copy()

    def split_agree(eval_df):
        agreeing    = []
        disagreeing = []
        for _, row in eval_df.iterrows():
            entry = {
                "question_id":               int(row["Question_ID"]),
                "question_text":             row.get("Question_Text", ""),
                "options":                   row["Options"],
                "correct_answer":            row["Correct_Answer"],
                "predicted_using_gt_report": row["Predicted_Answer_Using_GT"],
                "predicted_using_gen_report": row["Predicted_Answer_Using_Gen"],
            }
            if row["Predicted_Answer_Using_GT"] == row["Predicted_Answer_Using_Gen"]:
                agreeing.append(entry)
            else:
                disagreeing.append(entry)
        return agreeing, disagreeing

    gt_agreeing, gt_disagreeing     = split_agree(gt_eval)
    gen_agreeing, gen_disagreeing   = split_agree(gen_eval)

    # ---- disagreement scores (omission / hallucination) --------------------
    def disagree_pct(disagree_count, total):
        if total == 0:
            return None
        return round(disagree_count / total * 100, 4)

    omission_score      = disagree_pct(len(gt_disagreeing),  len(gt_agreeing)  + len(gt_disagreeing))
    hallucination_score = disagree_pct(len(gen_disagreeing), len(gen_agreeing) + len(gen_disagreeing))

    # ---- assemble ----------------------------------------------------------
    return {
        "sample_id":  sample_idx,
        "gt_report":  gt_report,
        "gen_report": gen_report,

        "gt_reference": {
            "all_questions":      gt_all_questions,
            "all_questions_count": len(gt_all_questions),
            "filtered_questions": gt_filtered_list,
            "filtered_count":     len(gt_filtered_list),
            "eval_agreeing_questions":    gt_agreeing,
            "eval_disagreeing_questions": gt_disagreeing,
            "agreeing_count":    len(gt_agreeing),
            "disagreeing_count": len(gt_disagreeing),
            "omission_score_pct": omission_score,
        },

        "gen_reference": {
            "all_questions":      gen_all_questions,
            "all_questions_count": len(gen_all_questions),
            "filtered_questions": gen_filtered_list,
            "filtered_count":     len(gen_filtered_list),
            "eval_agreeing_questions":    gen_agreeing,
            "eval_disagreeing_questions": gen_disagreeing,
            "agreeing_count":    len(gen_agreeing),
            "disagreeing_count": len(gen_disagreeing),
            "hallucination_score_pct": hallucination_score,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compile ICARE per-sample results to JSON.")
    parser.add_argument("--base_dir",  required=True, help="Pipeline output base directory")
    parser.add_argument("--input_csv", required=True, help="Original reports CSV (id, gt_report, gen_report)")
    parser.add_argument("--output",    required=True, help="Path for the output JSON file")
    parser.add_argument("--summary_csv", default="", help="Optional path for per-sample summary CSV")
    parser.add_argument("--timing_file", default="", help="Optional JSON file with pipeline step durations (seconds)")
    args = parser.parse_args()

    data_type = "shuffled_ans_choices_data"
    base = f"{args.base_dir}/{data_type}"

    # ---- load inputs -------------------------------------------------------
    reports_df = pd.read_csv(args.input_csv)

    gt_mcqa  = _load_mcqa_json(f"{base}/gt_reports_as_ref/mcqa_data.json")
    gen_mcqa = _load_mcqa_json(f"{base}/gen_reports_as_ref/mcqa_data.json")

    gt_filtered  = _load_filtered(f"{base}/gt_reports_as_ref/mcqa_filtering/filtered_questions_shuffled.csv")
    gen_filtered = _load_filtered(f"{base}/gen_reports_as_ref/mcqa_filtering/filtered_questions_shuffled.csv")

    gt_eval  = _load_eval_predictions(f"{base}/gt_reports_as_ref/mcqa_eval/mcqa_eval_answer_predictions.csv")
    gen_eval = _load_eval_predictions(f"{base}/gen_reports_as_ref/mcqa_eval/mcqa_eval_answer_predictions.csv")

    # Merge question text into eval predictions (it lives in filtered_questions)
    qt_cols = ["Report_ID", "Question_ID", "Question_Text"]
    gt_eval  = gt_eval.merge(gt_filtered[qt_cols],  on=["Report_ID", "Question_ID"], how="left")
    gen_eval = gen_eval.merge(gen_filtered[qt_cols], on=["Report_ID", "Question_ID"], how="left")

    # ---- build per-sample entries ------------------------------------------
    samples = []
    for i, row in reports_df.iterrows():
        entry = build_sample_entry(
            sample_idx          = i,
            gt_report           = row["ground_truth_report"],
            gen_report          = row["generated_report"],
            gt_all_questions    = gt_mcqa[i]["questions"]  if i < len(gt_mcqa)  else [],
            gen_all_questions   = gen_mcqa[i]["questions"] if i < len(gen_mcqa) else [],
            gt_filtered_df      = gt_filtered,
            gen_filtered_df     = gen_filtered,
            gt_eval_df          = gt_eval,
            gen_eval_df         = gen_eval,
        )
        samples.append(entry)

    timing = None
    if args.timing_file:
        with open(args.timing_file) as f:
            timing = json.load(f)

    output = {"samples": samples}
    if timing is not None:
        output = {"timing": timing, **output}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(samples)} samples to {args.output}")

    summary_rows = []
    for i, row in reports_df.iterrows():
        s = samples[i]
        gt_total = s["gt_reference"]["agreeing_count"] + s["gt_reference"]["disagreeing_count"]
        gen_total = s["gen_reference"]["agreeing_count"] + s["gen_reference"]["disagreeing_count"]
        gt_agree_pct = round(s["gt_reference"]["agreeing_count"] / gt_total * 100, 4) if gt_total else None
        gen_agree_pct = round(s["gen_reference"]["agreeing_count"] / gen_total * 100, 4) if gen_total else None
        summary_rows.append({
            "sample_id": i,
            "study_id": row["id"] if "id" in row else i,
            "gt_agreement_pct": gt_agree_pct,
            "gen_agreement_pct": gen_agree_pct,
            "total_gt_questions": gt_total,
            "total_gen_questions": gen_total,
        })

    summary_csv = args.summary_csv or str(Path(args.output).with_name("icare_results_summary.csv"))
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Wrote summary CSV to {summary_csv}")
    # Quick sanity print
    for s in samples:
        print(
            f"  sample {s['sample_id']:>2} | "
            f"gt_filtered={s['gt_reference']['filtered_count']:>2}  "
            f"gt_agree={s['gt_reference']['agreeing_count']:>2}  "
            f"gt_disagree={s['gt_reference']['disagreeing_count']:>2}  "
            f"omission={s['gt_reference']['omission_score_pct']}%  || "
            f"gen_filtered={s['gen_reference']['filtered_count']:>2}  "
            f"gen_agree={s['gen_reference']['agreeing_count']:>2}  "
            f"gen_disagree={s['gen_reference']['disagreeing_count']:>2}  "
            f"hallucination={s['gen_reference']['hallucination_score_pct']}%"
        )


if __name__ == "__main__":
    main()
