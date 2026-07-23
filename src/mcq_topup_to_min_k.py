"""
Top up filtered MCQs until each report has at least --min_k keepers.

After the normal one-shot generate + filter pass, some reports may have
fewer than min_k report-dependent questions. This script:

  1. Keeps existing filtered questions.
  2. For reports with n < min_k, generates additional questions with an
     anti-repeat prompt (reuses build_mcq_prompt from mcq_generation).
  3. Filters only the new candidates (same with/without-report rule).
  4. Appends keepers and rewrites filtered_questions*.csv.

Does not modify the original one-shot generation path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from config import Config
from mcq_filtering import get_model_prediction, shuffle_filtered_answers
from mcq_generation import build_mcq_prompt, make_llama_request, parse_mcq
from utils import ensure_dir


def _load_mcqa(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _counts_by_report(df: pd.DataFrame) -> Dict[int, int]:
    if df is None or len(df) == 0:
        return {}
    return df.groupby("Report_ID").size().astype(int).to_dict()


def _parse_options(options):
    if isinstance(options, dict):
        return options
    if isinstance(options, str):
        return eval(options)
    raise TypeError(f"Unexpected Options type: {type(options)}")


def generate_topup_batch(
    report: str,
    previous_question_texts: List[str],
    batch_n: int,
    seed: int,
    max_attempts: int = 3,
) -> List[dict]:
    """Generate up to batch_n new MCQs, avoiding previous_question_texts."""
    collected: List[dict] = []
    for _ in range(max_attempts):
        if len(collected) >= batch_n:
            break
        need = batch_n - len(collected)
        prompt = build_mcq_prompt(report, need, previous_question_texts + [q["question_text"] for q in collected])
        try:
            response = make_llama_request(
                prompt=prompt,
                url=Config.API_URL,
                api_key=Config.API_KEY,
                max_tokens=Config.GENERATION_MAX_TOKENS,
                temperature=Config.DEFAULT_TEMPERATURE,
                timeout=Config.GENERATION_TIMEOUT,
                model=Config.MODEL_NAME,
                seed=seed,
                top_p=Config.DEFAULT_TOP_P,
                n=Config.DEFAULT_N,
                stream=False,
            )
        except Exception as e:
            print(f"  top-up generation error: {e}", flush=True)
            continue
        if not response:
            continue
        mcq_text = response["choices"][0]["message"]["content"]
        parsed = parse_mcq(mcq_text)
        for mcq in parsed:
            if (
                mcq.get("question_text")
                and all(mcq.get("options", {}).values())
                and mcq.get("correct_answer")
            ):
                collected.append(mcq)
            if len(collected) >= batch_n:
                break
    return collected[:batch_n]


def filter_candidate_questions(
    report: str,
    report_id: int,
    questions: List[dict],
    start_index: int,
    seed: int,
) -> pd.DataFrame:
    """Apply the same keep-rule as mcq_filtering to a list of new questions."""
    rows = []
    for offset, question in enumerate(questions):
        options = question["options"]
        q_text = question["question_text"]
        correct = question["correct_answer"]
        pred_with = get_model_prediction(
            report, q_text, options, "using_report", seed=seed
        )
        pred_without = get_model_prediction(
            report, q_text, options, "without_using_report", seed=seed
        )
        rows.append(
            {
                "Index": start_index + offset,
                "Report_ID": report_id,
                "Question_ID": question.get("question_id", offset),
                "Question_Text": q_text,
                "Options": str(options),
                "Correct_Answer": correct,
                "Predicted_Answer_with_report": pred_with,
                "Predicted_Answer_without_report": pred_without,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = (
        (df["Predicted_Answer_with_report"] == df["Correct_Answer"])
        & (df["Predicted_Answer_without_report"] != df["Correct_Answer"])
    )
    return df.loc[keep].copy()


def topup_filter_dir(
    mcqa_json: str,
    filter_dir: str,
    min_k: int,
    max_rounds: int,
    batch_size: int,
    seed: int,
) -> dict:
    ensure_dir(filter_dir)
    data = _load_mcqa(mcqa_json)
    mcq_data = data["mcq_data"]

    filtered_path = os.path.join(filter_dir, "filtered_questions.csv")
    shuffled_path = os.path.join(filter_dir, "filtered_questions_shuffled.csv")

    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path)
    else:
        filtered_df = pd.DataFrame(
            columns=[
                "Index",
                "Report_ID",
                "Question_ID",
                "Question_Text",
                "Options",
                "Correct_Answer",
                "Predicted_Answer_with_report",
                "Predicted_Answer_without_report",
            ]
        )

    if os.path.exists(shuffled_path):
        shuffled_df = pd.read_csv(shuffled_path)
    else:
        shuffled_df = filtered_df.copy()

    before_counts = _counts_by_report(filtered_df)
    n_reports = len(mcq_data)
    short_ids = [rid for rid in range(n_reports) if before_counts.get(rid, 0) < min_k]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mcqa_json": mcqa_json,
        "filter_dir": filter_dir,
        "min_k": min_k,
        "max_rounds": max_rounds,
        "batch_size": batch_size,
        "seed": seed,
        "n_reports": n_reports,
        "n_short_before": len(short_ids),
        "counts_before": {str(k): int(v) for k, v in sorted(before_counts.items())},
        "rounds": [],
    }

    print(
        f"Top-up: {len(short_ids)}/{n_reports} reports have < {min_k} filtered questions",
        flush=True,
    )

    for round_idx in range(1, max_rounds + 1):
        counts = _counts_by_report(filtered_df)
        short_ids = [rid for rid in range(n_reports) if counts.get(rid, 0) < min_k]
        if not short_ids:
            print(f"Round {round_idx}: all reports >= {min_k}. Stopping.", flush=True)
            break

        print(f"Round {round_idx}: topping up {len(short_ids)} reports...", flush=True)
        next_index = int(filtered_df["Index"].max()) + 1 if len(filtered_df) else 0
        new_kept_rows: List[pd.DataFrame] = []
        round_stats = {"round": round_idx, "n_short": len(short_ids), "per_report": []}

        for rid in tqdm(short_ids, desc=f"topup_round_{round_idx}"):
            report = mcq_data[rid]["report"]
            existing = filtered_df[filtered_df["Report_ID"] == rid]
            prev_texts = existing["Question_Text"].astype(str).tolist()
            n_have = len(prev_texts)
            # Ask for a full batch (overshoot); do not request exactly k-n only.
            batch = generate_topup_batch(report, prev_texts, batch_size, seed=seed)
            for j, q in enumerate(batch):
                q["question_id"] = int(existing["Question_ID"].max()) + 1 + j if len(existing) else j

            kept = filter_candidate_questions(
                report, rid, batch, start_index=next_index, seed=seed
            )
            n_kept = len(kept)
            next_index += max(len(batch), 1)
            round_stats["per_report"].append(
                {
                    "Report_ID": rid,
                    "had": n_have,
                    "generated": len(batch),
                    "kept": n_kept,
                    "after": n_have + n_kept,
                }
            )
            if n_kept == 0:
                continue
            new_kept_rows.append(kept)

        if new_kept_rows:
            added = pd.concat(new_kept_rows, ignore_index=True)
            filtered_df = pd.concat([filtered_df, added], ignore_index=True)
            added_shuffled = shuffle_filtered_answers(added, seed=seed)
            shuffled_df = pd.concat([shuffled_df, added_shuffled], ignore_index=True)

        summary["rounds"].append(round_stats)
        still_short = sum(
            1
            for rid in range(n_reports)
            if _counts_by_report(filtered_df).get(rid, 0) < min_k
        )
        print(
            f"Round {round_idx} done: still short = {still_short}/{n_reports}",
            flush=True,
        )
        if still_short == 0:
            break

    after_counts = _counts_by_report(filtered_df)
    still_short_ids = [rid for rid in range(n_reports) if after_counts.get(rid, 0) < min_k]
    summary["n_short_after"] = len(still_short_ids)
    summary["short_after_report_ids"] = still_short_ids
    summary["counts_after"] = {str(k): int(v) for k, v in sorted(after_counts.items())}
    summary["reached_min_k_fraction"] = (
        (n_reports - len(still_short_ids)) / n_reports if n_reports else 1.0
    )

    # Backup originals once
    if os.path.exists(filtered_path) and not os.path.exists(filtered_path + ".pre_topup"):
        os.rename(filtered_path, filtered_path + ".pre_topup")
    if os.path.exists(shuffled_path) and not os.path.exists(shuffled_path + ".pre_topup"):
        os.rename(shuffled_path, shuffled_path + ".pre_topup")

    filtered_df.to_csv(filtered_path, index=False)
    shuffled_df.to_csv(shuffled_path, index=False)

    summary_path = os.path.join(filter_dir, "topup_to_min_k_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {filtered_path}", flush=True)
    print(f"Wrote {shuffled_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(
        f"Coverage: {n_reports - len(still_short_ids)}/{n_reports} reports >= {min_k} "
        f"({100 * summary['reached_min_k_fraction']:.1f}%)",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Top up filtered questions until each report has >= min_k keepers"
    )
    parser.add_argument("--mcqa-json", required=True, help="Path to mcqa_data.json")
    parser.add_argument(
        "--filter-dir",
        required=True,
        help="mcqa_filtering dir with filtered_questions.csv",
    )
    parser.add_argument("--min_k", type=int, default=8, help="Minimum filtered questions per report")
    parser.add_argument("--max_rounds", type=int, default=3, help="Max top-up rounds")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="New questions requested per short report per round",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.min_k < 1:
        print("--min_k must be >= 1", file=sys.stderr)
        sys.exit(1)

    topup_filter_dir(
        mcqa_json=args.mcqa_json,
        filter_dir=args.filter_dir,
        min_k=args.min_k,
        max_rounds=args.max_rounds,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
