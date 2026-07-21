"""
Post-filter MCQ question counts for RexVal ICARE runs.

Counts come from filtered_questions_shuffled.csv (questions actually used in eval),
NOT per_report_statistics.csv (that file counts pre-filter questions ~60).

Usage (on bigpurple):
    cd ICARE_score
    conda activate rrg-eval-clean
    python scripts/rexval_data/analyze_rexval_question_counts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/gpfs/data/oermannlab/users/rd3571")
OUT_DIR = BASE / "ICARE_score/outputs/rexval/rexval_test_200"
REXVAL_CSV = BASE / "cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
LABELS_CSV = (
    BASE / "cxr_report_datasets/rexval_physionet_labels"
    / "physionet.org/files/rexval-dataset/1.0.0"
    / "6_valid_raters_per_rater_error_categories.csv"
)

METHODS = {
    "llama": BASE / "ICARE_score/outputs/rexval/rexval_test_200/eval_seed_123/shuffled_ans_choices_data",
    "opus46": BASE / "ICARE_score/outputs/rexval/rexval_test_200_opus46/eval_seed_123/shuffled_ans_choices_data",
    "sonnet46": BASE / "ICARE_score/outputs/rexval/rexval_test_200_sonnet46/eval_seed_123/shuffled_ans_choices_data",
    "gpt54": BASE / "ICARE_score/outputs/rexval/rexval_test_200_gpt54/eval_seed_123/shuffled_ans_choices_data",
}
CANDIDATES = ["radgraph", "bertscore", "s_emb", "bleu"]
THRESHOLDS = [0, 5, 8, 10, 12, 15, 18, 20, 25]


def post_filter_qcounts(eval_dir, ref):
    """Number of post-filter questions per Report_ID (0..199)."""
    path = eval_dir / f"{ref}_reports_as_ref/mcqa_filtering/filtered_questions_shuffled.csv"
    if not path.exists():
        return pd.Series(index=range(200), dtype=float)
    return pd.read_csv(path).groupby("Report_ID").size().reindex(range(200))


def load_method_counts(eval_dir):
    q_gt = post_filter_qcounts(eval_dir, "gt")
    q_gen = post_filter_qcounts(eval_dir, "gen")
    return {"gt": q_gt, "gen": q_gen, "min": np.minimum(q_gt, q_gen)}


def load_ap_avg(eval_dir):
    gt = pd.read_csv(
        eval_dir / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
    ).set_index("Report_ID").reindex(range(200))["Agreement_Percentage"]
    gen = pd.read_csv(
        eval_dir / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
    ).set_index("Report_ID").reindex(range(200))["Agreement_Percentage"]
    return (gt + gen) / 2


def shared_keep_ids(min_q_by_method, threshold):
    """Report_IDs where every method has min(gt,gen) post-filter count >= threshold."""
    keep = set(range(200))
    for mq in min_q_by_method.values():
        keep &= set(mq[mq >= threshold].dropna().index.astype(int))
    return keep


def summarize_distribution(name, counts):
    rows = []
    for ref, vals in counts.items():
        v = vals.dropna()
        rows.append({
            "method": name,
            "ref": ref,
            "n": len(v),
            "min": int(v.min()),
            "p10": np.percentile(v, 10),
            "p25": np.percentile(v, 25),
            "median": np.median(v),
            "mean": v.mean(),
            "p75": np.percentile(v, 75),
            "max": int(v.max()),
            "lt5": int((v < 5).sum()),
            "lt10": int((v < 10).sum()),
            "lt15": int((v < 15).sum()),
        })
    return rows


def avg_kendall_tau(df, dis_col):
    taus = []
    for cand in CANDIDATES:
        sub = df[df["origin"] == cand]
        x = sub[dis_col].values.astype(float)
        y = sub["mean_clin_sig_errors"].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 5:
            taus.append(stats.kendalltau(x[m], y[m])[0])
    return float(np.mean(taus)) if taus else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_counts = {name: load_method_counts(path) for name, path in METHODS.items()}
    min_q = {name: c["min"] for name, c in all_counts.items()}

    # --- distribution table ---
    rows = []
    for name, counts in all_counts.items():
        rows.extend(summarize_distribution(name, counts))
    dist = pd.DataFrame(rows)
    dist_path = OUT_DIR / "filtered_question_count_distribution.csv"
    dist.to_csv(dist_path, index=False)

    print("POST-FILTER question counts (filtered_questions_shuffled.csv)")
    print(dist.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # --- how many reports are low per method / shared ---
    print("\nReports with min(gt,gen) post-filter Q below threshold:")
    print(f"{'threshold':>10s}", " ".join(f"{m:>8s}" for m in METHODS), f"{'all4':>8s}")
    for t in [5, 8, 10, 12, 15, 20]:
        per_method = [int((min_q[m] < t).sum()) for m in METHODS]
        shared_low = 200 - len(shared_keep_ids(min_q, t))
        print(f"{t:10d}", " ".join(f"{n:8d}" for n in per_method), f"{shared_low:8d}")

    # --- histogram ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.arange(0, 45, 2)
    for ax, (name, counts) in zip(axes.ravel(), all_counts.items()):
        for ref, color in [("gt", "#1565C0"), ("gen", "#E65100"), ("min", "#2E7D32")]:
            vals = counts[ref].dropna()
            ax.hist(vals, bins=bins, alpha=0.45, label=ref, color=color, edgecolor="white")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("post-filter questions per report")
        ax.set_ylabel("# reports")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle("RexVal post-filter question counts by method", fontsize=14, fontweight="bold")
    fig.tight_layout()
    hist_path = OUT_DIR / "filtered_question_count_distribution.png"
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {dist_path}")
    print(f"Saved: {hist_path}")

    # --- correlation sweep with SHARED filter (same Report_IDs for all ICARE methods) ---
    rexval_df = pd.read_csv(REXVAL_CSV).rename(columns={"Unnamed: 0": "row_id"})
    labels_df = pd.read_csv(LABELS_CSV)
    clin_sig = labels_df[labels_df["clinically_significant"] == True]
    per_rater = (
        clin_sig.groupby(["study_number", "candidate_type", "rater_index"], as_index=False)["num_errors"]
        .sum().rename(columns={"num_errors": "errors", "candidate_type": "origin"})
    )
    error_scores = (
        per_rater.groupby(["study_number", "origin"], as_index=False)["errors"]
        .mean().rename(columns={"errors": "mean_clin_sig_errors"})
    )
    base = rexval_df[["row_id", "study_number", "origin"]].merge(
        error_scores, on=["study_number", "origin"], how="left"
    )

    ap = {name: load_ap_avg(path) for name, path in METHODS.items()}
    dis_cols = {
        "llama": "dis_llama",
        "opus46": "dis_opus46",
        "sonnet46": "dis_sonnet46",
        "gpt54": "dis_gpt54",
    }
    for name, s in ap.items():
        base[dis_cols[name]] = 1 - base["row_id"].map(s.to_dict()) / 100

    corr_rows = []
    for thr in THRESHOLDS:
        keep = shared_keep_ids(min_q, thr)
        sub = base[base["row_id"].isin(keep)].copy()
        row = {"min_q_all_methods": thr, "n_reports": len(keep), "n_rows": len(sub)}
        for name, col in dis_cols.items():
            row[f"tau_{name}"] = avg_kendall_tau(sub, col)
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr_path = OUT_DIR / "correlation_shared_min_questions.csv"
    corr.to_csv(corr_path, index=False)

    print("\nDoctor-error correlation (avg Kendall tau) with SHARED report filter")
    print("(keep Report_ID only if min(gt,gen) >= threshold for llama AND opus AND sonnet AND gpt54)")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {corr_path}")
    print("\nTo regenerate main plots with the same shared filter:")
    print("  MIN_FILTERED_QUESTIONS=10 python scripts/rexval_data/plot_rexval_correlation.py")


if __name__ == "__main__":
    main()
