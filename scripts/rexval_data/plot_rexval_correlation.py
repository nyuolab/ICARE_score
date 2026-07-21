"""
RexVal analysis — three figures in one script:

  Figure 1: Correlation table (Kendall τ / Pearson r vs clinically significant errors)
            per candidate type (radgraph, bertscore, s_emb, bleu models)
  Figure 2: Per-rater preference alignment forest plot (top-1 agreement)
  Figure 3: Decisive-consensus alignment forest plot (≥4-of-6 raters)

All metrics loaded from produced output files.

Usage:
    cd ICARE_score
    python scripts/rexval_data/plot_rexval_correlation.py

    # Optional: drop low post-filter question reports from ALL ICARE methods (fair comparison)
    MIN_FILTERED_QUESTIONS=10 python scripts/rexval_data/plot_rexval_correlation.py

    # See post-filter question distributions first:
    python scripts/rexval_data/analyze_rexval_question_counts.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = Path("/gpfs/data/oermannlab/users/rd3571")
EVAL_DIR  = BASE / "ICARE_score/outputs/rexval/rexval_test_200/eval_seed_123/shuffled_ans_choices_data"
ALLQUES_EVAL_DIR = BASE / "ICARE_score/outputs/rexval/rexval_test_200_allques/eval_seed_123/shuffled_ans_choices_data"
SEQ_EVAL_DIR = BASE / "ICARE_score/outputs/rexval/rexval_test_200/eval_seed_123_sequential_b10/shuffled_ans_choices_data"
OPUS_EVAL_DIR   = BASE / "ICARE_score/outputs/rexval/rexval_test_200_opus46/eval_seed_123/shuffled_ans_choices_data"
SONNET_EVAL_DIR = BASE / "ICARE_score/outputs/rexval/rexval_test_200_sonnet46/eval_seed_123/shuffled_ans_choices_data"
GPT54_EVAL_DIR  = BASE / "ICARE_score/outputs/rexval/rexval_test_200_gpt54/eval_seed_123/shuffled_ans_choices_data"
PRED_DIR  = BASE / "ICARE_score/outputs/rexval/predefined/eval_seed_123/mcqa_eval"
BASELINES = BASE / "ICARE_score/outputs/rexval/rexval_test_200/baselines"
REXVAL_CSV = BASE / "cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
LABELS_CSV = (BASE / "cxr_report_datasets/rexval_physionet_labels"
              / "physionet.org/files/rexval-dataset/1.0.0"
              / "6_valid_raters_per_rater_error_categories.csv")
OUT_DIR   = BASE / "ICARE_score/outputs/rexval/rexval_test_200"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
RNG    = np.random.default_rng(42)
# Isolated stream for the new ICARE_SEQ variant so its bootstrap draws never
# shift the shared RNG's position and perturb already-reported metrics' CIs.
RNG_SEQ = np.random.default_rng(42)
CANDIDATES = ["radgraph", "bertscore", "s_emb", "bleu"]
CAND_LABELS = {"radgraph": "RadGraph", "bertscore": "BERTScore",
               "s_emb": "SembScore", "bleu": "BLEU"}

# ---------------------------------------------------------------------------
# Load RexVal reports CSV
# ---------------------------------------------------------------------------
rexval_df = pd.read_csv(REXVAL_CSV).rename(columns={"Unnamed: 0": "row_id"})
# row_id 0..199, study_number 0..49, origin = candidate type

# ---------------------------------------------------------------------------
# Load and aggregate clinically significant error labels
# ---------------------------------------------------------------------------
labels_df  = pd.read_csv(LABELS_CSV)
clin_sig   = labels_df[labels_df["clinically_significant"] == True].copy()

# Sum errors across categories per (study, candidate, rater)
per_rater_err = (
    clin_sig
    .groupby(["study_number", "candidate_type", "rater_index"], as_index=False)["num_errors"]
    .sum()
    .rename(columns={"num_errors": "errors", "candidate_type": "origin"})
)

# Mean across raters per (study, candidate)
error_scores = (
    per_rater_err
    .groupby(["study_number", "origin"], as_index=False)["errors"]
    .mean()
    .rename(columns={"errors": "mean_clin_sig_errors"})
)

RATERS = sorted(per_rater_err["rater_index"].unique())

# ---------------------------------------------------------------------------
# Load ICARE (shuffled) — gt, gen, avg
# ---------------------------------------------------------------------------
gt_stats  = pd.read_csv(EVAL_DIR / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
gen_stats = pd.read_csv(EVAL_DIR / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")

ap_gt  = gt_stats.loc[range(200), "Agreement_Percentage"].values
ap_gen = gen_stats.loc[range(200), "Agreement_Percentage"].values
ap_avg = (ap_gt + ap_gen) / 2

# ---------------------------------------------------------------------------
# Load ICARE (shuffled, sequential MCQ generation) — gt, gen, avg
# ---------------------------------------------------------------------------
gt_stats_seq  = pd.read_csv(SEQ_EVAL_DIR / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
gen_stats_seq = pd.read_csv(SEQ_EVAL_DIR / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")

ap_gt_seq  = gt_stats_seq.loc[range(200), "Agreement_Percentage"].values
ap_gen_seq = gen_stats_seq.loc[range(200), "Agreement_Percentage"].values
ap_seq     = (ap_gt_seq + ap_gen_seq) / 2

# ---------------------------------------------------------------------------
# Load ICARE model-ablation runs (Opus / Sonnet / GPT-5.4)
# ---------------------------------------------------------------------------
def _ap_avg(eval_dir):
    """Mean GT/GEN agreement for Report_IDs 0..199; NaN if either side missing."""
    gt  = pd.read_csv(eval_dir / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    gen = pd.read_csv(eval_dir / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    ids = range(200)
    avg = (
        gt.reindex(ids)["Agreement_Percentage"]
        + gen.reindex(ids)["Agreement_Percentage"]
    ) / 2
    n_miss = int(avg.isna().sum())
    if n_miss:
        miss = avg[avg.isna()].index.tolist()
        print(f"WARNING: {eval_dir} incomplete — {n_miss}/200 rows NaN (missing Report_IDs: {miss})")
    return avg.values

ap_opus   = _ap_avg(OPUS_EVAL_DIR)
ap_sonnet = _ap_avg(SONNET_EVAL_DIR)
ap_gpt54  = _ap_avg(GPT54_EVAL_DIR)
ap_allques = _ap_avg(ALLQUES_EVAL_DIR)

# ---------------------------------------------------------------------------
# Load ICARE predefined
# ---------------------------------------------------------------------------
pred_stats = pd.read_csv(PRED_DIR / "mcq_eval_report_level_stats.csv").set_index("Report_ID")
ap_pred    = pred_stats.loc[range(200), "Agreement_Percentage"].values

# ---------------------------------------------------------------------------
# Load CRIMSON baseline
# ---------------------------------------------------------------------------
with open(BASELINES / "crimson_results.json") as f:
    crimson_json = json.load(f)
crimson_scores = np.array([r["crimson_score"] for r in crimson_json["results"]])  # (200,)

# ---------------------------------------------------------------------------
# Load GREEN baseline
# ---------------------------------------------------------------------------
green_df     = pd.read_csv(BASELINES / "green/rexval_prepared_green_results.csv")
green_scores = green_df["green_score"].values  # (200,)

# ---------------------------------------------------------------------------
# Load AlignScore baseline
# ---------------------------------------------------------------------------
alignscore_df     = pd.read_csv(BASELINES / "alignscore/rexval_prepared_alignscore_results.csv")
alignscore_scores = alignscore_df["alignscore"].values  # (200,)

# ---------------------------------------------------------------------------
# Load RRG baselines (BLEU, BERTScore, SembScore, RadGraph, RadCliQ-v1)
# ---------------------------------------------------------------------------
rrg_df    = pd.read_csv(BASELINES / "rexval_test_200/rexval_prepared_results.csv")
bleu_sc   = rrg_df["bleu_score"].values
bert_sc   = rrg_df["bertscore"].values
semb_sc   = rrg_df["semb_score"].values
rg_sc     = rrg_df["radgraph_combined"].values
rcq_sc    = rrg_df["RadCliQ-v1"].values   # higher = more errors = worse

# ---------------------------------------------------------------------------
# Build merged DataFrame aligned by row_id (0..199)
# ---------------------------------------------------------------------------
merged = rexval_df[["row_id", "study_number", "origin"]].copy()
merged["ap_gt"]     = ap_gt
merged["ap_gen"]    = ap_gen
merged["ap_avg"]    = ap_avg
merged["ap_allques"] = ap_allques
merged["ap_seq"]    = ap_seq
merged["ap_opus"]   = ap_opus
merged["ap_sonnet"] = ap_sonnet
merged["ap_gpt54"]  = ap_gpt54
merged["ap_pred"]   = ap_pred
merged["crimson"]    = crimson_scores
merged["green"]      = green_scores
merged["alignscore"] = alignscore_scores
merged["bleu"]       = bleu_sc
merged["bertscore"] = bert_sc
merged["semb"]      = semb_sc
merged["radgraph"]  = rg_sc
merged["radcliq"]   = rcq_sc

# Merge with error labels
merged = merged.merge(error_scores, on=["study_number", "origin"], how="left")

# "Disagreement" convention for correlation table: higher = worse report
# Quality metrics negated; ICARE as 1-agreement; RadCliQ-v1 raw (already higher=worse)
merged["dis_gt"]      = 1 - merged["ap_gt"]   / 100
merged["dis_gen"]     = 1 - merged["ap_gen"]  / 100
merged["dis_avg"]     = 1 - merged["ap_avg"]  / 100
merged["dis_allques"] = 1 - merged["ap_allques"] / 100
merged["dis_seq"]     = 1 - merged["ap_seq"]  / 100
merged["dis_opus"]    = 1 - merged["ap_opus"] / 100
merged["dis_sonnet"]  = 1 - merged["ap_sonnet"] / 100
merged["dis_gpt54"]   = 1 - merged["ap_gpt54"] / 100
merged["dis_pred"]    = 1 - merged["ap_pred"] / 100
merged["neg_crimson"]    = -merged["crimson"]
merged["neg_green"]      = -merged["green"]
merged["neg_alignscore"] = -merged["alignscore"]
merged["neg_bleu"]       = -merged["bleu"]
merged["neg_bert"]    = -merged["bertscore"]
merged["neg_semb"]    = -merged["semb"]
merged["neg_rg"]      = -merged["radgraph"]
# merged["radcliq"]  — already higher=worse, use raw

# ---------------------------------------------------------------------------
# Optional shared filter: same Report_IDs for llama / opus / sonnet / gpt54
# Uses post-filter counts from filtered_questions_shuffled.csv (see analyze script).
# ---------------------------------------------------------------------------
MIN_FILTERED_QUESTIONS = int(os.environ.get("MIN_FILTERED_QUESTIONS", "0"))


def _post_filter_min_q(eval_dir):
    def _one(ref):
        p = eval_dir / f"{ref}_reports_as_ref/mcqa_filtering/filtered_questions_shuffled.csv"
        if not p.exists():
            return pd.Series(index=range(200), dtype=float)
        return pd.read_csv(p).groupby("Report_ID").size().reindex(range(200))
    return np.minimum(_one("gt"), _one("gen"))


if MIN_FILTERED_QUESTIONS > 0:
    _icare_dirs = {
        "llama": EVAL_DIR,
        "opus46": OPUS_EVAL_DIR,
        "sonnet46": SONNET_EVAL_DIR,
        "gpt54": GPT54_EVAL_DIR,
    }
    _keep_ids = set(range(200))
    for _name, _edir in _icare_dirs.items():
        _mq = _post_filter_min_q(_edir)
        _keep_ids &= set(_mq[_mq >= MIN_FILTERED_QUESTIONS].dropna().index.astype(int))
    _n_before = len(merged)
    merged = merged[merged["row_id"].isin(_keep_ids)].copy()
    print(
        f"MIN_FILTERED_QUESTIONS={MIN_FILTERED_QUESTIONS}: "
        f"kept {len(_keep_ids)}/200 reports, {len(merged)}/{_n_before} rows"
    )

# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(x, y, stat_fn, n=N_BOOT, rng=None):
    rng   = RNG if rng is None else rng
    idx   = np.arange(len(x))
    boots = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(stat_fn(x[s], y[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return lo, hi

def compute_corr_by_cand(df, score_col, rng=None):
    """Compute Kendall τ and Pearson r vs errors for each candidate type."""
    results = {}
    for cand in CANDIDATES:
        sub = df[df["origin"] == cand].copy()
        x = sub[score_col].values.astype(float)
        y = sub["mean_clin_sig_errors"].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            results[cand] = dict(tau=np.nan, r=np.nan,
                                 tau_lo=np.nan, tau_hi=np.nan,
                                 r_lo=np.nan, r_hi=np.nan)
            continue
        tau = stats.kendalltau(x, y)[0]
        r   = stats.pearsonr(x, y)[0]
        tau_lo, tau_hi = bootstrap_ci(x, y, lambda a, b: stats.kendalltau(a, b)[0], rng=rng)
        r_lo,   r_hi   = bootstrap_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0], rng=rng)
        results[cand] = dict(tau=tau, r=r,
                             tau_lo=tau_lo, tau_hi=tau_hi,
                             r_lo=r_lo, r_hi=r_hi)
    return results

# (label, disagreement_col)  — higher column value = worse report
CORR_METRICS = [
    ("BLEU",             "neg_bleu"),
    ("BERTScore",        "neg_bert"),
    ("SembScore",        "neg_semb"),
    ("RadGraph",         "neg_rg"),
    ("RadCliQ-v1",       "radcliq"),
    ("GREEN",            "neg_green"),
    ("AlignScore",       "neg_alignscore"),
    ("CRIMSON",          "neg_crimson"),
    ("ICARE_AVG",        "dis_avg"),
    ("ICARE_ALLQUES",    "dis_allques"),
    ("ICARE_SEQ",        "dis_seq"),
    ("ICARE_OPUS46",     "dis_opus"),
    ("ICARE_SONNET46",   "dis_sonnet"),
    ("ICARE_GPT54",      "dis_gpt54"),
    ("ICARE_PREDEFINED", "dis_pred"),
]

print("Computing correlations (Kendall τ / Pearson r vs clinically significant errors)...")
corr_results = {}
for label, col in CORR_METRICS:
    rng = RNG_SEQ if label == "ICARE_SEQ" else None
    corr_results[label] = compute_corr_by_cand(merged, col, rng=rng)
    avg_tau = np.mean([corr_results[label][c]["tau"] for c in CANDIDATES])
    avg_r   = np.mean([corr_results[label][c]["r"]   for c in CANDIDATES])
    print(f"  {label:22s}: avg τ={avg_tau:.3f}  avg r={avg_r:.3f}")

# ===========================================================================
# Table: Kendall τ and Pearson r (pandas DataFrame, saved as CSV)
# ===========================================================================
def fmt(val, lo, hi):
    return f"{val:.2f} [{lo:.2f}, {hi:.2f}]"

rows = []
for label, col in CORR_METRICS:
    row = {"Metric": label}
    for cand in CANDIDATES:
        c = corr_results[label][cand]
        row[f"{CAND_LABELS[cand]}\nKendall τ"] = fmt(c["tau"], c["tau_lo"], c["tau_hi"])
        row[f"{CAND_LABELS[cand]}\nPearson r"] = fmt(c["r"],   c["r_lo"],   c["r_hi"])
    rows.append(row)

corr_table = pd.DataFrame(rows).set_index("Metric")

print("\nKendall τ and Pearson r (95% CI) | n=50 per column")
print("Positive = metric correctly detects worse reports\n")
print(corr_table.to_string())

out_csv = OUT_DIR / "rexval_correlation_table.csv"
corr_table.to_csv(out_csv)
print(f"\nCorrelation table saved: {out_csv}")

# ---------------------------------------------------------------------------
# LaTeX table — single table* with two tabulars
# ---------------------------------------------------------------------------
CAND_HEADER = {"radgraph": "RadGraph", "bertscore": "BERTScore",
               "s_emb": "SembScore", "bleu": "BLEU"}

# (candidate list, include top-span "Correlation with Significant Error Counts" row)
TABLE_SPLITS = [
    (["radgraph", "bertscore"], True),
    (["s_emb",    "bleu"],      False),
]

def latex_fmt(val, lo, hi):
    return f"${val:.2f}\\;[{lo:.2f},{hi:.2f}]$"

def _col_top2(values):
    """Return the two largest distinct values (higher = better)."""
    unique = []
    for v in sorted(values, reverse=True):
        if not any(np.isclose(v, u) for u in unique):
            unique.append(v)
        if len(unique) == 2:
            break
    return unique

def latex_fmt_highlight(val, lo, hi, rank):
    """rank 0 = best (green), 1 = second-best (grey), else plain."""
    s = latex_fmt(val, lo, hi)
    if rank == 0:
        return f"\\cellcolor{{green!25}}{s}"
    if rank == 1:
        return f"\\cellcolor{{gray!25}}{s}"
    return s

LATEX_LABELS = {
    "ICARE_AVG":        r"\textbf{ICARE}$_{\textbf{AVG}}$",
    "ICARE_ALLQUES":    r"\textbf{ICARE}$_{\textbf{ALLQ}}$",
    "ICARE_SEQ":        r"\textbf{ICARE}$_{\textbf{SEQ}}$",
    "ICARE_OPUS46":     r"\textbf{ICARE}$_{\textbf{OPUS}}$",
    "ICARE_SONNET46":   r"\textbf{ICARE}$_{\textbf{SONNET}}$",
    "ICARE_GPT54":      r"\textbf{ICARE}$_{\textbf{GPT54}}$",
    "ICARE_PREDEFINED": r"\textbf{ICARE}$_{\textbf{PRE}}$",
    "CRIMSON":          r"CRIMSON*",
}

def build_tabular(cands, include_top_header):
    n_data = len(cands) * 2
    col_spec = "l" + "c" * n_data
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    if include_top_header:
        lines.append(f" & \\multicolumn{{{n_data}}}{{c}}{{\\textbf{{Correlation with Significant Error Counts}}}} \\\\")
        lines.append(f"\\cmidrule(lr){{2-{n_data + 1}}}")

    # Candidate group headers
    grp_headers = [""]
    for cand in cands:
        grp_headers.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{CAND_HEADER[cand]}}}}}")
    lines.append(" & ".join(grp_headers) + r" \\")

    # Cmidrules under each group
    cmidrules = []
    for ci in range(len(cands)):
        start = 2 + ci * 2
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start + 1}}}")
    lines.append(" ".join(cmidrules))

    # Sub-header: Metric | Kendall τ | Pearson r | ...
    sub_headers = [r"\textbf{Metric}"]
    for _ in cands:
        sub_headers += [r"Kendall $\tau$", r"Pearson $r$"]
    lines.append(" & ".join(sub_headers) + r" \\")
    lines.append(r"\midrule")

    # Pre-compute per-column top-2 values for green/grey highlighting
    col_top2_tau = {
        cand: _col_top2([corr_results[lbl][cand]["tau"] for lbl, _ in CORR_METRICS])
        for cand in cands
    }
    col_top2_r = {
        cand: _col_top2([corr_results[lbl][cand]["r"] for lbl, _ in CORR_METRICS])
        for cand in cands
    }

    def _rank(val, top2):
        if top2 and np.isclose(val, top2[0]):
            return 0
        if len(top2) > 1 and np.isclose(val, top2[1]):
            return 1
        return -1

    # Data rows
    prev_is_icare = False
    for label, _ in CORR_METRICS:
        is_icare = label.startswith("ICARE")
        if is_icare and not prev_is_icare:
            lines.append(r"\midrule")
        prev_is_icare = is_icare

        display_label = LATEX_LABELS.get(label, label)
        lines.append(display_label)
        for ci, cand in enumerate(cands):
            c = corr_results[label][cand]
            tau_str = latex_fmt_highlight(
                c["tau"], c["tau_lo"], c["tau_hi"], _rank(c["tau"], col_top2_tau[cand]))
            r_str   = latex_fmt_highlight(
                c["r"],   c["r_lo"],   c["r_hi"],   _rank(c["r"],   col_top2_r[cand]))
            suffix = r" \\" if ci == len(cands) - 1 else ""
            lines.append(f" & {tau_str} & {r_str}{suffix}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

def build_full_latex_table():
    lines = []
    lines.append(r"% Requires \usepackage[table]{xcolor} in the main document preamble.")
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Kendall $\tau$ and Pearson $r$ (95\% CI) between automatic metrics and radiologist-derived"
        "\n"
        r"clinically significant error counts ($n{=}50$ per column). Columns refer to different candidate"
        "\n"
        r"reports on RexVal, each chosen to optimize a specific metric. Positive values indicate the metric"
        "\n"
        r"correctly detects worse reports. *CRIMSON results are averaged across 5 runs.}"
    )
    lines.append(r"\label{tab:rexval_correlation}")
    for idx, (cands, include_top) in enumerate(TABLE_SPLITS):
        lines.append(build_tabular(cands, include_top))
        if idx < len(TABLE_SPLITS) - 1:
            lines.append("")
            lines.append(r"\vspace{6pt}")
            lines.append("")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

latex_str = build_full_latex_table()
out_tex = OUT_DIR / "rexval_correlation_table.tex"
out_tex.write_text(latex_str)
print(f"LaTeX table saved:      {out_tex}")
print()
print(latex_str)

# ===========================================================================
# Figure 1: ICARE disagreement vs clinically significant errors (scatter)
# ===========================================================================
print("\nGenerating Figure 1: ICARE vs errors scatter plot...")

ICARE_BLUE  = "#1565C0"
GRAY        = "#555555"

fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, cand in zip(axes1, CANDIDATES):
    sub = merged[merged["origin"] == cand].copy()
    sub["dis_avg_pct"] = 100 - sub["ap_avg"]   # disagreement in % units
    ax.scatter(sub["dis_avg_pct"], sub["mean_clin_sig_errors"],
               alpha=0.7, edgecolors="k", linewidths=0.4, s=55, color="steelblue")
    m, b = np.polyfit(sub["dis_avg_pct"], sub["mean_clin_sig_errors"], 1)
    xs = np.linspace(sub["dis_avg_pct"].min(), sub["dis_avg_pct"].max(), 100)
    ax.plot(xs, m * xs + b, "r--", linewidth=1.5)
    ax.set_xlabel("ICARE disagreement (100 − AVG %)", fontsize=13, fontweight="bold")
    ax.set_title(CAND_LABELS[cand], fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

axes1[0].set_ylabel("Mean clin. sig. errors", fontsize=14, fontweight="bold")
fig1.suptitle(
    "ICARE disagreement vs Radiologist Clinically Significant Errors (n=50 per panel)",
    y=1.02, fontsize=15, fontweight="bold")
fig1.tight_layout()
out1 = OUT_DIR / "icare_vs_errors_scatter.png"
fig1.savefig(out1, dpi=600, bbox_inches="tight")
print(f"Figure 1 saved: {out1}")
plt.close()

# ===========================================================================
# Forest plots: preference alignment (top-1 agreement with radiologists)
# ===========================================================================

# ---------------------------------------------------------------------------
# Per-rater top-1: candidate with fewest errors per (study, rater)
# ---------------------------------------------------------------------------
rater_top1_by_rater = {}
for rater in RATERS:
    rt = (
        per_rater_err[per_rater_err["rater_index"] == rater]
        .copy()
        .assign(_rank=lambda df: df.groupby("study_number")["errors"].rank(
            ascending=True, method="min"))
    )
    rt1 = (rt[rt["_rank"] == 1]
           .drop_duplicates("study_number")[["study_number", "origin"]]
           .rename(columns={"origin": "rater_winner"}))
    rater_top1_by_rater[rater] = rt1

# Inter-rater pairwise top-1 agreement (ceiling)
ir_agrees = []
rater_list = list(RATERS)
for i, r1 in enumerate(rater_list):
    for r2 in rater_list[i+1:]:
        df_ir = rater_top1_by_rater[r1].merge(
            rater_top1_by_rater[r2].rename(columns={"rater_winner": "rater_winner_2"}),
            on="study_number")
        ir_agrees.append((df_ir["rater_winner"] == df_ir["rater_winner_2"]).mean() * 100)
inter_rater_pct = np.mean(ir_agrees)
print(f"\nInter-rater pairwise top-1 agreement: {inter_rater_pct:.1f}%")

# ---------------------------------------------------------------------------
# Metric top-1 selection helper
# ---------------------------------------------------------------------------
def get_metric_top1(df_merged, col, ascending):
    """Return DataFrame[study_number, metric_winner] — the top-1 candidate per study."""
    ranked = df_merged.copy()
    ranked["_r"] = ranked.groupby("study_number")[col].rank(
        ascending=ascending, method="min")
    top1 = (ranked[ranked["_r"] == 1]
            .drop_duplicates("study_number")[["study_number", "origin"]]
            .rename(columns={"origin": "metric_winner"}))
    return top1

# (label, score_col, ascending) — ascending=True means lower score = better
FOREST_METRICS = [
    ("ICARE_AVG ◄",       "ap_avg",    False),
    ("ICARE_ALLQUES",     "ap_allques", False),
    ("ICARE_SEQ",         "ap_seq",    False),
    ("ICARE_OPUS46",      "ap_opus",   False),
    ("ICARE_SONNET46",    "ap_sonnet", False),
    ("ICARE_GPT54",       "ap_gpt54",  False),
    ("ICARE_PREDEFINED",  "ap_pred",   False),
    ("CRIMSON",           "crimson",    False),
    ("GREEN",             "green",      False),
    ("AlignScore",        "alignscore", False),
    ("BERTScore",         "bertscore",  False),
    ("SembScore",          "semb",      False),
    ("RadGraph",          "radgraph",  False),
    ("BLEU",              "bleu",      False),
    ("RadCliQ-v1",        "radcliq",   True),   # lower = fewer errors = better
]
N_STUDIES = 50

# ---------------------------------------------------------------------------
# Figure 2: Per-rater preference alignment
# Bootstrap across studies (n=50) for stable CIs
# ---------------------------------------------------------------------------
print("\nPer-rater alignment (bootstrap across studies):")
# rater_top1_by_rater doesn't change across metrics/bootstraps — index once.
rater_series_by_rater = {
    r: rater_top1_by_rater[r].set_index("study_number")["rater_winner"] for r in RATERS
}
forest_per_rater = []
for label, col, asc in FOREST_METRICS:
    rater_rng = RNG_SEQ if label == "ICARE_SEQ" else RNG
    metric_top1 = get_metric_top1(merged, col, asc)
    per_rater_pcts = []
    for rater in RATERS:
        df_r = rater_top1_by_rater[rater].merge(metric_top1, on="study_number")
        pct  = (df_r["rater_winner"] == df_r["metric_winner"]).mean() * 100
        per_rater_pcts.append(pct)

    mean_pct = np.mean(per_rater_pcts)

    # Precompute common study set once — intersection across all raters and metric
    m1_idx = metric_top1.set_index("study_number")
    all_sets = [set(rater_top1_by_rater[r]["study_number"]) for r in RATERS] + [set(m1_idx.index)]
    common_arr = np.array(sorted(set.intersection(*all_sets)))

    # Align to common_arr once per metric — plain numpy arrays for fast bootstrap indexing
    metric_arr = m1_idx.loc[common_arr, "metric_winner"].values
    rater_arrs = {r: rater_series_by_rater[r].loc[common_arr].values for r in RATERS}

    # Bootstrap across studies — resample positionally from common_arr
    boot_means = []
    for _ in range(N_BOOT):
        pos = rater_rng.choice(len(common_arr), len(common_arr), replace=True)
        boot_per_rater = []
        for rater in RATERS:
            hits = rater_arrs[rater][pos] == metric_arr[pos]
            boot_per_rater.append(hits.mean() * 100)
        boot_means.append(np.mean(boot_per_rater))

    lo, hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    forest_per_rater.append(dict(label=label, pcts=per_rater_pcts,
                                  mean=mean_pct, lo=lo, hi=hi))
    print(f"  {label:22s}: {mean_pct:.1f}%  [{lo:.1f}, {hi:.1f}]  "
          f"raters={[round(p, 1) for p in per_rater_pcts]}")

# Draw Figure 2
RATER_MARKERS = ["o", "s", "D", "^", "v", "P"]
RATER_COLORS  = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]  # Wong colorblind-safe palette

n_fm   = len(forest_per_rater)
y_pos  = np.arange(n_fm)[::-1]
jitter = np.linspace(-0.25, 0.25, len(RATERS))

fig2, ax2 = plt.subplots(figsize=(13, 9))

for row, y in zip(forest_per_rater, y_pos):
    is_avg   = row["label"] == "ICARE_AVG ◄"
    is_icare = row["label"].startswith("ICARE")
    color    = "#1565C0" if is_icare else "#555555"
    lw       = 2.0 if is_avg else 1.4

    for ri, pct in enumerate(row["pcts"]):
        ax2.plot(pct, y + jitter[ri], marker=RATER_MARKERS[ri],
                 color=RATER_COLORS[ri], ms=5, alpha=0.8, zorder=3)

    ax2.plot([row["lo"], row["hi"]], [y, y], color=color, lw=lw,
             solid_capstyle="round", zorder=4)
    ax2.plot([row["lo"], row["hi"]], [y, y], color=color, marker="|",
             ms=7, mew=lw, zorder=4)
    ax2.plot(row["mean"], y, marker="*" if is_avg else "o",
             color=color, ms=11 if is_avg else 8, zorder=5)

    ax2.text(-1, y, row["label"], ha="right", va="center", fontsize=12,
             color=color if is_icare else "black", fontweight="bold")
    ax2.text(max(row["hi"], row["mean"]) + 1.5, y,
             f"{row['mean']:.1f}%  [{row['lo']:.1f}, {row['hi']:.1f}]",
             ha="left", va="center", fontsize=11, color=color, fontweight="bold")

# Reference lines — labels in header band
ax2.axvline(25, color="dimgray", linestyle="--", linewidth=1.2, zorder=2)
ax2.axvline(inter_rater_pct, color="crimson", linestyle=":", linewidth=1.4, zorder=2)
ax2.text(25.5, n_fm + 0.1, "Chance (25%)", va="bottom", fontsize=12, fontweight="bold", color="dimgray")
ax2.text(inter_rater_pct + 0.5, n_fm + 0.1,
         f"Inter-rater ({inter_rater_pct:.0f}%)", va="bottom", ha="left",
         fontsize=12, fontweight="bold", color="crimson")

# Separator between ICARE and baseline rows
n_icare = sum(1 for r in forest_per_rater if r["label"].startswith("ICARE"))
ax2.axhline(y_pos[n_icare - 1] - 0.5, color="lightgray", linewidth=0.8)

# Legend
rater_handles = [
    plt.Line2D([0], [0], marker=RATER_MARKERS[i], color=RATER_COLORS[i],
               ms=6, ls="", label=f"Rater {RATERS[i]}")
    for i in range(len(RATERS))
]
rater_handles += [plt.Line2D([0], [0], marker="o", color="#555555",
                              ms=7, ls="-", lw=1.4, label="Mean [95% CI]")]
ax2.legend(handles=rater_handles, fontsize=11, loc="upper left",
           bbox_to_anchor=(0.01, 0.98), framealpha=0.9)

ax2.set_xlim(0, 100)
ax2.set_ylim(-0.8, n_fm + 0.4)
ax2.set_xlabel("Alignment with individual radiologist preference (%) — top-1 candidate",
               fontsize=13, fontweight="bold")
ax2.set_title("Per-rater preference alignment — all metrics\n"
              "Dots = individual raters  |  Bar = mean ± 95% CI across raters",
              fontsize=13)
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.tick_params(axis="x", labelsize=12)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.grid(axis="x", alpha=0.25, zorder=0)
plt.tight_layout()
out2 = OUT_DIR / "rexval_forest_plot_per_rater.png"
fig2.savefig(out2, dpi=600, bbox_inches="tight")
print(f"\nFigure 2 saved: {out2}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Decisive-consensus alignment (≥4-of-6 raters agree on top-1)
# ---------------------------------------------------------------------------
def build_decisive_consensus(threshold):
    vote_counts = (
        pd.concat([rt1.assign(rater=r) for r, rt1 in rater_top1_by_rater.items()])
        .groupby(["study_number", "rater_winner"])
        .size()
        .reset_index(name="votes")
    )
    idx      = vote_counts.groupby("study_number")["votes"].idxmax()
    winners  = vote_counts.loc[idx, ["study_number", "rater_winner", "votes"]].reset_index(drop=True)
    decisive = winners[winners["votes"] >= threshold].reset_index(drop=True)
    return decisive[["study_number", "rater_winner"]], len(decisive)

THRESHOLD = 4
decisive_df, N_DECISIVE = build_decisive_consensus(THRESHOLD)
print(f"\nDecisive cases (≥{THRESHOLD}/6 raters agree on top-1): n={N_DECISIVE}")

def consensus_alignment_ci(metric_top1_df, dec_df, n_boot=N_BOOT, rng=None):
    rng = RNG if rng is None else rng
    df = dec_df.merge(metric_top1_df, on="study_number")
    hits = (df["rater_winner"] == df["metric_winner"]).values.astype(float)
    pct  = hits.mean() * 100
    boots = [rng.choice(hits, len(hits), replace=True).mean() * 100
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pct, lo, hi

print(f"\nDecisive-consensus alignment (≥{THRESHOLD}/6, n={N_DECISIVE}):")
forest_consensus = []
for label, col, asc in FOREST_METRICS:
    metric_top1 = get_metric_top1(merged, col, asc)
    pct, lo, hi = consensus_alignment_ci(
        metric_top1, decisive_df, rng=(RNG_SEQ if label == "ICARE_SEQ" else None))
    forest_consensus.append(dict(label=label, pct=pct, lo=lo, hi=hi))
    print(f"  {label:22s}: {pct:.1f}%  [{lo:.1f}, {hi:.1f}]")

# Inter-rater agreement against consensus (ceiling for Fig 3)
ir_cons_agrees = []
for rater in RATERS:
    rt1 = rater_top1_by_rater[rater].rename(columns={"rater_winner": "metric_winner"})
    pct, _, _ = consensus_alignment_ci(rt1, decisive_df)
    ir_cons_agrees.append(pct)
ir_cons_pct = np.mean(ir_cons_agrees)
print(f"\nInter-rater vs consensus (mean per rater): {ir_cons_pct:.1f}%")

# Draw Figure 3
fig3, ax3 = plt.subplots(figsize=(13, 8))
n_fd  = len(forest_consensus)
y_pos = np.arange(n_fd)[::-1]

for row, y in zip(forest_consensus, y_pos):
    is_avg   = row["label"] == "ICARE_AVG ◄"
    is_icare = row["label"].startswith("ICARE")
    color    = "#1565C0" if is_icare else "#555555"
    lw       = 2.0 if is_avg else 1.4
    ms       = 9   if is_avg else 7

    ax3.plot([row["lo"], row["hi"]], [y, y], color=color, lw=lw,
             solid_capstyle="round", zorder=3)
    ax3.plot([row["lo"], row["hi"]], [y, y], color=color, marker="|",
             ms=ms * 0.8, mew=lw, zorder=3)
    ax3.plot(row["pct"], y, "o", color=color, ms=ms, zorder=4)

    ax3.text(-1, y, row["label"], ha="right", va="center", fontsize=12,
             color=color if is_icare else "black", fontweight="bold")
    ax3.text(max(row["hi"], row["pct"]) + 1.5, y,
             f"{row['pct']:.1f}%  [{row['lo']:.1f}, {row['hi']:.1f}]",
             ha="left", va="center", fontsize=11, color=color, fontweight="bold")

ax3.axvline(25, color="dimgray", linestyle="--", linewidth=1.2, zorder=2,
            label="Chance (25%)")
ax3.axvline(ir_cons_pct, color="crimson", linestyle=":", linewidth=1.4, zorder=2)
ax3.text(ir_cons_pct + 0.5, n_fd + 0.1,
         f"Inter-rater ({ir_cons_pct:.0f}%)", va="bottom", ha="left",
         fontsize=12, fontweight="bold", color="crimson")
ax3.text(25.5, n_fd + 0.1, "Chance (25%)", va="bottom", fontsize=12, fontweight="bold", color="dimgray")

n_icare = sum(1 for r in forest_consensus if r["label"].startswith("ICARE"))
ax3.axhline(y_pos[n_icare - 1] - 0.5, color="lightgray", linewidth=0.8)

ax3.set_xlim(0, 120)
ax3.set_ylim(-0.8, n_fd + 0.4)
ax3.set_xlabel(
    f"Alignment with ≥{THRESHOLD}-of-6 radiologist consensus  (n = {N_DECISIVE} studies)",
    fontsize=13, fontweight="bold")
ax3.set_title("Sample-level alignment: decisive consensus only",
              fontsize=13, fontweight="bold")
ax3.set_yticks([])
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.tick_params(axis="x", labelsize=12)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.grid(axis="x", alpha=0.25, zorder=0)
plt.tight_layout()
out3 = OUT_DIR / "rexval_forest_plot_decisive_consensus.png"
fig3.savefig(out3, dpi=600, bbox_inches="tight")
print(f"\nFigure 3 saved: {out3}")
plt.close()

print("\nDone. Outputs:")
print(f"  {out_csv}")
print(f"  {out1}")
print(f"  {out2}")
print(f"  {out3}")



# ============================================================
# CSV EXPORT — ReXVal forest plots
# Add these blocks at the END of plot_rexval_correlation.py
# (after the existing Figure 2 and Figure 3 code)
# All variables are already computed in the script.
# ============================================================

import pandas as pd

# ────────────────────────────────────────────────────────────
# EXPORT A: panel_rexval_forest_per_rater.csv
# Source: forest_per_rater  (computed just before Figure 2)
# Each row = one metric
# Columns: metric, mean, ci_lo, ci_hi,
#          rater_0 … rater_5  (positional, matching RATERS list),
#          inter_rater_pct, n_studies
# ────────────────────────────────────────────────────────────

rows_rx_forest = []
for row in forest_per_rater:
    d = {
        "metric":          row["label"],
        "mean":            round(row["mean"], 4),
        "ci_lo":           round(row["lo"],   4),
        "ci_hi":           round(row["hi"],   4),
        "inter_rater_pct": round(inter_rater_pct, 1),
        "n_studies":       N_STUDIES,
    }
    for ri, pct in enumerate(row["pcts"]):
        d[f"rater_{RATERS[ri]}"] = round(pct, 4)   # rater_0 … rater_5
    rows_rx_forest.append(d)

df_rx_forest = pd.DataFrame(rows_rx_forest)
df_rx_forest.to_csv(OUT_DIR / "panel_rexval_forest_per_rater.csv", index=False)
print("Saved panel_rexval_forest_per_rater.csv")
print(df_rx_forest.to_string(index=False))


# ────────────────────────────────────────────────────────────
# EXPORT B: panel_rexval_forest_decisive.csv
# Source: forest_consensus  (computed just before Figure 3)
# Each row = one metric
# Columns: metric, pct, ci_lo, ci_hi, n_decisive, threshold,
#          inter_rater_consensus_pct
# ────────────────────────────────────────────────────────────

rows_rx_dec = []
for row in forest_consensus:
    rows_rx_dec.append({
        "metric":                   row["label"],
        "pct":                      round(row["pct"], 4),
        "ci_lo":                    round(row["lo"],  4),
        "ci_hi":                    round(row["hi"],  4),
        "n_decisive":               N_DECISIVE,
        "threshold":                THRESHOLD,
        "inter_rater_consensus_pct": round(ir_cons_pct, 1),
    })

df_rx_dec = pd.DataFrame(rows_rx_dec)
df_rx_dec.to_csv(OUT_DIR / "panel_rexval_forest_decisive.csv", index=False)
print("\nSaved panel_rexval_forest_decisive.csv")
print(df_rx_dec.to_string(index=False))


# ────────────────────────────────────────────────────────────
# EXPORT C: panel_rexval_correlation.csv
# Source: corr_results  (the Kendall/Pearson table)
# One row per (metric, candidate_type)
# Columns: metric, candidate, kendall_tau, kendall_lo, kendall_hi,
#          pearson_r, pearson_lo, pearson_hi
# ────────────────────────────────────────────────────────────

rows_rx_corr = []
for label, _ in CORR_METRICS:
    for cand in CANDIDATES:
        c = corr_results[label][cand]
        rows_rx_corr.append({
            "metric":      label,
            "candidate":   CAND_LABELS[cand],
            "kendall_tau": round(c["tau"],    4),
            "kendall_lo":  round(c["tau_lo"], 4),
            "kendall_hi":  round(c["tau_hi"], 4),
            "pearson_r":   round(c["r"],      4),
            "pearson_lo":  round(c["r_lo"],   4),
            "pearson_hi":  round(c["r_hi"],   4),
        })

df_rx_corr = pd.DataFrame(rows_rx_corr)
df_rx_corr.to_csv(OUT_DIR / "panel_rexval_correlation.csv", index=False)
print("\nSaved panel_rexval_correlation.csv")
print(df_rx_corr.to_string(index=False))


# ────────────────────────────────────────────────────────────
# EXPORT D: panel_rexval_scatter.csv
# Source: merged DataFrame  (used for Figure 1 scatter)
# Columns: study_number, origin, dis_avg_pct, mean_clin_sig_errors
# Useful if you want to recreate the scatter panel in the combined figure
# ────────────────────────────────────────────────────────────

scatter_df = merged[["study_number", "origin",
                      "ap_avg", "mean_clin_sig_errors"]].copy()
scatter_df["dis_avg_pct"] = 100 - scatter_df["ap_avg"]
scatter_df = scatter_df.drop(columns=["ap_avg"])
scatter_df.to_csv(OUT_DIR / "panel_rexval_scatter.csv", index=False)
print("\nSaved panel_rexval_scatter.csv")
print(scatter_df.head(8).to_string(index=False))