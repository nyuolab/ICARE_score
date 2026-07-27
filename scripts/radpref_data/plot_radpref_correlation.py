"""
RadPref analysis — three figures in one script:

  Figure 1: Correlation plot (Kendall τ / Pearson r vs radiologist preferences)
  Figure 2: Per-rater preference alignment forest plot
  Figure 3: Decisive-consensus alignment forest plot

All baselines except RaTEScore are loaded from produced output files.
RaTEScore retains the hard-coded paper value (no produced output).

Usage:
    cd ICARE_score
    python scripts/radpref_data/plot_radpref_correlation.py
"""

from typing import Tuple

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/gpfs/data/oermannlab/users/rd3571")
ICARE_SHUFFLED_OUT   = BASE / "ICARE_score/outputs/radpref/eval_seed_123/shuffled_ans_choices_data"
ICARE_ALLQUES_OUT    = BASE / "ICARE_score/outputs/radpref/eval_seed_123_allques/shuffled_ans_choices_data"
ICARE_TOPK20_OUT     = BASE / "ICARE_score/outputs/radpref/eval_seed_123_topk20/shuffled_ans_choices_data"
ICARE_RAD_PROMPT_OUT = BASE / "ICARE_score/outputs/radpref/eval_seed_123_prompt_radiology/shuffled_ans_choices_data"
ICARE_GEN_PROMPT_OUT = BASE / "ICARE_score/outputs/radpref/eval_seed_123_prompt_generic/shuffled_ans_choices_data"
ICARE_SEQ_OUT         = BASE / "ICARE_score/outputs/radpref/eval_seed_123_sequential_b10/shuffled_ans_choices_data"
ICARE_PREDEFINED_OUT = BASE / "ICARE_score/outputs/radpref/predefined/eval_seed_123/mcqa_eval"
BASELINES_DIR        = BASE / "ICARE_score/outputs/radpref/baselines"
RADPREF_CSV          = BASE / "cxr_report_datasets/radpref/radpref_icare.csv"
ANN_DIR              = BASE / "CRIMSON/RadPref/radiologist_annotations"
OUT_DIR              = BASE / "ICARE_score/outputs/radpref"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT      = 2000
RNG         = np.random.default_rng(42)
# Isolated stream for the new ICARE_seq variant so its bootstrap draws never
# shift the shared RNG's position and perturb already-reported metrics' CIs.
RNG_SEQ     = np.random.default_rng(42)
N           = 100   # cases per candidate (rows 0–99 = C1, rows 100–199 = C2)

# ---------------------------------------------------------------------------
# Load radiologist ratings
# ---------------------------------------------------------------------------
icare_csv = pd.read_csv(RADPREF_CSV)
case_ids  = icare_csv[icare_csv["candidate"] == "C1"]["case_id"].values

rating_diffs = []
for uid in [1, 2, 3]:
    ann   = json.load(open(ANN_DIR / f"user_{uid}.json"))["annotations"]
    diffs = [ann[cid]["C1"]["rating"] - ann[cid]["C2"]["rating"] for cid in case_ids]
    rating_diffs.append(np.array(diffs))

avg_rating_diff = np.mean(rating_diffs, axis=0)

# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def corr_pair(x, y):
    tau = kendalltau(x, y)[0]
    r   = pearsonr(x, y)[0]
    return tau, r

def bootstrap_ci(x, y, stat_fn, n=N_BOOT, rng=None):
    rng   = RNG if rng is None else rng
    idx   = np.arange(len(x))
    boots = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(stat_fn(x[s], y[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return lo, hi

def compute_metric_corrs(score_diff, compute_ci=False, rng=None):
    """Kendall tau and Pearson r vs each rater (+ averaged). Optionally bootstrap CIs."""
    kendall_vals, pearson_vals = [], []
    kendall_cis,  pearson_cis  = [], []
    for rd in rating_diffs + [avg_rating_diff]:
        tau, r = corr_pair(score_diff, rd)
        kendall_vals.append(tau)
        pearson_vals.append(r)
        if compute_ci:
            tau_lo, tau_hi = bootstrap_ci(score_diff, rd, lambda a, b: kendalltau(a, b)[0], rng=rng)
            r_lo,   r_hi   = bootstrap_ci(score_diff, rd, lambda a, b: pearsonr(a, b)[0], rng=rng)
            kendall_cis.append((tau_lo, tau_hi))
            pearson_cis.append((r_lo,   r_hi))
    result = {"kendall": kendall_vals, "pearson": pearson_vals}
    if compute_ci:
        result["kendall_ci"] = kendall_cis
        result["pearson_ci"] = pearson_cis
    return result

# ---------------------------------------------------------------------------
# Load ICARE (shuffled) — gt, gen, and averaged
# ---------------------------------------------------------------------------
gt_df  = pd.read_csv(ICARE_SHUFFLED_OUT / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
gen_df = pd.read_csv(ICARE_SHUFFLED_OUT / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
combined = gt_df.join(gen_df, how="left", lsuffix="_gt", rsuffix="_gen")
combined["icare"] = combined[["Agreement_Percentage_gt", "Agreement_Percentage_gen"]].mean(axis=1)

icare_gt_diff  = combined.loc[range(0, N), "Agreement_Percentage_gt"].values  - combined.loc[range(N, 2*N), "Agreement_Percentage_gt"].values
icare_gen_diff = combined.loc[range(0, N), "Agreement_Percentage_gen"].values - combined.loc[range(N, 2*N), "Agreement_Percentage_gen"].values
icare_diff     = combined.loc[range(0, N), "icare"].values                    - combined.loc[range(N, 2*N), "icare"].values

def load_icare_shuffled_combined(shuffled_out: Path) -> pd.DataFrame:
    gt_df  = pd.read_csv(shuffled_out / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    gen_df = pd.read_csv(shuffled_out / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    comb = gt_df.join(gen_df, how="left", lsuffix="_gt", rsuffix="_gen")
    comb["icare"] = comb[["Agreement_Percentage_gt", "Agreement_Percentage_gen"]].mean(axis=1)
    return comb


def load_icare_shuffled_diff(shuffled_out: Path) -> np.ndarray:
    comb = load_icare_shuffled_combined(shuffled_out)
    return comb.loc[range(0, N), "icare"].values - comb.loc[range(N, 2*N), "icare"].values


def load_icare_shuffled_c1_c2(shuffled_out: Path) -> Tuple[np.ndarray, np.ndarray]:
    comb = load_icare_shuffled_combined(shuffled_out)
    return (
        comb.loc[range(0, N), "icare"].values,
        comb.loc[range(N, 2*N), "icare"].values,
    )

icare_allques_diff = load_icare_shuffled_diff(ICARE_ALLQUES_OUT) if ICARE_ALLQUES_OUT.is_dir() else None
if icare_allques_diff is None:
    print(f"SKIP: {ICARE_ALLQUES_OUT} not found — omitting ICARE (all Q)")
icare_topk20_diff = load_icare_shuffled_diff(ICARE_TOPK20_OUT) if ICARE_TOPK20_OUT.is_dir() else None
if icare_topk20_diff is None:
    print(f"SKIP: {ICARE_TOPK20_OUT} not found — omitting ICARE (topk20)")
icare_rad_prompt_diff = load_icare_shuffled_diff(ICARE_RAD_PROMPT_OUT)
icare_gen_prompt_diff = load_icare_shuffled_diff(ICARE_GEN_PROMPT_OUT)

# ---------------------------------------------------------------------------
# Load ICARE (shuffled, sequential MCQ generation) — gt, gen, and averaged
# ---------------------------------------------------------------------------
gt_df_seq  = pd.read_csv(ICARE_SEQ_OUT / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
gen_df_seq = pd.read_csv(ICARE_SEQ_OUT / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
combined_seq = gt_df_seq.join(gen_df_seq, how="left", lsuffix="_gt", rsuffix="_gen")
combined_seq["icare_seq"] = combined_seq[["Agreement_Percentage_gt", "Agreement_Percentage_gen"]].mean(axis=1)

icare_seq_diff = combined_seq.loc[range(0, N), "icare_seq"].values - combined_seq.loc[range(N, 2*N), "icare_seq"].values

# ---------------------------------------------------------------------------
# Load ICARE predefined (single Agreement_Percentage, no gt/gen split)
# ---------------------------------------------------------------------------
pred_df         = pd.read_csv(ICARE_PREDEFINED_OUT / "mcq_eval_report_level_stats.csv").set_index("Report_ID")
icare_pred_diff = (pred_df.loc[range(0, N), "Agreement_Percentage"].values
                   - pred_df.loc[range(N, 2*N), "Agreement_Percentage"].values)

# ---------------------------------------------------------------------------
# Load CRIMSON baseline
# ---------------------------------------------------------------------------
with open(BASELINES_DIR / "crimson_results.json") as f:
    crimson_results = json.load(f)["results"]
crimson_scores = np.array([r["crimson_score"] for r in crimson_results])
crimson_diff   = crimson_scores[:N] - crimson_scores[N:]

# ---------------------------------------------------------------------------
# Load GREEN baseline
# ---------------------------------------------------------------------------
green_df   = pd.read_csv(BASELINES_DIR / "green/radpref_prepared_green_results.csv")
green_diff = green_df["green_score"].values[:N] - green_df["green_score"].values[N:]

# ---------------------------------------------------------------------------
# Load AlignScore baseline
# ---------------------------------------------------------------------------
alignscore_df   = pd.read_csv(BASELINES_DIR / "alignscore/radpref_prepared_alignscore_results.csv")
alignscore_diff = alignscore_df["alignscore"].values[:N] - alignscore_df["alignscore"].values[N:]

# ---------------------------------------------------------------------------
# Load RRG baselines (SembScore, RadGraph, BERTScore)
# ---------------------------------------------------------------------------
rrg_df         = pd.read_csv(BASELINES_DIR / "radpref/radpref_prepared_results.csv")
sembscore_diff  = rrg_df["semb_score"].values[:N]        - rrg_df["semb_score"].values[N:]
radgraph_diff  = rrg_df["radgraph_combined"].values[:N] - rrg_df["radgraph_combined"].values[N:]
bertscore_diff = rrg_df["bertscore"].values[:N]         - rrg_df["bertscore"].values[N:]

# ---------------------------------------------------------------------------
# Compute all metric correlations from data
# ---------------------------------------------------------------------------
print("Computing correlations from output files...")
computed = {
    "SembScore":        compute_metric_corrs(sembscore_diff,   compute_ci=True),
    "RadGraph":         compute_metric_corrs(radgraph_diff,    compute_ci=True),
    "BERTScore":        compute_metric_corrs(bertscore_diff,   compute_ci=True),
    "GREEN":            compute_metric_corrs(green_diff,       compute_ci=True),
    "AlignScore":       compute_metric_corrs(alignscore_diff,  compute_ci=True),
    "CRIMSON":          compute_metric_corrs(crimson_diff,     compute_ci=True),
    "ICARE_predefined": compute_metric_corrs(icare_pred_diff,  compute_ci=True),
    "ICARE\n(inline)":  compute_metric_corrs(icare_diff,       compute_ci=True),
    **({"ICARE\n(topk20)": compute_metric_corrs(icare_topk20_diff, compute_ci=True, rng=np.random.default_rng(45))}
       if icare_topk20_diff is not None else {}),
    "ICARE\n(rad file)": compute_metric_corrs(icare_rad_prompt_diff, compute_ci=True, rng=np.random.default_rng(43)),
    "ICARE\n(generic)": compute_metric_corrs(icare_gen_prompt_diff, compute_ci=True, rng=np.random.default_rng(44)),
    **({"ICARE\n(all Q)": compute_metric_corrs(icare_allques_diff, compute_ci=True, rng=np.random.default_rng(46))}
       if icare_allques_diff is not None else {}),
    "ICARE_seq":        compute_metric_corrs(icare_seq_diff,   compute_ci=True, rng=RNG_SEQ),
}

rater_labels = ["Rater 1", "Rater 2", "Rater 3", "Averaged"]
for name, vals in computed.items():
    print(f"\n{name}:")
    for i, lab in enumerate(rater_labels):
        ci_str = ""
        if "kendall_ci" in vals:
            ci_str = (f"  K_CI=[{vals['kendall_ci'][i][0]:.2f},{vals['kendall_ci'][i][1]:.2f}]"
                      f"  P_CI=[{vals['pearson_ci'][i][0]:.2f},{vals['pearson_ci'][i][1]:.2f}]")
        print(f"  {lab}: K={vals['kendall'][i]:.2f}  P={vals['pearson'][i]:.2f}{ci_str}")

# Inter-rater correlations with bootstrap CIs
pairs       = [(0, 1), (0, 2), (1, 2)]
pair_labels = ["R1 vs R2", "R1 vs R3", "R2 vs R3"]
ir_kendall, ir_pearson = [], []
ir_kendall_ci, ir_pearson_ci = [], []
for i, j in pairs:
    tau, r = corr_pair(rating_diffs[i], rating_diffs[j])
    ir_kendall.append(tau)
    ir_pearson.append(r)
    tau_lo, tau_hi = bootstrap_ci(rating_diffs[i], rating_diffs[j], lambda a, b: kendalltau(a, b)[0])
    r_lo,   r_hi   = bootstrap_ci(rating_diffs[i], rating_diffs[j], lambda a, b: pearsonr(a, b)[0])
    ir_kendall_ci.append((tau_lo, tau_hi))
    ir_pearson_ci.append((r_lo,   r_hi))

print("\nInter-rater:")
for lab, tau, r, kci, pci in zip(pair_labels, ir_kendall, ir_pearson, ir_kendall_ci, ir_pearson_ci):
    print(f"  {lab}: K={tau:.2f} [{kci[0]:.2f},{kci[1]:.2f}]  P={r:.2f} [{pci[0]:.2f},{pci[1]:.2f}]")

# ---------------------------------------------------------------------------
# Metric table — RaTEScore hard-coded (no produced output files)
# ---------------------------------------------------------------------------
paper = {
    "SembScore": computed["SembScore"],
    "RadGraph":  computed["RadGraph"],
    "BERTScore": computed["BERTScore"],
    # hard-coded from CRIMSON paper (no produced output files)
    "RaTEScore": {
        "kendall": [0.51, 0.53, 0.54, 0.53],
        "pearson": [0.64, 0.65, 0.66, 0.68],
    },
    "GREEN":               computed["GREEN"],
    "AlignScore":          computed["AlignScore"],
    "CRIMSON":             computed["CRIMSON"],
    "ICARE\n(predefined)": computed["ICARE_predefined"],
    "ICARE\n(inline)":    computed["ICARE\n(inline)"],
    **({"ICARE\n(topk20)": computed["ICARE\n(topk20)"]} if icare_topk20_diff is not None else {}),
    "ICARE\n(rad file)":  computed["ICARE\n(rad file)"],
    "ICARE\n(generic)":   computed["ICARE\n(generic)"],
    **({"ICARE\n(all Q)": computed["ICARE\n(all Q)"]} if icare_allques_diff is not None else {}),
    "ICARE\n(sequential)": computed["ICARE_seq"],
}

interrater = {
    "kendall":    ir_kendall,
    "pearson":    ir_pearson,
    "kendall_ci": ir_kendall_ci,
    "pearson_ci": ir_pearson_ci,
}

# ---------------------------------------------------------------------------
# Plot — all per-rater bars; averaged score label only (metrics + inter-rater)
# ---------------------------------------------------------------------------
DARK_RED = "#8B1A1A"
IR_COLOR = "#707070"
AVG_IDX  = 3   # "Averaged" entry in rater_labels

MARKERS  = ["o", "s", "D", "*"]
OFFSETS  = [-0.27, -0.09, 0.09, 0.27]
SIZES    = [14, 14, 14, 22]
ZORDERS  = [2, 2, 2, 4]

IR_PAIR_MARKERS = ["o", "D", "^"]
IR_PAIR_OFFSETS = [-0.27, -0.09, 0.09]
IR_AVG_OFFSET   = 0.27
IR_PAIR_SIZE    = 14
IR_AVG_SIZE     = 22

FS_TITLE     = 32
FS_YLABEL    = 28
FS_TICK      = 24
FS_XLABEL    = 22
FS_VALUE_AVG = 20
FS_LEGEND    = 22

metric_names = list(paper.keys())
n_metrics    = len(metric_names)

YLIMS = {"kendall": (0.0, 0.98), "pearson": (0.0, 1.02)}

fig, axes = plt.subplots(2, 1, figsize=(28, 14), sharex=True)
fig.suptitle("Correlation with Radiologist Preferences (RadPref)",
             fontsize=FS_TITLE, fontweight="bold", y=0.98)
fig.subplots_adjust(hspace=0.16, bottom=0.36, top=0.94)

for corr_key, ax in zip(["kendall", "pearson"], axes):
    ylabel = "Kendall $\\tau_b$" if corr_key == "kendall" else "Pearson $r$"
    ax.set_ylabel(ylabel, fontsize=FS_YLABEL, fontweight="bold", labelpad=14)
    ylo, yhi = YLIMS[corr_key]
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(np.arange(0.0, yhi + 0.01, 0.2))
    ax.tick_params(axis="y", labelsize=FS_TICK, width=1.5, length=6)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.35, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for mi, mname in enumerate(metric_names):
        vals = paper[mname][corr_key]
        ci   = paper[mname].get(f"{corr_key}_ci", None)

        for ri, (val, marker, offset, ms, zord) in enumerate(
                zip(vals, MARKERS, OFFSETS, SIZES, ZORDERS)):
            xpos = mi + offset
            yerr = None
            y_top = val
            if ci is not None:
                lo, hi = ci[ri]
                yerr  = [[val - lo], [hi - val]]
                y_top = hi

            ax.errorbar(
                xpos, val,
                yerr=yerr,
                fmt=marker,
                color=DARK_RED,
                markersize=ms,
                markeredgewidth=1.0,
                markeredgecolor="white" if ri == AVG_IDX else DARK_RED,
                capsize=7, capthick=2.0, elinewidth=2.0,
                zorder=zord,
            )

            if ri == AVG_IDX:
                ax.text(xpos, y_top + 0.018, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=FS_VALUE_AVG,
                        fontweight="bold", color=DARK_RED, zorder=5)

    # Vertical divider before Inter-rater
    ax.axvline(n_metrics - 0.5, color="black", linewidth=1.5)

    # Inter-rater column — 3 pairwise bars + 1 averaged (mean) bar
    ir_x     = n_metrics
    ir_vals  = interrater[corr_key]
    ir_cis   = interrater[f"{corr_key}_ci"]
    for val, ci, marker, offset in zip(
            ir_vals, ir_cis, IR_PAIR_MARKERS, IR_PAIR_OFFSETS):
        lo, hi = ci
        ax.errorbar(ir_x + offset, val,
                    yerr=[[val - lo], [hi - val]],
                    fmt=marker,
                    color=IR_COLOR,
                    markersize=IR_PAIR_SIZE, markeredgewidth=1.0,
                    capsize=7, capthick=2.0, elinewidth=2.0,
                    zorder=3)

    ir_mean = np.mean(ir_vals)
    ir_lo   = np.mean([c[0] for c in ir_cis])
    ir_hi   = np.mean([c[1] for c in ir_cis])
    ax.errorbar(
        ir_x + IR_AVG_OFFSET, ir_mean,
        yerr=[[ir_mean - ir_lo], [ir_hi - ir_mean]],
        fmt="*",
        color=IR_COLOR,
        markersize=IR_AVG_SIZE,
        markeredgewidth=1.0,
        markeredgecolor="white",
        capsize=7, capthick=2.0, elinewidth=2.0,
        zorder=4,
    )
    ax.text(ir_x + IR_AVG_OFFSET, ir_hi + 0.018, f"{ir_mean:.2f}",
            ha="center", va="bottom", fontsize=FS_VALUE_AVG,
            fontweight="bold", color=IR_COLOR, zorder=5)

# X-axis tick labels
ax_xticks  = list(range(n_metrics)) + [n_metrics]
ax_xlabels = metric_names + ["Inter-rater"]
for ax in axes:
    ax.set_xticks(ax_xticks)
    ax.set_xticklabels(ax_xlabels, fontsize=FS_XLABEL, fontweight="bold",
                       rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=FS_XLABEL, width=1.5, length=6)
    ax.set_xlim(-0.65, n_metrics + 0.75)

legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DARK_RED,
               markersize=16, label="Rater 1"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=DARK_RED,
               markersize=16, label="Rater 2"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=DARK_RED,
               markersize=16, label="Rater 3"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=DARK_RED,
               markersize=22, label="Averaged"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=IR_COLOR,
               markersize=16, label="Rater 1 vs Rater 2"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=IR_COLOR,
               markersize=16, label="Rater 1 vs Rater 3"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=IR_COLOR,
               markersize=16, label="Rater 2 vs Rater 3"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=IR_COLOR,
               markersize=22, markeredgecolor=IR_COLOR,
               label="Inter-rater (averaged)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=4,
    framealpha=1.0,
    edgecolor="#aaaaaa",
    handlelength=1.8,
    handletextpad=1.0,
    columnspacing=2.2,
    labelspacing=1.4,
    borderpad=0.8,
    prop={"size": FS_LEGEND, "weight": "bold"},
)

out_png = OUT_DIR / "radpref_correlation_with_icare.png"
out_pdf = OUT_DIR / "radpref_correlation_with_icare.pdf"
plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.3)
plt.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.3)
print(f"\nFigure 1 saved: {out_png}")
print(f"Figure 1 saved: {out_pdf}")
plt.close()

# ===========================================================================
# Figure 2 & 3 — Forest plots (preference alignment)
# ===========================================================================

# ---------------------------------------------------------------------------
# Raw C1 / C2 score arrays (reuse already-loaded data)
# ---------------------------------------------------------------------------
icare_inline_C1, icare_inline_C2 = load_icare_shuffled_c1_c2(ICARE_SHUFFLED_OUT)
icare_rad_C1, icare_rad_C2 = load_icare_shuffled_c1_c2(ICARE_RAD_PROMPT_OUT)
icare_gen_C1, icare_gen_C2 = load_icare_shuffled_c1_c2(ICARE_GEN_PROMPT_OUT)
if icare_allques_diff is not None:
    icare_allques_C1, icare_allques_C2 = load_icare_shuffled_c1_c2(ICARE_ALLQUES_OUT)
if icare_topk20_diff is not None:
    icare_topk20_C1, icare_topk20_C2 = load_icare_shuffled_c1_c2(ICARE_TOPK20_OUT)
icare_seq_C1  = combined_seq.loc[range(0, N), "icare_seq"].values
icare_seq_C2  = combined_seq.loc[range(N, 2*N), "icare_seq"].values
icare_pred_C1 = pred_df.loc[range(0, N), "Agreement_Percentage"].values
icare_pred_C2 = pred_df.loc[range(N, 2*N), "Agreement_Percentage"].values
crimson_C1, crimson_C2 = crimson_scores[:N], crimson_scores[N:]
green_C1  = green_df["green_score"].values[:N];         green_C2  = green_df["green_score"].values[N:]
as_C1     = alignscore_df["alignscore"].values[:N];     as_C2     = alignscore_df["alignscore"].values[N:]
bert_C1   = rrg_df["bertscore"].values[:N];             bert_C2   = rrg_df["bertscore"].values[N:]
semb_C1   = rrg_df["semb_score"].values[:N];      semb_C2   = rrg_df["semb_score"].values[N:]
rg_C1     = rrg_df["radgraph_combined"].values[:N]; rg_C2   = rrg_df["radgraph_combined"].values[N:]
bleu_C1   = rrg_df["bleu_score"].values[:N];      bleu_C2   = rrg_df["bleu_score"].values[N:]
rcq_C1    = -rrg_df["RadCliQ-v1"].values[:N];     rcq_C2    = -rrg_df["RadCliQ-v1"].values[N:]  # negated: RadCliQ is an error metric (higher=worse)

# (label, C1_scores, C2_scores) — higher score = better
INLINE_LABEL = "ICARE (inline) ◄"
FOREST_METRICS = [
    (INLINE_LABEL,        icare_inline_C1, icare_inline_C2),
    *([("ICARE (topk20)", icare_topk20_C1, icare_topk20_C2)] if icare_topk20_diff is not None else []),
    ("ICARE (rad file)",  icare_rad_C1,    icare_rad_C2),
    ("ICARE (generic)",   icare_gen_C1,    icare_gen_C2),
    *([("ICARE (all Q)", icare_allques_C1, icare_allques_C2)] if icare_allques_diff is not None else []),
    ("ICARE (sequential)", icare_seq_C1,   icare_seq_C2),
    ("ICARE (predefined)", icare_pred_C1,  icare_pred_C2),
    ("CRIMSON",           crimson_C1,    crimson_C2),
    ("GREEN",             green_C1,      green_C2),
    ("AlignScore",        as_C1,         as_C2),
    ("BERTScore",         bert_C1,       bert_C2),
    ("SembScore",         semb_C1,       semb_C2),
    ("RadGraph",          rg_C1,         rg_C2),
    ("BLEU",              bleu_C1,       bleu_C2),
    ("RadCliQ-v1",        rcq_C1,        rcq_C2),
]

# Per-rater binary preference: +1 = C1 preferred, -1 = C2 preferred, 0 = tie
rater_prefs = {uid: np.sign(rd) for uid, rd in zip([1, 2, 3], rating_diffs)}

def metric_pref(C1, C2):
    return np.sign(C1 - C2)

def alignment_pct(m_pref, r_pref):
    mask = r_pref != 0
    if mask.sum() == 0:
        return np.nan
    return (m_pref[mask] == r_pref[mask]).mean() * 100

# Inter-rater pairwise top-1 agreement (ceiling line)
ir_agrees = []
rater_ids = [1, 2, 3]
for i, r1 in enumerate(rater_ids):
    for r2 in rater_ids[i+1:]:
        p1, p2 = rater_prefs[r1], rater_prefs[r2]
        mask = (p1 != 0) & (p2 != 0)
        if mask.sum() > 0:
            ir_agrees.append((p1[mask] == p2[mask]).mean() * 100)
inter_rater_pct = np.mean(ir_agrees)
print(f"\nInter-rater pairwise top-1 agreement: {inter_rater_pct:.1f}%")

# ---------------------------------------------------------------------------
# Figure 2: Per-rater preference alignment
# ---------------------------------------------------------------------------
RATER_MARKERS = ["o", "s", "D"]
RATER_COLORS  = ["#0072B2", "#D55E00", "#009E73"]  # Wong colorblind-safe palette
print("\nPer-rater alignment (bootstrap across cases):")
forest_per_rater = []
for label, C1, C2 in FOREST_METRICS:
    rng = RNG_SEQ if label == "ICARE (sequential)" else RNG
    mp = metric_pref(C1, C2)
    per_rater_pcts = [alignment_pct(mp, rater_prefs[uid]) for uid in rater_ids]
    mean = np.mean(per_rater_pcts)
    # Bootstrap across cases (n=100) for stable CIs
    boot_means = []
    for _ in range(N_BOOT):
        idx = rng.choice(N, N, replace=True)
        boot_per_rater = []
        for uid in rater_ids:
            rp = rater_prefs[uid][idx]
            mask = rp != 0
            if mask.sum() > 0:
                boot_per_rater.append((mp[idx][mask] == rp[mask]).mean() * 100)
        boot_means.append(np.mean(boot_per_rater))
    lo, hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    forest_per_rater.append(dict(label=label, pcts=per_rater_pcts, mean=mean, lo=lo, hi=hi))
    print(f"  {label:22s}: {mean:.1f}%  [{lo:.1f}, {hi:.1f}]  "
          f"raters={[round(p, 1) for p in per_rater_pcts]}")

fig2, ax2 = plt.subplots(figsize=(12, 10))
n_fm   = len(forest_per_rater)
y_pos  = np.arange(n_fm)[::-1]
jitter = np.linspace(-0.20, 0.20, len(rater_ids))

for row, y in zip(forest_per_rater, y_pos):
    is_avg   = row["label"] == INLINE_LABEL
    is_icare = row["label"].startswith("ICARE")
    color    = "#1565C0" if is_icare else "#555555"
    lw       = 2.0 if is_avg else 1.4

    for r_i, pct in enumerate(row["pcts"]):
        ax2.plot(pct, y + jitter[r_i], marker=RATER_MARKERS[r_i],
                 color=RATER_COLORS[r_i], ms=6, alpha=0.8, zorder=3)

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

ax2.axvline(50, color="dimgray", linestyle="--", linewidth=1.2, zorder=2)
ax2.axvline(inter_rater_pct, color="crimson", linestyle=":", linewidth=1.4, zorder=2)
# Labels sit in the header band above all metric rows
ax2.text(50.5, n_fm + 0.1, "Chance (50%)", va="bottom", fontsize=12, fontweight="bold", color="dimgray")
ax2.text(inter_rater_pct + 0.5, n_fm + 0.1,
         f"Inter-rater ({inter_rater_pct:.0f}%)", va="bottom", ha="left",
         fontsize=12, fontweight="bold", color="crimson")

n_icare = sum(1 for r in forest_per_rater if r["label"].startswith("ICARE"))
ax2.axhline(y_pos[n_icare - 1] - 0.5, color="lightgray", linewidth=0.8)

rater_handles = [
    plt.Line2D([0],[0], marker=RATER_MARKERS[i], color=RATER_COLORS[i],
               ms=7, ls="", label=f"Rater {rater_ids[i]}")
    for i in range(len(rater_ids))
]
rater_handles += [plt.Line2D([0],[0], marker="o", color="#555555",
                              ms=8, ls="-", lw=1.5, label="Mean [95% CI]")]
ax2.legend(handles=rater_handles, fontsize=11, loc="center left",
           bbox_to_anchor=(0.01, 0.25), framealpha=0.9)
ax2.set_xlim(0, 115)
ax2.set_ylim(-0.8, n_fm + 0.4)   # header band above top row for reference labels
ax2.set_xlabel("Alignment with individual radiologist preference (%)",
               fontsize=13, fontweight="bold")
ax2.set_title("Per-rater preference alignment — all metrics\n"
              "Dots = individual raters  |  Bar = mean ± 95% CI across raters", fontsize=13)
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.tick_params(axis="x", labelsize=12)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.grid(axis="x", alpha=0.25, zorder=0)
plt.tight_layout()
out2 = OUT_DIR / "forest_plot_per_rater.png"
plt.savefig(out2, dpi=600, bbox_inches="tight")
print(f"\nFigure 2 saved: {out2}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Decisive-consensus alignment
# With 3 raters + 2 candidates, unanimous (3/3) is the strictest threshold;
# fall back to majority (≥2/3) if fewer than 25 decisive cases.
# ---------------------------------------------------------------------------
def build_consensus(rater_prefs_dict, threshold):
    decisive_idx, consensus_sign = [], []
    for i in range(N):
        prefs = [rater_prefs_dict[uid][i] for uid in rater_ids
                 if rater_prefs_dict[uid][i] != 0]
        if not prefs:
            continue
        c1_votes = sum(1 for p in prefs if p > 0)
        c2_votes = sum(1 for p in prefs if p < 0)
        if c1_votes >= threshold:
            decisive_idx.append(i)
            consensus_sign.append(1)
        elif c2_votes >= threshold:
            decisive_idx.append(i)
            consensus_sign.append(-1)
    return np.array(decisive_idx), np.array(consensus_sign)

dec_idx, dec_cons = build_consensus(rater_prefs, 2)
THRESHOLD  = 2
N_DECISIVE = len(dec_idx)
print(f"\nDecisive cases (≥2/3 majority): n={N_DECISIVE}")

def consensus_alignment_ci(m_pref, n_boot=N_BOOT, rng=None):
    rng   = RNG if rng is None else rng
    hits  = (m_pref[dec_idx] == dec_cons).astype(float)
    pct   = hits.mean() * 100
    boots = [rng.choice(hits, len(hits), replace=True).mean() * 100
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pct, lo, hi

print(f"\nDecisive-consensus alignment (≥{THRESHOLD}/3, n={N_DECISIVE}):")
forest_data = []
for label, C1, C2 in FOREST_METRICS:
    mp = metric_pref(C1, C2)
    pct, lo, hi = consensus_alignment_ci(mp, rng=(RNG_SEQ if label == "ICARE (sequential)" else None))
    forest_data.append(dict(label=label, pct=pct, lo=lo, hi=hi))
    print(f"  {label:22s}: {pct:.1f}%  [{lo:.1f}, {hi:.1f}]")

fig3, ax3 = plt.subplots(figsize=(10, 8))
n_fd  = len(forest_data)
y_pos = np.arange(n_fd)[::-1]

for row, y in zip(forest_data, y_pos):
    is_avg   = row["label"] == INLINE_LABEL
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

ax3.axvline(50, color="dimgray", linestyle="--", linewidth=1.2, zorder=2,
            label="Chance (50%)")
n_icare = sum(1 for r in forest_data if r["label"].startswith("ICARE"))
ax3.axhline(y_pos[n_icare - 1] - 0.5, color="lightgray", linewidth=0.8)

ax3.set_xlim(0, 115)
ax3.set_ylim(-0.8, n_fd - 0.2)
ax3.set_xlabel(f"Alignment with ≥{THRESHOLD}-of-3 radiologist consensus  (n = {N_DECISIVE})",
               fontsize=13, fontweight="bold")
ax3.set_title("Sample-level alignment: decisive consensus only",
              fontsize=13, fontweight="bold")
ax3.set_yticks([])
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.tick_params(axis="x", labelsize=12)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.legend(fontsize=12, loc="lower right")
ax3.grid(axis="x", alpha=0.25, zorder=0)
plt.tight_layout()
out3 = OUT_DIR / "forest_plot_decisive_consensus.png"
plt.savefig(out3, dpi=600, bbox_inches="tight")
print(f"\nFigure 3 saved: {out3}")



# ============================================================
# CSV EXPORT — RadPref forest plots
# Add these blocks at the END of plot_radpref_correlation.py
# (after the existing Figure 2 and Figure 3 code)
# All variables are already computed in the script.
# ============================================================

import pandas as pd

# ────────────────────────────────────────────────────────────
# EXPORT A: panel_radpref_forest_per_rater.csv
# Source: forest_per_rater  (computed just before Figure 2)
# Columns: metric, mean, ci_lo, ci_hi, rater_1, rater_2, rater_3,
#          inter_rater_pct
# ────────────────────────────────────────────────────────────

rows_rp_forest = []
for row in forest_per_rater:
    d = {
        "metric":          row["label"],
        "mean":            round(row["mean"], 4),
        "ci_lo":           round(row["lo"],   4),
        "ci_hi":           round(row["hi"],   4),
        "inter_rater_pct": round(inter_rater_pct, 1),
    }
    for ri, pct in enumerate(row["pcts"]):
        d[f"rater_{ri+1}"] = round(pct, 4)   # rater_1, rater_2, rater_3
    rows_rp_forest.append(d)

df_rp_forest = pd.DataFrame(rows_rp_forest)
df_rp_forest.to_csv(OUT_DIR / "panel_radpref_forest_per_rater.csv", index=False)
print("Saved panel_radpref_forest_per_rater.csv")
print(df_rp_forest.to_string(index=False))


# ────────────────────────────────────────────────────────────
# EXPORT B: panel_radpref_forest_decisive.csv
# Source: forest_data  (computed just before Figure 3)
# Columns: metric, pct, ci_lo, ci_hi, n_decisive, threshold
# ────────────────────────────────────────────────────────────

rows_rp_dec = []
for row in forest_data:
    rows_rp_dec.append({
        "metric":     row["label"],
        "pct":        round(row["pct"], 4),
        "ci_lo":      round(row["lo"],  4),
        "ci_hi":      round(row["hi"],  4),
        "n_decisive": N_DECISIVE,
        "threshold":  THRESHOLD,
    })

df_rp_dec = pd.DataFrame(rows_rp_dec)
df_rp_dec.to_csv(OUT_DIR / "panel_radpref_forest_decisive.csv", index=False)
print("\nSaved panel_radpref_forest_decisive.csv")
print(df_rp_dec.to_string(index=False))


# ────────────────────────────────────────────────────────────
# EXPORT C: panel_radpref_correlation.csv
# Source: computed  (variable name in THIS script)
# RadPref has no candidate loop — correlations are per rater
# (rater_labels = ["Rater 1", "Rater 2", "Rater 3", "Averaged"])
# One row per (metric, rater)
# Columns: metric, rater, kendall_tau, kendall_lo, kendall_hi,
#          pearson_r, pearson_lo, pearson_hi
# Note: RaTEScore has no CIs (hard-coded), filled with nan
# ────────────────────────────────────────────────────────────

rater_labels_export = ["Rater 1", "Rater 2", "Rater 3", "Averaged"]

rows_rp_corr = []
for label in paper:
    clean_label = label.replace("\n", " ")   # "ICARE\n(predefined)" → "ICARE (predefined)"
    for ri, rater in enumerate(rater_labels_export):
        if label in computed and "kendall_ci" in computed[label]:
            c = computed[label]
            rows_rp_corr.append({
                "metric":      clean_label,
                "rater":       rater,
                "kendall_tau": round(c["kendall"][ri],       4),
                "kendall_lo":  round(c["kendall_ci"][ri][0], 4),
                "kendall_hi":  round(c["kendall_ci"][ri][1], 4),
                "pearson_r":   round(c["pearson"][ri],       4),
                "pearson_lo":  round(c["pearson_ci"][ri][0], 4),
                "pearson_hi":  round(c["pearson_ci"][ri][1], 4),
            })
        else:
            # RaTEScore: hard-coded point estimates, no CI
            rows_rp_corr.append({
                "metric":      clean_label,
                "rater":       rater,
                "kendall_tau": paper[label]["kendall"][ri],
                "kendall_lo":  float("nan"),
                "kendall_hi":  float("nan"),
                "pearson_r":   paper[label]["pearson"][ri],
                "pearson_lo":  float("nan"),
                "pearson_hi":  float("nan"),
            })

df_rp_corr = pd.DataFrame(rows_rp_corr)
df_rp_corr.to_csv(OUT_DIR / "panel_radpref_correlation.csv", index=False)
print("\nSaved panel_radpref_correlation.csv")
print(df_rp_corr.to_string(index=False))