import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/rexval"

BASELINES_CSV  = f"{BASE}/rexval_test_200/baselines/rexval_test_200/rexval_prepared_results.csv"
ORIGIN_MAP_CSV = f"{BASE}/rexval_test_200/baselines/rexval_prepared.csv"

ICARE_GEN_CSV  = f"{BASE}/rexval_test_200/eval_seed_123/orig_data/gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
ICARE_GT_CSV   = f"{BASE}/rexval_test_200/eval_seed_123/orig_data/gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv"
ICARE_PRE_CSV  = f"{BASE}/predefined/eval_seed_123/mcqa_eval/mcq_eval_report_level_stats.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
origins = pd.read_csv(ORIGIN_MAP_CSV)[["id", "origin"]]

baseline = (
    pd.read_csv(BASELINES_CSV)
    .merge(origins, left_on="study_id", right_on="id")
)

def load_icare(path):
    df = pd.read_csv(path).merge(origins, left_on="Report_ID", right_on="id")
    df["score"] = df["Agreement_Percentage"] / 100.0
    return df

icare_gen = load_icare(ICARE_GEN_CSV)
icare_gt  = load_icare(ICARE_GT_CSV)
icare_pre = load_icare(ICARE_PRE_CSV)

# ICARE-AVG: per-report average of GEN and GT scores
icare_avg = (
    icare_gen[["Report_ID", "origin", "score"]]
    .merge(icare_gt[["Report_ID", "score"]], on="Report_ID", suffixes=("_gen", "_gt"))
)
icare_avg["score"] = (icare_avg["score_gen"] + icare_avg["score_gt"]) / 2

# ── Aggregation: mean ± 95% CI per origin (percentile bootstrap) ──────────────
N_BOOT       = 10_000
_RNG         = np.random.default_rng(42)
ORIGIN_ORDER = ["bertscore", "bleu", "radgraph", "s_emb"]

def _boot_ci_half(x):
    x = np.asarray(x)
    boot = _RNG.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (hi - lo) / 2

def agg(df, col):
    g = df.groupby("origin")[col]
    mean = g.mean().reindex(ORIGIN_ORDER)
    ci   = g.apply(_boot_ci_half).reindex(ORIGIN_ORDER)
    return mean.values, ci.values

METRIC_SPECS = [
    # (display_label,  dataframe,   column)
    ("BLEU",          baseline,  "bleu_score"),
    ("BERTScore",     baseline,  "bertscore"),
    ("SembScore",     baseline,  "semb_score"),
    ("RadGraph",      baseline,  "radgraph_combined"),
    ("ICARE-GT",      icare_gt,  "score"),
    ("ICARE-GEN",     icare_gen, "score"),
    ("ICARE-AVG",     icare_avg, "score"),
    ("ICARE-PRE_DEF", icare_pre, "score"),
]

means_mat = []   # shape (n_metrics, n_origins)
cis_mat   = []
labels    = []

for label, df, col in METRIC_SPECS:
    m, c = agg(df, col)
    means_mat.append(m)
    cis_mat.append(c)
    labels.append(label)

means_mat = np.array(means_mat)   # (n_metrics, n_origins)
cis_mat   = np.array(cis_mat)

# ── Plot ───────────────────────────────────────────────────────────────────────
sns.set_context("talk")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(20, 7))

# Colorblind-friendly (Wong) palette
COLORS = {
    "bertscore": "#56B4E9",
    "bleu":      "#E69F00",
    "radgraph":  "#009E73",
    "s_emb":     "#CC79A7",
}
LEGEND_LABELS = {
    "bertscore": "BERTScore-sel",
    "bleu":      "BLEU-sel",
    "radgraph":  "RadGraph-sel",
    "s_emb":     "SembScore-sel",
}

n_metrics = len(labels)
n_origins = len(ORIGIN_ORDER)
bar_width = 0.20
x = np.arange(n_metrics)

for oi, origin in enumerate(ORIGIN_ORDER):
    positions = x + bar_width * oi
    means = means_mat[:, oi]
    cis   = cis_mat[:, oi]

    ax.bar(
        positions, means,
        width=bar_width,
        label=LEGEND_LABELS[origin],
        color=COLORS[origin],
        edgecolor="black",
        linewidth=0.5,
        yerr=cis,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
        zorder=3,
    )

    for p, h, ci_val in zip(positions, means, cis):
        ax.text(
            p + bar_width / 2,
            h + ci_val + 0.012,
            f"{h:.2f}",
            ha="center", va="bottom",
            fontsize=8, weight="bold",
            color=COLORS[origin],
        )

# X-axis ticks centered on each group
group_centers = x + bar_width * (n_origins - 1) / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=14, weight="bold")

# Highlight ICARE metrics in dark red
HIGHLIGHT = {"ICARE-GT", "ICARE-GEN", "ICARE-AVG"}
for tick_label in ax.get_xticklabels():
    if tick_label.get_text() in HIGHLIGHT:
        tick_label.set_color("#8B0000")

# Dashed separator before ICARE section
icare_start_idx = labels.index("ICARE-GT")
boundary_x = group_centers[icare_start_idx] - bar_width * 2.0
ax.axvline(x=boundary_x, color="gray", linestyle="--", linewidth=2)

# Axis formatting
ax.set_xlabel("Evaluation Metric", fontsize=18, weight="bold", labelpad=4)
ax.set_ylabel("Score (0–1 scale, 95% CI)", fontsize=18, weight="bold", labelpad=4)
ax.tick_params(axis="y", labelsize=14)
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")
ax.set_title(
    "Quantitative Evaluation on RexVal-200 Across Metrics",
    fontsize=20, fontweight="bold", pad=20,
)
ax.set_xlim(-0.15, n_metrics - 1 + bar_width * n_origins + 0.1)
ax.set_ylim(bottom=0)

# Legend below plot
ax.legend(
    title="Report type (selection metric)",
    title_fontsize=14,
    fontsize=13,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.34),
    ncol=4,
    frameon=True,
    fancybox=True,
    edgecolor="gray",
)

sns.despine()
plt.tight_layout()

plt.savefig("rexval_metrics_quantitative.pdf", dpi=600, bbox_inches="tight")
plt.savefig("rexval_metrics_quantitative.png", dpi=600, bbox_inches="tight")
print("Saved rexval_metrics_quantitative.pdf / .png")
plt.show()
