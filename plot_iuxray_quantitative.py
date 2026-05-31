import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────
BASE = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray"

MODELS = {
    "mimic-cxr-findings-baseline":         "CheXpertPlus_MIMIC",
    "chexpert-mimic-cxr-findings-baseline": "CheXpertPlus_CheX_MIMIC",
    "maira-2":                              "MAIRA2",
}
MODEL_SEED = "model_seed_1"
EVAL_SEEDS = ["eval_seed_101", "eval_seed_123", "eval_seed_202", "eval_seed_456", "eval_seed_789"]
N_BOOT = 10000
RNG    = np.random.default_rng(42)

# ── Loaders ────────────────────────────────────────────────────────────────────
def load_baseline(model_dir):
    """Per-report traditional metrics (BLEU, BERTScore, SembScore, RadGraph, RadCliQ-v1)."""
    files = [f for f in glob.glob(f"{BASE}/baselines/{model_dir}/{MODEL_SEED}/{model_dir}/*_results.csv")
             if "_green_results" not in f]
    assert files, f"No baseline results found for {model_dir}"
    return pd.read_csv(files[0]).set_index("index")


def load_green(model_dir):
    """Per-report GREEN scores (falls back to green_old if green/ not yet populated)."""
    for folder in ("green", "green_old"):
        files = glob.glob(f"{BASE}/baselines/{model_dir}/{MODEL_SEED}/{folder}/*_green_results.csv")
        if files:
            df = pd.read_csv(files[0])
            return df.set_index(df.index)["green_score"]
    raise FileNotFoundError(f"No GREEN results found for {model_dir}")


def load_icare(model_dir, ref_type=None, predefined=False):
    """
    Per-report ICARE score averaged across all 5 eval seeds → indexed by Report_ID.

    For dynamic ICARE, ref_type is 'gen_reports_as_ref' or 'gt_reports_as_ref'.
    For predefined ICARE, set predefined=True.
    """
    frames = []
    for es in EVAL_SEEDS:
        if predefined:
            path = f"{BASE}/{model_dir}/{MODEL_SEED}/{es}/predefined/mcqa_eval/mcq_eval_report_level_stats.csv"
        else:
            path = f"{BASE}/{model_dir}/{MODEL_SEED}/{es}/orig_data/{ref_type}/mcqa_eval/mcq_eval_report_level_stats.csv"
        df = pd.read_csv(path)[["Report_ID", "Agreement_Percentage"]].set_index("Report_ID")
        frames.append(df.rename(columns={"Agreement_Percentage": es}))
    combined = pd.concat(frames, axis=1)
    # Average across 5 eval seeds; divide by 100 → 0–1 scale
    return combined.mean(axis=1) / 100.0


def bootstrap_ci(series, n_boot=N_BOOT):
    """Returns (mean, 95% CI half-width) via percentile bootstrap."""
    s = series.dropna().values
    boot_means = np.array([
        np.mean(RNG.choice(s, size=len(s), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(s)), (hi - lo) / 2


# ── Aggregate per model ────────────────────────────────────────────────────────
METRIC_ORDER = [
    "GREEN", "BLEU-2", "BERTScore", "SembScore", "RadGraph", "1/RadCliQ-v1",
    "ICARE-GT", "ICARE-GEN", "ICARE-AVG", "ICARE-PRE_DEF",
]

means = {}
cis   = {}

for model_dir, display_name in MODELS.items():
    bl  = load_baseline(model_dir)
    gr  = load_green(model_dir)

    icare_gen = load_icare(model_dir, ref_type="gen_reports_as_ref")
    icare_gt  = load_icare(model_dir, ref_type="gt_reports_as_ref")
    icare_pre = load_icare(model_dir, predefined=True)
    # ICARE-AVG: per-report mean of GEN and GT, aligned on Report_ID
    icare_avg = pd.concat([icare_gen, icare_gt], axis=1).mean(axis=1)

    n = len(bl)  # number of reports for this model

    metric_scores = {
        "GREEN":         gr.values[:n],
        "BLEU-2":        bl["bleu_score"].values,
        "BERTScore":     bl["bertscore"].values,
        "SembScore":     bl["semb_score"].values,
        "RadGraph":      bl["radgraph_combined"].values,
        "ICARE-GT":      icare_gt.reindex(bl.index).values,
        "ICARE-GEN":     icare_gen.reindex(bl.index).values,
        "ICARE-AVG":     icare_avg.reindex(bl.index).values,
        "ICARE-PRE_DEF": icare_pre.reindex(bl.index).values,
    }

    means[display_name] = {}
    cis[display_name]   = {}
    for metric, scores in metric_scores.items():
        m, c = bootstrap_ci(pd.Series(scores))
        means[display_name][metric] = m
        cis[display_name][metric]   = c

    # 1/RadCliQ-v1: bootstrap on 1/mean to avoid per-sample inversion blowup
    rv1 = bl["RadCliQ-v1"].dropna().values
    boot_inv = np.array([
        1.0 / np.mean(RNG.choice(rv1, size=len(rv1), replace=True))
        for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot_inv, [2.5, 97.5])
    means[display_name]["1/RadCliQ-v1"] = 1.0 / rv1.mean()
    cis[display_name]["1/RadCliQ-v1"]   = (hi - lo) / 2

    print(f"{display_name}: n={n} reports")
    for metric in METRIC_ORDER:
        m, c = means[display_name][metric], cis[display_name][metric]
        print(f"  {metric:<18} {m:.4f} ± {c:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
sns.set_context("talk")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(20, 7))

COLORS = ["#E69F00", "#56B4E9", "#009E73"]
display_names = list(MODELS.values())

n_metrics = len(METRIC_ORDER)
n_models  = len(MODELS)
bar_width = 0.27
x = np.arange(n_metrics)

for idx, display_name in enumerate(display_names):
    positions = x + bar_width * idx
    m_vals = [means[display_name][m] for m in METRIC_ORDER]
    c_vals = [cis[display_name][m]   for m in METRIC_ORDER]

    ax.bar(
        positions, m_vals,
        width=bar_width,
        label=display_name,
        color=COLORS[idx],
        edgecolor="black",
        linewidth=0.5,
        yerr=c_vals,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
        zorder=3,
    )

    for p, h, ci_val in zip(positions, m_vals, c_vals):
        ax.text(
            p,
            h + ci_val + 0.012,
            f"{h:.2f}",
            ha="center", va="bottom",
            fontsize=12, weight="bold",
            color=COLORS[idx],
        )

# X-axis
group_centers = x + bar_width * (n_models - 1) / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(METRIC_ORDER, rotation=30, ha="right", fontsize=14, weight="bold")

HIGHLIGHT = {"ICARE-GT", "ICARE-GEN", "ICARE-AVG"}
for tick_label in ax.get_xticklabels():
    if tick_label.get_text() in HIGHLIGHT:
        tick_label.set_color("#8B0000")

# Dashed separator before ICARE section
icare_start = METRIC_ORDER.index("ICARE-GT")
ax.axvline(x=group_centers[icare_start] - bar_width * 2.0, color="gray", linestyle="--", linewidth=2)

ax.set_xlabel("Evaluation Metric", fontsize=18, weight="bold", labelpad=10)
ax.set_ylabel("Score (95% CI)", fontsize=18, weight="bold", labelpad=4)
ax.tick_params(axis="y", labelsize=14)
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")
ax.set_title(
    "Quantitative Evaluation of RRG Models Across Metrics (IU Xray)",
    fontsize=20, fontweight="bold", pad=20,
)
ax.set_xlim(-0.15, n_metrics - 1 + bar_width * n_models + 0.1)
ax.set_ylim(bottom=0)

ax.legend(
    title="Model",
    title_fontsize=18,
    fontsize=16,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.48),
    ncol=3,
    frameon=True,
    fancybox=True,
    edgecolor="gray",
)

sns.despine()
plt.tight_layout()
fig.subplots_adjust(bottom=0.28)

plt.savefig("iuxray_metrics_quantitative.pdf", dpi=600, bbox_inches="tight")
plt.savefig("iuxray_metrics_quantitative.png", dpi=600, bbox_inches="tight")
print("\nSaved iuxray_metrics_quantitative.pdf / .png")
plt.show()
