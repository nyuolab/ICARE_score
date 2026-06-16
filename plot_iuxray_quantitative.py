import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import logsumexp

# ── Config ─────────────────────────────────────────────────────────────────────
BASE = "/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray"

MODELS = {
    "mimic-cxr-findings-baseline":         "CheXpertPlus_MIMIC",
    "chexpert-mimic-cxr-findings-baseline": "CheXpertPlus_CheX_MIMIC",
    "maira-2":                              "MAIRA2",
}
MODEL_SEED = "model_seed_1"
EVAL_SEEDS = ["eval_seed_101", "eval_seed_123", "eval_seed_202", "eval_seed_456", "eval_seed_789"]
N_BOOT = 2000
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


def load_crimson(model_dir):
    """Per-report CRIMSON scores, indexed by study_id."""
    path = f"{BASE}/baselines/{model_dir}/{MODEL_SEED}/crimson/crimson_results.json"
    with open(path) as f:
        results = json.load(f)["results"]
    return pd.Series({r["id"]: r["crimson_score"] for r in results})


def load_alignscore(model_dir):
    """Per-report AlignScore scores, indexed by report index."""
    files = glob.glob(f"{BASE}/baselines/{model_dir}/{MODEL_SEED}/alignscore/*_alignscore_results.csv")
    assert files, f"No AlignScore results found for {model_dir}"
    return pd.read_csv(files[0]).set_index("index")["alignscore"]


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
    "BLEU-2", "BERTScore", "SembScore", "1/RadCliQ-v1", "RadGraph",
    "AlignScore", "GREEN", "CRIMSON",
    "ICARE-GT", "ICARE-GEN", "ICARE-AVG", "ICARE-PRE_DEF",
]

METRIC_LABELS = {
    "BLEU-2":        "BLEU-2 (2002)",
    "BERTScore":     "BertScore (2019)",
    "SembScore":     "SembScore (2020)",
    "1/RadCliQ-v1":  "1/RadCliQ-v1 (2022)",
    "RadGraph":      "RadGraph (2022)",
    "AlignScore":    "AlignScore (2023)",
    "GREEN":         "GREEN (2024)",
    "CRIMSON":       "Crimson (2026)",
    "ICARE-GT":      "ICARE-GT (2025)",
    "ICARE-GEN":     "ICARE-GEN (2025)",
    "ICARE-AVG":     "ICARE-AVG (2025)",
    "ICARE-PRE_DEF": "ICARE-Predefined (2025)",
}

means = {}
cis   = {}

# Store per-report scores for BT analysis
all_scores = {}

for model_dir, display_name in MODELS.items():
    bl  = load_baseline(model_dir)
    gr  = load_green(model_dir)
    as_ = load_alignscore(model_dir)
    cr  = load_crimson(model_dir)

    icare_gen = load_icare(model_dir, ref_type="gen_reports_as_ref")
    icare_gt  = load_icare(model_dir, ref_type="gt_reports_as_ref")
    icare_pre = load_icare(model_dir, predefined=True)
    # ICARE-AVG: per-report mean of GEN and GT, aligned on Report_ID
    icare_avg = pd.concat([icare_gen, icare_gt], axis=1).mean(axis=1)

    n = len(bl)  # number of reports for this model

    metric_scores = {
        "GREEN":         gr.values[:n],
        "AlignScore":    as_.reindex(bl.index).values,
        "CRIMSON":       cr.reindex(bl["study_id"]).values,
        "BLEU-2":        bl["bleu_score"].values,
        "BERTScore":     bl["bertscore"].values,
        "SembScore":     bl["semb_score"].values,
        "RadGraph":      bl["radgraph_combined"].values,
        "ICARE-GT":      icare_gt.reindex(bl.index).values,
        "ICARE-GEN":     icare_gen.reindex(bl.index).values,
        "ICARE-AVG":     icare_avg.reindex(bl.index).values,
        "ICARE-PRE_DEF": icare_pre.reindex(bl.index).values,
    }

    # Store per-report scores (with index = Report_ID) for BT fitting
    all_scores[display_name] = {
        k: pd.Series(v, index=bl.index)
        for k, v in metric_scores.items()
    }
    # Also store RadCliQ-v1 raw (lower-is-better)
    all_scores[display_name]["RadCliQ-v1_raw"] = bl["RadCliQ-v1"]

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

# ── Plot 1: Quantitative bar chart (unchanged) ─────────────────────────────────
TITLE_SIZE       = 25
LABEL_SIZE       = 20
TICK_SIZE        = 16
BAR_LABEL_SIZE   = 11
LEGEND_SIZE      = 16
LEGEND_TITLE_SIZE = 18

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
            fontsize=BAR_LABEL_SIZE, weight="bold",
            color=COLORS[idx],
        )

# X-axis
group_centers = x + bar_width * (n_models - 1) / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(
    [METRIC_LABELS[m] for m in METRIC_ORDER],
    rotation=30, ha="right", fontsize=TICK_SIZE, weight="bold",
)

ICARE_COLOR = "#1f77b4"
HIGHLIGHT = {"ICARE-GT", "ICARE-GEN", "ICARE-AVG", "ICARE-PRE_DEF"}
for tick_label, metric in zip(ax.get_xticklabels(), METRIC_ORDER):
    if metric in HIGHLIGHT:
        tick_label.set_color(ICARE_COLOR)

# Dashed separator before ICARE section
icare_start = METRIC_ORDER.index("ICARE-GT")
ax.axvline(x=group_centers[icare_start] - bar_width * 2.0, color="gray", linestyle="--", linewidth=2)

ax.set_xlabel("Evaluation Metric", fontsize=LABEL_SIZE, weight="bold", labelpad=5)
ax.set_ylabel("Score (95% CI)", fontsize=LABEL_SIZE, weight="bold", labelpad=22)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")
ax.set_title(
    "Quantitative Evaluation of RRG Models Across Metrics (IU Xray)",
    fontsize=TITLE_SIZE, fontweight="bold", pad=20,
)
ax.set_xlim(-0.15, n_metrics - 1 + bar_width * n_models + 0.1)
ax.set_ylim(bottom=0)

ax.legend(
    title="Model",
    title_fontsize=LEGEND_TITLE_SIZE,
    fontsize=LEGEND_SIZE,
    loc="upper right",
    ncol=3,
    frameon=True,
    fancybox=True,
    edgecolor="gray",
)

sns.despine()
plt.tight_layout()

plt.savefig("iuxray_metrics_quantitative.pdf", dpi=600, bbox_inches="tight")
plt.savefig("iuxray_metrics_quantitative.png", dpi=600, bbox_inches="tight")
print("\nSaved iuxray_metrics_quantitative.pdf / .png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ── Plot 2: Bradley–Terry ranking comparison across metrics (full IU-Xray) ───
# ══════════════════════════════════════════════════════════════════════════════

# ── Hardcoded clinician Davidson BT from colab (119 consensus samples) ────────
# Source: Section 4.5 of icare_human_eval_analysis_section4 colab
CLINICIAN_BT = {
    "CheXpertPlus_MIMIC":      {"Davidson score": 0.101619, "Normalized strength": 0.368020,
                                 "95% CI low": -0.224, "95% CI high": 0.451},
    "MAIRA2":                  {"Davidson score": -0.048256, "Normalized strength": 0.316797,
                                 "95% CI low": -0.386, "95% CI high": 0.259},
    "CheXpertPlus_CheX_MIMIC": {"Davidson score": -0.053363, "Normalized strength": 0.315183,
                                 "95% CI low": -0.373, "95% CI high": 0.256},
}

# ── Davidson BT on full IU-Xray test set ─────────────────────────────────────
ALL_MODELS_BT = list(MODELS.values())
K_BT          = len(ALL_MODELS_BT)
MODEL_IDX_BT  = {m: i for i, m in enumerate(ALL_MODELS_BT)}

def davidson_nll(params, a_idx, b_idx, outcome, K, l2=1e-4):
    theta = np.zeros(K)
    theta[:K-1] = params[:K-1]
    gamma = params[K-1]
    ta, tb = theta[a_idx], theta[b_idx]
    log_a   = ta
    log_b   = tb
    log_tie = gamma + 0.5 * (ta + tb)
    log_den = logsumexp(np.vstack([log_a, log_b, log_tie]), axis=0)
    log_p_a   = log_a   - log_den
    log_p_b   = log_b   - log_den
    log_p_tie = log_tie - log_den
    log_prob  = np.where(outcome == 1.0, log_p_a,
                np.where(outcome == 0.0, log_p_b, log_p_tie))
    return -np.sum(log_prob) + l2 * np.sum(params**2)


def build_pairwise_comparisons(metric_key, lower_is_better=False):
    """
    Build all pairwise (model_a, model_b, outcome) rows from full test set scores.
    Each report yields C(3,2)=3 pairs. Outcome: 1.0=a wins, 0.0=b wins, 0.5=tie.
    report_id is used as the cluster unit for bootstrap.
    """
    rows = []
    model_list = list(MODELS.values())
    model_dirs = list(MODELS.keys())

    # Collect per-report scores across all models; align on shared indices
    score_series = {}
    for mdir, mname in MODELS.items():
        if metric_key == "RadCliQ-v1_raw":
            s = all_scores[mname]["RadCliQ-v1_raw"].dropna()
        else:
            s = all_scores[mname][metric_key]
            s = pd.to_numeric(s, errors="coerce").dropna()
        score_series[mname] = s

    # Find common report indices
    common_idx = score_series[model_list[0]].index
    for mname in model_list[1:]:
        common_idx = common_idx.intersection(score_series[mname].index)

    for rid in common_idx:
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                ma, mb = model_list[i], model_list[j]
                sa = score_series[ma].loc[rid]
                sb = score_series[mb].loc[rid]
                if np.isnan(sa) or np.isnan(sb):
                    continue
                if lower_is_better:
                    if sa < sb:   outcome = 1.0
                    elif sa > sb: outcome = 0.0
                    else:         outcome = 0.5
                else:
                    if sa > sb:   outcome = 1.0
                    elif sa < sb: outcome = 0.0
                    else:         outcome = 0.5
                rows.append({"report_id": rid, "model1": ma, "model2": mb, "outcome": outcome})

    return pd.DataFrame(rows)


def fit_davidson_fullset(bt_df, source_name, n_boot=2000, seed=42):
    """
    Fit Davidson BT on full test set pairwise comparisons.
    Cluster bootstrap by report_id.
    """
    if len(bt_df) == 0:
        print(f"  Skipping {source_name}: no comparisons.")
        return None

    a_idx   = bt_df["model1"].map(MODEL_IDX_BT).to_numpy()
    b_idx   = bt_df["model2"].map(MODEL_IDX_BT).to_numpy()
    outcome = bt_df["outcome"].to_numpy()

    res = minimize(davidson_nll, np.zeros(K_BT),
                   args=(a_idx, b_idx, outcome, K_BT),
                   method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-10})

    theta = np.zeros(K_BT)
    theta[:K_BT-1] = res.x[:K_BT-1]
    theta -= theta.mean()

    # Cluster bootstrap by report_id
    rng   = np.random.default_rng(seed)
    boot  = {m: [] for m in ALL_MODELS_BT}
    rids  = bt_df["report_id"].unique()

    for _ in range(n_boot):
        sampled = rng.choice(rids, size=len(rids), replace=True)
        bdf = pd.concat(
            [bt_df[bt_df["report_id"] == r] for r in sampled],
            ignore_index=True
        )
        try:
            ba = bdf["model1"].map(MODEL_IDX_BT).to_numpy()
            bb = bdf["model2"].map(MODEL_IDX_BT).to_numpy()
            bo = bdf["outcome"].to_numpy()
            br = minimize(davidson_nll, np.zeros(K_BT),
                          args=(ba, bb, bo, K_BT),
                          method="L-BFGS-B",
                          options={"maxiter": 5000, "ftol": 1e-10})
            bt = np.zeros(K_BT)
            bt[:K_BT-1] = br.x[:K_BT-1]
            bt -= bt.mean()
            for m, s in zip(ALL_MODELS_BT, bt):
                boot[m].append(s)
        except Exception:
            continue

    ci_rows = []
    for m in ALL_MODELS_BT:
        vals = np.array(boot[m])
        lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 0 else (np.nan, np.nan)
        ci_rows.append({"Model": m, "95% CI low": lo, "95% CI high": hi})

    out = pd.DataFrame({
        "Source":          source_name,
        "Model":           ALL_MODELS_BT,
        "Davidson score":  theta,
    })
    out["Normalized strength"] = np.exp(out["Davidson score"]) / np.exp(out["Davidson score"]).sum()
    out = out.merge(pd.DataFrame(ci_rows), on="Model", how="left")
    out = out.sort_values("Davidson score", ascending=False).reset_index(drop=True)
    out.insert(1, "Rank", range(1, len(out) + 1))
    return out


# ── Metrics for BT analysis ───────────────────────────────────────────────────
BT_METRIC_SPECS = [
    ("BLEU-2",        "BLEU-2 (2002)",        False),
    ("BERTScore",     "BertScore (2019)",      False),
    ("SembScore",     "SembScore (2020)",      False),
    ("RadCliQ-v1_raw","1/RadCliQ-v1 (2022)",  True),
    ("RadGraph",      "RadGraph (2022)",       False),
    ("AlignScore",    "AlignScore (2023)",     False),
    ("GREEN",         "GREEN (2024)",          False),
    ("CRIMSON",       "Crimson (2026)",        False),
    ("ICARE-GT",      "ICARE-GT (2025)",       False),
    ("ICARE-GEN",     "ICARE-GEN (2025)",      False),
    ("ICARE-AVG",     "ICARE-AVG (2025)",      False),
    ("ICARE-PRE_DEF", "ICARE-Predefined (2025)", False),
]

print("\nFitting Davidson BT on full IU-Xray test set...")
bt_tables = []
for metric_key, metric_display, lower_is_better in BT_METRIC_SPECS:
    print(f"  {metric_display}...", end=" ", flush=True)
    comps = build_pairwise_comparisons(metric_key, lower_is_better)
    result = fit_davidson_fullset(comps, source_name=metric_display)
    if result is not None:
        bt_tables.append(result)
        top = result.iloc[0]["Model"]
        print(f"done → top: {top}")

# Add hardcoded clinician BT (first in the combined frame)
clin_rows = []
for m, vals in CLINICIAN_BT.items():
    clin_rows.append({
        "Source":              "Clinicians\n(119 samples)",
        "Rank":                sorted(CLINICIAN_BT.keys(),
                                      key=lambda k: -CLINICIAN_BT[k]["Davidson score"]).index(m) + 1,
        "Model":               m,
        "Davidson score":      vals["Davidson score"],
        "Normalized strength": vals["Normalized strength"],
        "95% CI low":          vals["95% CI low"],
        "95% CI high":         vals["95% CI high"],
    })
clin_df = pd.DataFrame(clin_rows)
combined_bt = pd.concat([clin_df] + bt_tables, ignore_index=True)

# Ranking summary
print("\nBT ranking summary (full test set):")
for grp, gdf in combined_bt.groupby("Source", sort=False):
    ranking = " > ".join(gdf.sort_values("Davidson score", ascending=False)["Model"])
    print(f"  {grp}: {ranking}")


# ── CI helper ─────────────────────────────────────────────────────────────────
def bt_ci_to_norm_strength_err(group_df):
    """Convert BT-score CI bounds through softmax to asymmetric normalized-strength error bars."""
    models       = group_df["Model"].tolist()
    point_scores = group_df["Davidson score"].values.copy()
    err_lo = np.zeros(len(models))
    err_hi = np.zeros(len(models))
    for i, row in group_df.reset_index(drop=True).iterrows():
        lo_s = row.get("95% CI low",  np.nan)
        hi_s = row.get("95% CI high", np.nan)
        if np.isnan(lo_s):
            continue
        point_norm = row["Normalized strength"]
        scores_lo = point_scores.copy()
        scores_lo[i] = lo_s
        norm_lo = np.exp(scores_lo) / np.exp(scores_lo).sum()
        scores_hi = point_scores.copy()
        scores_hi[i] = hi_s
        norm_hi = np.exp(scores_hi) / np.exp(scores_hi).sum()
        err_lo[i] = max(0.0, point_norm - norm_lo[i])
        err_hi[i] = max(0.0, norm_hi[i] - point_norm)
    return err_lo, err_hi


# ── Plot layout ───────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "CheXpertPlus_MIMIC":      "#E69F00",
    "CheXpertPlus_CheX_MIMIC": "#56B4E9",
    "MAIRA2":                  "#009E73",
}

# Source order: clinicians first, then metrics in METRIC_ORDER key order
SOURCE_ORDER = (
    ["Clinicians\n(119 samples)"]
    + [disp for _, disp, _ in BT_METRIC_SPECS
       if disp in combined_bt["Source"].unique()]
)

models    = ALL_MODELS_BT
n_sources = len(SOURCE_ORDER)
n_models  = len(models)
# Compress x spacing so 13 groups fit in the same canvas as 12-group quant plot
x         = np.arange(n_sources) * (n_metrics / n_sources)
width     = bar_width * (n_metrics / n_sources)
offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

sns.set_context("talk")
sns.set_style("whitegrid")
fig2, ax2 = plt.subplots(figsize=(20, 7))

for mi, model in enumerate(models):
    scores  = []
    ci_low  = []
    ci_high = []

    for source in SOURCE_ORDER:
        src_df = combined_bt[(combined_bt["Source"] == source) & (combined_bt["Model"] == model)]
        if len(src_df) == 0:
            scores.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
            continue

        scores.append(src_df["Normalized strength"].values[0])
        group_df = combined_bt[combined_bt["Source"] == source].copy()
        err_lo_arr, err_hi_arr = bt_ci_to_norm_strength_err(group_df)
        model_idx = group_df.reset_index(drop=True)["Model"].tolist().index(model)
        ci_low.append(err_lo_arr[model_idx])
        ci_high.append(err_hi_arr[model_idx])

    scores  = np.array(scores,  dtype=float)
    ci_low  = np.array(ci_low,  dtype=float)
    ci_high = np.array(ci_high, dtype=float)

    bars = ax2.bar(
        x + offsets[mi],
        scores,
        width=width,
        label=model,
        color=MODEL_COLORS.get(model, "#aaaaaa"),
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    ax2.errorbar(
        x + offsets[mi],
        scores,
        yerr=[ci_low, ci_high],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=4,
    )

    for bi, (bar, score) in enumerate(zip(bars, scores)):
        if not np.isnan(score):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ci_high[bi] if not np.isnan(ci_high[bi]) else 0) + 0.012,
                f"{score:.2f}",
                ha="center", va="bottom",
                fontsize=BAR_LABEL_SIZE, fontweight="bold",
                color=MODEL_COLORS.get(model, "#aaaaaa"),
            )

uniform = 1 / n_models
ax2.axhline(uniform, color="gray", linestyle="--", linewidth=1.2,
            zorder=2, label=f"Uniform (1/{n_models})")

# Highlight clinician column
ax2.axvspan(x[0] - width * 2, x[0] + width * 2, alpha=0.07, color="gray", zorder=0)

# Dashed separator before ICARE section (same logic as quant plot)
icare_bt_sources = [d for _, d, _ in BT_METRIC_SPECS if "ICARE" in d]
if icare_bt_sources:
    first_icare_idx = SOURCE_ORDER.index(icare_bt_sources[0])
    ax2.axvline(x=x[first_icare_idx] - width * 2, color="gray", linestyle="--", linewidth=1.5, zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(SOURCE_ORDER, fontsize=TICK_SIZE, fontweight="bold", rotation=30, ha="right")

# Color ICARE x-labels
for tick_lbl, src in zip(ax2.get_xticklabels(), SOURCE_ORDER):
    if "ICARE" in src:
        tick_lbl.set_color("#1f77b4")

ax2.set_xlabel("Evaluation Metric", fontsize=LABEL_SIZE, weight="bold", labelpad=5)
ax2.set_ylabel("BT Normalized Strength (95% CI)", fontsize=LABEL_SIZE, weight="bold", labelpad=22)
ax2.tick_params(axis="y", labelsize=TICK_SIZE)
for tick in ax2.get_yticklabels():
    tick.set_fontweight("bold")
ax2.set_title(
    "Bradley–Terry Model Rankings: Clinicians vs. Automatic Metrics (IU-Xray)",
    fontsize=TITLE_SIZE, fontweight="bold", pad=20,
)
ax2.set_xlim(x[0] - width * 2, x[-1] + width * 2)
ax2.set_ylim(0, max(0.6, combined_bt["Normalized strength"].max() + 0.14))

ax2.legend(
    title="Model",
    title_fontsize=LEGEND_TITLE_SIZE,
    fontsize=LEGEND_SIZE,
    loc="upper right",
    ncol=3,
    frameon=True,
    fancybox=True,
    edgecolor="gray",
)

ax2.grid(axis="y", alpha=0.3, zorder=0)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

sns.despine()
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# ── Save all results to CSV ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# 1. Quantitative means + CIs (one row per model × metric)
quant_rows = []
for model_name in display_names:
    for metric in METRIC_ORDER:
        quant_rows.append({
            "model":       model_name,
            "metric":      metric,
            "metric_label": METRIC_LABELS.get(metric, metric),
            "mean":        means[model_name][metric],
            "ci_halfwidth": cis[model_name][metric],
            "ci_low":      means[model_name][metric] - cis[model_name][metric],
            "ci_high":     means[model_name][metric] + cis[model_name][metric],
        })
quant_df = pd.DataFrame(quant_rows)
quant_df.to_csv("iuxray_quant_results.csv", index=False)
print("Saved iuxray_quant_results.csv")

# 2. BT ranking results (one row per source × model)
bt_out = combined_bt[[
    "Source", "Rank", "Model", "Davidson score",
    "Normalized strength", "95% CI low", "95% CI high"
]].copy()
bt_out.to_csv("iuxray_bt_results.csv", index=False)
print("Saved iuxray_bt_results.csv")
plt.savefig("iuxray_bt_ranking_comparison.pdf", dpi=600, bbox_inches="tight")
plt.savefig("iuxray_bt_ranking_comparison.png", dpi=600, bbox_inches="tight")
print("\nSaved iuxray_bt_ranking_comparison.pdf / .png")
plt.show()