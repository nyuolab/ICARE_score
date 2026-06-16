from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
OUT_DIR  = Path("/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "iuxray_bt_results.csv"

TITLE_SIZE        = 28
LABEL_SIZE        = 22
TICK_SIZE         = 16
BAR_LABEL_SIZE    = 9.5
LEGEND_SIZE       = 16
LEGEND_TITLE_SIZE = 18

FIG_W = 20
FIG_H = 8.8   # slightly taller than before to make space for legend below

sns.set_style("whitegrid")
sns.set_context("talk")

MODEL_ORDER = [
    "CheXpertPlus_MIMIC",
    "CheXpertPlus_CheX_MIMIC",
    "MAIRA2",
]

MODEL_COLORS = {
    "CheXpertPlus_MIMIC":      "#E69F00",
    "CheXpertPlus_CheX_MIMIC": "#56B4E9",
    "MAIRA2":                  "#009E73",
}

ICARE_TICK_COLOR = "#1f77b4"

# Display names for consistency
DISPLAY_SOURCE_NAMES = {
    "Clinicians": "Clinicians\n(119 samples)",
    "Clinicians (119 samples)": "Clinicians\n(119 samples)",
    "BLEU-2": "BLEU-2 (2002)",
    "BLEU": "BLEU-2 (2002)",
    "BertScore": "BERTScore (2019)",
    "BertScore (2019)": "BERTScore (2019)",
    "BERTScore": "BERTScore (2019)",
    "SembScore": "SembScore (2020)",
    "SembScore (2020)": "SembScore (2020)",
    "1/RadCliQ-v1": "1/RadCliQ-v1 (2022)",
    "1/RadCliQ-v1 (2022)": "1/RadCliQ-v1 (2022)",
    "RadGraph": "RadGraph (2022)",
    "RadGraph (2022)": "RadGraph (2022)",
    "AlignScore": "AlignScore (2023)",
    "AlignScore (2023)": "AlignScore (2023)",
    "GREEN": "GREEN (2024)",
    "GREEN (2024)": "GREEN (2024)",
    "Crimson": "CRIMSON (2026)",
    "Crimson (2026)": "CRIMSON (2026)",
    "CRIMSON": "CRIMSON (2026)",
    "CRIMSON (2026)": "CRIMSON (2026)",
    "ICARE-GT": "ICARE-GT (2025)",
    "ICARE-GT (2025)": "ICARE-GT (2025)",
    "ICARE-GEN": "ICARE-GEN (2025)",
    "ICARE-GEN (2025)": "ICARE-GEN (2025)",
    "ICARE-AVG": "ICARE-AVG (2025)",
    "ICARE-AVG (2025)": "ICARE-AVG (2025)",
    "ICARE_AVG": "ICARE-AVG (2025)",
    "ICARE_PREDEFINED": "ICARE-Predefined (2025)",
    "ICARE-Predefined": "ICARE-Predefined (2025)",
    "ICARE-Predefined (2025)": "ICARE-Predefined (2025)",
}

SOURCE_ORDER = [
    "Clinicians\n(119 samples)",
    "BLEU-2 (2002)",
    "BERTScore (2019)",
    "SembScore (2020)",
    "1/RadCliQ-v1 (2022)",
    "RadGraph (2022)",
    "AlignScore (2023)",
    "GREEN (2024)",
    "CRIMSON (2026)",
    "ICARE-GT (2025)",
    "ICARE-GEN (2025)",
    "ICARE-AVG (2025)",
    "ICARE-Predefined (2025)",
]

# =========================================================
# LOAD CSV
# =========================================================
bt_df = pd.read_csv(CSV_PATH)

print("Columns in CSV:")
print(bt_df.columns.tolist())
print("\nFirst few rows:")
print(bt_df.head())

# Standardize source display names
bt_df["Source"] = bt_df["Source"].map(lambda x: DISPLAY_SOURCE_NAMES.get(str(x), str(x)))

# Keep only rows from desired source list
bt_df = bt_df[bt_df["Source"].isin(SOURCE_ORDER)].copy()

# Standardize model names a bit if needed
bt_df["Model"] = bt_df["Model"].astype(str)

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================
required_cols = [
    "Source",
    "Model",
    "Davidson score",
    "Normalized strength",
    "95% CI low",
    "95% CI high",
]
missing_cols = [c for c in required_cols if c not in bt_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {missing_cols}")

# =========================================================
# HELPER:
# Convert Davidson score CI to normalized-strength CI
# =========================================================
def bt_ci_to_norm_strength_err(group_df, model_order):
    """
    Convert CI on Davidson scores into CI on normalized strengths
    via softmax transformation.

    group_df must contain exactly one row per model for a single source.
    """
    group_df = group_df.copy()
    group_df["Model"] = pd.Categorical(group_df["Model"], categories=model_order, ordered=True)
    group_df = group_df.sort_values("Model").reset_index(drop=True)

    point_scores = group_df["Davidson score"].values.astype(float)
    point_norms  = group_df["Normalized strength"].values.astype(float)

    err_lo = np.zeros(len(group_df))
    err_hi = np.zeros(len(group_df))

    for i, row in group_df.iterrows():
        lo_s = row["95% CI low"]
        hi_s = row["95% CI high"]

        if pd.isna(lo_s) or pd.isna(hi_s):
            err_lo[i] = np.nan
            err_hi[i] = np.nan
            continue

        # Lower bound transform
        scores_lo = point_scores.copy()
        scores_lo[i] = lo_s
        exp_lo = np.exp(scores_lo - np.max(scores_lo))
        norm_lo = exp_lo / exp_lo.sum()

        # Upper bound transform
        scores_hi = point_scores.copy()
        scores_hi[i] = hi_s
        exp_hi = np.exp(scores_hi - np.max(scores_hi))
        norm_hi = exp_hi / exp_hi.sum()

        err_lo[i] = max(0.0, point_norms[i] - norm_lo[i])
        err_hi[i] = max(0.0, norm_hi[i] - point_norms[i])

    return group_df, err_lo, err_hi

# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

n_sources = len(SOURCE_ORDER)
n_models = len(MODEL_ORDER)

x = np.arange(n_sources)
bar_width = 0.27
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

uniform = 1 / 3

csv_rows = []

for mi, model in enumerate(MODEL_ORDER):
    scores = []
    err_los = []
    err_his = []

    for source in SOURCE_ORDER:
        src_group = bt_df[bt_df["Source"] == source].copy()

        if src_group.empty:
            scores.append(np.nan)
            err_los.append(np.nan)
            err_his.append(np.nan)
            continue

        src_group, group_err_lo, group_err_hi = bt_ci_to_norm_strength_err(src_group, MODEL_ORDER)

        row = src_group[src_group["Model"] == model]
        if row.empty:
            scores.append(np.nan)
            err_los.append(np.nan)
            err_his.append(np.nan)
            continue

        idx = row.index[0]
        norm_strength = float(row["Normalized strength"].iloc[0])
        lo = group_err_lo[idx]
        hi = group_err_hi[idx]
        scores.append(norm_strength)
        err_los.append(lo)
        err_his.append(hi)

        csv_rows.append({
            "Source": source,
            "Model": model,
            "Normalized strength": norm_strength,
            "CI_low_err": lo,
            "CI_high_err": hi,
            "CI_low_abs": norm_strength - lo,
            "CI_high_abs": norm_strength + hi,
        })

    scores = np.array(scores, dtype=float)
    err_los = np.array(err_los, dtype=float)
    err_his = np.array(err_his, dtype=float)

    bars = ax.bar(
        x + offsets[mi],
        scores,
        width=bar_width,
        color=MODEL_COLORS[model],
        label=model,
        edgecolor="none",
        zorder=3,
    )

    ax.errorbar(
        x + offsets[mi],
        scores,
        yerr=[err_los, err_his],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=4,
    )

    # numeric labels above bars
    for bi, (bar, score) in enumerate(zip(bars, scores)):
        if not np.isnan(score):
            upper_err = 0.0 if np.isnan(err_his[bi]) else err_his[bi]

            label_y = score + upper_err + 0.012

            # If the label would sit too close to the uniform line,
            # gently move it above the line.
            if abs(label_y - uniform) < 0.020:
                label_y = uniform + 0.020

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_SIZE,
                fontweight="bold",
                color=MODEL_COLORS[model],
                zorder=10,
                clip_on=False,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=0.6,
                ),
            )

# =========================================================
# Reference lines / styling
# =========================================================
# =========================================================
# Reference lines / styling
# =========================================================
ax.axhline(
    uniform,
    color="gray",
    linestyle="--",
    linewidth=2.0,
    alpha=0.95,
    zorder=2,
    label="Uniform (1/3)",
)

# Shade clinician column
if "Clinicians\n(119 samples)" in SOURCE_ORDER:
    idx0 = SOURCE_ORDER.index("Clinicians\n(119 samples)")
    ax.axvspan(idx0 - 0.55, idx0 + 0.55, color="gray", alpha=0.08, zorder=0)

# separator before ICARE metrics
icare_start_idx = SOURCE_ORDER.index("ICARE-GT (2025)")
ax.axvline(
    icare_start_idx - 0.5,
    color=ICARE_TICK_COLOR,
    linestyle=":",
    linewidth=2.2,
    alpha=0.9,
    zorder=2,
)
# =========================================================
# Axes / title
# =========================================================
ax.set_title(
    "Bradley–Terry Model Rankings: Clinicians vs. Automatic Metrics (IU-Xray)",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    pad=20,
)

# ax.set_xlabel(
#     "Evaluation Metric",
#     fontsize=LABEL_SIZE,
#     fontweight="bold",
#     labelpad=18
# )

ax.set_ylabel(
    "BT Normalized Strength",
    fontsize=LABEL_SIZE,
    fontweight="bold",
    labelpad=24,
)

ax.set_xticks(x)
ax.set_xticklabels(
    SOURCE_ORDER,
    rotation=30,
    ha="right",
    fontsize=TICK_SIZE,
    fontweight="bold",
)

# color ICARE tick labels
for tick_label, source in zip(ax.get_xticklabels(), SOURCE_ORDER):
    if "ICARE" in source:
        tick_label.set_color(ICARE_TICK_COLOR)

ax.tick_params(axis="y", labelsize=TICK_SIZE)
for tick in ax.get_yticklabels():
    tick.set_fontweight("bold")

ax.set_xlim(-0.6, n_sources - 0.4)
ax.set_ylim(0, max(0.62, bt_df["Normalized strength"].max() + 0.14))

# =========================================================
# Legend below, 1 row
# =========================================================
handles, labels = ax.get_legend_handles_labels()

# Put uniform first, then models
uniform_handle = None
model_handles = []
model_labels = []

for h, l in zip(handles, labels):
    if l == "Uniform (1/3)":
        uniform_handle = h
    elif l in MODEL_ORDER:
        model_handles.append(h)
        model_labels.append(l)

legend_handles = [uniform_handle] + model_handles if uniform_handle is not None else model_handles
legend_labels  = ["Uniform (1/3)"] + model_labels if uniform_handle is not None else model_labels

ax.legend(
    legend_handles,
    legend_labels,
    title="Model",
    title_fontsize=LEGEND_TITLE_SIZE,
    fontsize=LEGEND_SIZE,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.43),
    ncol=4,   # uniform + 3 models in one row
    frameon=True,
    fancybox=True,
    edgecolor="gray",
)

# =========================================================
# Final polish
# =========================================================
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.grid(axis="x", visible=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.43)

# Save CSV with all plotted values
csv_out = pd.DataFrame(csv_rows, columns=[
    "Source", "Model", "Normalized strength",
    "CI_low_err", "CI_high_err", "CI_low_abs", "CI_high_abs",
])
csv_out.to_csv(OUT_DIR / "Fig3a_bradley_terry_rankings.csv", index=False)
print(f"Saved {OUT_DIR}/Fig3a_bradley_terry_rankings.csv")

# Save
plt.savefig(OUT_DIR / "iuxray_bt_ranking_comparison_from_csv.pdf", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "iuxray_bt_ranking_comparison_from_csv.png", dpi=600, bbox_inches="tight")

plt.show()