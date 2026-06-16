#!/usr/bin/env python3
"""
Plot controlled deletion experiment results.

X-axis  : perturbation levels L0–L4 (categorical).
Y-axis  : score retained (%) for ICARE, BLEU, BERTScore, RadGraph
          (each value divided by its L0 baseline so L0 = 100%).
          −RadCliQ-v0 (higher = better) for the RadCliQ panel — not normalised
          because raw RadCliQ crosses zero and score-retained is undefined.

Also prints and saves a mapping table:
  Level | Clinical-token budget | Mean total tokens deleted | Exact-matched reports

Set LAYOUT to "2x4" (default) or "1x8" at the top of the file.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_ROOT = Path("/gpfs/data/oermannlab/users/rd3571/ICARE_score")
LAYOUT = "2x4"   # "2x4" or "1x8"

MODEL = "maira-2"
MODEL_SEED = 1
EVAL_SEED = 123

BASE_DIR = (
    SCRIPT_ROOT
    / "outputs"
    / "IU_xray"
    / MODEL
    / f"model_seed_{MODEL_SEED}"
    / f"eval_seed_{EVAL_SEED}"
)
BASELINE_DIR = (
    SCRIPT_ROOT
    / "outputs"
    / "IU_xray"
    / "baselines_controlled_perturbation"
    / MODEL
    / f"model_seed_{MODEL_SEED}"
)
PLOT_DIR = SCRIPT_ROOT / "outputs" / "IU_xray" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["clinical_ctrl", "nonclinical", "random_ctrl"]
DEGREES = [0, 10, 20, 30, 40]
LEVEL_LABELS = ["L0", "L1", "L2", "L3", "L4"]

LABEL_MAP = {
    "clinical_ctrl": "Clinical words",
    "nonclinical": "Non-clinical words",
    "random_ctrl": "Random words",
}
COLOR_MAP = {
    "clinical_ctrl": "#d62728",
    "nonclinical": "#2ca02c",
    "random_ctrl": "#1f77b4",
}
LS_MAP = {
    "clinical_ctrl": "-",
    "nonclinical": "--",
    "random_ctrl": ":",
}
MARKER_MAP = {
    "clinical_ctrl": "o",
    "nonclinical": "s",
    "random_ctrl": "^",
}

RETAINED_METRICS = [
    "ICARE",
    "BLEU-2",
    "BERTScore",
    "RadGraph",
    "GREEN",
    "AlignScore",
    "SembScore",
]

# Panel placement for 2x4: (metric, axis index, show_xlabel, show_ylabel)
PANEL_ORDER_2X4 = [
    ("ICARE", 0, False, True),
    ("BLEU-2", 1, False, False),
    ("BERTScore", 2, False, False),
    ("RadGraph", 3, False, False),
    ("GREEN", 4, True, True),
    ("AlignScore", 5, True, False),
    ("SembScore", 6, True, False),
    ("RadCliQ-v0", 7, True, False),
]

# Panel placement for 1x8: (metric, axis index, show_xlabel, show_ylabel, show_legend)
PANEL_ORDER_1X8 = [
    ("ICARE", 0, True, True, True),
    ("BLEU-2", 1, True, True, False),
    ("BERTScore", 2, True, True, False),
    ("RadGraph", 3, True, True, False),
    ("GREEN", 4, True, True, False),
    ("AlignScore", 5, True, True, False),
    ("SembScore", 6, True, True, False),
    ("RadCliQ-v0", 7, True, True, False),
]

if LAYOUT == "2x4":
    TICK_SIZE = 15
    LABEL_SIZE = 17
    TITLE_SIZE = 19
    LEGEND_SIZE = 16
    SUPTITLE_SIZE = 22
    FIGSIZE = (20, 8)
    OUTPUT_STEM = "controlled_perturbation_result_2x4"
else:
    TICK_SIZE = 18
    LABEL_SIZE = 20
    TITLE_SIZE = 22
    LEGEND_SIZE = 16
    SUPTITLE_SIZE = 24
    FIGSIZE = (48, 7)
    OUTPUT_STEM = "controlled_perturbation_result"


# ── Helpers ───────────────────────────────────────────────────────────────────
def empty_metric_dict():
    return {cond: [] for cond in CONDITIONS}


def bootstrap_ci(values, n_boot=2000, rng_seed=42):
    values = pd.Series(values).dropna().values
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(rng_seed)
    boot = [rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_boot)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def normalize_to_retained(values, ci_lo=None, ci_hi=None):
    """Convert raw values to score retained (%), using clinical_ctrl L0 as reference."""
    ref = values["clinical_ctrl"][0]
    if ref is None or abs(ref) < 1e-12:
        return
    for cond in CONDITIONS:
        values[cond] = [(v / ref * 100 if v is not None else None) for v in values[cond]]
        if ci_lo is not None:
            ci_lo[cond] = [(v / ref * 100 if v is not None else None) for v in ci_lo[cond]]
        if ci_hi is not None:
            ci_hi[cond] = [(v / ref * 100 if v is not None else None) for v in ci_hi[cond]]


# ── Mapping table ─────────────────────────────────────────────────────────────
def create_mapping_table():
    orig_path = BASE_DIR / "perturbed_reports_clinical_ctrl_level" / "perturbed_0percent.csv"
    rows = []

    if not orig_path.exists():
        print(f"Warning: original report file not found: {orig_path}")
        return pd.DataFrame()

    orig_df = pd.read_csv(orig_path)
    orig_words = orig_df["generated_report"].astype(str).apply(lambda s: len(s.split()))

    for i, deg in enumerate(DEGREES):
        level = LEVEL_LABELS[i]
        clin_path = BASE_DIR / "perturbed_reports_clinical_ctrl_level" / f"perturbed_{deg}percent.csv"
        nonclin_path = BASE_DIR / "perturbed_reports_nonclinical_level" / f"perturbed_{deg}percent.csv"
        rand_path = BASE_DIR / "perturbed_reports_random_ctrl_level" / f"perturbed_{deg}percent.csv"

        mean_deleted = "N/A"
        exact_matched = "N/A"

        if clin_path.exists():
            clin_words = pd.read_csv(clin_path)["generated_report"].astype(str).apply(
                lambda s: len(s.split()))
            clin_deleted = orig_words - clin_words
            mean_deleted = f"{(clin_deleted / orig_words * 100).mean():.1f}%"

            if nonclin_path.exists() and rand_path.exists():
                nonclin_words = pd.read_csv(nonclin_path)["generated_report"].astype(str).apply(
                    lambda s: len(s.split()))
                rand_words = pd.read_csv(rand_path)["generated_report"].astype(str).apply(
                    lambda s: len(s.split()))
                nonclin_deleted = orig_words - nonclin_words
                rand_deleted = orig_words - rand_words
                exact_matched = str(int(
                    ((clin_deleted == nonclin_deleted) & (clin_deleted == rand_deleted)).sum()
                ))

        rows.append((level, f"{deg}%", mean_deleted, exact_matched))

    print("\nMapping Table")
    print(f"{'Level':<6} {'Clinical budget':<17} {'Mean % tokens deleted':<24} "
          f"{'Exact-matched reports'}")
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:<6} {row[1]:<17} {row[2]:<24} {row[3]}")

    table = pd.DataFrame(rows, columns=[
        "Level",
        "Clinical-token budget (%)",
        "Mean % of report tokens deleted",
        "Exact-matched reports",
    ])
    out_path = PLOT_DIR / "controlled_perturbation_mapping_table.csv"
    table.to_csv(out_path, index=False)
    print(f"Mapping table saved → {out_path}")
    return table


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_icare_mean(cond, deg):
    if deg == 0:
        path = (
            BASE_DIR / "shuffled_ans_choices_data" / "gt_reports_as_ref"
            / "mcqa_eval_perturbed_gen_reports_clinical_ctrl_level"
            / "perturbation_degree0" / "mcq_eval_report_level_stats_aggregated.csv"
        )
    else:
        path = (
            BASE_DIR / "shuffled_ans_choices_data" / "gt_reports_as_ref"
            / f"mcqa_eval_perturbed_gen_reports_{cond}_level"
            / f"perturbation_degree{deg}" / "mcq_eval_report_level_stats_aggregated.csv"
        )
    if not path.exists():
        return None
    return float(pd.read_csv(path).iloc[0]["Mean_Agreement"])


def load_icare_ci(cond, deg):
    if deg == 0:
        path = (
            BASE_DIR / "shuffled_ans_choices_data" / "gt_reports_as_ref"
            / "mcqa_eval_perturbed_gen_reports_clinical_ctrl_level"
            / "perturbation_degree0" / "mcq_eval_report_level_stats.csv"
        )
    else:
        path = (
            BASE_DIR / "shuffled_ans_choices_data" / "gt_reports_as_ref"
            / f"mcqa_eval_perturbed_gen_reports_{cond}_level"
            / f"perturbation_degree{deg}" / "mcq_eval_report_level_stats.csv"
        )
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    col = "Agreement_Percentage" if "Agreement_Percentage" in df.columns else df.columns[-1]
    return bootstrap_ci(df[col])


def load_baseline_summary(cond, deg):
    path = BASELINE_DIR / cond / f"perturbation_degree{deg}" / "summary_of_averages.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).iloc[0]


def load_baseline_ci(cond, deg, col, scale=1.0):
    path = (
        BASELINE_DIR / cond / f"perturbation_degree{deg}" / cond
        / "hybrid_perturbed_gen_orig_gt_results.csv"
    )
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if col not in df.columns:
        return None, None
    return bootstrap_ci(df[col].dropna().values * scale)


def load_green(cond, deg):
    path = (
        BASELINE_DIR / cond / f"perturbation_degree{deg}" / "green"
        / "hybrid_perturbed_gen_orig_gt_green_results.csv"
    )
    if not path.exists():
        return None, None, None
    values = pd.read_csv(path)["green_score"].dropna().values * 100
    lo, hi = bootstrap_ci(values)
    return float(values.mean()), lo, hi


def load_alignscore(cond, deg):
    path = (
        BASELINE_DIR / cond / f"perturbation_degree{deg}" / "alignscore"
        / "hybrid_perturbed_gen_orig_gt_alignscore_results.csv"
    )
    if not path.exists():
        return None, None, None
    values = pd.read_csv(path)["alignscore"].dropna().values * 100
    lo, hi = bootstrap_ci(values)
    return float(values.mean()), lo, hi


def load_all_metric_data():
    metrics = {
        name: (empty_metric_dict(), empty_metric_dict(), empty_metric_dict())
        for name in RETAINED_METRICS + ["RadCliQ-v0"]
    }

    for cond in CONDITIONS:
        for deg in DEGREES:
            icare, icare_lo, icare_hi = metrics["ICARE"]
            icare[cond].append(load_icare_mean(cond, deg))
            lo, hi = load_icare_ci(cond, deg)
            icare_lo[cond].append(lo)
            icare_hi[cond].append(hi)

            row = load_baseline_summary(cond, deg)
            for name, col, scale in [
                ("BLEU-2", "bleu_score", 100),
                ("BERTScore", "bertscore", 100),
                ("RadGraph", "radgraph_combined", 100),
                ("SembScore", "semb_score", 100),
                ("RadCliQ-v0", "RadCliQ-v0", 1),
            ]:
                data, lo_d, hi_d = metrics[name]
                data[cond].append(row[col] * scale if row is not None else None)
                lo, hi = load_baseline_ci(cond, deg, col, scale=scale)
                lo_d[cond].append(lo)
                hi_d[cond].append(hi)

            green, green_lo, green_hi = metrics["GREEN"]
            m, lo, hi = load_green(cond, deg)
            green[cond].append(m)
            green_lo[cond].append(lo)
            green_hi[cond].append(hi)

            alignscore, alignscore_lo, alignscore_hi = metrics["AlignScore"]
            m, lo, hi = load_alignscore(cond, deg)
            alignscore[cond].append(m)
            alignscore_lo[cond].append(lo)
            alignscore_hi[cond].append(hi)

    for name in RETAINED_METRICS:
        data, lo, hi = metrics[name]
        normalize_to_retained(data, lo, hi)

    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────
def configure_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": LABEL_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "font.weight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, show_xlabel=False, show_ylabel=False, ylabel="Score retained (%)"):
    x = np.arange(len(DEGREES))
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xlabel("Perturbation level" if show_xlabel else "", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=LABEL_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=1.5)


def plot_retained_metric(ax, title, data, ci_lo, ci_hi,
                         show_xlabel=False, show_ylabel=False, ylim=None, legend=False):
    x = np.arange(len(DEGREES))
    for cond in CONDITIONS:
        y = np.array(data[cond], dtype=float)
        ax.plot(
            x, y,
            color=COLOR_MAP[cond], linestyle=LS_MAP[cond], marker=MARKER_MAP[cond],
            linewidth=2.8, markersize=7.5, label=LABEL_MAP[cond],
        )
        lo = np.array(ci_lo[cond], dtype=float)
        hi = np.array(ci_hi[cond], dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, color=COLOR_MAP[cond], alpha=0.13)

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    style_axis(ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(fontsize=LEGEND_SIZE, loc="lower left")


def plot_radcliq_metric(ax, data, ci_lo, ci_hi,
                        show_xlabel=False, show_ylabel=False, ylim=None, legend=False):
    x = np.arange(len(DEGREES))
    for cond in CONDITIONS:
        y = -np.array(data[cond], dtype=float)
        lo = -np.array(ci_hi[cond], dtype=float)
        hi = -np.array(ci_lo[cond], dtype=float)
        ax.plot(
            x, y,
            color=COLOR_MAP[cond], linestyle=LS_MAP[cond], marker=MARKER_MAP[cond],
            linewidth=2.8, markersize=7.5, label=LABEL_MAP[cond],
        )
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, color=COLOR_MAP[cond], alpha=0.13)

    ax.set_title("RadCliQ-v0", fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    style_axis(
        ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel,
        ylabel="−RadCliQ-v0\n(higher = better)",
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(fontsize=LEGEND_SIZE, loc="lower left")


def get_retained_ylim(metric_data):
    values = []
    for name in RETAINED_METRICS:
        data, _, _ = metric_data[name]
        for cond in CONDITIONS:
            values.extend(v for v in data[cond] if v is not None)
    values = np.array(values, dtype=float)
    values = values[np.isfinite(values)]
    ymin = np.floor((values.min() - 1.5) / 2) * 2
    ymax = np.ceil((values.max() + 1.5) / 2) * 2
    return ymin, ymax


def get_radcliq_ylim(metric_data):
    data, _, _ = metric_data["RadCliQ-v0"]
    values = []
    for cond in CONDITIONS:
        y = -np.array(data[cond], dtype=float)
        values.extend(y[np.isfinite(y)])
    values = np.array(values, dtype=float)
    return values.min() - 0.08, values.max() + 0.08


def add_shared_legend(fig):
    handles = [
        Line2D(
            [0], [0],
            color=COLOR_MAP[cond], linestyle=LS_MAP[cond], marker=MARKER_MAP[cond],
            linewidth=2.8, markersize=7.5, label=LABEL_MAP[cond],
        )
        for cond in CONDITIONS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )


def create_figure(metric_data):
    configure_plot_style()
    radcliq_ylim = get_radcliq_ylim(metric_data)

    if LAYOUT == "2x4":
        fig, axes = plt.subplots(2, 4, figsize=FIGSIZE)
        axes = axes.ravel()
        panel_order = PANEL_ORDER_2X4
        use_shared_legend = True
    else:
        fig, axes = plt.subplots(1, 8, figsize=FIGSIZE, sharey=False)
        panel_order = PANEL_ORDER_1X8
        use_shared_legend = False

    fig.suptitle(
        "Controlled content deletion on IU-Xray reports",
        fontsize=SUPTITLE_SIZE, fontweight="bold",
        y=0.98 if LAYOUT == "2x4" else 1.02,
    )

    for panel in panel_order:
        metric_name = panel[0]
        ax_idx = panel[1]
        show_xlabel = panel[2]
        show_ylabel = panel[3]
        show_legend = panel[4] if len(panel) > 4 else False

        data, lo, hi = metric_data[metric_name]
        if metric_name == "RadCliQ-v0":
            plot_radcliq_metric(
                axes[ax_idx], data, lo, hi,
                show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                ylim=radcliq_ylim, legend=show_legend,
            )
        else:
            plot_retained_metric(
                axes[ax_idx], metric_name, data, lo, hi,
                show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                ylim=None, legend=show_legend,
            )

    if use_shared_legend:
        add_shared_legend(fig)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.16,
                            wspace=0.28, hspace=0.38)
    else:
        plt.tight_layout()

    return fig


ALL_METRICS = RETAINED_METRICS + ["RadCliQ-v0"]


def save_data_to_csv(metric_data):
    rows = []
    for metric in ALL_METRICS:
        data, ci_lo, ci_hi = metric_data[metric]
        normalized = metric in RETAINED_METRICS
        for cond in CONDITIONS:
            for i, (deg, level) in enumerate(zip(DEGREES, LEVEL_LABELS)):
                rows.append({
                    "metric":     metric,
                    "condition":  cond,
                    "level":      level,
                    "degree":     deg,
                    "mean":       data[cond][i],
                    "ci_lo":      ci_lo[cond][i],
                    "ci_hi":      ci_hi[cond][i],
                    "normalized": normalized,
                })
    out_path = PLOT_DIR / "controlled_perturbation_data.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Data CSV saved → {out_path}")


def main():
    print("Creating controlled perturbation figure.")
    print(f"LAYOUT       : {LAYOUT}")
    print(f"BASE_DIR     : {BASE_DIR}")
    print(f"BASELINE_DIR : {BASELINE_DIR}")
    print(f"PLOT_DIR     : {PLOT_DIR}")

    create_mapping_table()
    metric_data = load_all_metric_data()
    save_data_to_csv(metric_data)
    fig = create_figure(metric_data)

    out_png = PLOT_DIR / f"{OUTPUT_STEM}.png"
    out_pdf = PLOT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
