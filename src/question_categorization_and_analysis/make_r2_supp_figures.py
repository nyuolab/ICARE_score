#!/usr/bin/env python3
"""Final R2 supplementary figures/tables/Source Data for Exp 4 & 5 (model_seeds_all only).

Reads existing results/revision_r2/exp{4,5}/model_seeds_all/ outputs.
Does not recompute scores, re-cluster, or call LLMs for naming.
Re-runnable: picks up cluster_names_K{K}.json when present.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# ---------------------------------------------------------------------------
# Manuscript model display names (cluster_analysis.py)
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "mimic-cxr-findings-baseline": "CheXpertPlus_MIMIC",
    "chexpert-mimic-cxr-findings-baseline": "CheXpertPlus_Chex_MIMIC",
    "maira-2": "MAIRA2",
}
MODEL_ORDER = [
    "mimic-cxr-findings-baseline",
    "chexpert-mimic-cxr-findings-baseline",
    "maira-2",
]

# Fig. 4 side-of-diagonal colours (colour-blind safer red/blue)
COLOR_BELOW = "#d62728"  # omission-dominated: ICARE-GT < ICARE-GEN
COLOR_ABOVE = "#1f77b4"  # hallucination-dominated: ICARE-GT > ICARE-GEN
COLOR_ON = "#7f7f7f"     # on diagonal

# Fig. 4 style from cluster_analysis.create_cluster_summary_table_and_scatterplots
# Size mapping: s = GT_Count / 30 (absolute; same scale in every panel)
FIG4_SIZE_DIVISOR = 30.0
MARKER_ALPHA = 0.7
MARKER_EDGECOLOR = "black"
MARKER_EDGEWIDTH = 0.5
FIG4_DIAG_LW = 1.5
FIG4_DIAG_ALPHA = 0.7
AX_LIM = (0.0, 1.0)
# Per-panel size in the submitted Fig. 4 figure (figsize 24x28, 3 left panels)
FIG4_FIGSIZE = (24.0, 28.0)  # exact submitted Fig. 4
FIG4_WIDTH_RATIOS = (1.5, 1.0)
FIG4_HSPACE = 0.3
FIG4_PANEL_W = 24.0 * 1.5 / 2.5  # left-column panel width (in)
FIG4_PANEL_H = 28.0 / 3.0         # left-column panel height (in)
ERRORBAR_ZORDER = 2
MARKER_ZORDER = 3
LABEL_ZORDER = 4

KS = [10, 15, 20, 25, 30]
KS_GRID = [10, 15, 20, 25, 30]
MM = 1 / 25.4
CONCEPTS = ("Lung Lobe Abnormality", "Calcified Granuloma Location")


def _tex(s) -> str:
    return str(s).replace('_', r'\_')


def side_of_diagonal(gt: float, gen: float) -> str:
    if not (np.isfinite(gt) and np.isfinite(gen)):
        return "undefined"
    if gt < gen:
        return "below"
    if gt > gen:
        return "above"
    return "on"


def side_color(side: str) -> str:
    return {"below": COLOR_BELOW, "above": COLOR_ABOVE, "on": COLOR_ON}.get(side, COLOR_ON)


def display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def load_names(exp4: Path, k: int) -> Dict[int, str]:
    """Load cluster_names_K{k}.json if present (K=20 always; others when named)."""
    for base in (exp4 / "model_seeds_all", exp4):
        p = base / f"cluster_names_K{k}.json"
        if p.is_file():
            raw = json.loads(p.read_text())
            return {int(k_): str(v) for k_, v in raw.items()}
    return {}


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Binomial Wilson 95% CI from existing rate + count (no rerun)."""
    if n <= 0 or not np.isfinite(p):
        return float("nan"), float("nan")
    p = float(np.clip(p, 0.0, 1.0))
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)




def setup_style() -> None:
    """Exact Fig. 4 rcParams from cluster_analysis.py."""
    mpl.rcParams.update(
        {
            "font.size": 20,
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 100,
            "savefig.dpi": 600,
        }
    )


def fig4_point_sizes(counts) -> np.ndarray:
    """Submitted Fig. 4 size mapping: s = count / 30."""
    return np.asarray(counts, dtype=float) / FIG4_SIZE_DIVISOR


def style_fig4_axes(ax, *, show_xlabel: bool = True, show_ylabel: bool = True,
                    show_xticklabels: bool = True, show_yticklabels: bool = True,
                    xlabel_fontsize: float = 22, ylabel_fontsize: float = 22) -> None:
    ax.set_xlim(*AX_LIM)
    ax.set_ylim(*AX_LIM)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=20)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_weight("bold")
        lab.set_fontsize(20)
    # Fig. 4 removes the redundant zero on the y-axis
    yticks = [t for t in ax.get_yticks() if abs(t) > 1e-12]
    yticks = [t for t in yticks if AX_LIM[0] - 1e-9 <= t <= AX_LIM[1] + 1e-9]
    ax.set_yticks(yticks)
    if show_xlabel:
        ax.set_xlabel("ICARE-GEN", fontsize=xlabel_fontsize, weight="bold", labelpad=15)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("ICARE-GT", fontsize=ylabel_fontsize, weight="bold", labelpad=15)
    else:
        ax.set_ylabel("")
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)


def draw_fig4_diagonal(ax) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=FIG4_DIAG_LW, alpha=FIG4_DIAG_ALPHA, zorder=1)


def panel_letter_fig4(ax, letter: str) -> None:
    ax.text(
        -0.15, 1.05, f"({letter})", transform=ax.transAxes,
        fontsize=30, fontweight="bold", clip_on=False,
    )


def label_clusters_fig4(ax, xs, ys, labels) -> None:
    """Exact Fig. 4 ID labelling: annotate offset (5, 5) pt, fontsize 18."""
    for x, y, lab in zip(xs, ys, labels):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.annotate(
            str(int(lab)),
            (float(x), float(y)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=18,
            weight="bold",
            ha="left",
            alpha=0.8,
            zorder=LABEL_ZORDER,
            clip_on=False,
        )


def add_correlation_box(ax, xs, ys) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return
    corr = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
    ax.text(
        0.05, 0.95, f"Correlation: {corr:.3f}",
        transform=ax.transAxes, fontsize=20, weight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        verticalalignment="top", zorder=6,
    )


def colour_legend_handles():
    return [
        Patch(facecolor=COLOR_BELOW, edgecolor=MARKER_EDGECOLOR, linewidth=MARKER_EDGEWIDTH,
              label="Below diagonal (ICARE-GT < ICARE-GEN)"),
        Patch(facecolor=COLOR_ABOVE, edgecolor=MARKER_EDGECOLOR, linewidth=MARKER_EDGEWIDTH,
              label="Above diagonal (ICARE-GT > ICARE-GEN)"),
    ]


def size_legend_handles(example_counts):
    """Size key using Fig. 4 formula s = n / 30 (no pyplot axes side effects)."""
    handles = []
    for c in example_counts:
        s = float(c) / FIG4_SIZE_DIVISOR
        ms = 2.0 * float(np.sqrt(max(s, 1e-9) / np.pi))
        handles.append(
            Line2D(
                [0], [0],
                linestyle="None",
                marker="o",
                markersize=ms,
                markerfacecolor="0.55",
                markeredgecolor=MARKER_EDGECOLOR,
                markeredgewidth=MARKER_EDGEWIDTH,
                alpha=MARKER_ALPHA,
                label=f"n = {int(c):,}",
            )
        )
    return handles



def draw_fig4_table(ax, names: Dict[int, str]) -> None:
    """Exact Fig. 4 (d) table style from cluster_analysis.py."""
    ax.axis("off")
    ax.text(-0.15, 1.0, "(d)", fontsize=30, fontweight="bold")
    table_data = [[str(i), names.get(i, "")] for i in range(20)]
    table = ax.table(
        cellText=table_data,
        colLabels=["ID", "Cluster Name"],
        loc="center",
        cellLoc="left",
        bbox=[0.02, 0.02, 1.3, 0.98],
        colWidths=[0.24, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=24)
            cell.set_facecolor("#E6E6E6")
            cell._text.set_horizontalalignment("center")
            if col == 0:
                cell.get_text().set_text("Cluster ID")
        else:
            cell.set_text_props(weight="bold", fontsize=22)
            if col == 0:
                cell._text.set_horizontalalignment("center")
            else:
                cell._text.set_horizontalalignment("left")
                cell.get_text().set_wrap(True)
                cur = cell.get_text().get_text()
                if len(cur) > 50:
                    cell.get_text().set_text(cur[:47] + "...")
        cell.PAD = 0.1


def scatter_fig4(ax, xs, ys, counts, colors):
    sizes = fig4_point_sizes(counts)
    ax.scatter(
        xs, ys, s=sizes, c=colors, alpha=MARKER_ALPHA,
        edgecolors=MARKER_EDGECOLOR, linewidths=MARKER_EDGEWIDTH, zorder=MARKER_ZORDER,
    )



def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')} and .png")


def pooled_n_gt_exp5(exp5: Path) -> pd.DataFrame:
    """Pooled GT question count across all 5 eval seeds (Fig. 4 count definition)."""
    sc = pd.read_csv(exp5 / "model_seeds_all" / "scores_by_eval_seed.csv")
    g = sc.groupby(["Model", "Cluster"], as_index=False)["n_gt"].sum()
    g = g.rename(columns={"n_gt": "pooled_n_gt"})
    return g





def build_supp_fig_a(exp5: Path, names: Dict[int, str], out: Path) -> pd.DataFrame:
    """Fig. 4 layout (3 stacked left + spanning table); colours + SD error bars only."""
    summ = pd.read_csv(exp5 / "model_seeds_all" / "across_seed_summary.csv")
    pooled = pooled_n_gt_exp5(exp5)
    summ = summ.merge(pooled, on=["Model", "Cluster"], how="left")

    # Exact Fig. 4 figure size + GridSpec; thin bottom strip for one-line legend
    fig = plt.figure(figsize=(FIG4_FIGSIZE[0], FIG4_FIGSIZE[1] + 0.9))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=list(FIG4_WIDTH_RATIOS),
        height_ratios=[1, 1, 1, 0.08],
        hspace=FIG4_HSPACE,
        wspace=0.15,
        left=0.08,
        right=0.92,
        top=0.96,
        bottom=0.04,
    )
    letters = ["a", "b", "c"]
    source_rows = []

    for idx, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[idx, 0])
        sub = summ[summ.Model == model].copy()
        xs = sub.GEN_mean.to_numpy(dtype=float)
        ys = sub.GT_mean.to_numpy(dtype=float)
        labs = [int(c) for c in sub.Cluster]
        cnt = sub.pooled_n_gt.fillna(sub.mean_n_gt).to_numpy(dtype=float)
        colors = [side_color(side_of_diagonal(g, e)) for g, e in zip(ys, xs)]

        for _, r in sub.iterrows():
            side = side_of_diagonal(r.GT_mean, r.GEN_mean)
            ax.errorbar(
                r.GEN_mean,
                r.GT_mean,
                xerr=r.GEN_sd,
                yerr=r.GT_sd,
                fmt="none",
                ecolor=side_color(side),
                elinewidth=0.8,
                capsize=1.2,
                alpha=0.85,
                zorder=ERRORBAR_ZORDER,
            )
            source_rows.append(
                {
                    "panel": letters[idx],
                    "model": display(model),
                    "cluster_id": int(r.Cluster),
                    "cluster_name": names.get(int(r.Cluster), r.cluster_name),
                    "ICARE_GEN_mean": r.GEN_mean,
                    "ICARE_GEN_sd": r.GEN_sd,
                    "ICARE_GT_mean": r.GT_mean,
                    "ICARE_GT_sd": r.GT_sd,
                    "pooled_n_gt": float(r.pooled_n_gt) if pd.notna(r.pooled_n_gt) else float(r.mean_n_gt),
                    "side_of_diagonal": side,
                    "marker_size_pt2": float(
                        fig4_point_sizes(
                            [r.pooled_n_gt if pd.notna(r.pooled_n_gt) else r.mean_n_gt]
                        )[0]
                    ),
                }
            )

        scatter_fig4(ax, xs, ys, cnt, colors)
        label_clusters_fig4(ax, xs, ys, labs)
        draw_fig4_diagonal(ax)
        style_fig4_axes(ax)
        ax.set_title(display(model), fontsize=24, weight="bold", pad=20)
        panel_letter_fig4(ax, letters[idx])
        add_correlation_box(ax, xs, ys)

    ax_table = fig.add_subplot(gs[0:3, 1])
    draw_fig4_table(ax_table, names)

    leg_ax = fig.add_subplot(gs[3, :])
    leg_ax.axis("off")
    err_proxy = Line2D(
        [0], [0], color="0.25", lw=1.0, label="Error bars: SD across 5 evaluation seeds"
    )
    leg_ax.legend(
        handles=colour_legend_handles() + [err_proxy],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=16,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    print(
        f"Supp Fig A axes={len(fig.axes)} figsize={fig.get_size_inches()} "
        f"panel~{FIG4_PANEL_W:.2f}x{FIG4_PANEL_H:.2f}in"
    )
    save_fig(fig, out / "SuppFigA_exp5_seed_consistency")
    return pd.DataFrame(source_rows)


def build_supp_table_a(exp5: Path, names: Dict[int, str], out: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summ = pd.read_csv(exp5 / "model_seeds_all" / "across_seed_summary.csv")
    lab = pd.read_csv(exp5 / "model_seeds_all" / "label_consistency.csv")
    jac_sum = pd.read_csv(exp5 / "model_seeds_all" / "report_jaccard_summary.csv")
    jac = pd.read_csv(exp5 / "model_seeds_all" / "report_jaccard.csv")
    meta_path = exp5 / "model_seeds_all" / "mode_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    rows = []
    footnotes = []
    for _, r in summ.iterrows():
        cid = int(r.Cluster)
        side_counts = {
            "below": int(r.n_seeds_omission),
            "above": int(r.n_seeds_hallucination),
            "on": int(r.n_seeds_balanced),
        }
        n_same_side = max(side_counts.values())
        both_low = (r.GT_mean < 0.3) and (r.GEN_mean < 0.3)
        mark = []
        if bool(r.any_low_n):
            mark.append("†")
            footnotes.append((cid, display(r.Model), "†"))
        if both_low:
            mark.append("‡")
            footnotes.append((cid, display(r.Model), "‡"))
        rows.append(
            {
                "cluster_id": cid,
                "cluster_name": names.get(cid, r.cluster_name) + ("".join(mark) if mark else ""),
                "model": display(r.Model),
                "ICARE_GT": f"{r.GT_mean:.3f} ± {r.GT_sd:.3f}",
                "ICARE_GEN": f"{r.GEN_mean:.3f} ± {r.GEN_sd:.3f}",
                "ICARE_GT_mean": r.GT_mean,
                "ICARE_GT_sd": r.GT_sd,
                "ICARE_GEN_mean": r.GEN_mean,
                "ICARE_GEN_sd": r.GEN_sd,
                "seeds_same_side": f"{n_same_side}/5",
                "n_seeds_below": side_counts["below"],
                "n_seeds_above": side_counts["above"],
                "n_seeds_on": side_counts["on"],
                "any_low_n": bool(r.any_low_n),
                "both_rates_lt_0.3": both_low,
                "mean_n_gt": r.mean_n_gt,
                "mean_n_gen": r.mean_n_gen,
            }
        )
    detail = pd.DataFrame(rows)

    # Wide table for LaTeX: one row per cluster
    wide_rows = []
    for cid in range(20):
        row = {"cluster_id": cid, "cluster_name": names.get(cid, "")}
        marks = set()
        for model in MODEL_ORDER:
            sub = detail[(detail.cluster_id == cid) & (detail.model == display(model))]
            if sub.empty:
                continue
            r = sub.iloc[0]
            pref = display(model)
            row[f"{pref}_ICARE_GT"] = r.ICARE_GT
            row[f"{pref}_ICARE_GEN"] = r.ICARE_GEN
            row[f"{pref}_seeds_same_side"] = r.seeds_same_side
            if r.any_low_n:
                marks.add("†")
            if r["both_rates_lt_0.3"]:
                marks.add("‡")
        row["cluster_name"] = names.get(cid, "") + ("".join(sorted(marks)) if marks else "")
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)

    # Summary rows
    def jac_ci(model: str, metric: str, pair_type: str) -> Tuple[float, float, float]:
        sub = jac[(jac.Model == model) & (jac.metric == metric) & (jac.pair_type == pair_type)]
        vals = sub.jaccard.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(0)
        boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)])
        return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    overlap = meta.get("overlap", {})
    n_common = overlap.get("intersection_all_5", 9569)
    n_union = overlap.get("union_all_5", overlap.get("total_unique_questions", 78708))

    summary_rows = []
    for model in MODEL_ORDER:
        lc = lab[lab.Model == model].iloc[0]
        same_gt = jac_ci(model, "gt_reports_as_ref", "same_report_across_seeds")
        diff_gt = jac_ci(model, "gt_reports_as_ref", "different_reports")
        same_gen = jac_ci(model, "gen_reports_as_ref", "same_report_across_seeds")
        diff_gen = jac_ci(model, "gen_reports_as_ref", "different_reports")
        summary_rows.append(
            {
                "model": display(model),
                "pct_clusters_same_side_all_5_seeds": lc.pct_same_label_all_clusters,
                "median_across_seed_SD_ICARE_GT": lc.median_GT_sd,
                "median_across_seed_SD_ICARE_GEN": lc.median_GEN_sd,
                "jaccard_same_report_gt_mean": same_gt[0],
                "jaccard_same_report_gt_CI95": f"[{same_gt[1]:.3f}, {same_gt[2]:.3f}]",
                "jaccard_diff_report_gt_mean": diff_gt[0],
                "jaccard_diff_report_gt_CI95": f"[{diff_gt[1]:.3f}, {diff_gt[2]:.3f}]",
                "jaccard_same_report_gen_mean": same_gen[0],
                "jaccard_same_report_gen_CI95": f"[{same_gen[1]:.3f}, {same_gen[2]:.3f}]",
                "jaccard_diff_report_gen_mean": diff_gen[0],
                "jaccard_diff_report_gen_CI95": f"[{diff_gen[1]:.3f}, {diff_gen[2]:.3f}]",
                "unique_questions_common_all_5_seeds": f"{n_common} / {n_union}",
            }
        )
    summary = pd.DataFrame(summary_rows)

    wide.to_csv(out / "SuppTableA_exp5.csv", index=False)
    summary.to_csv(out / "SuppTableA_exp5_summary.csv", index=False)
    detail.to_csv(out / "SuppTableA_exp5_long.csv", index=False)

    # LaTeX
    lines = [
        r"% Supp Table A — Exp 5 evaluation-seed consistency (K=20)",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Evaluation-seed consistency of ICARE cluster scores (K=20).}",
        r"\label{tab:supp_exp5}",
        r"\begin{tabular}{ll" + "ccc" * 3 + "}",
        r"\toprule",
        r"ID & Name & "
        + " & ".join(
            [
                r"\multicolumn{3}{c}{" + _tex(display(m)) + "}"
                for m in MODEL_ORDER
            ]
        )
        + r" \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r" & & "
        + " & ".join([r"GT & GEN & side"] * 3)
        + r" \\",
        r"\midrule",
    ]
    for _, r in wide.iterrows():
        cells = [str(int(r.cluster_id)), _tex(r.cluster_name)[:40]]
        for m in MODEL_ORDER:
            pref = display(m)
            cells += [
                str(r.get(f"{pref}_ICARE_GT", "")),
                str(r.get(f"{pref}_ICARE_GEN", "")),
                str(r.get(f"{pref}_seeds_same_side", "")),
            ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize",
        r"$^\dagger$Cluster with $<$30 questions on either side in any of the 5 evaluation seeds. ",
        r"$^\ddagger$Cluster with ICARE-GT and ICARE-GEN both $<$0.3. ",
        r"GT/GEN: across-seed mean $\pm$ SD. side: number of seeds on the modal side of the diagonal.",
        r"\end{flushleft}",
        r"\end{table*}",
        "",
        r"% Summary",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Exp 5 summary across models.}",
        r"\label{tab:supp_exp5_summary}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & \% same side (5/5) & med.\ SD GT & med.\ SD GEN & Q common \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{_tex(r.model)} & {r.pct_clusters_same_side_all_5_seeds:.1f} & "
            f"{r.median_across_seed_SD_ICARE_GT:.3f} & {r.median_across_seed_SD_ICARE_GEN:.3f} & "
            f"{r.unique_questions_common_all_5_seeds} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    # Jaccard latex block
    lines += [
        r"% Jaccard",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Report-level Jaccard of clusters probed (Exp 5).}",
        r"\label{tab:supp_exp5_jaccard}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Metric & Same report (95\% CI) & Different reports (95\% CI) \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{_tex(r.model)} & GT-ref & "
            f"{r.jaccard_same_report_gt_mean:.3f} {r.jaccard_same_report_gt_CI95} & "
            f"{r.jaccard_diff_report_gt_mean:.3f} {r.jaccard_diff_report_gt_CI95} \\\\"
        )
        lines.append(
            f" & GEN-ref & "
            f"{r.jaccard_same_report_gen_mean:.3f} {r.jaccard_same_report_gen_CI95} & "
            f"{r.jaccard_diff_report_gen_mean:.3f} {r.jaccard_diff_report_gen_CI95} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out / "SuppTableA_exp5.tex").write_text("\n".join(lines))
    print(f"Wrote SuppTableA → {out}")
    return wide, summary


# ========================= Exp 4 helpers =========================

def load_scores(exp4: Path, k: int) -> pd.DataFrame:
    p = exp4 / "model_seeds_all" / f"scores_K{k}.csv"
    return pd.read_csv(p)


def load_correspondence(exp4: Path, k: int) -> np.ndarray:
    npy = exp4 / "model_seeds_all" / f"correspondence_K{k}.npy"
    if npy.is_file():
        return np.load(npy)
    csv = pd.read_csv(exp4 / "model_seeds_all" / f"correspondence_K{k}.csv", index_col=0)
    return csv.to_numpy(dtype=float)


def spearman_gt_gen(scores_k20: pd.DataFrame, scores_new: pd.DataFrame, mat: np.ndarray, model: str) -> Tuple[float, float, int]:
    """Question-weighted Spearman of ICARE-GT and ICARE-GEN vs K=20 (n=20 clusters)."""
    so = scores_k20[scores_k20.Model == model].set_index("Cluster")
    sn = scores_new[scores_new.Model == model].set_index("Cluster")
    orig_gt, map_gt, orig_gen, map_gen = [], [], [], []
    for oid in range(20):
        if oid not in so.index:
            continue
        w = mat[oid].astype(float)
        tot = w.sum()
        if tot <= 0:
            continue
        w = w / tot
        gts, gens, ww = [], [], []
        for j, weight in enumerate(w):
            if weight <= 0 or j not in sn.index:
                continue
            if not np.isfinite(sn.loc[j, "GT"]) or not np.isfinite(sn.loc[j, "GEN"]):
                continue
            gts.append(float(sn.loc[j, "GT"]))
            gens.append(float(sn.loc[j, "GEN"]))
            ww.append(weight)
        if not ww:
            continue
        ww = np.asarray(ww)
        ww = ww / ww.sum()
        if not np.isfinite(so.loc[oid, "GT"]) or not np.isfinite(so.loc[oid, "GEN"]):
            continue
        orig_gt.append(float(so.loc[oid, "GT"]))
        map_gt.append(float(np.dot(ww, gts)))
        orig_gen.append(float(so.loc[oid, "GEN"]))
        map_gen.append(float(np.dot(ww, gens)))
    n = len(orig_gt)
    if n < 3:
        return float("nan"), float("nan"), n
    rho_gt = float(spearmanr(orig_gt, map_gt).correlation)
    rho_gen = float(spearmanr(orig_gen, map_gen).correlation)
    return rho_gt, rho_gen, n


def below_above_counts(scores: pd.DataFrame, model: str, k: int) -> dict:
    sub = scores[scores.Model == model]
    below = int((sub.GT < sub.GEN).sum())
    above = int((sub.GT > sub.GEN).sum())
    on = int((sub.GT == sub.GEN).sum())
    return {
        "n_below": below,
        "n_above": above,
        "n_on": on,
        "pct_below": 100.0 * below / k,
        "pct_above": 100.0 * above / k,
    }


def build_exp4_metrics(exp4: Path) -> pd.DataFrame:
    cons = pd.read_csv(exp4 / "model_seeds_all" / "consistency_by_K.csv")
    s20 = load_scores(exp4, 20)
    rows = []
    for k in KS:
        sk = load_scores(exp4, k)
        mat = load_correspondence(exp4, k)
        for model in MODEL_ORDER:
            ba = below_above_counts(sk, model, k)
            csub = cons[(cons.K == k) & (cons.Model == model)]
            pct_agree = float(csub.pct_label_match_best.iloc[0]) if len(csub) else float("nan")
            rho_gt, rho_gen, n_sp = spearman_gt_gen(s20, sk, mat, model)
            rows.append(
                {
                    "K": k,
                    "Model": model,
                    "model_display": display(model),
                    **ba,
                    "label_agreement_pct": pct_agree,
                    "spearman_ICARE_GT": rho_gt,
                    "spearman_ICARE_GEN": rho_gen,
                    "spearman_n": n_sp,
                }
            )
    return pd.DataFrame(rows)



def concept_cluster_ids(exp4: Path) -> Dict[int, Dict[str, int]]:
    """K -> {concept: matched cluster_id} for highlight outlines."""
    ct = pd.read_csv(exp4 / "model_seeds_all" / "concept_tracking.csv")
    out: Dict[int, Dict[str, int]] = {}
    for k in KS_GRID:
        out[k] = {}
        for concept in CONCEPTS:
            sub = ct[(ct.K == k) & (ct.concept == concept) & (ct.is_matched)]
            if len(sub):
                out[k][concept] = int(sub.cluster_id.iloc[0])
    return out





def build_supp_fig_b(exp4: Path, metrics: pd.DataFrame, names20: Dict[int, str], out: Path) -> Dict[str, pd.DataFrame]:
    """5x3 grid; each panel uses Fig. 4 panel size and scatter style."""
    concept_ids = concept_cluster_ids(exp4)
    scores_by_k = {k: load_scores(exp4, k) for k in KS_GRID}

    nrows, ncols = len(KS_GRID), len(MODEL_ORDER)
    fig_w = ncols * FIG4_PANEL_W
    fig_h = nrows * FIG4_PANEL_H + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=FIG4_HSPACE,
        wspace=0.18,
        left=0.05,
        right=0.99,
        top=0.95,
        bottom=0.03,
    )
    fig.legend(
        handles=colour_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=True,
        fontsize=16,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    source_d = []

    for ri, k in enumerate(KS_GRID):
        sk = scores_by_k[k]
        name_map = load_names(exp4, k)
        for ci, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            sub = sk[sk.Model == model].copy()
            xs = sub.GEN.to_numpy(dtype=float)
            ys = sub.GT.to_numpy(dtype=float)
            labs = [int(c) for c in sub.Cluster]
            cnt = sub.n_gt.to_numpy(dtype=float)
            colors = [side_color(side_of_diagonal(g, e)) for g, e in zip(ys, xs)]

            scatter_fig4(ax, xs, ys, cnt, colors)
            label_clusters_fig4(ax, xs, ys, labs)
            draw_fig4_diagonal(ax)
            style_fig4_axes(
                ax,
                show_xlabel=(ri == nrows - 1),
                show_ylabel=(ci == 0),
                show_xticklabels=(ri == nrows - 1),
                show_yticklabels=(ci == 0),
            )
            if ri == 0:
                ax.set_title(display(model), fontsize=24, weight="bold", pad=20)
            if ci == 0:
                if k == 20:
                    ax.set_ylabel(
                        f"K = {k} (Fig. 4)\nICARE-GT",
                        fontsize=22,
                        weight="bold",
                        labelpad=15,
                    )
                else:
                    ax.set_ylabel(
                        f"K = {k}\nICARE-GT",
                        fontsize=22,
                        weight="bold",
                        labelpad=15,
                    )
            if k == 20:
                ax.set_facecolor("#eeeeee")
            if ri == 0 and ci == 0:
                panel_letter_fig4(ax, "a")
            add_correlation_box(ax, xs, ys)

            for _, r in sub.iterrows():
                side = side_of_diagonal(r.GT, r.GEN)
                source_d.append(
                    {
                        "K": k,
                        "model": display(model),
                        "cluster_id": int(r.Cluster),
                        "cluster_name": name_map.get(int(r.Cluster), ""),
                        "ICARE_GEN": r.GEN,
                        "ICARE_GT": r.GT,
                        "n_gt": int(r.n_gt),
                        "n_gen": int(r.n_gen),
                        "side_of_diagonal": side,
                        "marker_size_pt2": float(fig4_point_sizes([r.n_gt])[0]),
                        "highlight_concept": next(
                            (
                                c
                                for c, cid in concept_ids.get(k, {}).items()
                                if cid == int(r.Cluster)
                            ),
                            "",
                        ),
                    }
                )

    print(
        f"Supp Fig B axes={len(fig.axes)} (expect {nrows * ncols}) "
        f"figsize={fig.get_size_inches()} panel={FIG4_PANEL_W:.2f}x{FIG4_PANEL_H:.2f}in"
    )
    if len(fig.axes) != nrows * ncols:
        print(f"[WARN] unexpected axes count {len(fig.axes)}")
    save_fig(fig, out / "SuppFigB_exp4_k_sensitivity")
    return {
        "metrics": metrics,
        "fig_d": pd.DataFrame(source_d),
        "fig_a": metrics[["K", "model_display", "pct_below"]].copy(),
        "fig_b": metrics[["K", "model_display", "label_agreement_pct"]].copy(),
        "fig_c": metrics[["K", "model_display", "spearman_ICARE_GT", "spearman_ICARE_GEN"]].copy(),
    }


def build_supp_table_b(exp4: Path, metrics: pd.DataFrame, names20: Dict[int, str], out: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    part1 = metrics.copy()
    part1.to_csv(out / "SuppTableB_exp4_part1.csv", index=False)

    ct = pd.read_csv(exp4 / "model_seeds_all" / "concept_tracking.csv")
    concepts = ["Lung Lobe Abnormality", "Calcified Granuloma Location"]
    rows = []
    for k in KS:
        sk = load_scores(exp4, k)
        name_map = load_names(exp4, k)
        for concept in concepts:
            for model in MODEL_ORDER:
                sub = ct[(ct.K == k) & (ct.concept == concept) & (ct.Model == model) & (ct.is_matched)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                cid = int(r.cluster_id)
                sc = sk[(sk.Model == model) & (sk.Cluster == cid)]
                if sc.empty:
                    continue
                s = sc.iloc[0]
                gt_lo, gt_hi = wilson_ci(float(s.GT), int(s.n_gt))
                gen_lo, gen_hi = wilson_ci(float(s.GEN), int(s.n_gen))
                side = side_of_diagonal(s.GT, s.GEN)
                mark = "†" if bool(s.low_n) else ""
                # best-matching K=20 name
                if k == 20:
                    k20_name = names20.get(cid, name_map.get(cid, ""))
                    share = 1.0
                    k20_id = cid
                else:
                    k20_id = int(s.best_k20_cluster) if "best_k20_cluster" in s and pd.notna(s.best_k20_cluster) else -1
                    k20_name = names20.get(k20_id, str(s.get("best_k20_name", "")))
                    share = float(s.best_k20_share) if "best_k20_share" in s and pd.notna(s.best_k20_share) else float(r.instance_share)
                rows.append(
                    {
                        "K": k,
                        "concept": concept,
                        "model": display(model),
                        "matched_cluster_id": cid,
                        "matched_cluster_name_or_k20": (name_map.get(cid) or k20_name) + mark,
                        "best_k20_id": k20_id,
                        "best_k20_name": k20_name,
                        "share": share if k != 20 else float(r.instance_share),
                        "ICARE_GT": float(s.GT),
                        "ICARE_GT_CI95": f"[{gt_lo:.3f}, {gt_hi:.3f}]",
                        "ICARE_GEN": float(s.GEN),
                        "ICARE_GEN_CI95": f"[{gen_lo:.3f}, {gen_hi:.3f}]",
                        "n_gt": int(s.n_gt),
                        "n_gen": int(s.n_gen),
                        "side_of_diagonal": side,
                        "low_n": bool(s.low_n),
                        "delta_boot_CI95": f"[{s.CI_low:.3f}, {s.CI_high:.3f}]",
                    }
                )
    part2 = pd.DataFrame(rows)
    part2.to_csv(out / "SuppTableB_exp4_part2.csv", index=False)

    # LaTeX part 1
    lines = [
        r"% Supp Table B Part 1",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Exp 4: sensitivity to number of clusters $K$.}",
        r"\label{tab:supp_exp4_part1}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"$K$ & Model & Below ($n$) & Below (\%) & Above ($n$) & Above (\%) & Agree w/ $K$=20 (\%) & Spearman GT / GEN \\",
        r"\midrule",
    ]
    for _, r in part1.sort_values(["K", "Model"]).iterrows():
        lines.append(
            f"{int(r.K)} & {_tex(r.model_display)} & "
            f"{int(r.n_below)} & {r.pct_below:.1f} & {int(r.n_above)} & {r.pct_above:.1f} & "
            f"{r.label_agreement_pct:.1f} & {r.spearman_ICARE_GT:.3f} / {r.spearman_ICARE_GEN:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [
        r"% Supp Table B Part 2",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Exp 4: Lung Lobe Abnormality and Calcified Granuloma Location across $K$.}",
        r"\label{tab:supp_exp4_part2}",
        r"\begin{tabular}{lllrlrrll}",
        r"\toprule",
        r"$K$ & Concept & Model & ID & K20 name (share) & ICARE-GT [95\% CI] & ICARE-GEN [95\% CI] & $n_{\mathrm{gt}}/n_{\mathrm{gen}}$ & Side \\",
        r"\midrule",
    ]
    for _, r in part2.sort_values(["concept", "K", "model"]).iterrows():
        side_lab = {"below": "below", "above": "above", "on": "on"}[r.side_of_diagonal]
        lines.append(
            f"{int(r.K)} & {r.concept} & {_tex(r.model)} & {int(r.matched_cluster_id)} & "
            f"{_tex(r.best_k20_name)[:28]} ({r.share:.2f}) & "
            f"{r.ICARE_GT:.3f} {r.ICARE_GT_CI95} & {r.ICARE_GEN:.3f} {r.ICARE_GEN_CI95} & "
            f"{int(r.n_gt)}/{int(r.n_gen)}{'†' if r.low_n else ''} & {side_lab} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize $^\dagger$$n_{\mathrm{gt}}$ or $n_{\mathrm{gen}}$ $<$ 30. "
        r"Rate 95\% CIs are Wilson intervals from instance counts; bootstrap CIs in source data are for GT$-$GEN.",
        r"\end{flushleft}",
        r"\end{table*}",
    ]
    (out / "SuppTableB_exp4.tex").write_text("\n".join(lines))
    print(f"Wrote SuppTableB → {out}")
    return part1, part2


def build_supp_table_c(exp4: Path, names20: Dict[int, str], out: Path) -> pd.DataFrame:
    rows = []
    for k in [10, 15, 25, 30]:
        sk = load_scores(exp4, k)
        name_map = load_names(exp4, k)
        # correspondence is shared across models; take first model for mapping cols
        model0 = MODEL_ORDER[0]
        sub = sk[sk.Model == model0]
        for _, r in sub.iterrows():
            cid = int(r.Cluster)
            if "best_k20_cluster" in r and pd.notna(r.best_k20_cluster):
                k20_id = int(r.best_k20_cluster)
                share = float(r.best_k20_share)
                k20_name = names20.get(k20_id, str(r.get("best_k20_name", "")))
            else:
                k20_id, share, k20_name = -1, float("nan"), ""
            local_name = name_map.get(cid, "")
            rows.append(
                {
                    "K": k,
                    "cluster_id": cid,
                    "cluster_name": local_name,  # filled when cluster_names_K{K}.json present
                    "best_k20_id": k20_id,
                    "best_k20_name": k20_name,
                    "best_k20_share": share,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "SuppTableC_exp4_cluster_keys.csv", index=False)

    lines = [
        r"% Supp Table C — cluster keys",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Exp 4 cluster keys: best-matching $K$=20 name and share.}",
        r"\label{tab:supp_exp4_keys}",
        r"\begin{tabular}{rrlrl}",
        r"\toprule",
        r"$K$ & ID & Name (if available) & Best $K$=20 ID & Best $K$=20 name (share) \\",
        r"\midrule",
    ]
    for _, r in df.sort_values(["K", "cluster_id"]).iterrows():
        nm = str(r.cluster_name) if r.cluster_name else "—"
        lines.append(
            f"{int(r.K)} & {int(r.cluster_id)} & {_tex(nm)[:32]} & "
            f"{int(r.best_k20_id)} & {_tex(r.best_k20_name)[:36]} ({r.best_k20_share:.2f}) \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "SuppTableC_exp4_cluster_keys.tex").write_text("\n".join(lines))
    print(f"Wrote SuppTableC → {out}")
    return df


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_xlsx_stdlib(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """Minimal XLSX writer (no openpyxl) for Source Data sheets."""
    import zipfile
    from xml.sax.saxutils import escape

    def col_name(i: int) -> str:
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    def sheet_xml(df: pd.DataFrame) -> str:
        rows_xml = []
        # header
        cells = []
        for j, col in enumerate(df.columns):
            ref = f"{col_name(j)}1"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(col))}</t></is></c>'
            )
        rows_xml.append(f'<row r="1">{"".join(cells)}</row>')
        for i, row in enumerate(df.itertuples(index=False), start=2):
            cells = []
            for j, val in enumerate(row):
                ref = f"{col_name(j)}{i}"
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                if isinstance(val, (int, np.integer)):
                    cells.append(f'<c r="{ref}"><v>{int(val)}</v></c>')
                elif isinstance(val, (float, np.floating)):
                    cells.append(f'<c r="{ref}"><v>{float(val)}</v></c>')
                elif isinstance(val, (bool, np.bool_)):
                    cells.append(f'<c r="{ref}" t="b"><v>{1 if val else 0}</v></c>')
                else:
                    cells.append(
                        f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(val))}</t></is></c>'
                    )
            rows_xml.append(f'<row r="{i}">{"".join(cells)}</row>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
        )

    names = list(sheets.keys())
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(
            f'<sheet name="{escape(n[:31])}" sheetId="{i+1}" r:id="rId{i+1}"/>'
            for i, n in enumerate(names)
        )
        + "</sheets></workbook>"
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{i+1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            f'Target="xl/worksheets/sheet{i+1}.xml"/>'
            for i in range(len(names))
        )
        # fix: workbook rels should point to worksheets with worksheet type
        + "</Relationships>"
    )
    # Correct workbook relationships
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{i+1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i+1}.xml"/>'
            for i in range(len(names))
        )
        + "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(len(names))
        )
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        for i, (name, df) in enumerate(sheets.items(), start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(df))


def write_source_data(out: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    xlsx = out / "SourceData_R2.xlsx"
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception:
        _write_xlsx_stdlib(xlsx, sheets)
    # Always also dump CSVs for transparency
    sd = out / "SourceData_R2_csv"
    sd.mkdir(exist_ok=True)
    for name, df in sheets.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
        df.to_csv(sd / f"{safe}.csv", index=False)
    print(f"Wrote {xlsx}")





def write_captions(out: Path, concept_ids: Optional[Dict[int, Dict[str, int]]] = None) -> None:
    concept_ids = concept_ids or {}
    concept_lines = []
    for k in KS_GRID:
        ids = concept_ids.get(k, {})
        lung = ids.get("Lung Lobe Abnormality", "NA")
        calc = ids.get("Calcified Granuloma Location", "NA")
        concept_lines.append(
            f"- K = {k}: Lung Lobe Abnormality = cluster {lung}; "
            f"Calcified Granuloma Location = cluster {calc}."
        )
    concept_block = "\n".join(concept_lines)
    text = f"""# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Point style matches submitted Fig. 4 (`cluster_analysis.create_cluster_summary_table_and_scatterplots`): marker area = GT question count / 30, black edges, ID labels offset (5, 5) pt. Colour encodes side of the diagonal (the only intentional style difference from Fig. 4). Question counts are also reported in Supplementary Tables A and B.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Point size uses pooled GT question count / 30 (same formula as Fig. 4); colour encodes side of the diagonal. Points are labelled by K=20 cluster ID as in Fig. 4. The dashed line is the identity. (d) Cluster ID → name key.

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡).

## Supp Fig. B
**Supplementary Figure B. ICARE cluster scores across the number of clusters K (Fig. 4–style grid).**  
Each panel matches the submitted Fig. 4 scatter style. Rows: K = 10, 15, 20, 25, 30 (K=20 shaded and labelled Fig. 4); columns: CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC, MAIRA2. Point size = GT question count / 30; colour encodes side of the diagonal.

Best-matching cluster IDs for the tracked concepts:
{concept_block}

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN. Part 2: Lung Lobe Abnormality and Calcified Granuloma Location tracking.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. The K=20 name key is Fig. 4d.
"""
    (out / "captions.md").write_text(text)
    print(f"Wrote {out / 'captions.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root (ICARE_score_rev_stats)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/revision_r2/final)",
    )
    args = ap.parse_args()
    root: Path = args.root
    out: Path = args.out or (root / "results" / "revision_r2" / "final")
    exp4 = root / "results" / "revision_r2" / "exp4"
    exp5 = root / "results" / "revision_r2" / "exp5"
    out.mkdir(parents=True, exist_ok=True)

    print("=== Model-name mapping (from cluster_analysis.py) ===")
    for raw, disp in MODEL_DISPLAY_NAMES.items():
        print(f"  {raw!r} → {disp!r}")
    print("Plot order:", [display(m) for m in MODEL_ORDER])
    print("=== Marker / layout settings (submitted Fig. 4) ===")
    print(f"  size: s = count / {FIG4_SIZE_DIVISOR:g}")
    print(f"  alpha={MARKER_ALPHA}, edgecolors={MARKER_EDGECOLOR!r}, lw={MARKER_EDGEWIDTH}")
    print(f"  labels: annotate xytext=(5,5), fontsize=18")
    print(f"  axes lim={AX_LIM}; fonts from Fig. 4 (size 20 bold)")

    setup_style()
    names20 = load_names(exp4, 20)
    if not names20:
        print("ERROR: cluster_names_K20.json not found", file=sys.stderr)
        return 1

    fig_a_src = build_supp_fig_a(exp5, names20, out)
    tab_a_wide, tab_a_sum = build_supp_table_a(exp5, names20, out)

    metrics = build_exp4_metrics(exp4)
    metrics.to_csv(out / "exp4_metrics_for_supp.csv", index=False)
    fig_b_src = build_supp_fig_b(exp4, metrics, names20, out)
    tab_b1, tab_b2 = build_supp_table_b(exp4, metrics, names20, out)
    tab_c = build_supp_table_c(exp4, names20, out)

    write_captions(out, concept_cluster_ids(exp4))
    sheets = {
        "FigA_panels_abc": fig_a_src,
        "FigA_key": pd.DataFrame({"cluster_id": list(range(20)), "cluster_name": [names20.get(i, "") for i in range(20)]}),
        "TableA_clusters": tab_a_wide,
        "TableA_summary": tab_a_sum,
        "FigB_grid": fig_b_src["fig_d"],
        "TableB_metrics_lines": fig_b_src["metrics"],
        "TableB_part1": tab_b1,
        "TableB_part2": tab_b2,
        "TableC_keys": tab_c,
    }
    write_source_data(out, sheets)

    print("\n=== Outputs ===")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(root)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
