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
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

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

# Fig. 4 (cluster_analysis.py) marker / axis settings reused:
#   s = GT_Count / 30   (same formula, scale, GT count definition)
#   alpha=0.7, edgecolors="black", linewidth=0.5
#   diagonal: "k--", linewidth=1.5, alpha=0.7
#   grid(True); default spines (full box); bold fonts; model names as titles
FIG4_SIZE_DIVISOR = 30.0
FIG4_ALPHA = 0.7
FIG4_EDGE = "black"
FIG4_EDGE_LW = 0.5
FIG4_DIAG_LW = 1.5
FIG4_DIAG_ALPHA = 0.7
AX_LIM = (-0.02, 1.03)

KS = [10, 15, 20, 25, 30]
KS_GRID = [10, 15, 20, 25, 30]  # include K=20 (Fig. 4 setting)
MM = 1 / 25.4
FIG_W = 180 * MM  # double-column
FIG_B_H = 230 * MM
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
    """Fig. 4 uses bold global fonts at large figure size; scale to ≥7 pt at 180 mm."""
    mpl.rcParams.update(
        {
            "font.size": 8,
            "font.weight": "bold",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def fig4_point_size(gt_count) -> np.ndarray:
    """Manuscript Fig. 4: s = GT_Count / 30 (no sqrt / min-size compression)."""
    return np.asarray(gt_count, dtype=float) / FIG4_SIZE_DIVISOR


def apply_fig4_axes(ax, *, show_xlabel: bool = True, show_ylabel: bool = True,
                     show_xticklabels: bool = True, show_yticklabels: bool = True) -> None:
    ax.set_xlim(*AX_LIM)
    ax.set_ylim(*AX_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    # Fixed major ticks (avoids gutter spill of auto-ticks under equal aspect)
    ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if show_xlabel:
        ax.set_xlabel("ICARE-GEN", fontweight="bold")
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("ICARE-GT", fontweight="bold")
    else:
        ax.set_ylabel("")
    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_yticklabels:
        ax.set_yticklabels([])
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_weight("bold")


def draw_diagonal(ax) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=FIG4_DIAG_LW, alpha=FIG4_DIAG_ALPHA, zorder=1)


def panel_letter_outside(ax, letter: str) -> None:
    """Top-left, outside the axes (Fig. 4 uses ~(-0.15, 1.05))."""
    ax.text(
        -0.14,
        1.06,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def offset_cluster_labels(ax, xs, ys, labels, fontsize: float = 6.5) -> None:
    """Short leader-line offsets; simple repulsion so IDs stay readable (no adjustText)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(labels)
    # Initial offsets in points, alternating
    base = [
        (5, 5),
        (-8, 5),
        (5, -8),
        (-8, -8),
        (10, 0),
        (-10, 0),
        (0, 10),
        (0, -10),
        (8, 8),
        (-10, 6),
    ]
    offsets = [base[i % len(base)] for i in range(n)]
    # Repel labels that share nearly the same data position
    for _ in range(40):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                # screen-ish distance using data coords + offset proxy
                dx = (xs[i] - xs[j]) + (offsets[i][0] - offsets[j][0]) * 0.004
                dy = (ys[i] - ys[j]) + (offsets[i][1] - offsets[j][1]) * 0.004
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 0.045:
                    # push apart in offset space
                    push = 3.0
                    if abs(dx) + abs(dy) < 1e-9:
                        ang = (i + j) * 0.7
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / (dist + 1e-9), dy / (dist + 1e-9)
                    oi = list(offsets[i])
                    oj = list(offsets[j])
                    oi[0] += push * ux
                    oi[1] += push * uy
                    oj[0] -= push * ux
                    oj[1] -= push * uy
                    # clip offset magnitude
                    for o in (oi, oj):
                        mag = (o[0] ** 2 + o[1] ** 2) ** 0.5
                        if mag > 18:
                            o[0], o[1] = o[0] * 18 / mag, o[1] * 18 / mag
                    offsets[i], offsets[j] = tuple(oi), tuple(oj)
                    moved = True
        if not moved:
            break
    for x, y, lab, (ox, oy) in zip(xs, ys, labels, offsets):
        ax.annotate(
            str(lab),
            (x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            color="0.1",
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.4, shrinkA=0, shrinkB=1.5),
            zorder=5,
            clip_on=False,
        )


def size_legend_handles(example_counts=(1500, 8000, 25000)):
    """Legend markers using Fig. 4 size mapping s = count/30 (true scatter sizes)."""
    handles = []
    for c in example_counts:
        h = plt.scatter(
            [],
            [],
            s=float(c) / FIG4_SIZE_DIVISOR,
            c="0.45",
            alpha=FIG4_ALPHA,
            edgecolors=FIG4_EDGE,
            linewidths=FIG4_EDGE_LW,
            label=f"n≈{int(c):,}",
        )
        handles.append(h)
    return handles


def colour_legend_handles():
    return [
        Patch(facecolor=COLOR_BELOW, edgecolor=FIG4_EDGE, linewidth=FIG4_EDGE_LW, label="Below diagonal (ICARE-GT < ICARE-GEN)"),
        Patch(facecolor=COLOR_ABOVE, edgecolor=FIG4_EDGE, linewidth=FIG4_EDGE_LW, label="Above diagonal (ICARE-GT > ICARE-GEN)"),
    ]


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.pdf')} and .png")


# ========================= Exp 5 / Supp Fig A & Table A =========================


def pooled_n_gt_exp5(exp5: Path) -> pd.DataFrame:
    """Pooled GT question count across all 5 eval seeds (Fig. 4 count definition)."""
    sc = pd.read_csv(exp5 / "model_seeds_all" / "scores_by_eval_seed.csv")
    g = sc.groupby(["Model", "Cluster"], as_index=False)["n_gt"].sum()
    g = g.rename(columns={"n_gt": "pooled_n_gt"})
    return g


def build_supp_fig_a(exp5: Path, names: Dict[int, str], out: Path) -> pd.DataFrame:
    summ = pd.read_csv(exp5 / "model_seeds_all" / "across_seed_summary.csv")
    pooled = pooled_n_gt_exp5(exp5)
    summ = summ.merge(pooled, on=["Model", "Cluster"], how="left")

    # Fig. 4-like layout: 3 stacked model panels + key column
    fig = plt.figure(figsize=(FIG_W, FIG_W * 1.15))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.45, 1.0], height_ratios=[1, 1, 1], figure=fig
    )
    gs.update(hspace=0.38, wspace=0.35)
    letters = ["a", "b", "c"]
    source_rows = []

    # Shared legend axis strip above key
    legend_ax = None
    for i, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[i, 0])
        sub = summ[summ.Model == model].copy()
        xs, ys, labs, sides = [], [], [], []
        sizes = fig4_point_size(sub.pooled_n_gt.fillna(sub.mean_n_gt))
        colors = []
        for _, r in sub.iterrows():
            side = side_of_diagonal(r.GT_mean, r.GEN_mean)
            colors.append(side_color(side))
            xs.append(r.GEN_mean)
            ys.append(r.GT_mean)
            labs.append(int(r.Cluster))
            sides.append(side)
            ax.errorbar(
                r.GEN_mean,
                r.GT_mean,
                xerr=r.GEN_sd,
                yerr=r.GT_sd,
                fmt="none",
                ecolor=side_color(side),
                elinewidth=0.7,
                capsize=1.5,
                alpha=0.75,
                zorder=2,
            )
            source_rows.append(
                {
                    "panel": letters[i],
                    "model": display(model),
                    "cluster_id": int(r.Cluster),
                    "cluster_name": names.get(int(r.Cluster), r.cluster_name),
                    "ICARE_GEN_mean": r.GEN_mean,
                    "ICARE_GEN_sd": r.GEN_sd,
                    "ICARE_GT_mean": r.GT_mean,
                    "ICARE_GT_sd": r.GT_sd,
                    "pooled_n_gt": r.pooled_n_gt,
                    "side_of_diagonal": side,
                }
            )
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=colors,
            alpha=FIG4_ALPHA,
            edgecolors=FIG4_EDGE,
            linewidths=FIG4_EDGE_LW,
            zorder=3,
        )
        draw_diagonal(ax)
        apply_fig4_axes(ax)
        ax.set_title(display(model), fontweight="bold", pad=8)
        panel_letter_outside(ax, letters[i])
        offset_cluster_labels(ax, xs, ys, labs, fontsize=6.5)
        if i == 0:
            legend_ax = ax

    # Colour + size legend outside data: place in figure coordinates near key column top
    fig.legend(
        handles=colour_legend_handles() + size_legend_handles(
            example_counts=(
                int(np.nanpercentile(summ.pooled_n_gt, 25)),
                int(np.nanpercentile(summ.pooled_n_gt, 50)),
                int(np.nanpercentile(summ.pooled_n_gt, 90)),
            )
        ),
        loc="upper left",
        bbox_to_anchor=(0.58, 0.98),
        frameon=True,
        framealpha=0.95,
        fontsize=6.5,
        title="Point colour / size (Fig. 4: n/30)",
        title_fontsize=7,
    )

    # (d) key panel spanning right column
    axd = fig.add_subplot(gs[:, 1])
    axd.axis("off")
    axd.text(-0.08, 1.02, "(d)", transform=axd.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", clip_on=False)
    rows = [[str(cid), names.get(cid, "")] for cid in range(20)]
    table = axd.table(
        cellText=rows,
        colLabels=["ID", "Cluster name"],
        loc="center",
        cellLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.88],
        colWidths=[0.14, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E6E6E6")
            cell.set_text_props(weight="bold", fontsize=7)
        else:
            cell.set_text_props(weight="bold", fontsize=6.5)
        if col == 0:
            cell.get_text().set_ha("center")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05)
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
    """Full-page Fig. 4-style grid only (K=10,15,20,25,30 × 3 models)."""
    concept_ids = concept_cluster_ids(exp4)
    fig = plt.figure(figsize=(FIG_W, FIG_B_H))
    nrows, ncols = len(KS_GRID), len(MODEL_ORDER)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.28, wspace=0.18, figure=fig)

    source_d = []
    # Legend outside: top of figure
    fig.legend(
        handles=colour_legend_handles()
        + size_legend_handles(example_counts=(1000, 8000, 25000))
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="w",
                markeredgecolor="black",
                markeredgewidth=1.8,
                markersize=8,
                label="Lung Lobe / Calcified Granuloma (best match)",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=6,
        title="Colour = side of diagonal; size = GT question count / 30 (Fig. 4)",
        title_fontsize=6.5,
    )

    for ri, k in enumerate(KS_GRID):
        sk = load_scores(exp4, k)
        name_map = load_names(exp4, k)
        highlight = set(concept_ids.get(k, {}).values())
        for ci, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            sub = sk[sk.Model == model].copy()
            xs = sub.GEN.to_numpy(dtype=float)
            ys = sub.GT.to_numpy(dtype=float)
            labs = [int(c) for c in sub.Cluster]
            sizes = fig4_point_size(sub.n_gt)
            colors = [side_color(side_of_diagonal(g, e)) for g, e in zip(ys, xs)]
            # Draw non-highlight first, then highlight with thicker edge
            for is_hi in (False, True):
                mask = np.array([lab in highlight for lab in labs], dtype=bool)
                if is_hi:
                    m = mask
                    lw = 1.8
                else:
                    m = ~mask
                    lw = FIG4_EDGE_LW
                if not m.any():
                    continue
                ax.scatter(
                    xs[m],
                    ys[m],
                    s=sizes[m],
                    c=np.array(colors, dtype=object)[m].tolist(),
                    alpha=FIG4_ALPHA,
                    edgecolors=FIG4_EDGE,
                    linewidths=lw,
                    zorder=3 + int(is_hi),
                )
            draw_diagonal(ax)
            apply_fig4_axes(
                ax,
                show_xlabel=(ri == nrows - 1),
                show_ylabel=False,  # custom K= row label below
                show_xticklabels=(ri == nrows - 1),
                show_yticklabels=(ci == 0),
            )
            offset_cluster_labels(ax, xs, ys, labs, fontsize=5.5 if k >= 25 else 6.0)

            if ri == 0:
                ax.set_title(display(model), fontweight="bold", pad=6, fontsize=8)
            else:
                ax.set_title("")
            if ci == 0:
                ylab = f"K={k}\nICARE-GT"
                if k == 20:
                    ylab = f"K={k} (Fig. 4)\nICARE-GT"
                ax.set_ylabel(ylab, fontsize=7, fontweight="bold")
            if ri == nrows - 1:
                ax.set_xlabel("ICARE-GEN", fontsize=7, fontweight="bold")

            if ri == 0 and ci == 0:
                panel_letter_outside(ax, "a")

            # Light band to mark Fig. 4 row
            if k == 20:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.4)
                    spine.set_color("0.15")

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
                        "highlight_concept": next(
                            (c for c, cid in concept_ids.get(k, {}).items() if cid == int(r.Cluster)),
                            "",
                        ),
                    }
                )

    # Tighten top for legend
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.08, right=0.98)
    save_fig(fig, out / "SuppFigB_exp4_k_sensitivity")

    # Source sheets for grid only (line-plot sheets removed)
    grid_df = pd.DataFrame(source_d)
    return {
        "metrics": metrics,
        "fig_d": grid_df,
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



def write_captions(out: Path) -> None:
    text = """# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Point size follows the manuscript Fig. 4 mapping (GT question count / 30). Cluster point labels are IDs; names appear in key panels and tables.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Point size uses the pooled GT question count across all five seeds with the Fig. 4 size mapping (count / 30); colour encodes side of the diagonal (red: below / omission-dominated; blue: above / hallucination-dominated). Points are labelled by K=20 cluster ID. The dashed line is the identity. (d) Cluster ID → name key (same as Fig. 4d).

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡). Summary rows report the percentage of clusters on the same side of the diagonal in all five seeds; median across-seed SD of ICARE-GT and of ICARE-GEN; report-level Jaccard of clusters probed for the same report across seeds versus different reports (mean with 95% CI); and unique questions common to all five seeds (9,569 / 78,708).

## Supp Fig. B
**Supplementary Figure B. ICARE cluster scores across the number of clusters K (Fig. 4–style grid).**  
Each panel is a Fig. 4–style scatterplot of ICARE-GEN (x) versus ICARE-GT (y) for one model at one K. Rows: K = 10, 15, 20, 25, 30 (the K=20 row is the Fig. 4 setting); columns: CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC, MAIRA2. Point size follows Fig. 4 (GT question count / 30); colour encodes side of the diagonal; points are labelled by cluster ID. Clusters best-matching **Lung Lobe Abnormality** and **Calcified Granuloma Location** are highlighted with a thicker outline. Cluster ID → name keys for K ≠ 20 are in Supplementary Table C; the K=20 key is Fig. 4d / Supplementary Figure A(d).

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN (n = 20). Part 2: Lung Lobe Abnormality and Calcified Granuloma Location — best-matching cluster at each K (with share); per model, ICARE-GT and ICARE-GEN with 95% CIs, n_gt / n_gen, and side of the diagonal; † marks < 30 questions.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. Local names are filled when `cluster_names_K{K}.json` is present (LLM naming can be applied later without recomputing scores). The K=20 name key is Fig. 4d.
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
    print("=== Fig. 4 settings reused (cluster_analysis.py) ===")
    print(f"  point size: s = GT_Count / {FIG4_SIZE_DIVISOR:g}")
    print(f"  alpha={FIG4_ALPHA}, edgecolors={FIG4_EDGE!r}, linewidth={FIG4_EDGE_LW}")
    print(f"  diagonal: k--, lw={FIG4_DIAG_LW}, alpha={FIG4_DIAG_ALPHA}")
    print(f"  grid=True; default spines (full box); axes lim={AX_LIM}")
    print(f"  model titles from MODEL_DISPLAY_NAMES; fonts bold")

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

    write_captions(out)
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
