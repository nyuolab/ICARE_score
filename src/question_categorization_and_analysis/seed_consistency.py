#!/usr/bin/env python3
"""Exp 5: eval-seed consistency (Nature Communications R2, reviewer R4).

Stored K=20 labels/names only. No re-clustering / embedding / LLM.
Unit of comparison = eval_seed. Labels follow Fig. 4 (GT vs GEN diagonal).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_stats.bootstrap_utils import make_indices, percentile_ci  # noqa: E402

DEFAULT_FIG4 = Path(
    "/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/"
    "MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis"
)
DEFAULT_OUT = REPO_ROOT / "results" / "revision_r2" / "exp5"
EVAL_SEEDS = (123, 456, 789, 202, 101)
BOOT_SEED = 42
T_CRIT = 2.776  # t_{0.975, df=4}
CONCEPT_LUNG = "Lung Lobe Abnormality"
CONCEPT_CALC = "Calcified Granuloma Location"
LABEL_COLORS = {
    "omission-dominated": "#d62728",
    "hallucination-dominated": "#1f77b4",
    "balanced": "#7f7f7f",
}
MODEL_SHORT = {
    "chexpert-mimic-cxr-findings-baseline": "CheX+MIMIC",
    "mimic-cxr-findings-baseline": "MIMIC",
    "maira-2": "MAIRA-2",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 5: seed consistency")
    p.add_argument("--fig4-root", type=Path, default=DEFAULT_FIG4)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="200 bootstrap resamples")
    p.add_argument(
        "--model-seeds",
        type=str,
        default="all",
        help='"all" (pool model seeds 1-5 per eval seed) or "1" (model_seed==1 only)',
    )
    return p.parse_args()


def fig4_label(gt: float, gen: float) -> str:
    """Manuscript Fig. 4 rule: below diagonal = omission, above = hallucination."""
    if not (np.isfinite(gt) and np.isfinite(gen)):
        return "balanced"
    if gt < gen:
        return "omission-dominated"
    if gt > gen:
        return "hallucination-dominated"
    return "balanced"


def score_slice_with_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    n_boot: int,
    boot_seed: int = BOOT_SEED,
) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model_name"].unique()):
        mdf = df[df["model_name"] == model]
        reports = np.array(sorted(mdf["Report_ID"].unique()))
        n_rep = len(reports)
        rid_to_i = {r: i for i, r in enumerate(reports)}
        indices = make_indices(n_rep, n_boot, boot_seed) if n_rep > 0 else np.zeros((n_boot, 0), dtype=int)

        for cid in sorted(mdf[cluster_col].dropna().unique()):
            cdf = mdf[mdf[cluster_col] == cid]
            gt = cdf[cdf["metric"] == "gt_reports_as_ref"]
            gen = cdf[cdf["metric"] == "gen_reports_as_ref"]

            def _pack(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
                agree = np.zeros(n_rep, dtype=np.float64)
                count = np.zeros(n_rep, dtype=np.float64)
                if sub.empty or n_rep == 0:
                    return agree, count
                ok = (
                    sub["Predicted_Answer_Using_Gen"].to_numpy()
                    == sub["Predicted_Answer_Using_GT"].to_numpy()
                ).astype(np.float64)
                ridx = sub["Report_ID"].map(rid_to_i).to_numpy()
                np.add.at(agree, ridx, ok)
                np.add.at(count, ridx, 1.0)
                return agree, count

            gt_a, gt_n = _pack(gt)
            gen_a, gen_n = _pack(gen)

            def _rate(a: np.ndarray, n: np.ndarray, idx: np.ndarray) -> float:
                den = n[idx].sum()
                return float(a[idx].sum() / den) if den > 0 else float("nan")

            gt_point = _rate(gt_a, gt_n, np.arange(n_rep))
            gen_point = _rate(gen_a, gen_n, np.arange(n_rep))
            delta = gt_point - gen_point
            n_gt = int(gt_n.sum())
            n_gen = int(gen_n.sum())

            boot_delta = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                idx = indices[b]
                boot_delta[b] = _rate(gt_a, gt_n, idx) - _rate(gen_a, gen_n, idx)
            finite = boot_delta[np.isfinite(boot_delta)]
            if finite.size:
                lo, hi = percentile_ci(finite)
            else:
                lo, hi = float("nan"), float("nan")

            rows.append(
                {
                    "Model": model,
                    "Cluster": int(cid),
                    "GT": gt_point,
                    "GEN": gen_point,
                    "n_gt": n_gt,
                    "n_gen": n_gen,
                    "n": n_gt,
                    "delta": delta,
                    "CI_low": lo,
                    "CI_high": hi,
                    "label": fig4_label(gt_point, gen_point),
                    "low_n": bool(n_gt < 30 or n_gen < 30),
                }
            )
    return pd.DataFrame(rows)


def across_seed_summary(scores: pd.DataFrame, name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for (model, cid), g in scores.groupby(["Model", "Cluster"]):
        def _summ(col: str) -> Tuple[float, float, float, float]:
            mu = float(g[col].mean())
            sd = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
            half = T_CRIT * sd / np.sqrt(len(g))
            return mu, sd, mu - half, mu + half

        gt_m, gt_s, gt_lo, gt_hi = _summ("GT")
        gen_m, gen_s, gen_lo, gen_hi = _summ("GEN")
        d_m, d_s, d_lo, d_hi = _summ("delta")
        lab_counts = g["label"].value_counts().to_dict()
        rows.append(
            {
                "Model": model,
                "Cluster": int(cid),
                "cluster_name": name_map.get(int(cid), str(cid)),
                "GT_mean": gt_m,
                "GT_sd": gt_s,
                "GT_t_low": gt_lo,
                "GT_t_high": gt_hi,
                "GEN_mean": gen_m,
                "GEN_sd": gen_s,
                "GEN_t_low": gen_lo,
                "GEN_t_high": gen_hi,
                "delta_mean": d_m,
                "delta_sd": d_s,
                "delta_t_low": d_lo,
                "delta_t_high": d_hi,
                "n_seeds_omission": int(lab_counts.get("omission-dominated", 0)),
                "n_seeds_hallucination": int(lab_counts.get("hallucination-dominated", 0)),
                "n_seeds_balanced": int(lab_counts.get("balanced", 0)),
                "same_label_all_seeds": int(g["label"].nunique() == 1),
                "mean_n_gt": float(g["n_gt"].mean()),
                "mean_n_gen": float(g["n_gen"].mean()),
                "any_low_n": bool(g["low_n"].any()),
                "n_seeds_low_n": int(g["low_n"].sum()),
            }
        )
    return pd.DataFrame(rows)


def label_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in summary.groupby("Model"):
        def _pct(sub: pd.DataFrame) -> Tuple[float, float]:
            if len(sub) == 0:
                return float("nan"), float("nan")
            pct = 100.0 * sub["same_label_all_seeds"].mean()
            w = sub["mean_n_gt"].to_numpy()
            w_pct = (
                100.0 * np.average(sub["same_label_all_seeds"], weights=w)
                if w.sum() > 0
                else float("nan")
            )
            return float(pct), float(w_pct)

        pct_all, w_all = _pct(g)
        g_ok = g[~g["any_low_n"]]
        pct_ex, w_ex = _pct(g_ok)
        rows.append(
            {
                "Model": model,
                "pct_same_label_all_clusters": pct_all,
                "size_weighted_pct_all": w_all,
                "pct_same_label_excl_low_n": pct_ex,
                "size_weighted_pct_excl_low_n": w_ex,
                "n_low_n_clusters": int(g["any_low_n"].sum()),
                "median_delta_sd": float(g["delta_sd"].median()),
                "max_delta_sd": float(g["delta_sd"].max()),
                "median_GT_sd": float(g["GT_sd"].median()),
                "max_GT_sd": float(g["GT_sd"].max()),
                "median_GEN_sd": float(g["GEN_sd"].median()),
                "max_GEN_sd": float(g["GEN_sd"].max()),
                "sd_of_cluster_delta_means": float(g["delta_mean"].std(ddof=1)),
                "Lung_Lobe_Abnormality_same": int(
                    g.loc[g.cluster_name == CONCEPT_LUNG, "same_label_all_seeds"].iloc[0]
                )
                if (g.cluster_name == CONCEPT_LUNG).any()
                else None,
                "Calcified_Granuloma_Location_same": int(
                    g.loc[g.cluster_name == CONCEPT_CALC, "same_label_all_seeds"].iloc[0]
                )
                if (g.cluster_name == CONCEPT_CALC).any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def concept_summary(scores: pd.DataFrame, summary: pd.DataFrame, name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    id_by_name = {v: k for k, v in name_map.items()}
    for concept in (CONCEPT_LUNG, CONCEPT_CALC):
        cid = id_by_name.get(concept)
        if cid is None:
            continue
        for model in sorted(scores.Model.unique()):
            per = scores[(scores.Model == model) & (scores.Cluster == cid)].sort_values("eval_seed")
            for _, r in per.iterrows():
                rows.append(
                    {
                        "concept": concept,
                        "Model": model,
                        "eval_seed": int(r.eval_seed),
                        "GT": r.GT,
                        "GEN": r.GEN,
                        "n_gt": r.n_gt,
                        "n_gen": r.n_gen,
                        "delta": r.delta,
                        "CI_low": r.CI_low,
                        "CI_high": r.CI_high,
                        "label": r.label,
                        "low_n": r.low_n,
                        "scope": "per_seed",
                    }
                )
            sm = summary[(summary.Model == model) & (summary.Cluster == cid)]
            if len(sm):
                s = sm.iloc[0]
                rows.append(
                    {
                        "concept": concept,
                        "Model": model,
                        "eval_seed": None,
                        "GT": s.GT_mean,
                        "GEN": s.GEN_mean,
                        "n_gt": s.mean_n_gt,
                        "n_gen": s.mean_n_gen,
                        "delta": s.delta_mean,
                        "CI_low": s.delta_t_low,
                        "CI_high": s.delta_t_high,
                        "label": None,
                        "low_n": s.any_low_n,
                        "scope": "across_seed_mean",
                        "delta_sd": s.delta_sd,
                        "n_seeds_omission": s.n_seeds_omission,
                        "n_seeds_hallucination": s.n_seeds_hallucination,
                        "n_seeds_balanced": s.n_seeds_balanced,
                    }
                )
    return pd.DataFrame(rows)


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return float(len(a & b) / len(u)) if u else float("nan")


def report_level_jaccard(cq: pd.DataFrame, metric: str, n_ref_pairs: int = 1000) -> pd.DataFrame:
    rows = []
    sub = cq[cq["metric"] == metric]
    for model in sorted(sub["model_name"].unique()):
        mdf = sub[sub["model_name"] == model]
        for rid, g in mdf.groupby("Report_ID"):
            sets = {
                int(es): set(g.loc[g.eval_seed == es, "cluster_id"])
                for es in EVAL_SEEDS
                if (g.eval_seed == es).any()
            }
            if len(sets) < 2:
                continue
            for a, b in combinations(sorted(sets), 2):
                rows.append(
                    {
                        "Model": model,
                        "metric": metric,
                        "pair_type": "same_report_across_seeds",
                        "Report_ID_a": rid,
                        "Report_ID_b": rid,
                        "eval_seed_a": a,
                        "eval_seed_b": b,
                        "jaccard": _jaccard(sets[a], sets[b]),
                    }
                )
        by_rid = {rid: set(g["cluster_id"]) for rid, g in mdf.groupby("Report_ID")}
        rids = np.array(sorted(by_rid))
        rng = np.random.default_rng(BOOT_SEED)
        seen = set()
        n_pairs = 0
        while n_pairs < n_ref_pairs and len(rids) >= 2:
            i, j = rng.choice(len(rids), size=2, replace=False)
            a, b = int(rids[i]), int(rids[j])
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Model": model,
                    "metric": metric,
                    "pair_type": "different_reports",
                    "Report_ID_a": a,
                    "Report_ID_b": b,
                    "eval_seed_a": None,
                    "eval_seed_b": None,
                    "jaccard": _jaccard(by_rid[a], by_rid[b]),
                }
            )
            n_pairs += 1
    return pd.DataFrame(rows)


def jaccard_summary(jac: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(jac.Model.unique()):
        for metric in ["gt_reports_as_ref", "gen_reports_as_ref"]:
            same = jac[
                (jac.Model == model)
                & (jac.metric == metric)
                & (jac.pair_type == "same_report_across_seeds")
            ]["jaccard"].dropna()
            diff = jac[
                (jac.Model == model)
                & (jac.metric == metric)
                & (jac.pair_type == "different_reports")
            ]["jaccard"].dropna()
            gap = float(same.mean() - diff.mean()) if len(same) and len(diff) else float("nan")
            # bootstrap CI on gap: resample same and diff means
            rng = np.random.default_rng(BOOT_SEED)
            if len(same) and len(diff):
                boots = []
                s_arr, d_arr = same.to_numpy(), diff.to_numpy()
                for _ in range(2000):
                    boots.append(
                        float(rng.choice(s_arr, size=len(s_arr), replace=True).mean()
                              - rng.choice(d_arr, size=len(d_arr), replace=True).mean())
                    )
                glo, ghi = percentile_ci(np.asarray(boots))
            else:
                glo, ghi = float("nan"), float("nan")
            rows.append(
                {
                    "Model": model,
                    "metric": metric,
                    "same_mean": float(same.mean()) if len(same) else float("nan"),
                    "same_median": float(same.median()) if len(same) else float("nan"),
                    "same_q25": float(same.quantile(0.25)) if len(same) else float("nan"),
                    "same_q75": float(same.quantile(0.75)) if len(same) else float("nan"),
                    "diff_mean": float(diff.mean()) if len(diff) else float("nan"),
                    "gap_same_minus_diff": gap,
                    "gap_CI_low": glo,
                    "gap_CI_high": ghi,
                }
            )
    return pd.DataFrame(rows)


def question_overlap(cq: pd.DataFrame) -> Dict:
    sets = {int(es): set(cq.loc[cq.eval_seed == es, "Question_Text"]) for es in EVAL_SEEDS}
    inter = set.intersection(*sets.values())
    return {
        "unique_per_eval_seed": {es: len(sets[es]) for es in EVAL_SEEDS},
        "intersection_all_5": len(inter),
        "union_all_5": len(set.union(*sets.values())),
        "expected_intersection": 9569,
        "total_unique_questions": 78708,
    }


def omission_share_by_seed(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of clusters (and size-weighted by n_gt) that are omission-dominated per model×seed."""
    rows = []
    for (model, es), g in scores.groupby(["Model", "eval_seed"]):
        rows.append(
            {
                "Model": model,
                "eval_seed": int(es),
                "pct_omission_clusters": 100.0 * (g.label == "omission-dominated").mean(),
                "pct_hallucination_clusters": 100.0 * (g.label == "hallucination-dominated").mean(),
                "pct_balanced_clusters": 100.0 * (g.label == "balanced").mean(),
                "size_weighted_pct_omission": 100.0
                * np.average(
                    (g.label == "omission-dominated").astype(float),
                    weights=np.maximum(g.n_gt, 1),
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_fig4_style_seed(scores_seed: pd.DataFrame, name_map: Dict[int, str], out_stem: Path) -> None:
    models = sorted(scores_seed.Model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = scores_seed[scores_seed.Model == model]
        for _, r in sub.iterrows():
            color = LABEL_COLORS.get(r.label, "#333")
            size = 30 + 100 * (r.n / max(sub.n.max(), 1))
            ax.scatter(r.GEN, r.GT, s=size, c=color, alpha=0.85, edgecolors="none", zorder=3)
            name = name_map.get(int(r.Cluster), str(int(r.Cluster)))
            ax.text(r.GEN, r.GT, name, fontsize=5.5, ha="center", va="bottom")
        ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ICARE-GEN")
        ax.set_ylabel("ICARE-GT")
        ax.set_title(MODEL_SHORT.get(model, model))
    fig.suptitle(f"Exp 5: Fig. 4–style (eval_seed={int(scores_seed.eval_seed.iloc[0])})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_grid(scores: pd.DataFrame, name_map: Dict[int, str], out_path: Path) -> None:
    models = sorted(scores.Model.unique())
    seeds = list(EVAL_SEEDS)
    fig, axes = plt.subplots(len(seeds), len(models), figsize=(4 * len(models), 3.6 * len(seeds)))
    for i, es in enumerate(seeds):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = scores[(scores.eval_seed == es) & (scores.Model == model)]
            for _, r in sub.iterrows():
                color = LABEL_COLORS.get(r.label, "#333")
                size = 20 + 80 * (r.n / max(sub.n.max(), 1))
                ax.scatter(r.GEN, r.GT, s=size, c=color, alpha=0.85, edgecolors="none")
                ax.text(r.GEN, r.GT, str(int(r.Cluster)), fontsize=6, ha="center", va="center")
            ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(MODEL_SHORT.get(model, model))
            if j == 0:
                ax.set_ylabel(f"seed {es}\nICARE-GT")
            else:
                ax.set_ylabel("ICARE-GT")
            ax.set_xlabel("ICARE-GEN")
    fig.suptitle("Exp 5: GEN vs GT by eval seed (point labels = cluster ID)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_errorbars(summary: pd.DataFrame, out_path: Path) -> None:
    models = sorted(summary.Model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = summary[summary.Model == model]
        for _, r in sub.iterrows():
            if r.n_seeds_omission >= max(r.n_seeds_hallucination, r.n_seeds_balanced):
                c = LABEL_COLORS["omission-dominated"]
            elif r.n_seeds_hallucination >= r.n_seeds_balanced:
                c = LABEL_COLORS["hallucination-dominated"]
            else:
                c = LABEL_COLORS["balanced"]
            ax.errorbar(
                r.GEN_mean,
                r.GT_mean,
                xerr=[[r.GEN_mean - r.GEN_t_low], [r.GEN_t_high - r.GEN_mean]],
                yerr=[[r.GT_mean - r.GT_t_low], [r.GT_t_high - r.GT_mean]],
                fmt="o",
                color=c,
                ecolor=c,
                alpha=0.85,
                markersize=5,
            )
            ax.text(
                r.GEN_mean,
                r.GT_mean,
                r.cluster_name.replace(" ", "\n"),
                fontsize=5.5,
                ha="center",
                va="bottom",
            )
        ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ICARE-GEN (mean ± t-interval)")
        ax.set_ylabel("ICARE-GT (mean ± t-interval)")
        ax.set_title(MODEL_SHORT.get(model, model))
    fig.suptitle("Exp 5: across-seed means with t-interval error bars", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_box(jac: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    order = list(MODEL_SHORT.values())
    for ax, metric, title in zip(
        axes,
        ["gt_reports_as_ref", "gen_reports_as_ref"],
        ["GT-source questions", "GEN-source questions"],
    ):
        sub = jac[jac.metric == metric].copy()
        sub["Model_short"] = sub["Model"].map(lambda m: MODEL_SHORT.get(m, m))
        data, positions = [], []
        pos = 0.0
        for model in order:
            for ptype, offset in [("same_report_across_seeds", 0.0), ("different_reports", 0.35)]:
                vals = sub[(sub.Model_short == model) & (sub.pair_type == ptype)]["jaccard"].dropna()
                data.append(vals)
                positions.append(pos + offset)
            pos += 1.2
        bp = ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#4c78a8" if i % 2 == 0 else "#f58518")
        centres = [i * 1.2 + 0.175 for i in range(len(order))]
        ax.set_xticks(centres)
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlim(-0.4, centres[-1] + 0.6)
        ax.set_title(title)
        ax.set_ylabel("Jaccard (cluster sets)")
    fig.suptitle("Report-level cluster Jaccard: same report across seeds vs different reports")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def filter_model_seeds(cq: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return cq
    if mode == "1":
        return cq[cq["model_seed"] == 1].copy()
    raise ValueError(f"Unknown --model-seeds={mode!r}; use 'all' or '1'")


def mode_outdir(base: Path, mode: str) -> Path:
    return base / ("model_seeds_all" if mode == "all" else "model_seed_1")


def run_one_mode(
    cq_full: pd.DataFrame,
    name_map: Dict[int, str],
    mode: str,
    out_dir: Path,
    n_boot: int,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cq = filter_model_seeds(cq_full, mode)
    print(f"=== mode={mode} rows={len(cq)} n_boot={n_boot} ===")

    score_rows = []
    for es in EVAL_SEEDS:
        print(f"  eval_seed={es}")
        sub = cq[cq["eval_seed"] == es]
        sc = score_slice_with_bootstrap(sub, "cluster_id", n_boot)
        sc["eval_seed"] = es
        # attach names
        sc["cluster_name"] = sc["Cluster"].map(name_map)
        score_rows.append(sc)
        plot_fig4_style_seed(sc, name_map, out_dir / f"fig4_style_seed{es}")
    scores = pd.concat(score_rows, ignore_index=True)
    scores.to_csv(out_dir / "scores_by_eval_seed.csv", index=False)

    summary = across_seed_summary(scores, name_map)
    summary.to_csv(out_dir / "across_seed_summary.csv", index=False)
    lab = label_consistency(summary)
    lab.to_csv(out_dir / "label_consistency.csv", index=False)
    concepts = concept_summary(scores, summary, name_map)
    concepts.to_csv(out_dir / "concept_summary.csv", index=False)
    omit = omission_share_by_seed(scores)
    omit.to_csv(out_dir / "omission_share_by_seed.csv", index=False)

    jac = pd.concat(
        [
            report_level_jaccard(cq, "gt_reports_as_ref"),
            report_level_jaccard(cq, "gen_reports_as_ref"),
        ],
        ignore_index=True,
    )
    jac.to_csv(out_dir / "report_jaccard.csv", index=False)
    jac_sum = jaccard_summary(jac)
    jac_sum.to_csv(out_dir / "report_jaccard_summary.csv", index=False)

    overlap = question_overlap(cq)
    pd.DataFrame(
        [{"eval_seed": k, "n_unique_Q": v} for k, v in overlap["unique_per_eval_seed"].items()]
    ).to_csv(out_dir / "question_overlap.csv", index=False)
    (out_dir / "question_overlap.md").write_text(
        f"# Question overlap (mode={mode})\n\n"
        f"- per eval_seed: {overlap['unique_per_eval_seed']}\n"
        f"- intersection all 5: **{overlap['intersection_all_5']}** / {overlap['total_unique_questions']}\n"
        f"- expected: {overlap['expected_intersection']}\n"
    )

    plot_combined_grid(scores, name_map, out_dir / "combined_seed_grid.png")
    plot_scatter_errorbars(summary, out_dir / "scatter_mean_with_errorbars.png")
    plot_jaccard_box(jac, out_dir / "jaccard_boxplot.png")

    # flipping clusters
    flips = summary[summary.same_label_all_seeds == 0][
        ["Model", "Cluster", "cluster_name", "delta_mean", "delta_sd", "any_low_n",
         "n_seeds_omission", "n_seeds_hallucination", "n_seeds_balanced"]
    ].copy()
    flips["near_diagonal"] = flips["delta_mean"].abs() < 0.05
    flips.to_csv(out_dir / "flipping_clusters.csv", index=False)

    meta = {
        "mode": mode,
        "sklearn": sklearn.__version__,
        "n_boot": n_boot,
        "n_rows": int(len(cq)),
        "overlap": overlap,
        "label_consistency": lab.to_dict(orient="records"),
        "jaccard_summary": jac_sum.to_dict(orient="records"),
        "omission_share": omit.to_dict(orient="records"),
        "n_low_n_cluster_model": int(summary.any_low_n.sum()),
        "flips": flips.to_dict(orient="records"),
        "concepts": concepts[concepts.scope == "across_seed_mean"].to_dict(orient="records"),
    }
    (out_dir / "mode_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def write_top_summary(base: Path, metas: Dict[str, Dict]) -> None:
    lines = [
        "# Exp 5 summary — eval-seed consistency (final)",
        "",
        f"- sklearn: `{sklearn.__version__}`",
        "- Stored K=20 labels/names; no re-clustering.",
        "- Labels: Fig. 4 rule (GT < GEN → omission; GT > GEN → hallucination).",
        "- Modes: `model_seeds_all` (primary) and `model_seed_1`.",
        "",
        "## Side-by-side: both model-seed modes",
        "",
        "| Model | Mode | % same label (all) | % same excl low_n | median Δ SD | max Δ SD | # low_n clusters | Jaccard gap GT (same−diff) [CI] |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("all", "1"):
        meta = metas[mode]
        lab = {r["Model"]: r for r in meta["label_consistency"]}
        jac = {
            (r["Model"], r["metric"]): r for r in meta["jaccard_summary"]
        }
        for model in sorted(lab):
            r = lab[model]
            j = jac.get((model, "gt_reports_as_ref"), {})
            gap = j.get("gap_same_minus_diff")
            glo, ghi = j.get("gap_CI_low"), j.get("gap_CI_high")
            lines.append(
                f"| {model} | {mode} | {r['pct_same_label_all_clusters']:.1f}% | "
                f"{r['pct_same_label_excl_low_n']:.1f}% | {r['median_delta_sd']:.4f} | "
                f"{r['max_delta_sd']:.4f} | {r['n_low_n_clusters']} | "
                f"{gap:.3f} [{glo:.3f},{ghi:.3f}] |"
            )

    lines += ["", "## Tracked concepts (across-seed mean ± SD; label counts)", ""]
    for mode in ("all", "1"):
        lines.append(f"### mode={mode}")
        for c in metas[mode]["concepts"]:
            lines.append(
                f"- **{c['concept']}** / {c['Model']}: delta={c['delta']:.4f}±{c.get('delta_sd', float('nan')):.4f}; "
                f"labels omission/hall/bal="
                f"{c.get('n_seeds_omission')}/{c.get('n_seeds_hallucination')}/{c.get('n_seeds_balanced')}; "
                f"low_n={c.get('low_n')}"
            )
        lines.append("")

    lines += ["## Question overlap", ""]
    for mode in ("all", "1"):
        o = metas[mode]["overlap"]
        lines.append(
            f"- mode={mode}: intersection **{o['intersection_all_5']}** / {o['total_unique_questions']} "
            f"(expected {o['expected_intersection']}); per-seed {o['unique_per_eval_seed']}"
        )

    lines += [
        "",
        "## Figure paths",
        "",
        "Under `results/revision_r2/exp5/model_seeds_all/` and `.../model_seed_1/`:",
        "- `fig4_style_seed{123,456,789,202,101}.png/.pdf`",
        "- `combined_seed_grid.png`",
        "- `scatter_mean_with_errorbars.png`",
        "- `jaccard_boxplot.png`",
        "",
    ]
    (base / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    n_boot = 200 if args.quick else args.n_boot
    mode = args.model_seeds
    if mode not in {"all", "1"}:
        raise SystemExit("--model-seeds must be 'all' or '1'")

    fig4 = args.fig4_root
    base = args.output_dir
    base.mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)

    cq = pd.read_csv(fig4 / "clustered_questions_with_names.csv")
    with open(fig4 / "cluster_names.json") as f:
        name_map = {int(k): v for k, v in json.load(f).items()}

    out = mode_outdir(base, mode)
    meta = run_one_mode(cq, name_map, mode, out, n_boot)

    # If the other mode already has meta, refresh top-level summary
    metas = {mode: meta}
    other = "1" if mode == "all" else "all"
    other_meta = mode_outdir(base, other) / "mode_meta.json"
    if other_meta.exists():
        metas[other] = json.loads(other_meta.read_text())
        write_top_summary(base, metas)
    else:
        # write partial summary for this mode only
        write_top_summary(base, {mode: meta, other: {
            "label_consistency": [],
            "jaccard_summary": [],
            "concepts": [],
            "overlap": {"intersection_all_5": "pending", "total_unique_questions": 78708,
                        "expected_intersection": 9569, "unique_per_eval_seed": {}},
        }})

    print("Exp 5 done →", out)


if __name__ == "__main__":
    main()
