#!/usr/bin/env python3
"""Exp 5: eval-seed consistency (Nature Communications R2, reviewer R4).

Uses stored K=20 labels. For each eval_seed, pools all 5 model seeds (Fig. 4 Methods).
Does not re-cluster. Does not call process_questions().
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 5: seed consistency")
    p.add_argument("--fig4-root", type=Path, default=DEFAULT_FIG4)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-gate", action="store_true",
                   help="Skip re-running gate; copy from exp4 if present")
    return p.parse_args()


def dominance_label(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "omission-dominated"
    if ci_lo > 0:
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
        indices = make_indices(n_rep, n_boot, boot_seed)
        for cid in sorted(mdf[cluster_col].dropna().unique()):
            cdf = mdf[mdf[cluster_col] == cid]
            gt = cdf[cdf["metric"] == "gt_reports_as_ref"]
            gen = cdf[cdf["metric"] == "gen_reports_as_ref"]

            def _pack(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
                agree = np.zeros(n_rep, dtype=np.float64)
                count = np.zeros(n_rep, dtype=np.float64)
                if sub.empty:
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

            def _rate(a, n, idx):
                den = n[idx].sum()
                return float(a[idx].sum() / den) if den > 0 else float("nan")

            gt_point = _rate(gt_a, gt_n, np.arange(n_rep))
            gen_point = _rate(gen_a, gen_n, np.arange(n_rep))
            delta = gt_point - gen_point
            boot_delta = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                idx = indices[b]
                boot_delta[b] = _rate(gt_a, gt_n, idx) - _rate(gen_a, gen_n, idx)
            lo, hi = percentile_ci(boot_delta)
            rows.append(
                {
                    "Model": model,
                    "Cluster": int(cid),
                    "GT": gt_point,
                    "GEN": gen_point,
                    "n_gt": int(gt_n.sum()),
                    "n_gen": int(gen_n.sum()),
                    "n": int(gt_n.sum()),
                    "delta": delta,
                    "CI_low": lo,
                    "CI_high": hi,
                    "label": dominance_label(lo, hi),
                }
            )
    return pd.DataFrame(rows)


def across_seed_summary(scores: pd.DataFrame, name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for (model, cid), g in scores.groupby(["Model", "Cluster"]):
        def _summ(col):
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
            }
        )
    return pd.DataFrame(rows)


def label_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in summary.groupby("Model"):
        pct = 100.0 * g["same_label_all_seeds"].mean()
        w = g["mean_n_gt"].to_numpy()
        w_pct = 100.0 * np.average(g["same_label_all_seeds"], weights=w) if w.sum() else float("nan")
        rows.append(
            {
                "Model": model,
                "pct_clusters_same_label_all_seeds": pct,
                "size_weighted_pct_same_label": float(w_pct),
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


def instance_counts(cq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, es, cid), g in cq.groupby(["model_name", "eval_seed", "cluster_id"]):
        rows.append(
            {
                "Model": model,
                "eval_seed": int(es),
                "Cluster": int(cid),
                "n_instances": len(g),
            }
        )
    df = pd.DataFrame(rows)
    # flag clusters with <30 in any seed
    mins = df.groupby(["Model", "Cluster"])["n_instances"].min().reset_index(name="min_n_any_seed")
    df = df.merge(mins, on=["Model", "Cluster"])
    df["flag_lt_30_any_seed"] = df["min_n_any_seed"] < 30
    return df


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return float(len(a & b) / len(u)) if u else float("nan")


def report_level_jaccard(cq: pd.DataFrame, metric: str, n_ref_pairs: int = 1000) -> pd.DataFrame:
    """Jaccard of cluster sets across eval seeds for same report; plus different-report ref."""
    rows = []
    sub = cq[cq["metric"] == metric]
    for model in sorted(sub["model_name"].unique()):
        mdf = sub[sub["model_name"] == model]
        # same report across seeds
        same_vals = []
        for rid, g in mdf.groupby("Report_ID"):
            sets = {
                int(es): set(g.loc[g.eval_seed == es, "cluster_id"])
                for es in EVAL_SEEDS
                if (g.eval_seed == es).any()
            }
            if len(sets) < 2:
                continue
            for a, b in combinations(sorted(sets), 2):
                j = _jaccard(sets[a], sets[b])
                same_vals.append(j)
                rows.append(
                    {
                        "Model": model,
                        "metric": metric,
                        "pair_type": "same_report_across_seeds",
                        "Report_ID_a": rid,
                        "Report_ID_b": rid,
                        "eval_seed_a": a,
                        "eval_seed_b": b,
                        "jaccard": j,
                    }
                )
        # different reports reference (use one eval seed's cluster sets, or union across?)
        # Spec: Jaccard between different reports of the same model — use cluster sets
        # pooled over all seeds for each report (or per-seed?). Use all-seed union of clusters.
        by_rid = {
            rid: set(g["cluster_id"])
            for rid, g in mdf.groupby("Report_ID")
        }
        rids = np.array(sorted(by_rid))
        rng = np.random.default_rng(BOOT_SEED)
        n_pairs = 0
        seen = set()
        while n_pairs < n_ref_pairs and len(rids) >= 2:
            i, j = rng.choice(len(rids), size=2, replace=False)
            a, b = int(rids[i]), int(rids[j])
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            jacc = _jaccard(by_rid[a], by_rid[b])
            rows.append(
                {
                    "Model": model,
                    "metric": metric,
                    "pair_type": "different_reports",
                    "Report_ID_a": a,
                    "Report_ID_b": b,
                    "eval_seed_a": None,
                    "eval_seed_b": None,
                    "jaccard": jacc,
                }
            )
            n_pairs += 1
    return pd.DataFrame(rows)


def question_overlap(cq: pd.DataFrame) -> Dict:
    # Full table (all model seeds) — expected 9569
    sets = {int(es): set(cq.loc[cq.eval_seed == es, "Question_Text"]) for es in EVAL_SEEDS}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    per = {es: len(sets[es]) for es in EVAL_SEEDS}
    return {
        "unique_per_eval_seed": per,
        "intersection_all_5": len(inter),
        "union_all_5": len(union),
        "expected_intersection": 9569,
        "total_unique_questions": 78708,
    }


def plot_scatter_errorbars(summary: pd.DataFrame, out_path: Path) -> None:
    models = sorted(summary.Model.unique())
    short = {
        "chexpert-mimic-cxr-findings-baseline": "CheX+MIMIC",
        "mimic-cxr-findings-baseline": "MIMIC",
        "maira-2": "MAIRA-2",
    }
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = summary[summary.Model == model]
        for _, r in sub.iterrows():
            # colour by majority label across seeds
            if r.n_seeds_omission >= r.n_seeds_hallucination and r.n_seeds_omission >= r.n_seeds_balanced:
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
            ax.text(r.GEN_mean, r.GT_mean, r.cluster_name.replace(" ", "\n"), fontsize=5.5, ha="center", va="bottom")
        ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ICARE-GEN (mean across eval seeds)")
        ax.set_ylabel("ICARE-GT (mean across eval seeds)")
        ax.set_title(short.get(model, model))
    fig.suptitle("Exp 5: cluster means with across-seed t-interval error bars", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard_box(jac: pd.DataFrame, out_path: Path) -> None:
    # summarise for plotting: one box per model × pair_type (pool GT and GEN metrics as facets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    short = {
        "chexpert-mimic-cxr-findings-baseline": "CheX+MIMIC",
        "mimic-cxr-findings-baseline": "MIMIC",
        "maira-2": "MAIRA-2",
    }
    for ax, metric, title in zip(
        axes,
        ["gt_reports_as_ref", "gen_reports_as_ref"],
        ["GT-source questions", "GEN-source questions"],
    ):
        sub = jac[jac.metric == metric].copy()
        sub["Model_short"] = sub["Model"].map(lambda m: short.get(m, m))
        order = list(short.values())
        data = []
        labels = []
        positions = []
        pos = 0
        for model in order:
            for ptype, offset in [("same_report_across_seeds", 0), ("different_reports", 0.35)]:
                vals = sub[(sub.Model_short == model) & (sub.pair_type == ptype)]["jaccard"].dropna()
                data.append(vals)
                labels.append(f"{model}\n{'same' if 'same' in ptype else 'diff'}")
                positions.append(pos + offset)
            pos += 1.2
        bp = ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#4c78a8" if i % 2 == 0 else "#f58518")
        centres = [i * 1.2 + 0.175 for i in range(len(order))]
        ax.set_xticks(centres)
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlim(-0.4, centres[-1] + 0.6 if centres else 1)
        ax.set_title(title)
        ax.set_ylabel("Jaccard (cluster sets)")
    fig.suptitle("Report-level cluster Jaccard: same report across seeds vs different reports")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    lab: pd.DataFrame,
    summary: pd.DataFrame,
    overlap: Dict,
    jac: pd.DataFrame,
) -> None:
    lines = [
        "# Exp 5 summary — eval-seed consistency",
        "",
        f"- sklearn: `{sklearn.__version__}`",
        "- Unit: eval_seed; each seed pools all 5 model seeds (Fig. 4 Methods).",
        "- Stored K=20 labels; no re-clustering.",
        "",
        "## Label consistency per model",
        "",
    ]
    for _, r in lab.iterrows():
        lines.append(
            f"- **{r.Model}**: {r.pct_clusters_same_label_all_seeds:.1f}% clusters same label "
            f"in all 5 seeds (size-weighted {r.size_weighted_pct_same_label:.1f}%). "
            f"Lung Lobe Abnormality same={r.Lung_Lobe_Abnormality_same}; "
            f"Calcified Granuloma Location same={r.Calcified_Granuloma_Location_same}."
        )
    lines += ["", "## Tracked concepts across seeds", ""]
    for concept in (CONCEPT_LUNG, CONCEPT_CALC):
        sub = summary[summary.cluster_name == concept]
        for _, r in sub.iterrows():
            lines.append(
                f"- {concept} / {r.Model}: delta mean±SD={r.delta_mean:.4f}±{r.delta_sd:.4f}; "
                f"labels omission/hall/bal={r.n_seeds_omission}/{r.n_seeds_hallucination}/{r.n_seeds_balanced}"
            )
    sd_vals = summary["delta_sd"]
    lines += [
        "",
        f"## Across-seed SD (delta)",
        f"- range: {float(sd_vals.min()):.4f} – {float(sd_vals.max()):.4f} "
        f"(median {float(sd_vals.median()):.4f})",
        "",
        "## Report-level Jaccard vs reference",
        "",
    ]
    for model in sorted(jac.Model.unique()):
        for metric in ["gt_reports_as_ref", "gen_reports_as_ref"]:
            same = jac[(jac.Model == model) & (jac.metric == metric) & (jac.pair_type == "same_report_across_seeds")]["jaccard"]
            diff = jac[(jac.Model == model) & (jac.metric == metric) & (jac.pair_type == "different_reports")]["jaccard"]
            lines.append(
                f"- {model} / {metric}: same-report mean/median/IQR="
                f"{same.mean():.3f}/{same.median():.3f}/[{same.quantile(0.25):.3f},{same.quantile(0.75):.3f}]; "
                f"diff-report mean={diff.mean():.3f}"
            )
    lines += [
        "",
        "## Question overlap",
        f"- Unique Q per eval_seed: {overlap['unique_per_eval_seed']}",
        f"- Intersection all 5 seeds: **{overlap['intersection_all_5']}** / {overlap['total_unique_questions']} "
        f"(expected {overlap['expected_intersection']})",
        "",
        "## Unexpected / notes",
        "",
        "- See ../exp4/gate.md for model-seed MCQ nondeterminism (gate c).",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    n_boot = 200 if args.quick else args.n_boot
    fig4 = args.fig4_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    print(f"sklearn={sklearn.__version__} n_boot={n_boot}")
    cq = pd.read_csv(fig4 / "clustered_questions_with_names.csv")
    with open(fig4 / "cluster_names.json") as f:
        name_map = {int(k): v for k, v in json.load(f).items()}

    # Copy gate from exp4 if present
    exp4_gate = REPO_ROOT / "results" / "revision_r2" / "exp4" / "gate.md"
    if exp4_gate.exists():
        shutil.copy2(exp4_gate, out_dir / "gate.md")
        gjson = exp4_gate.with_suffix(".json")
        if gjson.exists():
            shutil.copy2(gjson, out_dir / "gate.json")

    score_rows = []
    for es in EVAL_SEEDS:
        print(f"=== eval_seed={es} (pooling all model seeds) ===")
        # Pool all model seeds for this eval_seed
        sub = cq[cq["eval_seed"] == es]
        sc = score_slice_with_bootstrap(sub, "cluster_id", n_boot)
        sc["eval_seed"] = es
        score_rows.append(sc)
    scores = pd.concat(score_rows, ignore_index=True)
    scores.to_csv(out_dir / "scores_by_eval_seed.csv", index=False)

    summary = across_seed_summary(scores, name_map)
    summary.to_csv(out_dir / "across_seed_summary.csv", index=False)
    lab = label_consistency(summary)
    lab.to_csv(out_dir / "label_consistency.csv", index=False)

    counts = instance_counts(cq)
    counts.to_csv(out_dir / "instance_counts.csv", index=False)

    jac_frames = [
        report_level_jaccard(cq, "gt_reports_as_ref"),
        report_level_jaccard(cq, "gen_reports_as_ref"),
    ]
    jac = pd.concat(jac_frames, ignore_index=True)
    jac.to_csv(out_dir / "report_jaccard.csv", index=False)

    overlap = question_overlap(cq)
    (out_dir / "question_overlap.csv").write_text(
        pd.DataFrame([{"eval_seed": k, "n_unique_Q": v} for k, v in overlap["unique_per_eval_seed"].items()]).to_csv(index=False)
    )
    (out_dir / "question_overlap.md").write_text(
        f"# Question overlap\n\n"
        f"- per eval_seed: {overlap['unique_per_eval_seed']}\n"
        f"- intersection all 5: **{overlap['intersection_all_5']}** / {overlap['total_unique_questions']}\n"
        f"- expected: {overlap['expected_intersection']}\n"
    )

    plot_scatter_errorbars(summary, out_dir / "scatter_mean_with_errorbars.png")
    plot_jaccard_box(jac, out_dir / "jaccard_boxplot.png")
    write_summary(out_dir / "summary.md", lab, summary, overlap, jac)
    print("Exp 5 done →", out_dir)


if __name__ == "__main__":
    main()
