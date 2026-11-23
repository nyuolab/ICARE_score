#!/usr/bin/env python3
"""Exp 4: cluster K sensitivity (Nature Communications R2, reviewer R1-C7).

Uses stored 2025 Fig. 4 embeddings. Does not call process_questions().
Fresh KMeans for every K including 20. Fig. 4 label rule (point estimate).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_stats.bootstrap_utils import make_indices, percentile_ci  # noqa: E402
from question_categorization_and_analysis.cluster_analysis import (  # noqa: E402
    calculate_agreement,
)
from question_categorization_and_analysis.question_embedding_and_clustering import (  # noqa: E402
    cluster_questions,
    name_clusters_with_llama,
)

DEFAULT_FIG4 = Path(
    "/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/"
    "MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis"
)
DEFAULT_OUT = REPO_ROOT / "results" / "revision_r2" / "exp4"
CONCEPT_LUNG = "Lung Lobe Abnormality"
CONCEPT_CALC = "Calcified Granuloma Location"
BOOT_SEED = 42
KS_DEFAULT = (10, 15, 20, 25, 30)
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
    p = argparse.ArgumentParser(description="Exp 4: cluster K sensitivity")
    p.add_argument("--fig4-root", type=Path, default=DEFAULT_FIG4)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="200 bootstrap resamples")
    p.add_argument(
        "--model-seeds",
        type=str,
        default="all",
        help=' "all" (pool seeds 1-5) or "1" (model_seed==1 only)',
    )
    p.add_argument("--ks", type=str, default="10,15,20,25,30")
    p.add_argument(
        "--name",
        action="store_true",
        help="Name non-20 clusters via Llama (implement only; do not run in final)",
    )
    return p.parse_args()


def fig4_label(gt: float, gen: float) -> str:
    """Manuscript Fig. 4 rule (point estimate; ignore CI/n)."""
    if not np.isfinite(gt) or not np.isfinite(gen):
        return "balanced"
    if gt < gen:
        return "omission-dominated"
    if gt > gen:
        return "hallucination-dominated"
    return "balanced"


def write_representatives(
    path: Path, reps: Dict[int, List[str]], sizes: Dict[int, int]
) -> None:
    with open(path, "w") as f:
        for cid in sorted(reps):
            f.write(f"=== Cluster {cid} ({sizes[cid]} questions) ===\n\n")
            for i, q in enumerate(reps[cid]):
                f.write(f"{i + 1}. {q}\n")
            f.write("\n")


def filter_model_seeds(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    if mode == "1":
        return df[df["model_seed"] == 1].copy()
    raise ValueError(f"Unknown --model-seeds={mode!r}; use 'all' or '1'")


def score_slice_with_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    n_boot: int,
    boot_seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """Per model × cluster: GT, GEN, n_*, delta, CI, label (Fig. 4), low_n."""
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

            def _rate(a: np.ndarray, n: np.ndarray, idx: np.ndarray) -> float:
                den = n[idx].sum()
                return float(a[idx].sum() / den) if den > 0 else float("nan")

            gt_point = _rate(gt_a, gt_n, np.arange(n_rep))
            gen_point = _rate(gen_a, gen_n, np.arange(n_rep))
            delta = gt_point - gen_point
            boot_delta = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                idx = indices[b]
                boot_delta[b] = _rate(gt_a, gt_n, idx) - _rate(gen_a, gen_n, idx)
            finite = boot_delta[np.isfinite(boot_delta)]
            if finite.size:
                lo, hi = percentile_ci(finite)
            else:
                lo, hi = float("nan"), float("nan")
            n_gt = int(gt_n.sum())
            n_gen = int(gen_n.sum())
            rows.append(
                {
                    "Model": model,
                    "Cluster": int(cid),
                    "GT": gt_point,
                    "GEN": gen_point,
                    "n_gt": n_gt,
                    "n_gen": n_gen,
                    "delta": delta,
                    "CI_low": lo,
                    "CI_high": hi,
                    "label": fig4_label(gt_point, gen_point),
                    "low_n": bool(n_gt < 30 or n_gen < 30),
                }
            )
    return pd.DataFrame(rows)


def correspondence_matrix(
    q_to_orig: Dict[str, int],
    q_to_new: Dict[str, int],
    n_orig: int,
    n_new: int,
) -> np.ndarray:
    mat = np.zeros((n_orig, n_new), dtype=np.float64)
    by_orig: Dict[int, List[str]] = defaultdict(list)
    for q, oid in q_to_orig.items():
        by_orig[int(oid)].append(q)
    for oid, qs in by_orig.items():
        if oid < 0 or oid >= n_orig:
            continue
        counts = np.zeros(n_new, dtype=np.float64)
        for q in qs:
            if q in q_to_new:
                nid = int(q_to_new[q])
                if 0 <= nid < n_new:
                    counts[nid] += 1
        s = counts.sum()
        if s > 0:
            mat[oid] = counts / s
    return mat


def best_match_k20_info(
    q_to_k20: Dict[str, int],
    q_to_new: Dict[str, int],
    k20_names: Dict[int, str],
    k: int,
) -> pd.DataFrame:
    """For each new cluster: best-matching K=20 cluster name and unique-Q share."""
    # share of new cluster's questions that belong to each orig cluster
    by_new: Dict[int, List[str]] = defaultdict(list)
    for q, nid in q_to_new.items():
        by_new[int(nid)].append(q)
    rows = []
    for nid in range(k):
        qs = by_new.get(nid, [])
        if not qs:
            rows.append(
                {
                    "Cluster": nid,
                    "best_k20_cluster": None,
                    "best_k20_name": None,
                    "best_k20_share": float("nan"),
                }
            )
            continue
        counts: Dict[int, int] = defaultdict(int)
        for q in qs:
            if q in q_to_k20:
                counts[int(q_to_k20[q])] += 1
        total = sum(counts.values())
        if total == 0:
            rows.append(
                {
                    "Cluster": nid,
                    "best_k20_cluster": None,
                    "best_k20_name": None,
                    "best_k20_share": float("nan"),
                }
            )
            continue
        best_oid = max(counts, key=counts.get)
        share = counts[best_oid] / total
        rows.append(
            {
                "Cluster": nid,
                "best_k20_cluster": int(best_oid),
                "best_k20_name": k20_names.get(int(best_oid), str(best_oid)),
                "best_k20_share": float(share),
            }
        )
    return pd.DataFrame(rows)


def concept_tracking(
    cq_k: pd.DataFrame,
    anchor_q: Dict[str, set],
    scores: pd.DataFrame,
    k: int,
    reps: Dict[int, List[str]],
) -> List[dict]:
    rows = []
    for concept, qset in anchor_q.items():
        sub = cq_k[cq_k["Question_Text"].isin(qset)]
        total = len(sub)
        if total == 0:
            continue
        shares = (sub.groupby("cluster_id").size().astype(float) / total).sort_values(
            ascending=False
        )
        matched = int(shares.index[0])
        for cid, share in shares.items():
            if float(share) < 0.20 and int(cid) != matched:
                continue
            for model in sorted(scores.Model.unique()):
                sc = scores[(scores.Model == model) & (scores.Cluster == int(cid))]
                lab = sc.label.iloc[0] if len(sc) else None
                low = bool(sc.low_n.iloc[0]) if len(sc) else None
                rows.append(
                    {
                        "K": k,
                        "concept": concept,
                        "cluster_id": int(cid),
                        "instance_share": float(share),
                        "is_matched": int(cid) == matched,
                        "Model": model,
                        "label": lab,
                        "low_n": low,
                        "representatives": " | ".join(reps.get(int(cid), [])),
                    }
                )
    return rows


def consistency_metrics(
    scores_k20: pd.DataFrame,
    scores_new: pd.DataFrame,
    q_to_k20: Dict[str, int],
    q_to_new: Dict[str, int],
    cq_new: pd.DataFrame,
    k: int,
) -> List[dict]:
    mat = correspondence_matrix(q_to_k20, q_to_new, 20, k)
    best_new = {oid: int(np.argmax(mat[oid])) for oid in range(20)}
    rows = []
    for model in sorted(scores_k20.Model.unique()):
        so = scores_k20[scores_k20.Model == model].set_index("Cluster")
        sn = scores_new[scores_new.Model == model].set_index("Cluster")
        agree = 0
        for oid in range(20):
            if oid not in so.index or best_new[oid] not in sn.index:
                continue
            if so.loc[oid, "label"] == sn.loc[best_new[oid], "label"]:
                agree += 1
        pct_a = 100.0 * agree / 20.0

        orig_q = {oid: {q for q, o in q_to_k20.items() if int(o) == oid} for oid in range(20)}
        mdf = cq_new[cq_new["model_name"] == model]
        orig_deltas, mapped_deltas = [], []
        for oid in range(20):
            if oid not in so.index:
                continue
            qs = orig_q[oid]
            sub = mdf[mdf["Question_Text"].isin(qs)]
            if sub.empty:
                continue
            shares = sub.groupby("cluster_id").size().astype(float)
            shares = shares / shares.sum()
            wmean = 0.0
            ok = True
            for cid, sh in shares.items():
                if int(cid) not in sn.index or not np.isfinite(sn.loc[int(cid), "delta"]):
                    ok = False
                    break
                wmean += float(sh) * float(sn.loc[int(cid), "delta"])
            if not ok or not np.isfinite(so.loc[oid, "delta"]):
                continue
            orig_deltas.append(float(so.loc[oid, "delta"]))
            mapped_deltas.append(wmean)
        n_used = len(orig_deltas)
        if n_used >= 3:
            rho, _ = spearmanr(orig_deltas, mapped_deltas)
            rho = float(rho) if np.isfinite(rho) else float("nan")
        else:
            rho = float("nan")

        lab_map = sn["label"].to_dict()
        inst = mdf.copy()
        inst["dom_label"] = inst["cluster_id"].map(lab_map)
        n_inst = len(inst)
        share_om = float((inst["dom_label"] == "omission-dominated").mean()) if n_inst else 0.0
        share_hall = (
            float((inst["dom_label"] == "hallucination-dominated").mean()) if n_inst else 0.0
        )
        n_low = int(sn["low_n"].sum()) if "low_n" in sn.columns else 0

        rows.append(
            {
                "K": k,
                "Model": model,
                "pct_label_match_best": pct_a,
                "spearman_delta": rho,
                "spearman_n_clusters_used": n_used,
                "share_omission_instances": share_om,
                "share_hallucination_instances": share_hall,
                "n_low_n_clusters": n_low,
            }
        )
    return rows


def plot_fig4_style(
    scores: pd.DataFrame,
    k: int,
    out_stem: Path,
    name_map: Optional[Dict[int, str]] = None,
    use_names: bool = False,
) -> None:
    models = sorted(scores.Model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = scores[scores.Model == model]
        for _, r in sub.iterrows():
            color = LABEL_COLORS.get(r.label, "#333")
            size = 30 + 90 * (r.n_gt / max(sub.n_gt.max(), 1))
            ax.scatter(r.GEN, r.GT, s=size, c=color, alpha=0.85, zorder=3, edgecolors="none")
            if use_names and name_map and int(r.Cluster) in name_map:
                lab = name_map[int(r.Cluster)].replace(" ", "\n")
            else:
                lab = str(int(r.Cluster))
            ax.text(r.GEN, r.GT, lab, fontsize=6 if use_names else 8, ha="center", va="center")
        ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ICARE-GEN")
        ax.set_ylabel("ICARE-GT")
        ax.set_title(MODEL_SHORT.get(model, model))
    fig.suptitle(f"Exp 4: K={k} (Fig. 4-style)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scatter_grid(
    all_scores: Dict[int, pd.DataFrame],
    out_path: Path,
    k20_names: Dict[int, str],
) -> None:
    ks = sorted(all_scores)
    models = sorted(all_scores[ks[0]].Model.unique())
    fig, axes = plt.subplots(len(ks), len(models), figsize=(4 * len(models), 3.4 * len(ks)))
    if len(ks) == 1:
        axes = np.array([axes])
    for i, k in enumerate(ks):
        sc = all_scores[k]
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = sc[sc.Model == model]
            for _, r in sub.iterrows():
                color = LABEL_COLORS.get(r.label, "#333")
                size = 20 + 70 * (r.n_gt / max(sub.n_gt.max(), 1))
                ax.scatter(r.GEN, r.GT, s=size, c=color, alpha=0.85, edgecolors="none")
                if k == 20:
                    lab = k20_names.get(int(r.Cluster), str(int(r.Cluster))).replace(" ", "\n")
                    fs = 5
                else:
                    lab = str(int(r.Cluster))
                    fs = 7
                ax.text(r.GEN, r.GT, lab, fontsize=fs, ha="center", va="center")
            ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(MODEL_SHORT.get(model, model))
            if j == 0:
                ax.set_ylabel(f"K={k}\nICARE-GT")
            else:
                ax.set_ylabel("ICARE-GT")
            ax.set_xlabel("ICARE-GEN")
    fig.suptitle("Exp 4: GEN vs GT by K (colour = Fig. 4 dominance label)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consistency_c(cons: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in cons.groupby("Model"):
        g = g.sort_values("K")
        ax.plot(
            g.K,
            100 * g.share_omission_instances,
            marker="o",
            label=f"{MODEL_SHORT.get(model, model)} omission",
        )
        ax.plot(
            g.K,
            100 * g.share_hallucination_instances,
            marker="s",
            ls="--",
            label=f"{MODEL_SHORT.get(model, model)} hallucination",
        )
    ax.set_xlabel("K")
    ax.set_ylabel("% of question instances")
    ax.set_title("Share of instances in omission- / hallucination-dominated clusters")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_to_stored_fig4(
    scores_k20: pd.DataFrame, fig4: Path, mode: str
) -> Dict[str, float]:
    """max |diff| vs stored agreement CSVs. Only meaningful for model_seeds_all."""
    gt_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gt_ref.csv")
    gen_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gen_ref.csv")
    # Rebuild from scores
    gt_r = scores_k20.rename(columns={"GT": "Agreement", "Cluster": "Cluster", "n_gt": "Count"})[
        ["Model", "Cluster", "Agreement", "Count"]
    ]
    gen_r = scores_k20.rename(columns={"GEN": "Agreement", "n_gen": "Count"})[
        ["Model", "Cluster", "Agreement", "Count"]
    ]
    m_gt = gt_saved.merge(gt_r, on=["Model", "Cluster"], suffixes=("_s", "_r"))
    m_gen = gen_saved.merge(gen_r, on=["Model", "Cluster"], suffixes=("_s", "_r"))
    return {
        "mode": mode,
        "max_abs_diff_gt": float((m_gt["Agreement_s"] - m_gt["Agreement_r"]).abs().max()),
        "max_abs_diff_gen": float((m_gen["Agreement_s"] - m_gen["Agreement_r"]).abs().max()),
    }


def run_name(mode_dir: Path, ks: Sequence[int], cq_by_k: Dict[int, pd.DataFrame]) -> None:
    for k in ks:
        if k == 20:
            continue
        reps_path = mode_dir / f"cluster_representatives_K{k}.txt"
        reps: Dict[int, List[str]] = {}
        cur = None
        for line in reps_path.read_text().splitlines():
            if line.startswith("=== Cluster"):
                cur = int(line.split()[2])
                reps[cur] = []
            elif line[:1].isdigit() and ". " in line and cur is not None:
                reps[cur].append(line.split(". ", 1)[1])
        scratch = mode_dir / f"naming_K{k}"
        scratch.mkdir(parents=True, exist_ok=True)
        names = name_clusters_with_llama(reps, str(scratch))
        if not names:
            print(f"--name failed for K={k}")
            continue
        names_out = {str(int(a)): b for a, b in names.items()}
        (mode_dir / f"cluster_names_K{k}.json").write_text(json.dumps(names_out, indent=2))
        df = cq_by_k[k].copy()
        df["cluster_name"] = df["cluster_id"].map({int(a): b for a, b in names_out.items()})
        df.to_csv(mode_dir / f"clustered_questions_with_names_K{k}.csv", index=False)


def run_one_mode(
    mode: str,
    mode_dir: Path,
    fig4: Path,
    emb: np.ndarray,
    unique_questions: List[str],
    combined: pd.DataFrame,
    cq_named: pd.DataFrame,
    q_to_stored: Dict[str, int],
    stored_names: Dict[int, str],
    anchor_q: Dict[str, set],
    ks: Sequence[int],
    n_boot: int,
    do_name: bool,
) -> Dict:
    mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n######## MODE model_seeds={mode} → {mode_dir} ########")
    df_base = filter_model_seeds(combined, mode)
    print(f"scoring rows after model-seed filter: {len(df_base)}")

    # Fresh clustering for every K (including 20)
    clusterings = {}
    for k in ks:
        print(f"=== clustering K={k} ===")
        labels, q2c, reps, sizes = cluster_questions(emb, unique_questions, n_clusters=k)
        clusterings[k] = {
            "labels": np.asarray(labels),
            "q2c": q2c,
            "reps": reps,
            "sizes": sizes,
        }
        write_representatives(mode_dir / f"cluster_representatives_K{k}.txt", reps, sizes)

    # K=20 reproduction vs stored
    lab_fresh = np.array([clusterings[20]["q2c"][q] for q in unique_questions])
    lab_stored = np.array([q_to_stored[q] for q in unique_questions])
    ari = float(adjusted_rand_score(lab_stored, lab_fresh))
    exact = float((lab_fresh == lab_stored).mean())
    print(f"K=20 fresh vs stored: ARI={ari} exact={exact}")

    # Copy stored names for K=20 display (do not overwrite 2025 tree)
    (mode_dir / "cluster_names_K20.json").write_text(
        json.dumps({str(k): v for k, v in stored_names.items()}, indent=2)
    )

    all_scores: Dict[int, pd.DataFrame] = {}
    cq_by_k: Dict[int, pd.DataFrame] = {}
    concept_rows: List[dict] = []
    cons_rows: List[dict] = []
    corr_rows: List[dict] = []

    for k in ks:
        q2c = clusterings[k]["q2c"]
        reps = clusterings[k]["reps"]
        cq_k = df_base.copy()
        cq_k["cluster_id"] = cq_k["Question_Text"].map(q2c)
        cq_k.to_csv(mode_dir / f"clustered_questions_K{k}.csv", index=False)
        cq_by_k[k] = cq_k

        scores = score_slice_with_bootstrap(cq_k, "cluster_id", n_boot)
        # attach best-matching K=20 name/share (unique-Q)
        # For matching names use STORED K=20 partition (manuscript concepts)
        match_info = best_match_k20_info(q_to_stored, q2c, stored_names, k)
        scores = scores.merge(match_info, on="Cluster", how="left")
        scores.to_csv(mode_dir / f"scores_K{k}.csv", index=False)
        all_scores[k] = scores

        concept_rows.extend(concept_tracking(cq_k, anchor_q, scores, k, reps))

        mat = correspondence_matrix(q_to_stored, q2c, 20, k)
        np.save(mode_dir / f"correspondence_K{k}.npy", mat)
        pd.DataFrame(
            mat,
            index=[f"orig_{i}" for i in range(20)],
            columns=[f"new_{i}" for i in range(k)],
        ).to_csv(mode_dir / f"correspondence_K{k}.csv")
        fig, ax = plt.subplots(figsize=(max(6, k * 0.35), 5))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel(f"new cluster (K={k})")
        ax.set_ylabel("stored K=20 cluster")
        ax.set_title(f"Overlap share (unique Question_Text), K={k}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(mode_dir / f"correspondence_heatmap_K{k}.png", dpi=120)
        plt.close(fig)

        ari_k = float(
            adjusted_rand_score(
                [q_to_stored[q] for q in unique_questions],
                [q2c[q] for q in unique_questions],
            )
        )
        corr_rows.append(
            {"K": k, "ARI_vs_stored_K20": ari_k, "mean_best_match_share": float(mat.max(axis=1).mean())}
        )

        # consistency vs FRESH K=20 labels/scores (as K=20 row everywhere)
        plot_fig4_style(
            scores,
            k,
            mode_dir / f"fig4_style_K{k}",
            name_map=stored_names if k == 20 else None,
            use_names=(k == 20),
        )

    # Recompute consistency properly now that scores_K20 exists
    scores_k20 = all_scores[20]
    cons_rows = []
    for k in ks:
        cons_rows.extend(
            consistency_metrics(
                scores_k20,
                all_scores[k],
                clusterings[20]["q2c"],
                clusterings[k]["q2c"],
                cq_by_k[k],
                k,
            )
        )

    concept_df = pd.DataFrame(concept_rows)
    concept_df.to_csv(mode_dir / "concept_tracking.csv", index=False)
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(mode_dir / "consistency_by_K.csv", index=False)
    corr_sum = pd.DataFrame(corr_rows)
    corr_sum.to_csv(mode_dir / "correspondence_summary.csv", index=False)

    plot_scatter_grid(all_scores, mode_dir / "scatter_grid.png", stored_names)
    plot_consistency_c(cons, mode_dir / "consistency_share_vs_K.png")

    fig4_cmp = None
    if mode == "all":
        fig4_cmp = compare_to_stored_fig4(scores_k20, fig4, mode)
        cq20 = cq_by_k[20].copy()
        cq20["cluster_name"] = cq20["cluster_id"].map(stored_names)
        gt_df = cq20[cq20.metric == "gt_reports_as_ref"]
        gen_df = cq20[cq20.metric == "gen_reports_as_ref"]
        gt_mc = calculate_agreement(gt_df, ["model_name", "cluster_id"]).rename(
            columns={"model_name": "Model", "cluster_id": "Cluster"}
        )
        gen_mc = calculate_agreement(gen_df, ["model_name", "cluster_id"]).rename(
            columns={"model_name": "Model", "cluster_id": "Cluster"}
        )
        gt_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gt_ref.csv")
        gen_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gen_ref.csv")
        if ari == 1.0:
            m_gt = gt_saved.merge(gt_mc, on=["Model", "Cluster"], suffixes=("_s", "_r"))
            m_gen = gen_saved.merge(gen_mc, on=["Model", "Cluster"], suffixes=("_s", "_r"))
            fig4_cmp["max_abs_diff_gt_calc_agreement"] = float(
                (m_gt["Agreement_s"] - m_gt["Agreement_r"]).abs().max()
            )
            fig4_cmp["max_abs_diff_gen_calc_agreement"] = float(
                (m_gen["Agreement_s"] - m_gen["Agreement_r"]).abs().max()
            )
    meta = {
        "ari_vs_stored": ari,
        "exact_label_match": exact,
        "model_seeds": mode,
        "sklearn_version": sklearn.__version__,
    }
    if fig4_cmp:
        meta.update(fig4_cmp)
    (mode_dir / "k20_reproduction.json").write_text(json.dumps(meta, indent=2))

    if do_name:
        run_name(mode_dir, ks, cq_by_k)

    return {
        "mode": mode,
        "ari": ari,
        "exact": exact,
        "fig4_cmp": fig4_cmp,
        "cons": cons,
        "concept_df": concept_df,
        "corr_sum": corr_sum,
        "all_scores": all_scores,
        "clusterings": clusterings,
    }


def write_top_summary(out_dir: Path, results: Dict[str, Dict], stored_names: Dict[int, str]) -> None:
    lines = [
        "# Exp 4 summary — cluster K sensitivity (final)",
        "",
        f"- sklearn: `{sklearn.__version__}`",
        "- Label rule: Fig. 4 point estimate (omission if GT < GEN; hallucination if GT > GEN).",
        "- Fresh `cluster_questions` for every K including 20; stored names used for K=20 display only.",
        "- Modes: `model_seeds_all` (primary) and `model_seed_1`.",
        "",
    ]

    # K=20 reproduction
    lines += ["## K=20 reproduction (fresh vs stored)", ""]
    for mode, res in results.items():
        lines.append(
            f"- **model_seeds={mode}**: ARI={res['ari']}, exact-label match={res['exact']}"
        )
        if res.get("fig4_cmp"):
            c = res["fig4_cmp"]
            lines.append(
                f"  - max |diff| vs stored Fig. 4 CSVs (from scores): "
                f"GT={c.get('max_abs_diff_gt')}, GEN={c.get('max_abs_diff_gen')}"
            )
            if "max_abs_diff_gt_calc_agreement" in c:
                lines.append(
                    f"  - max |diff| via calculate_agreement: "
                    f"GT={c['max_abs_diff_gt_calc_agreement']}, "
                    f"GEN={c['max_abs_diff_gen_calc_agreement']}"
                )
    lines.append("")

    # Side-by-side consistency table
    lines += ["## Consistency side-by-side (all vs seed_1)", ""]
    lines.append(
        "| K | Model | mode | label-match% vs K=20 | Spearman (n) | "
        "omission% | hallucination% | n_low_n |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    # merge
    frames = []
    for mode, res in results.items():
        c = res["cons"].copy()
        c["seeds_mode"] = mode
        frames.append(c)
    cons_all = pd.concat(frames, ignore_index=True)
    for _, r in cons_all.sort_values(["K", "Model", "seeds_mode"]).iterrows():
        sp = r.spearman_delta
        sp_s = f"{sp:.3f}" if sp == sp else "nan"
        lines.append(
            f"| {int(r.K)} | {MODEL_SHORT.get(r.Model, r.Model)} | {r['seeds_mode']} | "
            f"{r.pct_label_match_best:.1f} | {sp_s} ({int(r.spearman_n_clusters_used)}) | "
            f"{100*r.share_omission_instances:.1f} | {100*r.share_hallucination_instances:.1f} | "
            f"{int(r.n_low_n_clusters)} |"
        )
    lines.append("")

    # Concepts
    lines += ["## Matched clusters for tracked concepts", ""]
    for concept in (CONCEPT_LUNG, CONCEPT_CALC):
        lines.append(f"### {concept}")
        for mode, res in results.items():
            cdf = res["concept_df"]
            sub = cdf[(cdf.concept == concept) & (cdf.is_matched)]
            lines.append(f"#### model_seeds={mode}")
            for k, g in sub.groupby("K"):
                cid = int(g.cluster_id.iloc[0])
                share = float(g.instance_share.iloc[0])
                per_model = "; ".join(
                    f"{MODEL_SHORT.get(r.Model, r.Model)}: {r.label}"
                    + (" [low_n]" if r.low_n else "")
                    for _, r in g.drop_duplicates("Model").iterrows()
                )
                lines.append(f"- K={k}: matched cluster **{cid}** (share={share:.3f}); {per_model}")
                if int(k) != 20:
                    reps = g.representatives.iloc[0]
                    if isinstance(reps, str) and reps:
                        lines.append(f"  - representatives: {reps}")
            lines.append("")
        lines.append("")

    lines += [
        "## Figure paths",
        "",
        "Per mode under `results/revision_r2/exp4/model_seeds_all/` and `.../model_seed_1/`:",
        "- `fig4_style_K{10,15,20,25,30}.png/.pdf`",
        "- `scatter_grid.png`, `consistency_share_vs_K.png`",
        "- `correspondence_heatmap_K{K}.png`",
        "",
        "## Notes",
        "",
        "- `--name` is implemented but was not run in this final pass; K≠20 points use cluster IDs.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    n_boot = 200 if args.quick else args.n_boot
    ks = tuple(int(x) for x in args.ks.split(","))
    mode = args.model_seeds
    if mode not in {"all", "1"}:
        raise SystemExit("--model-seeds must be 'all' or '1'")

    fig4 = args.fig4_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    print(f"sklearn={sklearn.__version__} n_boot={n_boot} ks={ks} model_seeds={mode}")
    with open(fig4 / "question_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    combined = pd.read_csv(fig4 / "combined_mcqa_data.csv")
    cq_named = pd.read_csv(fig4 / "clustered_questions_with_names.csv")
    unique_questions = list(combined["Question_Text"].unique())
    assert emb.shape[0] == len(unique_questions)

    with open(fig4 / "cluster_names.json") as f:
        stored_names = {int(k): v for k, v in json.load(f).items()}
    q_to_stored = (
        cq_named.drop_duplicates("Question_Text")
        .set_index("Question_Text")["cluster_id"]
        .astype(int)
        .to_dict()
    )
    anchor_q = {
        CONCEPT_LUNG: set(cq_named.loc[cq_named.cluster_name == CONCEPT_LUNG, "Question_Text"].unique()),
        CONCEPT_CALC: set(cq_named.loc[cq_named.cluster_name == CONCEPT_CALC, "Question_Text"].unique()),
    }

    mode_dir = out_dir / ("model_seeds_all" if mode == "all" else "model_seed_1")
    res = run_one_mode(
        mode=mode,
        mode_dir=mode_dir,
        fig4=fig4,
        emb=emb,
        unique_questions=unique_questions,
        combined=combined,
        cq_named=cq_named,
        q_to_stored=q_to_stored,
        stored_names=stored_names,
        anchor_q=anchor_q,
        ks=ks,
        n_boot=n_boot,
        do_name=args.name,
    )

    # If the sibling mode already exists, write combined summary; else write partial
    results = {mode: res}
    sibling = "1" if mode == "all" else "all"
    sibling_dir = out_dir / ("model_seeds_all" if sibling == "all" else "model_seed_1")
    sibling_cons = sibling_dir / "consistency_by_K.csv"
    sibling_concept = sibling_dir / "concept_tracking.csv"
    sibling_k20 = sibling_dir / "k20_reproduction.json"
    if sibling_cons.exists() and sibling_concept.exists():
        sib = {
            "mode": sibling,
            "ari": None,
            "exact": None,
            "fig4_cmp": None,
            "cons": pd.read_csv(sibling_cons),
            "concept_df": pd.read_csv(sibling_concept),
            "corr_sum": pd.read_csv(sibling_dir / "correspondence_summary.csv")
            if (sibling_dir / "correspondence_summary.csv").exists()
            else pd.DataFrame(),
        }
        if sibling_k20.exists():
            meta = json.loads(sibling_k20.read_text())
            sib["ari"] = meta.get("ari_vs_stored")
            sib["exact"] = meta.get("exact_label_match")
            sib["fig4_cmp"] = {
                k: meta[k]
                for k in meta
                if k.startswith("max_abs") or k == "mode"
            } or None
        results[sibling] = sib
        # ensure current mode k20 meta saved for all
        if mode == "all":
            pass
        write_top_summary(out_dir, results, stored_names)
    else:
        # write a mode-local summary and a top-level stub
        write_top_summary(out_dir, results, stored_names)

    print("Exp 4 done →", mode_dir)


if __name__ == "__main__":
    main()
