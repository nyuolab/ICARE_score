#!/usr/bin/env python3
"""Exp 4: cluster K sensitivity (Nature Communications R2, reviewer R1-C7).

Reuses stored 2025 Fig. 4 embeddings/labels. Does not call process_questions().
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_stats.bootstrap_utils import make_indices, percentile_ci  # noqa: E402
from question_categorization_and_analysis.cluster_analysis import (  # noqa: E402
    analyze_clusters,
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
GEN_REPORT_CSV = {
    "maira-2": (
        "/gpfs/data/oermannlab/users/rd3571/RRG_models/maira-2/results/"
        "iuxray_report_gen_findings_frontal+lateral_seed{seed}_20250107_003058.csv"
    ),
    "chexpert-mimic-cxr-findings-baseline": (
        "/gpfs/data/oermannlab/users/rd3571/RRG_models/"
        "chexpert-mimic-cxr-findings-baseline/results/"
        "iuxray_report_gen_findings_frontal+lateral_seed{seed}_20250106_211756.csv"
    ),
    "mimic-cxr-findings-baseline": (
        "/gpfs/data/oermannlab/users/rd3571/RRG_models/"
        "mimic-cxr-findings-baseline/results/"
        "iuxray_report_gen_findings_frontal_seed{seed}_20250106_213559.csv"
    ),
}
ICARE_OUT = Path("/gpfs/data/oermannlab/users/rd3571/ICARE_score/outputs/IU_xray")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 4: cluster K sensitivity")
    p.add_argument("--fig4-root", type=Path, default=DEFAULT_FIG4)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="Use 200 bootstrap resamples")
    p.add_argument("--name", action="store_true", help="Name non-20 clusters via Llama API")
    p.add_argument("--ks", type=str, default="10,15,20,25,30")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--gate-only", action="store_true")
    return p.parse_args()


def dominance_label(ci_lo: float, ci_hi: float) -> str:
    if ci_hi < 0:
        return "omission-dominated"
    if ci_lo > 0:
        return "hallucination-dominated"
    return "balanced"


def write_representatives(
    path: Path,
    reps: Dict[int, List[str]],
    sizes: Dict[int, int],
) -> None:
    with open(path, "w") as f:
        for cid in sorted(reps):
            f.write(f"=== Cluster {cid} ({sizes[cid]} questions) ===\n\n")
            for i, q in enumerate(reps[cid]):
                f.write(f"{i + 1}. {q}\n")
            f.write("\n")


def score_slice_with_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    n_boot: int,
    boot_seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """Per model × cluster: GT, GEN, n_gt, n_gen, delta, CI, label.

    Bootstrap Report_IDs within model; keep all rows of a report together.
    """
    rows = []
    models = sorted(df["model_name"].unique())
    for model in models:
        mdf = df[df["model_name"] == model]
        reports = np.array(sorted(mdf["Report_ID"].unique()))
        n_rep = len(reports)
        rid_to_i = {r: i for i, r in enumerate(reports)}
        indices = make_indices(n_rep, n_boot, boot_seed)

        clusters = sorted(mdf[cluster_col].dropna().unique())
        for cid in clusters:
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
                num = a[idx].sum()
                den = n[idx].sum()
                return float(num / den) if den > 0 else float("nan")

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


def run_gate(
    fig4: Path,
    out_dir: Path,
    emb: np.ndarray,
    unique_questions: Sequence[str],
    cq: pd.DataFrame,
) -> Dict:
    gate_dir = out_dir / "_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    results: Dict = {"sklearn_version": sklearn.__version__, "pass_a": False, "pass_b": False}

    # (a) ARI
    stored_labels = (
        cq.drop_duplicates("Question_Text")
        .set_index("Question_Text")
        .loc[list(unique_questions), "cluster_id"]
        .to_numpy()
    )
    cluster_labels, _, _, _ = cluster_questions(emb, list(unique_questions), n_clusters=20)
    ari = float(adjusted_rand_score(stored_labels, cluster_labels))
    exact = float((np.asarray(cluster_labels) == stored_labels).mean())
    results["ari"] = ari
    results["exact_match_rate"] = exact
    results["pass_a"] = ari == 1.0
    print(f"GATE a: ARI={ari} exact={exact} pass={results['pass_a']}")

    # (b) recompute agreements
    _, gt_mc, gen_mc = analyze_clusters(cq, str(gate_dir))
    gt_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gt_ref.csv")
    gen_saved = pd.read_csv(fig4 / "analysis" / "model_cluster_agreement_gen_ref.csv")
    m_gt = gt_saved.merge(gt_mc, on=["Model", "Cluster"], suffixes=("_s", "_r"))
    m_gen = gen_saved.merge(gen_mc, on=["Model", "Cluster"], suffixes=("_s", "_r"))
    max_gt = float((m_gt["Agreement_s"] - m_gt["Agreement_r"]).abs().max())
    max_gen = float((m_gen["Agreement_s"] - m_gen["Agreement_r"]).abs().max())
    results["max_abs_diff_gt"] = max_gt
    results["max_abs_diff_gen"] = max_gen
    results["pass_b"] = max_gt < 1e-10 and max_gen < 1e-10
    print(f"GATE b: max|diff| GT={max_gt} GEN={max_gen} pass={results['pass_b']}")

    # (c) record-only
    results["gate_c"] = record_gate_c(cq, fig4)
    return results


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return float(len(a & b) / len(u)) if u else float("nan")


def record_gate_c(cq: pd.DataFrame, fig4: Path) -> Dict:
    """Record-only diagnostics across model seeds."""
    out: Dict = {"explanation": (
        "MCQ differences across model seeds are expected from LLM server batching "
        "nondeterminism (temperature 0 on a shared vLLM server). Proceed regardless."
    )}
    # (i) question-text and cluster Jaccard across model seeds
    rng = np.random.default_rng(42)
    models = sorted(cq["model_name"].unique())
    text_jacs = []
    cluster_jacs = []
    for model in models:
        mdf = cq[cq["model_name"] == model]
        # sample report×eval_seed pairs
        keys = mdf[["Report_ID", "eval_seed"]].drop_duplicates()
        sample = keys.sample(n=min(80, len(keys)), random_state=42)
        for _, row in sample.iterrows():
            rid, es = int(row.Report_ID), int(row.eval_seed)
            sets_q = {}
            sets_c = {}
            for ms in range(1, 6):
                sub = mdf[
                    (mdf.Report_ID == rid)
                    & (mdf.eval_seed == es)
                    & (mdf.model_seed == ms)
                ]
                sets_q[ms] = set(sub["Question_Text"])
                sets_c[ms] = set(sub["cluster_id"])
            for a, b in combinations(range(1, 6), 2):
                text_jacs.append(_jaccard(sets_q[a], sets_q[b]))
                cluster_jacs.append(_jaccard(sets_c[a], sets_c[b]))
    out["i_exact_text_jaccard_mean"] = float(np.nanmean(text_jacs))
    out["i_exact_text_jaccard_median"] = float(np.nanmedian(text_jacs))
    out["i_cluster_jaccard_mean"] = float(np.nanmean(cluster_jacs))
    out["i_cluster_jaccard_median"] = float(np.nanmedian(cluster_jacs))
    out["i_n_pairs"] = len(text_jacs)

    # (ii) option-text comparison on overlapping (Report_ID, Question_Text)
    # Use mcqa_eval_answer_predictions Options for one model×eval as sample
    opt_letter_match = []
    opt_text_match = []
    pred_letter_match = []
    pred_text_match = []
    try:
        model = "maira-2"
        es = 123
        paths = {
            ms: ICARE_OUT
            / model
            / f"model_seed_{ms}"
            / f"eval_seed_{es}"
            / "shuffled_ans_choices_data"
            / "gt_reports_as_ref"
            / "mcqa_eval"
            / "mcqa_eval_answer_predictions.csv"
            for ms in (1, 2)
        }
        if all(p.exists() for p in paths.values()):
            a = pd.read_csv(paths[1])
            b = pd.read_csv(paths[2])
            # Need Question_Text — join from filtered if needed
            filt = (
                ICARE_OUT
                / model
                / "model_seed_1"
                / f"eval_seed_{es}"
                / "shuffled_ans_choices_data"
                / "gt_reports_as_ref"
                / "mcqa_filtering"
                / "filtered_questions_shuffled.csv"
            )
            if filt.exists():
                fq = pd.read_csv(filt)[["Report_ID", "Question_ID", "Question_Text", "Options"]]
                a = a.merge(fq, on=["Report_ID", "Question_ID"], how="left", suffixes=("", "_fq"))
                if "Options" not in a.columns or a["Options"].isna().all():
                    a["Options"] = a.get("Options_fq", a.get("Options"))
                b = b.merge(
                    fq.rename(columns={"Question_Text": "Question_Text", "Options": "Options_b"}),
                    on=["Report_ID", "Question_ID"],
                    how="left",
                )
                # Prefer Options from predictions if present
            def _opt_map(options_val, letter):
                if pd.isna(options_val) or pd.isna(letter):
                    return None
                try:
                    opts = options_val if isinstance(options_val, dict) else ast.literal_eval(str(options_val))
                except Exception:
                    return None
                if not isinstance(opts, dict):
                    return None
                return opts.get(str(letter)) or opts.get(letter)

            # Merge a,b on Report_ID + Question_Text when available
            if "Question_Text" in a.columns and "Question_Text" in b.columns:
                m = a.merge(b, on=["Report_ID", "Question_Text"], suffixes=("_1", "_2"))
            else:
                m = a.merge(b, on=["Report_ID", "Question_ID"], suffixes=("_1", "_2"))
            m = m.head(5000)
            for _, r in m.iterrows():
                ca1, ca2 = r.get("Correct_Answer_1"), r.get("Correct_Answer_2")
                pg1, pg2 = r.get("Predicted_Answer_Using_GT_1"), r.get("Predicted_Answer_Using_GT_2")
                opt1 = r.get("Options_1", r.get("Options"))
                opt2 = r.get("Options_2", r.get("Options_b", r.get("Options")))
                opt_letter_match.append(ca1 == ca2)
                t1, t2 = _opt_map(opt1, ca1), _opt_map(opt2, ca2)
                if t1 is not None and t2 is not None:
                    opt_text_match.append(t1 == t2)
                pred_letter_match.append(pg1 == pg2)
                pt1, pt2 = _opt_map(opt1, pg1), _opt_map(opt2, pg2)
                if pt1 is not None and pt2 is not None:
                    pred_text_match.append(pt1 == pt2)
    except Exception as e:
        out["ii_error"] = str(e)
    if opt_letter_match:
        out["ii_correct_letter_match_rate"] = float(np.mean(opt_letter_match))
        out["ii_correct_option_text_match_rate"] = (
            float(np.mean(opt_text_match)) if opt_text_match else None
        )
        out["ii_pred_gt_letter_match_rate"] = float(np.mean(pred_letter_match))
        out["ii_pred_gt_option_text_match_rate"] = (
            float(np.mean(pred_text_match)) if pred_text_match else None
        )
        out["ii_n_overlap_rows"] = len(opt_letter_match)
        out["ii_note"] = (
            "If option-text match >> letter match, letter diffs are largely shuffle artifacts. "
            "If option-text still differs, content differs beyond shuffle."
        )

    # (iii) generated report identity across model seeds
    report_id_rates = {}
    for model, tmpl in GEN_REPORT_CSV.items():
        dfs = []
        ok = True
        for seed in range(1, 6):
            path = Path(tmpl.format(seed=seed))
            if not path.exists():
                ok = False
                report_id_rates[model] = {"error": f"missing {path}"}
                break
            d = pd.read_csv(path)
            # normalize id column
            id_col = "id" if "id" in d.columns else d.columns[0]
            gen_col = "generated_report" if "generated_report" in d.columns else None
            if gen_col is None:
                ok = False
                report_id_rates[model] = {"error": f"no generated_report in {path}"}
                break
            dfs.append(d[[id_col, gen_col]].rename(columns={id_col: "id", gen_col: f"gen_{seed}"}))
        if not ok:
            continue
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="id", how="inner")
        cols = [f"gen_{s}" for s in range(1, 6)]
        identical = merged[cols].nunique(axis=1) == 1
        report_id_rates[model] = {
            "n_reports": int(len(merged)),
            "pct_identical_across_seeds_1_5": float(100.0 * identical.mean()),
            "pairwise_seed1_vs_2_pct": float(
                100.0 * (merged["gen_1"] == merged["gen_2"]).mean()
            ),
        }
    out["iii_generated_report_identity"] = report_id_rates
    return out


def write_gate_md(path: Path, results: Dict) -> None:
    gc = results.get("gate_c", {})
    lines = [
        "# Gate results (Exp 4/5)",
        "",
        f"- sklearn: `{results.get('sklearn_version')}`",
        f"- **(a) ARI vs stored K=20 labels:** {results.get('ari')} "
        f"(exact match {results.get('exact_match_rate')}) — "
        f"**{'PASS' if results.get('pass_a') else 'FAIL'}**",
        f"- **(b) max |diff| vs stored agreement CSVs:** GT={results.get('max_abs_diff_gt')}, "
        f"GEN={results.get('max_abs_diff_gen')} — "
        f"**{'PASS' if results.get('pass_b') else 'FAIL'}**",
        "",
        "## (c) Record-only (do not stop)",
        "",
        gc.get("explanation", ""),
        "",
        "### (i) Question Jaccard across model seeds",
        f"- Exact-text Jaccard mean/median: {gc.get('i_exact_text_jaccard_mean')}/"
        f"{gc.get('i_exact_text_jaccard_median')} (n_pairs={gc.get('i_n_pairs')})",
        f"- K=20-cluster Jaccard mean/median: {gc.get('i_cluster_jaccard_mean')}/"
        f"{gc.get('i_cluster_jaccard_median')}",
        "",
        "### (ii) Letter vs option-text answer comparison",
        f"- Correct letter match: {gc.get('ii_correct_letter_match_rate')}",
        f"- Correct option-text match: {gc.get('ii_correct_option_text_match_rate')}",
        f"- Pred-GT letter match: {gc.get('ii_pred_gt_letter_match_rate')}",
        f"- Pred-GT option-text match: {gc.get('ii_pred_gt_option_text_match_rate')}",
        f"- n overlap rows: {gc.get('ii_n_overlap_rows')}",
        f"- note: {gc.get('ii_note', '')}",
        "",
        "### (iii) Generated report identity across model_seed 1–5",
        "```json",
        json.dumps(gc.get("iii_generated_report_identity", {}), indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    (path.parent / "gate.json").write_text(json.dumps(results, indent=2, default=str))


def concept_tracking(
    cq_k: pd.DataFrame,
    anchor_q: Dict[str, set],
    scores: pd.DataFrame,
    k: int,
) -> List[dict]:
    rows = []
    for concept, qset in anchor_q.items():
        sub = cq_k[cq_k["Question_Text"].isin(qset)]
        total = len(sub)
        if total == 0:
            continue
        shares = (
            sub.groupby("cluster_id").size().astype(float) / total
        ).sort_values(ascending=False)
        matched = int(shares.index[0])
        for cid, share in shares.items():
            if share < 0.20 and int(cid) != matched:
                continue
            for model in sorted(scores.Model.unique()):
                sc = scores[(scores.Model == model) & (scores.Cluster == int(cid))]
                lab = sc.label.iloc[0] if len(sc) else None
                rows.append(
                    {
                        "K": k,
                        "concept": concept,
                        "cluster_id": int(cid),
                        "instance_share": float(share),
                        "is_matched": int(cid) == matched,
                        "Model": model,
                        "label": lab,
                    }
                )
    return rows


def correspondence_matrix(
    q_to_orig: Dict[str, int],
    q_to_new: Dict[str, int],
    n_orig: int,
    n_new: int,
) -> np.ndarray:
    """Share of each original cluster's unique questions in each new cluster."""
    mat = np.zeros((n_orig, n_new), dtype=np.float64)
    by_orig: Dict[int, List[str]] = defaultdict(list)
    for q, oid in q_to_orig.items():
        by_orig[oid].append(q)
    for oid, qs in by_orig.items():
        counts = np.zeros(n_new, dtype=np.float64)
        for q in qs:
            if q in q_to_new:
                counts[int(q_to_new[q])] += 1
        s = counts.sum()
        if s > 0:
            mat[oid] = counts / s
    return mat


def consistency_metrics(
    scores_orig: pd.DataFrame,
    scores_new: pd.DataFrame,
    q_to_orig: Dict[str, int],
    q_to_new: Dict[str, int],
    cq_new: pd.DataFrame,
    k: int,
) -> List[dict]:
    """(a) label agreement via best-match; (b) Spearman on deltas; (c) instance shares."""
    rows = []
    # unique-Q best match orig -> new
    mat = correspondence_matrix(q_to_orig, q_to_new, 20, k)
    best_new = {oid: int(np.argmax(mat[oid])) for oid in range(20)}

    for model in sorted(scores_orig.Model.unique()):
        so = scores_orig[scores_orig.Model == model].set_index("Cluster")
        sn = scores_new[scores_new.Model == model].set_index("Cluster")
        # (a)
        agree = 0
        for oid in range(20):
            if oid not in so.index or best_new[oid] not in sn.index:
                continue
            if so.loc[oid, "label"] == sn.loc[best_new[oid], "label"]:
                agree += 1
        pct_a = 100.0 * agree / 20.0

        # (b) Spearman: orig delta vs question-weighted mean new delta
        # weight by unique-Q correspondence shares (instances for consistency b per approval: instances)
        # Use instance-weighted mapping: for each orig cluster, weighted mean of new deltas
        # by share of orig cluster's *instances* falling in each new cluster
        orig_q = {oid: {q for q, o in q_to_orig.items() if o == oid} for oid in range(20)}
        mdf = cq_new[cq_new["model_name"] == model]
        orig_deltas = []
        mapped_deltas = []
        for oid in range(20):
            if oid not in so.index:
                continue
            qs = orig_q[oid]
            sub = mdf[mdf["Question_Text"].isin(qs)]
            if sub.empty or best_new[oid] not in sn.index:
                continue
            shares = sub.groupby("cluster_id").size().astype(float)
            shares = shares / shares.sum()
            wmean = 0.0
            for cid, sh in shares.items():
                if int(cid) in sn.index:
                    wmean += float(sh) * float(sn.loc[int(cid), "delta"])
            orig_deltas.append(float(so.loc[oid, "delta"]))
            mapped_deltas.append(wmean)
        if len(orig_deltas) >= 3:
            rho, _ = spearmanr(orig_deltas, mapped_deltas)
        else:
            rho = float("nan")

        # (c) instance shares in omission / hallucination clusters
        lab_map = sn["label"].to_dict()
        inst = mdf.copy()
        inst["dom_label"] = inst["cluster_id"].map(lab_map)
        n_inst = len(inst)
        share_om = float((inst["dom_label"] == "omission-dominated").mean()) if n_inst else 0.0
        share_hall = (
            float((inst["dom_label"] == "hallucination-dominated").mean()) if n_inst else 0.0
        )

        rows.append(
            {
                "K": k,
                "Model": model,
                "pct_label_match_best": pct_a,
                "spearman_delta": float(rho) if rho == rho else None,
                "share_omission_instances": share_om,
                "share_hallucination_instances": share_hall,
            }
        )
    return rows


def plot_scatter_grid(
    all_scores: Dict[int, pd.DataFrame],
    concept_df: pd.DataFrame,
    out_path: Path,
) -> None:
    ks = sorted(all_scores)
    models = sorted(all_scores[ks[0]].Model.unique())
    short = {
        "chexpert-mimic-cxr-findings-baseline": "CheX+MIMIC",
        "mimic-cxr-findings-baseline": "MIMIC",
        "maira-2": "MAIRA-2",
    }
    fig, axes = plt.subplots(len(ks), len(models), figsize=(4 * len(models), 3.5 * len(ks)))
    if len(ks) == 1:
        axes = np.array([axes])
    for i, k in enumerate(ks):
        sc = all_scores[k]
        # matched concept clusters for this K
        matched = set(
            int(x)
            for x in concept_df.loc[
                (concept_df.K == k) & (concept_df.is_matched), "cluster_id"
            ].unique()
        )
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = sc[sc.Model == model]
            for _, r in sub.iterrows():
                color = LABEL_COLORS.get(r.label, "#333333")
                size = 20 + 80 * (r.n / max(sub.n.max(), 1))
                edge = "gold" if int(r.Cluster) in matched else "none"
                lw = 2.0 if int(r.Cluster) in matched else 0.0
                ax.scatter(
                    r.GEN,
                    r.GT,
                    s=size,
                    c=color,
                    edgecolors=edge,
                    linewidths=lw,
                    alpha=0.85,
                    zorder=3,
                )
                ax.text(r.GEN, r.GT, str(int(r.Cluster)), fontsize=7, ha="center", va="center")
            ax.plot([0, 1], [0, 1], ls="--", c="0.5", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("ICARE-GEN")
            ax.set_ylabel("ICARE-GT")
            if i == 0:
                ax.set_title(short.get(model, model))
            if j == 0:
                ax.set_ylabel(f"K={k}\nICARE-GT")
    fig.suptitle(
        "Exp 4: GEN vs GT by cluster (colour=dominance label; gold edge=matched concept)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consistency_c(cons: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    short = {
        "chexpert-mimic-cxr-findings-baseline": "CheX+MIMIC",
        "mimic-cxr-findings-baseline": "MIMIC",
        "maira-2": "MAIRA-2",
    }
    for model, g in cons.groupby("Model"):
        g = g.sort_values("K")
        ax.plot(
            g.K,
            100 * g.share_omission_instances,
            marker="o",
            label=f"{short.get(model, model)} omission",
        )
        ax.plot(
            g.K,
            100 * g.share_hallucination_instances,
            marker="s",
            ls="--",
            label=f"{short.get(model, model)} hallucination",
        )
    ax.set_xlabel("K")
    ax.set_ylabel("% of question instances")
    ax.set_title("Share of instances in omission- / hallucination-dominated clusters")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_name(out_dir: Path, ks: Sequence[int], cq_by_k: Dict[int, pd.DataFrame]) -> None:
    """Call name_clusters_with_llama for K≠20; write K-suffixed name files."""
    for k in ks:
        if k == 20:
            continue
        reps_path = out_dir / f"cluster_representatives_K{k}.txt"
        # rebuild reps dict from file
        reps: Dict[int, List[str]] = {}
        cur = None
        for line in reps_path.read_text().splitlines():
            if line.startswith("=== Cluster"):
                cur = int(line.split()[2])
                reps[cur] = []
            elif line[:1].isdigit() and ". " in line and cur is not None:
                reps[cur].append(line.split(". ", 1)[1])
        scratch = out_dir / f"naming_K{k}"
        scratch.mkdir(parents=True, exist_ok=True)
        names = name_clusters_with_llama(reps, str(scratch))
        if not names:
            print(f"--name: failed for K={k}")
            continue
        # normalize keys to str for json parity with stored format
        names_out = {str(int(k_)): v for k_, v in names.items()}
        (out_dir / f"cluster_names_K{k}.json").write_text(json.dumps(names_out, indent=2))
        df = cq_by_k[k].copy()
        df["cluster_name"] = df["cluster_id"].map({int(k_): v for k_, v in names_out.items()})
        df.to_csv(out_dir / f"clustered_questions_with_names_K{k}.csv", index=False)
        print(f"--name: wrote names for K={k}")


def write_summary(
    path: Path,
    gate: Dict,
    concept_df: pd.DataFrame,
    cons: pd.DataFrame,
    corr_sum: pd.DataFrame,
) -> None:
    lines = [
        "# Exp 4 summary — cluster K sensitivity",
        "",
        f"- sklearn: `{gate.get('sklearn_version')}`",
        f"- Gate (a) PASS={gate.get('pass_a')} ARI={gate.get('ari')}",
        f"- Gate (b) PASS={gate.get('pass_b')} max|diff| GT/GEN="
        f"{gate.get('max_abs_diff_gt')}/{gate.get('max_abs_diff_gen')}",
        "- Gate (c): record-only (see gate.md)",
        "",
        "## Matched clusters for tracked concepts",
        "",
    ]
    for concept in (CONCEPT_LUNG, CONCEPT_CALC):
        lines.append(f"### {concept}")
        sub = concept_df[(concept_df.concept == concept) & (concept_df.is_matched)]
        for k, g in sub.groupby("K"):
            labs = ", ".join(
                f"{r.Model.split('-')[0]}:{r.label} (cluster {r.cluster_id}, share={r.instance_share:.2f})"
                for _, r in g.iterrows()
            )
            # one row per model; cluster same
            cid = int(g.cluster_id.iloc[0])
            share = float(g.instance_share.iloc[0])
            per_model = "; ".join(
                f"{r.Model}: {r.label}" for _, r in g.drop_duplicates("Model").iterrows()
            )
            lines.append(f"- K={k}: matched cluster **{cid}** (share={share:.3f}); labels: {per_model}")
        # also ≥20% non-matched
        extra = concept_df[
            (concept_df.concept == concept) & (~concept_df.is_matched)
        ]
        if len(extra):
            lines.append(f"  Other clusters with ≥20% share: {sorted(extra.cluster_id.unique().tolist())}")
        lines.append("")

    lines += ["## Consistency (a)–(c) per K", ""]
    for _, r in cons.sort_values(["K", "Model"]).iterrows():
        lines.append(
            f"- K={int(r.K)} {r.Model}: (a) label-match={r.pct_label_match_best:.1f}% "
            f"(b) Spearman={r.spearman_delta} "
            f"(c) omission={100*r.share_omission_instances:.1f}% "
            f"hallucination={100*r.share_hallucination_instances:.1f}%"
        )
    lines += ["", "## Correspondence vs K=20", ""]
    for _, r in corr_sum.iterrows():
        lines.append(
            f"- K={int(r.K)}: ARI={r.ARI:.4f}; mean best-match share={r.mean_best_match_share:.4f}"
        )
    lines += ["", "## Unexpected / notes", "", "- See gate.md for model-seed MCQ nondeterminism.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    n_boot = 200 if args.quick else args.n_boot
    ks = tuple(int(x) for x in args.ks.split(","))
    fig4 = args.fig4_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    print(f"sklearn={sklearn.__version__} n_boot={n_boot} ks={ks}")
    print(f"fig4={fig4}")
    print(f"out={out_dir}")

    with open(fig4 / "question_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    combined = pd.read_csv(fig4 / "combined_mcqa_data.csv")
    cq = pd.read_csv(fig4 / "clustered_questions_with_names.csv")
    unique_questions = list(combined["Question_Text"].unique())
    assert emb.shape[0] == len(unique_questions)

    with open(fig4 / "cluster_names.json") as f:
        stored_names = {int(k): v for k, v in json.load(f).items()}

    if not args.skip_gate:
        gate = run_gate(fig4, out_dir, emb, unique_questions, cq)
        write_gate_md(out_dir / "gate.md", gate)
        if not (gate["pass_a"] and gate["pass_b"]):
            print("GATE a/b FAILED — stopping.")
            sys.exit(2)
        if args.gate_only:
            print("gate-only done")
            return
    else:
        gate = {"sklearn_version": sklearn.__version__, "pass_a": True, "pass_b": True}

    # Stored K=20 mapping
    q_to_k20 = (
        cq.drop_duplicates("Question_Text")
        .set_index("Question_Text")["cluster_id"]
        .astype(int)
        .to_dict()
    )
    anchor_q = {
        CONCEPT_LUNG: set(cq.loc[cq.cluster_name == CONCEPT_LUNG, "Question_Text"].unique()),
        CONCEPT_CALC: set(cq.loc[cq.cluster_name == CONCEPT_CALC, "Question_Text"].unique()),
    }

    # Baseline scores at K=20 (stored labels)
    scores_k20 = score_slice_with_bootstrap(cq, "cluster_id", n_boot)
    scores_k20.to_csv(out_dir / "scores_K20.csv", index=False)

    all_scores: Dict[int, pd.DataFrame] = {}
    cq_by_k: Dict[int, pd.DataFrame] = {}
    concept_rows: List[dict] = []
    cons_rows: List[dict] = []
    corr_rows: List[dict] = []

    for k in ks:
        print(f"=== K={k} ===")
        if k == 20:
            cluster_labels = np.array([q_to_k20[q] for q in unique_questions])
            # rebuild reps from stored file for completeness
            reps: Dict[int, List[str]] = {}
            sizes: Dict[int, int] = {}
            # still call cluster_questions for representatives consistency with pipeline
            _, q2c, reps, sizes = cluster_questions(emb, unique_questions, n_clusters=20)
            # force stored labels for assignments
            q2c = dict(q_to_k20)
            cluster_labels = np.array([q2c[q] for q in unique_questions])
            cq_k = cq.copy()
        else:
            cluster_labels, q2c, reps, sizes = cluster_questions(
                emb, unique_questions, n_clusters=k
            )
            cq_k = combined.copy()
            cq_k["cluster_id"] = cq_k["Question_Text"].map(q2c)

        # save assignments + reps
        cq_k.to_csv(out_dir / f"clustered_questions_K{k}.csv", index=False)
        write_representatives(out_dir / f"cluster_representatives_K{k}.txt", reps, sizes)
        cq_by_k[k] = cq_k

        if k == 20:
            scores = scores_k20.copy()
        else:
            scores = score_slice_with_bootstrap(cq_k, "cluster_id", n_boot)
        scores.to_csv(out_dir / f"scores_K{k}.csv", index=False)
        all_scores[k] = scores

        concept_rows.extend(concept_tracking(cq_k, anchor_q, scores, k))

        mat = correspondence_matrix(q_to_k20, q2c, 20, k)
        np.save(out_dir / f"correspondence_K{k}.npy", mat)
        pd.DataFrame(mat, index=[f"orig_{i}" for i in range(20)], columns=[f"new_{i}" for i in range(k)]).to_csv(
            out_dir / f"correspondence_K{k}.csv"
        )
        # heatmap
        fig, ax = plt.subplots(figsize=(max(6, k * 0.35), 5))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel(f"new cluster (K={k})")
        ax.set_ylabel("original K=20 cluster")
        ax.set_title(f"Overlap share (unique Question_Text), K={k}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / f"correspondence_heatmap_K{k}.png", dpi=120)
        plt.close(fig)

        ari = float(
            adjusted_rand_score(
                [q_to_k20[q] for q in unique_questions],
                [q2c[q] for q in unique_questions],
            )
        )
        mean_best = float(mat.max(axis=1).mean())
        corr_rows.append({"K": k, "ARI": ari, "mean_best_match_share": mean_best})

        cons_rows.extend(
            consistency_metrics(scores_k20, scores, q_to_k20, q2c, cq_k, k)
        )

    concept_df = pd.DataFrame(concept_rows)
    concept_df.to_csv(out_dir / "concept_tracking.csv", index=False)
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(out_dir / "consistency_by_K.csv", index=False)
    corr_sum = pd.DataFrame(corr_rows)
    corr_sum.to_csv(out_dir / "correspondence_summary.csv", index=False)

    plot_scatter_grid(all_scores, concept_df, out_dir / "scatter_grid.png")
    plot_consistency_c(cons, out_dir / "consistency_share_vs_K.png")
    write_summary(out_dir / "summary.md", gate, concept_df, cons, corr_sum)

    # K=20 named copy for parity
    with open(fig4 / "cluster_names.json") as f:
        (out_dir / "cluster_names_K20.json").write_text(f.read())

    if args.name:
        run_name(out_dir, ks, cq_by_k)

    print("Exp 4 done →", out_dir)


if __name__ == "__main__":
    main()
