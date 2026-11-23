#!/usr/bin/env python3
"""ReXVal alignment statistics: original, fractional, and pairwise protocols.

Additional analyses only. Does not modify ``plot_rexval_correlation.py`` or its
absolute-path plot outputs. Writes exclusively under ``--out-dir``.

Protocols
---------
original
    Submission top-1 rule (rank method=min + drop_duplicates; alphabetical /
    row-order tie-break). Preserved for reference.
fractional
    Expected agreement under independent uniform tie-breaking:
    |Sr ∩ Sm| / (|Sr| |Sm|). Chance remains 25% with 4 candidates.
pairwise
    Agreement on clinician-discordant candidate pairs only (1 / 0.5 / 0).
    Chance is 50%.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_stats.bootstrap_utils import (  # noqa: E402
    assert_finite_replicates,
    holm_adjust,
    make_indices,
    null_centered_pvalue,
    paired_comparison_table,
    percentile_ci,
)

BASE = Path("/gpfs/data/oermannlab/users/rd3571")
EVAL_DIR = BASE / "ICARE_score/outputs/rexval/rexval_test_200/eval_seed_123/shuffled_ans_choices_data"
PRED_DIR = BASE / "ICARE_score/outputs/rexval/predefined/eval_seed_123/mcqa_eval"
BASELINES = BASE / "ICARE_score/outputs/rexval/rexval_test_200/baselines"
REXVAL_CSV = BASE / "cxr_report_datasets/rexval/RexVal_test_icare_200.csv"
LABELS_CSV = (
    BASE
    / "cxr_report_datasets/rexval_physionet_labels"
    / "physionet.org/files/rexval-dataset/1.0.0"
    / "6_valid_raters_per_rater_error_categories.csv"
)

PUBLISHED_ORIGINAL = {
    "ICARE_AVG": 47.7,
    "ICARE_PREDEFINED": 52.3,
    "BERTScore": 68.7,
}

PROPOSED = ["ICARE_AVG", "ICARE_PREDEFINED"]
BASELINE_METRICS = [
    "CRIMSON",
    "GREEN",
    "AlignScore",
    "BERTScore",
    "SembScore",
    "RadGraph",
    "BLEU",
    "RadCliQ-v1",
]
ALL_METRICS = PROPOSED + BASELINE_METRICS
# higher-is-better columns in the score table
ASCENDING_FALSE = set(ALL_METRICS)  # RadCliQ stored already negated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["original", "fractional", "pairwise"],
        required=True,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to worktree outputs/.../clinical_alignment_stats/<mode>",
    )
    p.add_argument("--n-boot-ci", type=int, default=2000)
    p.add_argument("--n-boot-paired", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol-published", type=float, default=0.05)
    p.add_argument(
        "--exclude-four-way-ties",
        action="store_true",
        help="Fractional mode only: drop clinician 4-way ties from the average.",
    )
    return p.parse_args()


def load_study_tables() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], List[int]]:
    """Return study metadata, error tensor [S,C,R], metric scores [S,C], rater list."""
    rex = pd.read_csv(REXVAL_CSV).rename(columns={"Unnamed: 0": "row_id"})
    labels = pd.read_csv(LABELS_CSV)
    clin = labels[labels["clinically_significant"] == True].copy()  # noqa: E712
    per = (
        clin.groupby(["study_number", "candidate_type", "rater_index"], as_index=False)["num_errors"]
        .sum()
        .rename(columns={"num_errors": "errors", "candidate_type": "origin"})
    )
    raters = sorted(per["rater_index"].unique().tolist())
    order = list(rex[rex.study_number == 0]["origin"])
    assert order == ["radgraph", "bertscore", "s_emb", "bleu"]
    assert all(
        list(g["origin"]) == order for _, g in rex.groupby("study_number")
    ), "candidate row order must be stable across studies"

    S, C, R = 50, 4, len(raters)
    E = np.zeros((S, C, R), dtype=float)
    for s, o, r, v in per[["study_number", "origin", "rater_index", "errors"]].itertuples(index=False):
        E[int(s), order.index(o), raters.index(int(r))] = float(v)

    gt = pd.read_csv(EVAL_DIR / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    gen = pd.read_csv(EVAL_DIR / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    pred = pd.read_csv(PRED_DIR / "mcq_eval_report_level_stats.csv").set_index("Report_ID")
    with open(BASELINES / "crimson_results.json") as f:
        crimson = np.array([r["crimson_score"] for r in json.load(f)["results"]])
    green = pd.read_csv(BASELINES / "green/rexval_prepared_green_results.csv")["green_score"].to_numpy()
    align = pd.read_csv(BASELINES / "alignscore/rexval_prepared_alignscore_results.csv")["alignscore"].to_numpy()
    rrg = pd.read_csv(BASELINES / "rexval_test_200/rexval_prepared_results.csv")

    scores = {
        "ICARE_AVG": ((gt.loc[range(200), "Agreement_Percentage"].to_numpy()
                       + gen.loc[range(200), "Agreement_Percentage"].to_numpy()) / 2.0).reshape(S, C),
        "ICARE_PREDEFINED": pred.loc[range(200), "Agreement_Percentage"].to_numpy().reshape(S, C),
        "CRIMSON": crimson.reshape(S, C),
        "GREEN": green.reshape(S, C),
        "AlignScore": align.reshape(S, C),
        "BERTScore": rrg["bertscore"].to_numpy().reshape(S, C),
        "SembScore": rrg["semb_score"].to_numpy().reshape(S, C),
        "RadGraph": rrg["radgraph_combined"].to_numpy().reshape(S, C),
        "BLEU": rrg["bleu_score"].to_numpy().reshape(S, C),
        "RadCliQ-v1": (-rrg["RadCliQ-v1"].to_numpy()).reshape(S, C),  # higher=better
    }

    studies = pd.DataFrame({"study_number": np.arange(S, dtype=int)})
    studies["candidate_order"] = [",".join(order)] * S
    return studies, E, scores, raters


def rater_tied_sets(E: np.ndarray) -> np.ndarray:
    """Boolean [S, R, C] for clinician tied-best candidates."""
    return E == E.min(axis=1, keepdims=True)


def metric_tied_sets(scores: np.ndarray) -> np.ndarray:
    """Boolean [S, C] for metric tied-best (higher better)."""
    return scores == scores.max(axis=1, keepdims=True)


def original_pick_from_sets(tied: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Mirror plot script: among tied True entries, take smallest priority index."""
    # tied: [..., C]
    picks = np.empty(tied.shape[:-1], dtype=int)
    flat = tied.reshape(-1, tied.shape[-1])
    out = np.empty(flat.shape[0], dtype=int)
    for i, row in enumerate(flat):
        idx = np.flatnonzero(row)
        out[i] = idx[np.argmin(priority[idx])]
    return out.reshape(tied.shape[:-1])


def study_scores_original(
    E: np.ndarray,
    scores: Dict[str, np.ndarray],
    cand_order: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Per-study mean over raters of hit indicators; also per-study inter-rater %."""
    S, C, R = E.shape
    # Rater tie-break priority = alphabetical order of candidate names (plot script).
    alpha = np.array([sorted(cand_order).index(o) for o in cand_order])
    Sr = (E == E.min(axis=1, keepdims=True)).transpose(0, 2, 1)  # [S,R,C]
    rpick = original_pick_from_sets(Sr, alpha)  # [S,R]

    # Metric picks: first True in CSV row order (priority = 0..C-1)
    row_pri = np.arange(C)
    mpick = {}
    for m, sc in scores.items():
        Sm = metric_tied_sets(sc)
        mpick[m] = original_pick_from_sets(Sm, row_pri)  # [S]

    per_study = {}
    for m in ALL_METRICS:
        hits = (mpick[m][:, None] == rpick).astype(float)  # [S,R]
        per_study[m] = hits.mean(axis=1) * 100.0

    pairs = list(itertools.combinations(range(R), 2))
    ir = np.zeros(S)
    for s in range(S):
        ir[s] = np.mean([rpick[s, a] == rpick[s, b] for a, b in pairs]) * 100.0
    return per_study, ir


def study_scores_fractional(
    E: np.ndarray,
    scores: Dict[str, np.ndarray],
    exclude_four_way: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    Sr = (E == E.min(axis=1, keepdims=True)).transpose(0, 2, 1)  # [S,R,C]
    k = Sr.sum(axis=2)
    per_study = {}
    for m, sc in scores.items():
        Sm = metric_tied_sets(sc)  # [S,C]
        # |∩| / (|Sr||Sm|)
        inter = (Sr & Sm[:, None, :]).sum(axis=2).astype(float)
        denom = k * Sm.sum(axis=1, keepdims=True)
        frac = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        if exclude_four_way:
            mask = k != 4
            # mean over remaining raters; if none, nan
            vals = []
            for s in range(Sr.shape[0]):
                msk = mask[s]
                vals.append(frac[s, msk].mean() * 100.0 if msk.any() else np.nan)
            per_study[m] = np.array(vals, dtype=float)
        else:
            per_study[m] = frac.mean(axis=1) * 100.0

    R = Sr.shape[1]
    pairs = list(itertools.combinations(range(R), 2))
    ir = np.zeros(Sr.shape[0])
    for s in range(Sr.shape[0]):
        vals = []
        for a, b in pairs:
            if exclude_four_way and (k[s, a] == 4 or k[s, b] == 4):
                continue
            ka, kb = k[s, a], k[s, b]
            inter = np.logical_and(Sr[s, a], Sr[s, b]).sum()
            vals.append(inter / (ka * kb))
        ir[s] = (np.mean(vals) * 100.0) if vals else np.nan
    return per_study, ir


def study_scores_pairwise(
    E: np.ndarray,
    scores: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return per-study rates and denominator counts for weighting in bootstrap."""
    S, C, R = E.shape
    pairs = list(itertools.combinations(range(C), 2))
    num = {m: np.zeros(S) for m in ALL_METRICS}
    den = np.zeros(S)
    ir_num = np.zeros(S)
    ir_den = np.zeros(S)
    for s in range(S):
        for i, j in pairs:
            for r in range(R):
                ei, ej = E[s, i, r], E[s, j, r]
                if ei == ej:
                    continue
                better, worse = (i, j) if ei < ej else (j, i)
                den[s] += 1
                for m in ALL_METRICS:
                    x, y = scores[m][s, better], scores[m][s, worse]
                    if x > y:
                        num[m][s] += 1.0
                    elif x == y:
                        num[m][s] += 0.5
                for r2 in range(R):
                    if r2 == r:
                        continue
                    x, y = E[s, better, r2], E[s, worse, r2]
                    ir_den[s] += 1
                    if x < y:
                        ir_num[s] += 1.0
                    elif x == y:
                        ir_num[s] += 0.5
    per_study = {}
    for m in ALL_METRICS:
        per_study[m] = np.divide(
            100.0 * num[m], den, out=np.full(S, np.nan), where=den > 0
        )
    ir = np.divide(100.0 * ir_num, ir_den, out=np.full(S, np.nan), where=ir_den > 0)
    return per_study, ir, den


def bootstrap_mean_of_study_scores(
    per_study: Dict[str, np.ndarray],
    indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    out = {}
    for m, arr in per_study.items():
        reps = np.empty(indices.shape[0])
        for b, idx in enumerate(indices):
            vals = arr[idx]
            reps[b] = np.nanmean(vals)
        out[m] = reps
    return out


def bootstrap_ir(ir_study: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(ir_study[idx]) for idx in indices])


def joint_eligible_original(
    E: np.ndarray,
    scores: Dict[str, np.ndarray],
    cand_order: Sequence[str],
    proposed: str,
    baseline: str,
) -> Tuple[float, float]:
    """Restrict to studies where both metrics have a unique top score."""
    S, C, R = E.shape
    alpha = np.array([sorted(cand_order).index(o) for o in cand_order])
    Sr = (E == E.min(axis=1, keepdims=True)).transpose(0, 2, 1)
    rpick = original_pick_from_sets(Sr, alpha)
    Sm_p = metric_tied_sets(scores[proposed])
    Sm_b = metric_tied_sets(scores[baseline])
    unique = (Sm_p.sum(1) == 1) & (Sm_b.sum(1) == 1)
    if unique.sum() == 0:
        return float("nan"), float("nan")
    mp = original_pick_from_sets(Sm_p, np.arange(C))
    mb = original_pick_from_sets(Sm_b, np.arange(C))
    prop = (mp[:, None] == rpick).mean(axis=1) * 100.0
    base = (mb[:, None] == rpick).mean(axis=1) * 100.0
    return float(prop[unique].mean()), float(base[unique].mean())


def run_mode(args: argparse.Namespace) -> None:
    studies, E, scores, raters = load_study_tables()
    cand_order = ["radgraph", "bertscore", "s_emb", "bleu"]
    S = E.shape[0]
    assert list(studies["study_number"]) == list(range(S))

    if args.out_dir is None:
        args.out_dir = (
            REPO_ROOT
            / "outputs/rexval/rexval_test_200/clinical_alignment_stats"
            / args.mode
        )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "original":
        per_study, ir_study = study_scores_original(E, scores, cand_order)
        result_class = "original_protocol_reference"
        # Reproduction check against published points
        points = {m: float(np.nanmean(per_study[m])) for m in ALL_METRICS}
        for m, pub in PUBLISHED_ORIGINAL.items():
            if abs(points[m] - pub) > args.tol_published:
                raise AssertionError(
                    f"Original-protocol point mismatch for {m}: {points[m]} vs published {pub}"
                )
    elif args.mode == "fractional":
        per_study, ir_study = study_scores_fractional(
            E, scores, exclude_four_way=args.exclude_four_way_ties
        )
        result_class = (
            "alternative_fractional_no4way"
            if args.exclude_four_way_ties
            else "alternative_fractional"
        )
        points = {m: float(np.nanmean(per_study[m])) for m in ALL_METRICS}
    else:
        per_study, ir_study, den = study_scores_pairwise(E, scores)
        result_class = "alternative_pairwise"
        points = {m: float(np.nanmean(per_study[m])) for m in ALL_METRICS}

    idx_ci = make_indices(S, args.n_boot_ci, args.seed)
    idx_paired = make_indices(S, args.n_boot_paired, args.seed)
    boot_ci = bootstrap_mean_of_study_scores(per_study, idx_ci)
    boot_paired = bootstrap_mean_of_study_scores(per_study, idx_paired)
    # Allow NaNs only in fractional_no4way edge cases; otherwise require finite.
    for name, reps in boot_paired.items():
        if args.mode == "fractional" and args.exclude_four_way_ties:
            if np.isnan(reps).all():
                raise AssertionError(f"All NaN replicates for {name}")
        else:
            if not np.isfinite(reps).all():
                raise AssertionError(f"Non-finite replicates for {name}")

    rows = []
    for m in ALL_METRICS:
        lo, hi = percentile_ci(boot_ci[m][np.isfinite(boot_ci[m])])
        rows.append(
            {
                "dataset": "ReXVal",
                "protocol": args.mode,
                "metric": m,
                "n_studies": S,
                "point": points[m],
                "ci_lo": lo,
                "ci_hi": hi,
                "n_boot": args.n_boot_ci,
                "seed": args.seed,
                "result_class": result_class,
                "exclude_four_way_ties": bool(args.exclude_four_way_ties),
            }
        )
    ci_df = pd.DataFrame(rows)
    ci_path = out_dir / f"rexval_{args.mode}_alignment_ci.csv"
    ci_df.to_csv(ci_path, index=False)

    # Inter-rater
    ir_point = float(np.nanmean(ir_study))
    ir_boot = bootstrap_ir(ir_study, idx_ci)
    ir_boot_f = ir_boot[np.isfinite(ir_boot)]
    ir_lo, ir_hi = percentile_ci(ir_boot_f)
    ir_df = pd.DataFrame(
        [
            {
                "dataset": "ReXVal",
                "protocol": args.mode,
                "statistic": "inter_rater_agreement",
                "point": ir_point,
                "ci_lo": ir_lo,
                "ci_hi": ir_hi,
                "n_boot": args.n_boot_ci,
                "seed": args.seed,
                "result_class": result_class,
            }
        ]
    )
    ir_path = out_dir / f"rexval_{args.mode}_interrater_ci.csv"
    ir_df.to_csv(ir_path, index=False)

    paired_df = paired_comparison_table(
        points, boot_paired, PROPOSED, BASELINE_METRICS, dataset=f"ReXVal:{args.mode}"
    )
    paired_path = out_dir / f"rexval_{args.mode}_paired_comparisons.csv"
    paired_df.to_csv(paired_path, index=False)

    reps_path = out_dir / f"rexval_{args.mode}_paired_bootstrap_replicates.csv"
    pd.DataFrame({m: boot_paired[m] for m in ALL_METRICS}).to_csv(reps_path, index=False)

    # Study-level export with identifiers for provenance / shared-index clarity.
    export = studies.copy()
    export["inter_rater_study"] = ir_study
    for m in ALL_METRICS:
        export[f"align_{m}"] = per_study[m]
    export_path = out_dir / f"rexval_{args.mode}_study_aligned_scores.csv"
    export.to_csv(export_path, index=False)

    # Jointly eligible sensitivity (original mode only).
    if args.mode == "original":
        joint_rows = []
        for proposed in PROPOSED:
            for baseline in BASELINE_METRICS:
                # Point on jointly unique-top studies; bootstrap by resampling all
                # studies then recomputing unique mask inside — approximate via
                # study-level scores restricted each replicate.
                Sm_p = metric_tied_sets(scores[proposed])
                Sm_b = metric_tied_sets(scores[baseline])
                unique = (Sm_p.sum(1) == 1) & (Sm_b.sum(1) == 1)
                if unique.sum() < 5:
                    continue
                prop_pt = float(per_study[proposed][unique].mean())
                base_pt = float(per_study[baseline][unique].mean())
                prop_reps = []
                base_reps = []
                for idx in idx_paired:
                    u = unique[idx]
                    if u.sum() == 0:
                        prop_reps.append(np.nan)
                        base_reps.append(np.nan)
                    else:
                        prop_reps.append(per_study[proposed][idx][u].mean())
                        base_reps.append(per_study[baseline][idx][u].mean())
                prop_reps = np.array(prop_reps)
                base_reps = np.array(base_reps)
                ok = np.isfinite(prop_reps) & np.isfinite(base_reps)
                if ok.sum() < 100:
                    continue
                observed = prop_pt - base_pt
                boot_diff = prop_reps[ok] - base_reps[ok]
                lo, hi = percentile_ci(boot_diff)
                p = null_centered_pvalue(observed, boot_diff)
                joint_rows.append(
                    {
                        "Dataset": "ReXVal:original_jointly_eligible",
                        "Proposed metric": proposed,
                        "Baseline": baseline,
                        "n_eligible_studies": int(unique.sum()),
                        "Proposed alignment": prop_pt,
                        "Baseline alignment": base_pt,
                        "Difference (pp)": observed,
                        "CI low": lo,
                        "CI high": hi,
                        "p-value": p,
                    }
                )
        joint_df = pd.DataFrame(joint_rows)
        if len(joint_df):
            adj, reject = holm_adjust(joint_df["p-value"].to_numpy())
            joint_df["Holm-adjusted p (within dataset)"] = adj
            joint_df["Significant (within dataset)"] = reject
        joint_path = out_dir / "rexval_original_paired_comparisons_jointly_eligible.csv"
        joint_df.to_csv(joint_path, index=False)
    else:
        joint_path = None

    meta = {
        "dataset": "ReXVal",
        "mode": args.mode,
        "base_commit": "b0b38a0ae1506aea56ca58522e50bb1aca8c57ff",
        "n_studies": S,
        "n_raters": len(raters),
        "n_boot_ci": args.n_boot_ci,
        "n_boot_paired": args.n_boot_paired,
        "seed": args.seed,
        "point_estimates": points,
        "inter_rater_point": ir_point,
        "result_class": result_class,
        "inputs": {
            "rexval_csv": str(REXVAL_CSV),
            "labels": str(LABELS_CSV),
            "icare": str(EVAL_DIR),
            "predefined": str(PRED_DIR),
            "baselines": str(BASELINES),
        },
        "outputs": {
            "ci": str(ci_path),
            "paired": str(paired_path),
            "replicates": str(reps_path),
            "interrater": str(ir_path),
            "study_export": str(export_path),
            "jointly_eligible": str(joint_path) if joint_path else None,
        },
        "note": "Does not overwrite plot_rexval_correlation.py outputs.",
    }
    meta_path = out_dir / f"rexval_{args.mode}_analysis_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    print(f"ReXVal mode={args.mode} complete.")
    print(f"  inter-rater: {ir_point:.2f}% [{ir_lo:.2f}, {ir_hi:.2f}]")
    for m in ALL_METRICS:
        print(f"  {m:16s} {points[m]:6.2f}")
    print(f"  wrote {out_dir}")


def main() -> None:
    args = parse_args()
    run_mode(args)


if __name__ == "__main__":
    main()
