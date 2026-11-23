#!/usr/bin/env python3
"""RadPref paired preference-alignment statistics (additional analysis).

Reuses the same inputs and preference-alignment estimand as
``plot_radpref_correlation.py`` on submission commit b0b38a0, but:

  * aligns all scores explicitly by case_id before resampling
  * uses shared bootstrap indices across metrics
  * reports paired ICARE-vs-baseline comparisons with Holm correction
  * writes only under a dedicated output directory (does not touch plot outputs)

Original plotting script is left unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Allow ``python scripts/radpref_data/run_radpref_paired_stats.py`` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_stats.bootstrap_utils import (  # noqa: E402
    assert_finite_replicates,
    bootstrap_metric_replicates,
    holm_adjust,
    make_indices,
    mean_per_rater_from_matrix,
    paired_comparison_table,
    percentile_ci,
)

# Paths mirror plot_radpref_correlation.py (submission main).
BASE = Path("/gpfs/data/oermannlab/users/rd3571")
ICARE_SHUFFLED_OUT = BASE / "ICARE_score/outputs/radpref/eval_seed_123/shuffled_ans_choices_data"
ICARE_PREDEFINED_OUT = BASE / "ICARE_score/outputs/radpref/predefined/eval_seed_123/mcqa_eval"
BASELINES_DIR = BASE / "ICARE_score/outputs/radpref/baselines"
RADPREF_CSV = BASE / "cxr_report_datasets/radpref/radpref_icare.csv"
ANN_DIR = BASE / "CRIMSON/RadPref/radiologist_annotations"

# Published point estimates (manuscript Fig. 2d / prior audit), 1 d.p. check.
PUBLISHED_POINTS = {
    "ICARE_AVG": 85.3,
    "ICARE_PREDEFINED": 88.5,
    "GREEN": 85.0,
    "CRIMSON": 83.4,
    "BLEU": 79.8,
}

PROPOSED = ["ICARE_AVG", "ICARE_PREDEFINED"]
BASELINES = [
    "CRIMSON",
    "GREEN",
    "AlignScore",
    "BERTScore",
    "SembScore",
    "RadGraph",
    "BLEU",
    "RadCliQ-v1",
]
ALL_METRICS = PROPOSED + BASELINES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/radpref/clinical_alignment_stats",
        help="Directory for new outputs only (never the original plot OUT_DIR).",
    )
    p.add_argument("--n-boot-ci", type=int, default=2000)
    p.add_argument("--n-boot-paired", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol-published", type=float, default=0.05)
    return p.parse_args()


def load_case_table() -> pd.DataFrame:
    """One row per case_id with C1/C2 report indices 0..99 / 100..199."""
    icare_csv = pd.read_csv(RADPREF_CSV)
    c1 = icare_csv[icare_csv["candidate"] == "C1"].reset_index(drop=True)
    c2 = icare_csv[icare_csv["candidate"] == "C2"].reset_index(drop=True)
    if len(c1) != 100 or len(c2) != 100:
        raise AssertionError(f"Expected 100 C1 and 100 C2 rows, got {len(c1)}, {len(c2)}")
    if not np.array_equal(c1["case_id"].values, c2["case_id"].values):
        raise AssertionError("C1/C2 case_id order mismatch in radpref_icare.csv")
    # Explicit positional indices used by ICARE report-level stats (submission layout).
    out = pd.DataFrame(
        {
            "case_id": c1["case_id"].astype(str).values,
            "idx_c1": np.arange(100, dtype=int),
            "idx_c2": np.arange(100, 200, dtype=int),
        }
    )
    return out


def load_rater_prefs(case_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Return sign(C1-C2) preference per rater, keyed rater_1..3; 0 = tie."""
    prefs = {}
    for uid in [1, 2, 3]:
        ann = json.load(open(ANN_DIR / f"user_{uid}.json"))["annotations"]
        diffs = np.array([ann[cid]["C1"]["rating"] - ann[cid]["C2"]["rating"] for cid in case_ids])
        prefs[f"rater_{uid}"] = np.sign(diffs).astype(float)
    return prefs


def score_diff_from_array(scores: np.ndarray, idx_c1: np.ndarray, idx_c2: np.ndarray) -> np.ndarray:
    return scores[idx_c1] - scores[idx_c2]


def load_metric_diffs(cases: pd.DataFrame) -> Dict[str, np.ndarray]:
    idx1 = cases["idx_c1"].to_numpy()
    idx2 = cases["idx_c2"].to_numpy()

    gt = pd.read_csv(ICARE_SHUFFLED_OUT / "gt_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    gen = pd.read_csv(ICARE_SHUFFLED_OUT / "gen_reports_as_ref/mcqa_eval/mcq_eval_report_level_stats.csv").set_index("Report_ID")
    combined = gt.join(gen, how="left", lsuffix="_gt", rsuffix="_gen")
    icare = combined[["Agreement_Percentage_gt", "Agreement_Percentage_gen"]].mean(axis=1).to_numpy()
    pred = pd.read_csv(ICARE_PREDEFINED_OUT / "mcq_eval_report_level_stats.csv").set_index("Report_ID")
    pred_sc = pred.loc[range(200), "Agreement_Percentage"].to_numpy()

    with open(BASELINES_DIR / "crimson_results.json") as f:
        crimson = np.array([r["crimson_score"] for r in json.load(f)["results"]])
    green = pd.read_csv(BASELINES_DIR / "green/radpref_prepared_green_results.csv")["green_score"].to_numpy()
    align = pd.read_csv(BASELINES_DIR / "alignscore/radpref_prepared_alignscore_results.csv")["alignscore"].to_numpy()
    rrg = pd.read_csv(BASELINES_DIR / "radpref/radpref_prepared_results.csv")

    diffs = {
        "ICARE_AVG": score_diff_from_array(icare, idx1, idx2),
        "ICARE_PREDEFINED": score_diff_from_array(pred_sc, idx1, idx2),
        "CRIMSON": score_diff_from_array(crimson, idx1, idx2),
        "GREEN": score_diff_from_array(green, idx1, idx2),
        "AlignScore": score_diff_from_array(align, idx1, idx2),
        "BERTScore": score_diff_from_array(rrg["bertscore"].to_numpy(), idx1, idx2),
        "SembScore": score_diff_from_array(rrg["semb_score"].to_numpy(), idx1, idx2),
        "RadGraph": score_diff_from_array(rrg["radgraph_combined"].to_numpy(), idx1, idx2),
        "BLEU": score_diff_from_array(rrg["bleu_score"].to_numpy(), idx1, idx2),
        # RadCliQ is an error metric; negate so higher = better (plot script L393).
        "RadCliQ-v1": score_diff_from_array(-rrg["RadCliQ-v1"].to_numpy(), idx1, idx2),
    }
    for k, v in diffs.items():
        if v.shape != (100,):
            raise AssertionError(f"{k} diff shape {v.shape}, expected (100,)")
    return diffs


def build_alignment_matrix(metric_diff: np.ndarray, rater_prefs: Dict[str, np.ndarray]) -> np.ndarray:
    """(n_cases, n_raters) with 1/0/nan. Excludes clinician ties; metric ties count as mismatch."""
    m_pref = np.sign(metric_diff)
    cols = []
    for key in ["rater_1", "rater_2", "rater_3"]:
        rp = rater_prefs[key]
        col = np.full(len(rp), np.nan, dtype=float)
        mask = rp != 0
        col[mask] = (m_pref[mask] == rp[mask]).astype(float)
        cols.append(col)
    return np.column_stack(cols)


def build_joint_matrix(
    prop_diff: np.ndarray,
    base_diff: np.ndarray,
    rater_prefs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Jointly eligible: rater decisive and both metrics non-tied."""
    mp = np.sign(prop_diff)
    mb = np.sign(base_diff)
    prop_cols, base_cols = [], []
    for key in ["rater_1", "rater_2", "rater_3"]:
        rp = rater_prefs[key]
        mask = (rp != 0) & (mp != 0) & (mb != 0)
        pc = np.full(len(rp), np.nan, dtype=float)
        bc = np.full(len(rp), np.nan, dtype=float)
        pc[mask] = (mp[mask] == rp[mask]).astype(float)
        bc[mask] = (mb[mask] == rp[mask]).astype(float)
        prop_cols.append(pc)
        base_cols.append(bc)
    return np.column_stack(prop_cols), np.column_stack(base_cols)


def interrater_pct_on_cases(rater_prefs: Dict[str, np.ndarray], case_idx: np.ndarray) -> float:
    """Pairwise top-1 agreement among decisive rater pairs (plot script L421-430)."""
    ids = ["rater_1", "rater_2", "rater_3"]
    agrees = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            p1 = rater_prefs[a][case_idx]
            p2 = rater_prefs[b][case_idx]
            mask = (p1 != 0) & (p2 != 0)
            if mask.sum() == 0:
                continue
            agrees.append((p1[mask] == p2[mask]).mean() * 100.0)
    return float(np.mean(agrees)) if agrees else float("nan")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_case_table()
    case_ids = cases["case_id"].to_numpy()
    rater_prefs = load_rater_prefs(case_ids)
    metric_diffs = load_metric_diffs(cases)

    matrices = {m: build_alignment_matrix(metric_diffs[m], rater_prefs) for m in ALL_METRICS}
    points = {m: mean_per_rater_from_matrix(matrices[m]) for m in ALL_METRICS}

    # Provenance / reproduction checks.
    for m, pub in PUBLISHED_POINTS.items():
        if abs(points[m] - pub) > args.tol_published:
            raise AssertionError(f"Point estimate mismatch for {m}: got {points[m]}, published {pub}")

    n_cases = len(cases)
    assert n_cases == 100

    # Shared indices for CI and paired analyses.
    idx_ci = make_indices(n_cases, args.n_boot_ci, args.seed)
    idx_paired = make_indices(n_cases, args.n_boot_paired, args.seed)

    boot_ci = bootstrap_metric_replicates(matrices, idx_ci, mean_per_rater_from_matrix)
    assert_finite_replicates(boot_ci, "RadPref CI")
    boot_paired = bootstrap_metric_replicates(matrices, idx_paired, mean_per_rater_from_matrix)
    assert_finite_replicates(boot_paired, "RadPref paired")

    # Shared-index check across metrics.
    for m in ALL_METRICS[1:]:
        assert idx_paired.shape == (args.n_boot_paired, n_cases)

    rows_ci = []
    for m in ALL_METRICS:
        lo, hi = percentile_ci(boot_ci[m])
        rows_ci.append(
            {
                "dataset": "RadPref",
                "protocol": "original_per_rater_alignment",
                "metric": m,
                "n_cases": n_cases,
                "point": points[m],
                "ci_lo": lo,
                "ci_hi": hi,
                "n_boot": args.n_boot_ci,
                "seed": args.seed,
                "result_class": "additional_shared_index_ci",
            }
        )
    ci_df = pd.DataFrame(rows_ci)
    ci_path = out_dir / "radpref_alignment_ci.csv"
    ci_df.to_csv(ci_path, index=False)

    paired_df = paired_comparison_table(
        points, boot_paired, PROPOSED, BASELINES, dataset="RadPref"
    )
    paired_path = out_dir / "radpref_paired_comparisons.csv"
    paired_df.to_csv(paired_path, index=False)

    reps_df = pd.DataFrame({m: boot_paired[m] for m in ALL_METRICS})
    reps_path = out_dir / "radpref_paired_bootstrap_replicates.csv"
    reps_df.to_csv(reps_path, index=False)

    # Corrected inter-rater CI: resample cases, recompute pairwise agreement each time.
    ir_point = interrater_pct_on_cases(rater_prefs, np.arange(n_cases))
    ir_boot = np.array([interrater_pct_on_cases(rater_prefs, idx) for idx in idx_ci])
    ir_lo, ir_hi = percentile_ci(ir_boot)
    ir_df = pd.DataFrame(
        [
            {
                "dataset": "RadPref",
                "statistic": "inter_rater_pairwise_top1_agreement",
                "point": ir_point,
                "ci_lo": ir_lo,
                "ci_hi": ir_hi,
                "n_boot": args.n_boot_ci,
                "seed": args.seed,
                "note": "CI recomputed inside each case-level resample (not mean of endpoint CIs)",
                "result_class": "additional_corrected_interrater_ci",
            }
        ]
    )
    ir_path = out_dir / "radpref_interrater_ci.csv"
    ir_df.to_csv(ir_path, index=False)

    # Secondary: jointly eligible paired comparisons.
    joint_rows = []
    joint_boot = {m: np.empty(args.n_boot_paired) for m in ALL_METRICS}
    # Precompute joint matrices per proposed/baseline pair for point estimates;
    # for bootstrap, rebuild masks inside loop would be expensive — instead store
    # per-metric unrestricted matrices and apply joint mask via paired rebuild.
    for proposed in PROPOSED:
        for baseline in BASELINES:
            jp, jb = build_joint_matrix(metric_diffs[proposed], metric_diffs[baseline], rater_prefs)
            prop_pt = mean_per_rater_from_matrix(jp)
            base_pt = mean_per_rater_from_matrix(jb)
            prop_reps = np.array([mean_per_rater_from_matrix(jp[idx]) for idx in idx_paired])
            base_reps = np.array([mean_per_rater_from_matrix(jb[idx]) for idx in idx_paired])
            # Skip pairs with undefined point (should not happen often).
            if not np.isfinite(prop_pt) or not np.isfinite(base_pt):
                continue
            if not (np.isfinite(prop_reps).all() and np.isfinite(base_reps).all()):
                continue
            observed = prop_pt - base_pt
            boot_diff = prop_reps - base_reps
            lo, hi = percentile_ci(boot_diff)
            from clinical_stats.bootstrap_utils import null_centered_pvalue

            p = null_centered_pvalue(observed, boot_diff)
            joint_rows.append(
                {
                    "Dataset": "RadPref",
                    "Protocol": "jointly_eligible_sensitivity",
                    "Proposed metric": proposed,
                    "Baseline": baseline,
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
        joint_df = joint_df.sort_values(
            ["Proposed metric", "Difference (pp)"], ascending=[True, False]
        ).reset_index(drop=True)
    joint_path = out_dir / "radpref_paired_comparisons_jointly_eligible.csv"
    joint_df.to_csv(joint_path, index=False)

    # Case-level alignment export for provenance.
    case_export = cases.copy()
    for m in ALL_METRICS:
        case_export[f"{m}_diff"] = metric_diffs[m]
        case_export[f"{m}_pref"] = np.sign(metric_diffs[m])
    for k, v in rater_prefs.items():
        case_export[k] = v
    case_path = out_dir / "radpref_case_aligned_inputs.csv"
    case_export.to_csv(case_path, index=False)

    meta = {
        "dataset": "RadPref",
        "base_commit": "b0b38a0ae1506aea56ca58522e50bb1aca8c57ff",
        "n_cases": n_cases,
        "n_boot_ci": args.n_boot_ci,
        "n_boot_paired": args.n_boot_paired,
        "seed": args.seed,
        "inputs": {
            "radpref_csv": str(RADPREF_CSV),
            "annotations": str(ANN_DIR),
            "icare_shuffled": str(ICARE_SHUFFLED_OUT),
            "icare_predefined": str(ICARE_PREDEFINED_OUT),
            "baselines": str(BASELINES_DIR),
        },
        "point_estimates": points,
        "inter_rater_point": ir_point,
        "result_classes": {
            "reproduced_points": "point column in radpref_alignment_ci.csv",
            "additional_shared_index_ci": str(ci_path),
            "additional_paired_comparisons": str(paired_path),
            "additional_corrected_interrater_ci": str(ir_path),
            "sensitivity_jointly_eligible": str(joint_path),
        },
        "note": "Does not modify or overwrite plot_radpref_correlation.py outputs.",
    }
    meta_path = out_dir / "radpref_analysis_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=float))

    print("RadPref clinical-alignment stats complete.")
    print(f"  cases: {n_cases}")
    print(f"  inter-rater: {ir_point:.2f}% [{ir_lo:.2f}, {ir_hi:.2f}]")
    print("  point estimates:")
    for m in ALL_METRICS:
        print(f"    {m:16s} {points[m]:6.2f}")
    print(f"  wrote: {ci_path}")
    print(f"  wrote: {paired_path}")
    print(f"  wrote: {reps_path}")
    print(f"  wrote: {ir_path}")
    print(f"  wrote: {joint_path}")
    print(f"  wrote: {meta_path}")


if __name__ == "__main__":
    main()
