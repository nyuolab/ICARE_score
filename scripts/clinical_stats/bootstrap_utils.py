"""Reusable percentile / paired bootstrap helpers for alignment analyses.

Designed for RadPref / ReXVal clinical-alignment statistics. Keeps the same
hypothesis-testing convention as the RRG-Pref Colab analysis:
  - shared resampling indices across metrics
  - percentile CIs on paired differences
  - approximate two-sided null-centered bootstrap p-values
  - Holm adjustment within a comparison family
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def make_indices(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """Return (n_boot, n_units) indices into units [0, n_units)."""
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    return np.random.default_rng(seed).integers(0, n_units, size=(n_boot, n_units))


def percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def null_centered_pvalue(observed_diff: float, boot_diff: np.ndarray) -> float:
    """Approximate two-sided null-centered bootstrap p-value (RRG-Pref convention)."""
    boot_diff = np.asarray(boot_diff, dtype=float)
    if boot_diff.ndim != 1:
        raise ValueError("boot_diff must be 1-D")
    null = boot_diff - observed_diff
    b = boot_diff.size
    return float((1 + np.sum(np.abs(null) >= abs(observed_diff))) / (b + 1))


def holm_adjust(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjustment.

    Returns (adjusted_p, reject) with adjusted_p in the original order.
    Equivalent to statsmodels multipletests(..., method='holm').
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p.copy(), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, pv in enumerate(ranked):
        running = max(running, (m - i) * pv)
        adj[i] = min(1.0, running)
    # Enforce monotonicity from largest to smallest rank
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(m, dtype=float)
    out[order] = adj
    reject = out <= alpha
    return out, reject


def mean_per_rater_from_matrix(matrix: np.ndarray) -> float:
    """Mean of per-column alignment percentages; NaNs mark excluded observations.

    matrix shape: (n_units, n_raters) with values in {0, 1, nan}.
    """
    matrix = np.asarray(matrix, dtype=float)
    counts = np.sum(~np.isnan(matrix), axis=0)
    matches = np.nansum(matrix, axis=0)
    pct = np.divide(
        100.0 * matches,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    if np.all(np.isnan(pct)):
        return float("nan")
    return float(np.nanmean(pct))


def bootstrap_metric_replicates(
    matrices: Mapping[str, np.ndarray],
    indices: np.ndarray,
    agg_fn: Callable[[np.ndarray], float],
) -> Dict[str, np.ndarray]:
    """Apply agg_fn to each metric matrix on each bootstrap index row."""
    out: Dict[str, np.ndarray] = {}
    n_boot = indices.shape[0]
    for name, mat in matrices.items():
        reps = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            reps[b] = agg_fn(mat[indices[b]])
        out[name] = reps
    return out


def paired_comparison_table(
    point_estimates: Mapping[str, float],
    boot_replicates: Mapping[str, np.ndarray],
    proposed_metrics: Sequence[str],
    baseline_metrics: Sequence[str],
    dataset: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build ICARE-vs-baseline paired comparison table with Holm adjustment."""
    rows: List[dict] = []
    for proposed in proposed_metrics:
        for baseline in baseline_metrics:
            prop_mean = float(point_estimates[proposed])
            base_mean = float(point_estimates[baseline])
            observed = prop_mean - base_mean
            boot_diff = boot_replicates[proposed] - boot_replicates[baseline]
            lo, hi = percentile_ci(boot_diff)
            p = null_centered_pvalue(observed, boot_diff)
            rows.append(
                {
                    "Dataset": dataset,
                    "Proposed metric": proposed,
                    "Baseline": baseline,
                    "Proposed alignment": prop_mean,
                    "Baseline alignment": base_mean,
                    "Difference (pp)": observed,
                    "CI low": lo,
                    "CI high": hi,
                    "p-value": p,
                }
            )
    df = pd.DataFrame(rows)
    adj, reject = holm_adjust(df["p-value"].to_numpy(), alpha=alpha)
    df["Holm-adjusted p (within dataset)"] = adj
    df["Significant (within dataset)"] = reject
    return df.sort_values(
        ["Proposed metric", "Difference (pp)"], ascending=[True, False]
    ).reset_index(drop=True)


def assert_finite_replicates(boot: Mapping[str, np.ndarray], context: str) -> None:
    for name, reps in boot.items():
        if not np.isfinite(reps).all():
            bad = int(np.size(reps) - np.isfinite(reps).sum())
            raise AssertionError(f"{context}: {bad} non-finite bootstrap replicates for {name}")
