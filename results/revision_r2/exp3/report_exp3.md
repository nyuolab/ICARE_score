# Exp 3 report — Fig. 2a triptych on RadPref and ReXVal

## Provenance

- Script: `scripts/clinical_stats/run_exp3_score_gap.py`
- Loaders (import only): `scripts/radpref_data/run_radpref_paired_stats.py`, `scripts/rexval_data/run_rexval_alignment_stats.py`
- Metric scores for Exp 3 plots: RadPref / ReXVal `eval_seed_123` ICARE shuffled + predefined + baselines (same paths as Exp 2 paired stats)
- RRG-Pref pair sampling used ICARE-AVG gaps aggregated over eval seeds **123, 456, 789, 202, 101** (mean of `gt_reports_as_ref` and `gen_reports_as_ref` Agreement_Percentage); see `captions.md`

## Rules

- Per-(case/pair, rater) rows
- Metric tie + decisive rater → **misalignment** (stated in captions)
- Quintiles: `pd.qcut(q=5, duplicates="drop")`; two-bin metrics plotted at Q1 and Q5
- No hypothesis tests — descriptive figures only

## Tie fractions

```json
{
  "radpref": {
    "n_rows": 3000,
    "doctor_tie_fraction": 0.23,
    "metric_tie_fraction": 0.031,
    "n_doctor_ties": 69,
    "n_metric_ties": 93,
    "n_doctor_rows": 300
  },
  "rexval": {
    "n_rows": 18000,
    "doctor_tie_fraction": 0.544444,
    "metric_tie_fraction": 0.214333,
    "n_doctor_ties": 980,
    "n_metric_ties": 3858,
    "n_doctor_rows": 1800
  }
}
```

## ICARE trends (descriptive)

- **RadPref:** Indecision [Q1=0.433, Q2=0.283, Q3=0.167, Q4=0.150, Q5=0.117]; Alignment [Q1=0.383, Q2=0.467, Q3=0.767, Q4=0.783, Q5=0.883].
- **ReXVal:** Indecision [Q1=0.867, Q2=0.669, Q3=0.497, Q4=0.433, Q5=0.244]; Alignment [Q1=0.086, Q2=0.169, Q3=0.289, Q4=0.342, Q5=0.628].

On both independent datasets, ICARE's indecision rate fell and alignment rose with score gap.
