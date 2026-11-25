# Exp 3 captions (R1-C4)

## RRG-Pref sampling provenance (Fig. 2a source cohort)

The 153 RRG-Pref preference pairs were selected from IU-Xray model-pair ICARE summaries as follows:

1. **Pool construction** (`mcq_best_worst_analysis.py`, `--selection_mode all_combinations`): all model-seed report pairs with ICARE `Agreement_Percentage` averaged across **evaluation seeds 123, 456, 789, 202, 101** and across both `gt_reports_as_ref` and `gen_reports_as_ref` (ICARE-AVG). Inputs live under `MCQ_gen_data_our_eval_seed{seed}/IU_xray/.../shuffled_ans_choices_data/`. Shell defaults: model seeds 1–5 (`mcq_best_worst_analysis.sh`).
2. **Round 1 (n = 100):** drop identical high/low generated reports and duplicate `(Report_ID, Generated_Report_High, Generated_Report_Low)` pairs; importance-sample by ICARE score gap with **beta = 0.9955**, **sample_size = 100**, **seed = 12345** (`sampling_data_and_analysis.py` / `.sh`).
3. **Round 2 (n = +53 → 153):** targeted fill of ICARE |gap| bins **5–10 / 10–15 / 15–20 / >20** to **25** each (`--target_per_bin 25`, same seed). Combined file: `combined_targeted_samples.csv`.

Manuscript Fig. 2a then keeps the **119** samples with ≥2-of-3 clinician consensus (34 one-each splits dropped). Metric-tie with a decisive clinician vote is counted as **misalignment** (same rule used here).

Alignment-panel y-axis is 0–1.05 (Fig. 2a used 0–0.82); RadPref alignment exceeds 0.82 at large gaps.

## RadPref figure

**Header:** RadPref: score gap vs clinician certainty

**Unit of analysis:** (case, rater) rows (100 cases × 3 raters). Tied 1–5 ratings → “Can't say” / indecision; otherwise the rater prefers the higher-rated report. Score gap = |metric(C1) − metric(C2)|; quintiles per metric via `pd.qcut(..., duplicates="drop")`. Two-bin metrics (e.g. CRIMSON) are plotted at Q1 and Q5. Metric tie with a decisive rater counts as misalignment.

**Trend (descriptive):** Indecision [Q1=0.433, Q2=0.283, Q3=0.167, Q4=0.150, Q5=0.117]; Alignment [Q1=0.383, Q2=0.467, Q3=0.767, Q4=0.783, Q5=0.883].

## ReXVal figure

**Header:** ReXVal: score gap vs clinician certainty

**Unit of analysis:** all 6 candidate pairs per study × 6 raters (50 × 6 × 6). Equal clinically significant error counts → “Can't say”; otherwise the rater prefers fewer errors. Same quintile / two-bin / metric-tie rules as RadPref and Fig. 2a.

**Trend (descriptive):** Indecision [Q1=0.867, Q2=0.669, Q3=0.497, Q4=0.433, Q5=0.244]; Alignment [Q1=0.086, Q2=0.169, Q3=0.289, Q4=0.342, Q5=0.628].

## Tie fractions (Exp 3 rows)

- RadPref: doctor-tie 0.23, metric-tie 0.031 (n=3000 rows × 10 metrics counted separately in metric-tie aggregate).
- ReXVal: doctor-tie 0.544444, metric-tie 0.214333 (n=18000).
