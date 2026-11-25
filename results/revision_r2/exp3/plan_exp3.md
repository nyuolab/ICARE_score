# Exp 3 Step 0 — Plan (R1-C4 Fig. 2a triptych on RadPref & ReXVal)

**Status:** STOP after Step 0. Do **not** implement Steps 1–3 until approved.  
**Branch / worktree:** `revision/clinical-alignment-statistics` @ `/gpfs/data/oermannlab/users/rd3571/ICARE_score_rev_stats`  
**Protect:** hash `results/revision_r2/**` before/after implementation; touch only new `scripts/clinical_stats/run_exp3_score_gap.py`, one README line, and `results/revision_r2/exp3/`.  
**Compute:** srun on `cpu_short` (never login node or sp-0010).

---

## 1. How the 153 RRG-Pref pairs were selected

**Provenance (code, not Colab):**
- Sampler: `RRG_evaluation/MCQ_generation/RRGEval/src/human_evaluation/sampling_data_and_analysis.py`
- Launcher: `.../RRGEval/scripts/iuxray_data/human_evaluation/sampling_data_and_analysis.sh`
- Pool: `.../doctors_evaluation/mcq_eval_all_combinations_pairs_summary.csv` (IU-Xray model-pair MCQ summary)

**Round 1 — ICARE-gap weighted sample → 100 pairs**
- Drop pairs with identical high/low generated reports; drop duplicates on `(Report_ID, Generated_Report_High, Generated_Report_Low)`.
- Importance weights from ICARE `Agreement_Percentage_Difference` with `--beta 0.9955`, `--sample_size 100`, `--seed 12345`.
- Output: `mcq_eval_all_combinations_weighted_unique_no_identical_sample.csv` (**n = 100**).
- Observed |ICARE gap| bins after Round 1: `0–5: 53`, `5–10: 18`, `10–15: 8`, `15–20: 11`, `>20: 10`.

**Round 2 — ICARE-gap bin fill → +53 pairs (total 153)**
- `targeted_bin_sampling(..., target_per_bin=25)` on bins `5–10`, `10–15`, `15–20`, `>20` (shell uses `--target_per_bin 25`).
- Additional: `additional_targeted_samples.csv` (**n = 53**).
- Combined: `combined_targeted_samples.csv` (**n = 153**).
- Final |ICARE gap| bins: `0–5: 53`, `5–10: 25`, `10–15: 25`, `15–20: 25`, `>20: 25`.

**By criterion:** all **153** pairs were selected using **ICARE score-gap** criteria (100 weighted + 53 targeted bin fill). No other-metric selection criterion.

**Forms / analysis counts:**
- Google-form questions: Round 1 forms 50+50 = 100; Round 2 forms 27+26 = 53 → **153** unique samples.
- Metric+response CSVs under `.../google_form_data_*_crimson_alignscore/` and Round 2 twin dirs.
- Colab analysis (`icare_human_eval_analysis_section4_*_with_alignscore_and_crimson*.ipynb`): **153** unique samples → consensus ≥2-of-3 keeps **119** (42 decisive + 77 indecision); drops **34** 1-1-1 splits. Manuscript Fig. 2a uses these **119** consensus samples.

---

## 2. RRG-Pref Fig. 2a triptych: code + definitions

**Not on BigPurple repo.** Table + rates live in Colab; Nature-Comm styling is a separate Colab plotter.

| Piece | Location |
|-------|----------|
| Rates + `panel_triptych.csv` export | Local Colab: `~/Downloads/icare_human_eval_analysis_section4_(1)_with_alignscore_and_crimson (2).ipynb` — Section 2.1–2.2 + EXPORT cell |
| Fig. 2a style (colors, lw, header band) | Local Colab: `~/Downloads/colab_make_fig2_latest_restructured (3).ipynb` — reads Drive `figures_v2_rebuttal/final_aggregate_results_csv/panel_triptych.csv` |

**Score gap:** `|score_Text1 − score_Text2|` per metric (`load_metric_picks`). RadCliQ uses lower-is-better for pick direction; gap is still absolute difference.

**Quintiles (`robust_qcut`):** `pd.qcut(series, q=5, duplicates="drop")`, then relabel surviving categories `Q1…Qn`. Metrics with many tied gaps keep fewer bins (CRIMSON → 2 bins). Plotter maps CRIMSON’s second surviving bin from `Q2` x-position to **Q5** (same as submitted Fig. 2a).

**Doctor consensus (sample-level; Fig. 2a):** per Question ID, ≥2 of 3 votes → `R1` / `R2` / `Indecision` (`Can't say`); else drop.  
**Labels vs metric:** Indecision if consensus Indecision; Alignment if consensus equals metric pick (`Text1`→R1, `Text2`→R2); else Misalignment (includes metric **Tie** against a decisive consensus).

**Per-bin rates (panel_triptych.csv schema):**
`metric, quintile, indecision, alignment, misalignment, n_samples`  
with each rate = count(label) / n in that (metric, quintile) bin.

**Fig. 2a plot style to mirror (top-row triptych only for Exp 3):**
- rcParams: sans-serif Arial/Helvetica/DejaVu Sans; `axes.linewidth=1.2`; `pdf.fonttype=42`
- TITLE 24, LABEL 22, TICK 20, LEGEND 17; width 28 in; `wspace=0.34`
- Panels: Indecision (`ylim` 0–1.05), Alignment & Misalignment (`ylim` 0–0.82)
- Colors: ICARE `#1A6FAF`, ICARE-Predefined `#0B3C6F`, BLEU-2 `#FF7F0E`, SembScore `#2CA02C`, BERTScore `#9467BD`, GREEN `#D62728`, 1/RadCliQ-v1 `#8C564B`, RadGraph `#E377C2`, AlignScore `#7F7F7F`, CRIMSON `#BCBD22`
- Lines: ICARE lw 4.8 / ms 12.5; Predefined lw 3.7 / ms 10.8; others lw 1.7 / ms 6.2 α 0.6; marker `o`; ICARE on top; 2-bin metrics at Q1 & Q5
- Legend under middle panel (ncol 5, no frame, ICARE first); header band `#1A4A7A`

---

## 3. RadPref & ReXVal — paths, columns, counts, loaders

### RadPref
| Item | Path / fact |
|------|-------------|
| Cases | **100** cases × **2** candidates (C1, C2) |
| ICARE-ready CSV | `/gpfs/data/oermannlab/users/rd3571/cxr_report_datasets/radpref/radpref_icare.csv` (**200** rows: C1 then C2) |
| Ratings (1–5) | `/gpfs/data/oermannlab/users/rd3571/CRIMSON/RadPref/radiologist_annotations/user_{1,2,3}.json` — per case `annotations[case_id][C1|C2].rating` |
| Metric scores | `ICARE_score/outputs/radpref/` (eval_seed_123 shuffled + predefined + baselines) — same paths as `scripts/radpref_data/run_radpref_paired_stats.py` |
| **Reuse by import** | `load_case_table`, `load_rater_prefs`, `load_metric_diffs` from `scripts/radpref_data/run_radpref_paired_stats.py` (do not edit that file) |

**Exp 3 unit:** row = `(case, rater)`. Tied ratings (`rating_C1 == rating_C2`) → `"Can't say"`; else rater prefers higher rating. Metric prefers higher score (RadCliQ already negated higher-better in loader).

### ReXVal
| Item | Path / fact |
|------|-------------|
| Studies / candidates / raters | **50** studies × **4** candidates (`radgraph, bertscore, s_emb, bleu`) × **6** raters |
| Candidate texts | `cxr_report_datasets/rexval/RexVal_test_icare_200.csv` (**200** rows) |
| Error labels | `cxr_report_datasets/rexval_physionet_labels/physionet.org/files/rexval-dataset/1.0.0/6_valid_raters_per_rater_error_categories.csv` (clinically_significant errors summed → per study/candidate/rater count) |
| Metric scores | `ICARE_score/outputs/rexval/...` — same as `scripts/rexval_data/run_rexval_alignment_stats.py` |
| **Reuse by import** | `load_study_tables` from `run_rexval_alignment_stats.py` (do not edit) |

**Exp 3 unit:** all **C(4,2)=6** candidate pairs per study; row = `(pair, rater)`. Equal error counts → `"Can't say"`; else prefer fewer errors. Metric prefers higher score on the pair.

**Expected row counts (before binning):** RadPref 100×3 = **300**; ReXVal 50×6×6 = **1800**.

---

## 4. Proposed Step 1–2 implementation (awaiting approval)

**New script only:** `scripts/clinical_stats/run_exp3_score_gap.py`  
**Outputs only under:** `results/revision_r2/exp3/`

1. Import existing loaders; compute per-metric `|Δscore|`; `robust_qcut` as above (keep surviving bins; 2-bin metrics plotted at Q1/Q5).
2. Build doctor labels per Exp 3 rules (RadPref ratings / ReXVal errors); classify Indecision / Alignment / Misalignment vs metric pick (metric tie → Misalignment when doctor is decisive, matching RRG-Pref sample-level).
3. Write `radpref_panel_triptych.csv` and `rexval_panel_triptych.csv` (`metric, quintile, indecision, alignment, misalignment, n`).
4. Also write a small `tie_fractions.json` (fraction of rows with doctor tie / metric tie) for the Step 3 report — descriptive only, no hypothesis tests.
5. Plot 1×3 triptych per dataset with Fig. 2a style; headers:  
   `"RadPref: score gap vs clinician certainty"` / `"ReXVal: score gap vs clinician certainty"`.  
   Save PDF (fonttype 42) + PNG.
6. One README line under revision_r2 docs pointing to Exp 3.
7. Hash-check existing `results/revision_r2/**`; commit message:  
   `R2 revision: Exp 3 Fig. 2a triptych on RadPref and ReXVal`; push.

**Text policy (no tests):** describe trends only, e.g. “On both independent datasets, ICARE's indecision rate fell and alignment rose with score gap.” Do not claim ICARE beats other metrics.

---

## Open points for approval

1. Confirm Exp 3 uses **per-(case/pair, rater)** rows (as specified), not RRG-Pref’s ≥2-of-3 consensus aggregation.
2. Confirm metric-tie → Misalignment when doctor is decisive (RRG-Pref sample-level convention).
3. Confirm CRIMSON 2-bin → plot at Q1 and Q5.
4. Approve Steps 1–3 to run on `cpu_short`.
