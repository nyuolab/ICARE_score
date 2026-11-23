# Plan: Exp 4 (K sensitivity) and Exp 5 (seed consistency)

**Phase:** 1 (plan only) — no implementation until approval.  
**Worktree:** `/gpfs/data/oermannlab/users/rd3571/ICARE_score_rev_stats`  
**Branch:** `revision/clinical-alignment-statistics` @ `b0b38a0` (*clean code and update readme*)  
**Date:** 2026-11-23

---

## 1. Worktree status (confirmed)

```
On branch revision/clinical-alignment-statistics
Untracked files:
  scripts/clinical_stats/__init__.py
  scripts/clinical_stats/bootstrap_utils.py
  scripts/radpref_data/run_radpref_paired_stats.py
  scripts/rexval_data/run_rexval_alignment_stats.py
```

- No staged/modified tracked files.
- `__pycache__/` is **already** in `.gitignore` (line 5); Phase 2 will not need a new ignore rule unless we want a comment. `__pycache__` under `scripts/clinical_stats/` exists on disk and will stay untracked.
- `results/revision_r2/` did not exist in this worktree (discovery.md lives under `ICARE_score/`); this plan creates it here.

**Phase 2 first commit (as requested):** Exp 2 scripts as-is —  
`"R2 revision: paired bootstrap stats for RadPref and ReXVal (Exp 2)"`.

---

## 2. Inputs (read-only, 2025 Fig. 4 tree)

**Root:**  
`/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis/`

| File | Role | Notes |
|------|------|-------|
| `question_embeddings.pkl` | MedCPT embeddings | `(78708, 768)` float32; row *i* ↔ `combined_mcqa_data.csv["Question_Text"].unique()[i]` |
| `combined_mcqa_data.csv` | Pre-cluster instances | **993 403** rows |
| `clustered_questions_with_names.csv` | Fig. 4 labels + names | Same 993 403 rows + `cluster_id`, `cluster_name` |
| `cluster_names.json` | Name map | 20 names; **12 = Lung Lobe Abnormality**, **13 = Calcified Granuloma Location** |
| `cluster_representatives.txt` | 5 nearest-centroid Qs | Format to mirror for `cluster_representatives_K{K}.txt` |
| `analysis/model_cluster_agreement_{gt,gen}_ref.csv` | Stored Fig. 4 scores | 60 rows each (`Model`, `Cluster`, `Agreement`, `Count`, `Cluster_Name`) |

**Columns / dtypes (`combined` / `clustered_with_names`):**  
`Report_ID` int64, `Question_ID` int64, `Question_Text` object, `Correct_Answer` object, `Predicted_Answer_Using_{Gen,GT}` object, `dataset` object, `model_name` object, `model_seed` int64, `eval_seed` int64, `metric` object [, `cluster_id` int64, `cluster_name` object].

**Cardinality:**
- Unique `Question_Text`: **78 708** (matches embedding rows).
- Unique `Report_ID`: **590** (same for all 3 models).
- Models: `chexpert-mimic-cxr-findings-baseline`, `mimic-cxr-findings-baseline`, `maira-2`.
- `model_seed` ∈ {1,2,3,4,5}; `eval_seed` ∈ {123,456,789,202,101}; `metric` ∈ {`gt_reports_as_ref`, `gen_reports_as_ref`}.
- Anchor (K=20 stored): Lung Lobe Abnormality → cluster **12**, **3496** unique Q / **19603** rows; Calcified Granuloma Location → cluster **13**, **1758** unique Q / **12275** rows.
- Unique Q appearing in **all 5 eval seeds** (full table): **9 569 / 78 708** (matches expected).

**Env for runs:** `rrg-eval-clean` (sklearn **1.3.2**). Record `sklearn.__version__` in summaries.

**Agreement (Fig. 4):**  
`(Predicted_Answer_Using_Gen == Predicted_Answer_Using_GT)` mean over rows with `metric == gt_reports_as_ref` → ICARE-GT; same with `gen_reports_as_ref` → ICARE-GEN. **Pools all `model_seed` and `eval_seed`** (verified: chexpert cluster 0 GT Agreement = 0.820612… matches stored).

---

## 3. Functions to import (not copy / not modify)

### From `src/question_categorization_and_analysis/question_embedding_and_clustering.py`

```text
cluster_questions(embeddings, unique_questions, n_clusters=20)
  -> (cluster_labels, question_to_cluster, cluster_representatives, cluster_sizes)
  KMeans(n_clusters=n_clusters, random_state=42, n_init=10); 5 nearest-centroid reps.

name_clusters_with_llama(cluster_representatives, output_dir)
  -> dict | None
  Writes llm_naming_prompt.txt, llm_naming_responses.txt, cluster_names.json under output_dir.
  (Used only by Exp4 --name; implement path, do not run in Phase 3.)
```

### From `src/question_categorization_and_analysis/cluster_analysis.py`

```text
calculate_agreement(df, group_cols) -> DataFrame
  Columns: group_cols + Agreement + Count
  Agreement = mean(Gen==GT) within groups.

analyze_clusters(combined_df, output_dir)
  -> (cluster_results, gt_model_cluster_results, gen_model_cluster_results)
  Splits on metric; calls calculate_agreement; writes CSVs into output_dir.
  Gate (b): call with a throwaway dir under results/revision_r2/.../_gate/ (never the 2025 tree).
```

### From `scripts/clinical_stats/bootstrap_utils.py`

```text
make_indices(n_units: int, n_boot: int, seed: int) -> np.ndarray  # (n_boot, n_units)
percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]
```

Thin local wrappers will turn report-level resample indices into GT/GEN/delta replicates; we will **not** reuse RadPref matrix helpers (`mean_per_rater_from_matrix`, `paired_comparison_table`, etc.).

### Not used

- `process_questions()`, existing `*.sh` launchers (re-embed / overwrite `outputs/`).
- `create_combined_questions.py` (inputs already stored).
- `get_embeddings`, `visualize_clusters` (no re-embed / no t-SNE required).

---

## 4. Files to add / edit

### Phase 2 — commit 1 (existing untracked, as-is)

- `scripts/clinical_stats/{__init__.py,bootstrap_utils.py}`
- `scripts/radpref_data/run_radpref_paired_stats.py`
- `scripts/rexval_data/run_rexval_alignment_stats.py`

### Phase 2 — new files only

| Path | Role |
|------|------|
| `src/question_categorization_and_analysis/cluster_k_sensitivity.py` | Exp 4 |
| `src/question_categorization_and_analysis/seed_consistency.py` | Exp 5 |
| `scripts/iuxray_data/question_categorization_and_analysis/run_cluster_k_sensitivity.sh` | Exp 4 launcher |
| `scripts/iuxray_data/question_categorization_and_analysis/run_seed_consistency.sh` | Exp 5 launcher |

### Phase 2 — edit only

- `README.md` — new subsection under **Question categorization and cluster analysis**:  
  *“Revision analyses (Nature Communications R2): K sensitivity and seed consistency”*  
  (purpose, inputs, sbatch/srun, outputs, later `--name`).
- `.gitignore` — **no change needed** (`__pycache__/` already present); mention in commit message if anything else appears.

### Outputs (write-only under worktree)

```text
results/revision_r2/exp4/
  logs/
  gate.json | gate.md
  clustered_questions_K{K}.csv
  cluster_representatives_K{K}.txt
  scores_K{K}.csv                  # per model×cluster: GT, GEN, n, delta, CI, label
  concept_tracking.csv             # anchor shares / matched / ≥20%
  correspondence_K{K}.npy|.csv     # 20×K overlap (share)
  correspondence_heatmap_K{K}.png
  correspondence_summary.csv       # ARI vs K=20, mean best-match share
  consistency_by_K.csv             # (a)(b)(c) per model×K
  consistency_share_vs_K.png       # (c) lines
  scatter_grid.png                 # rows=K, cols=3 models
  summary.md
  (--name later:) cluster_names_K{K}.json, clustered_questions_with_names_K{K}.csv,
                  refreshed figures/tables using names

results/revision_r2/exp5/
  logs/
  gate.json | gate.md              # shared gate or symlink/copy from exp4
  scores_by_eval_seed.csv
  across_seed_summary.csv          # mean/SD/t-interval; label counts
  label_consistency.csv
  instance_counts.csv              # flag <30 in any seed
  report_jaccard.csv               # same-report across seeds + reference
  question_overlap.md/.csv         # per-seed unique Q + 9569/78708
  scatter_mean_with_errorbars.png  # per model; clusters labelled by name
  jaccard_boxplot.png
  summary.md
```

**K=20:** use **stored** `cluster_id` / names (no new KMeans for scores). Still run `cluster_questions(..., 20)` once for gate ARI and for correspondence baseline.

---

## 5. Shared definitions (both experiments)

**Per model × cluster (and × K or × eval_seed as applicable):**

1. Filter rows to that model (+ cluster assignment).  
2. `GT` = mean(Gen==GT) on `metric==gt_reports_as_ref`.  
3. `GEN` = mean(Gen==GT) on `metric==gen_reports_as_ref`.  
4. `n` = instance count used for that score (report both or the pooled count used in Fig. 4 — Fig. 4 stores Count per metric table separately; we will store `n_gt`, `n_gen`, and `n = n_gt + n_gen` or match Fig. 4’s per-ref Count; **default:** `n_gt`/`n_gen` plus `n_gt` as primary “n instances” for sizing, documenting in summary).  
5. `delta = GT - GEN`.  
6. **Bootstrap (B=2000 default, `--quick` → 200):** within model, `make_indices(n_reports, B, seed=42)` on unique `Report_ID`s; each replicate keeps **all** rows of sampled reports (all eval/model seeds present in the analysis slice). Recompute GT, GEN, delta; `percentile_ci` → 95% CI on delta.  
7. **Label:** omission-dominated if CI upper < 0; hallucination-dominated if CI lower > 0; else balanced.

**Reporting language:** prose/figures/tables use concept names (“Lung Lobe Abnormality”, “Calcified Granuloma Location”); at other K, “the cluster(s) containing most Lung Lobe Abnormality questions”. Cluster IDs only as keys / point labels.

**Fixed seeds:** KMeans `random_state=42`; bootstrap `seed=42`; Jaccard reference pairs `seed=42`.

---

## 6. Gate (before Exp 4/5 proper; srun on `cpu_short`, never login / never sp-0010)

| ID | Check | Pass criterion |
|----|-------|----------------|
| **a** | `cluster_questions(emb, unique_questions, n_clusters=20)` on 2025 pkl | ARI vs stored unique-Q labels = **1.0** |
| **b** | `analyze_clusters` / `calculate_agreement` on stored named table | max \|diff\| vs stored agreement CSVs ≈ 0 (report value; expect ~1e-16) |
| **c** | Model seeds 1–5 “identical outputs” | See **Ambiguity A** below |

If any gate fails → stop, write `gate.md`, do not proceed to full Exp 4/5 compute.

---

## 7. Exp 4 — `cluster_k_sensitivity.py` (R1-C7)

**CLI (sketch):**  
`--fig4-root`, `--output-dir` (default `results/revision_r2/exp4`), `--n-boot 2000`, `--quick`, `--name` (implement only), `--ks 10,15,20,25,30`.

**Steps:**

1. **Cluster:** For K∈{10,15,20,25,30}, call `cluster_questions(..., n_clusters=K)`. Map labels onto full instance table via `Question_Text`.  
   - K=20 scores/figures use **stored** labels; refit only for gate + ARI baseline.  
   - Write `clustered_questions_K{K}.csv` (`cluster_id` column) and `cluster_representatives_K{K}.txt` (same text format as stored reps).

2. **Scores:** Per K, model, cluster → GT, GEN, n, delta, CI, label (defs above). Pool **all** model_seed × eval_seed (Fig. 4).

3. **Concept tracking:** Anchor sets = unique `Question_Text` in stored clusters named Lung Lobe Abnormality / Calcified Granuloma Location. Per K, share of each anchor set’s **instances** in each new cluster; matched = max share; list clusters with share ≥ 20%; attach those clusters’ dominance labels per model.

4. **Correspondence:** For each K, 20×K matrix = share of each original cluster’s questions (unique Q or instances — **default: unique Question_Text**, consistent with ARI) falling in each new cluster; heatmap; ARI(new labels, K=20 stored) on unique Q; mean over original clusters of best-match share.

5. **Consistency (per model × K):**  
   (a) % of 20 original clusters whose dominance label equals best-matched new cluster’s label.  
   (b) Spearman across 20 original clusters: original delta vs question-weighted mean delta of new clusters their questions fall into.  
   (c) Share of question instances in omission- vs hallucination-dominated clusters vs K (one line per model).

6. **Figure:** scatter grid (rows=K, cols=3 models): GEN vs GT, diagonal, size∝count, colour=label, point label=cluster ID; highlight matched concept clusters.

7. **`--name` (implement, do not run):** For K≠20, load reps → call `name_clusters_with_llama` in a per-K scratch dir (API writes fixed filenames) → copy/rename to `cluster_names_K{K}.json` and build `clustered_questions_with_names_K{K}.csv`; refresh figures/tables with names; **no score recompute**. K=20 keeps stored names.

---

## 8. Exp 5 — `seed_consistency.py` (R4)

**CLI:** `--fig4-root`, `--output-dir` (default `results/revision_r2/exp5`), `--n-boot`, `--quick`.  
**Slice:** `model_seed == 1`; stored K=20 labels; **no re-clustering**.

1. Per eval_seed ∈ {123,456,789,202,101}, model, cluster → GT, GEN, n, delta, CI, label.  
2. Across 5 seeds: mean, SD, 95% t-interval `mean ± 2.776·SD/√5` for GT, GEN, delta; counts of each label; per model % clusters (and size-weighted %) with same label in all 5 seeds; highlight the two concept clusters by **name**.  
3. Instance counts per seed×model×cluster; flag any cluster with **&lt; 30** instances in any seed.  
4. **Report-level Jaccard:** for each (model, Report_ID), set of `cluster_id`s with ≥1 question in that seed; Jaccard over all 10 seed pairs; separately for GT-source (`gt_reports_as_ref`) and GEN-source (`gen_reports_as_ref`) rows. Summarise mean/median/IQR per model. **Reference:** 1000 random pairs of **different** reports (same model, fixed seed 42), same Jaccard definition.  
5. Unique `Question_Text` per eval_seed and intersection across all 5 (**expect 9 569 / 78 708** on full table; also report model_seed=1-only intersection for transparency).  
6. Figures: (i) per model, cluster mean GEN vs GT with across-seed error bars, diagonal, **name** labels; (ii) box plot same-report-across-seeds Jaccard vs different-report reference, per model.

---

## 9. Launchers

Both `.sh` files:

```bash
#SBATCH -p cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o results/revision_r2/<exp>/logs/%x_%j.out
# (and -e similarly)
```

- `cd` to repo root; load `.env` if present; `conda activate rrg-eval-clean`.  
- `export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=`.  
- Runnable via `sbatch` **or** `bash` inside `srun` on `cpu_short` (fallback `-p oermannlab` if busy).  
- **Never** request GPUs; **never** login-node compute; **never** `sp-0010` / job 29998975.

---

## 10. Runtime estimate (cpu_short, 8 CPU, 64G)

| Step | Estimate |
|------|----------|
| KMeans ×5 (K=10…30) | ~3–5 min (K=20 alone ~39 s in rrg-eval-clean) |
| Gate a–b | &lt; 2 min |
| Exp4 scores + 2000 report bootstraps (vectorized per-report counts) | ~20–45 min |
| Exp4 figures / correspondence | &lt; 5 min |
| Exp5 scores + bootstraps (5 seeds × 20 × 3) | ~15–30 min |
| Exp5 Jaccard + figures | &lt; 5 min |
| **`--quick` (B=200) end-to-end both** | **~15–25 min** |
| **Full B=2000 both (sbatch 6h)** | **~1–2 h** with margin |

---

## 11. Ambiguities / risks (need awareness before Phase 2)

### A. Gate (c) — **HIGH RISK**

Literal reading: “Model seeds 1–5 give identical outputs (same questions and predictions per model, report, eval seed).”

**Preliminary check on stored CSV:** false.  
- Question sets differ across `model_seed` (e.g. mean pairwise Jaccard ≈ 0.36 for maira-2, eval 123, gt_ref, 50 reports).  
- Even on overlapping `(Report_ID, Question_Text)`, `Correct_Answer` / predictions often differ (~27% match) — consistent with **different answer-choice shuffles** per run.  
- README: *“generated reports are identical across model seeds. The 5 model seeds exist to support clustering robustness analysis.”*

**Proposed gate (c) for approval:**  
(c1) Document MCQ non-identity (as above) in `gate.md`.  
(c2) Confirm Exp 5 uses **`model_seed=1` only** by design.  
(c3) Optional: if report CSVs are available, spot-check greedy RRG report identity across model seeds (README claim).  

If you require the literal MCQ identity check, Phase 2 will **stop** after reporting failure.

### B. `name_clusters_with_llama` side effects

Writes fixed names (`cluster_names.json`, etc.). Plan: call with per-K scratch `output_dir`, then rename to `*_K{K}.*`. No live API in Phase 3 (`--name` not run). Naming remains blocked until a 70B endpoint is up (discovery B.3).

### C. Correspondence unit

ARI is on unique questions; concept tracking is on **instances** (per discovery B.6). Correspondence heatmap: **unique Question_Text** shares (aligns with ARI). Will state both in summary if they diverge.

### D. `analyze_clusters` writes files

Gate (b) uses a disposable dir under `results/revision_r2/`; never writes into the 2025 RRG_evaluation tree.

### E. Import path for `bootstrap_utils`

Launchers run from repo root with `python src/...`. Exp scripts will add repo `scripts/` (or root) to `sys.path` to `from clinical_stats.bootstrap_utils import make_indices, percentile_ci` — same pattern as other script imports; **no edits** to `bootstrap_utils.py`.

### F. Output root

User/discovery mentioned `ICARE_score/results/revision_r2/`; this worktree will use **`ICARE_score_rev_stats/results/revision_r2/`** so outputs stay on the revision branch.

---

## 12. Phase 2 → 3 sequence (after approval)

1. Commit Exp 2 untracked scripts.  
2. Implement 4 new files + README subsection.  
3. Gate in `srun -p cpu_short` (or oermannlab); stop if fail.  
4. `--quick` both exps in srun; fix issues.  
5. `sbatch` both launchers (full B=2000); report job IDs.  
6. Commits: Exp 4; Exp 5 (+ README); push branch.  
7. On completion, paste both `summary.md` files.

---

## 13. Approval checklist

Please confirm or correct:

1. Gate (c) interpretation (**A** above) — proposed soft check vs literal stop.  
2. Correspondence matrix on **unique questions** (vs instances).  
3. Outputs under **this worktree** `results/revision_r2/`.  
4. Proceed to Phase 2 as written.

---

## 14. Approved decisions (2026-11-23)

1. **Gate (c) is record-only** — do not stop; do not drop rows. In `gate.md` record:
   - (i) exact-text and K=20-cluster Jaccard of questions across model seeds (same model, Report_ID, eval_seed);
   - (ii) whether answer differences persist when comparing **option text** rather than letters;
   - (iii) % identical generated-report text across model_seed 1–5 (stored RRG CSVs).
   - Explanation: MCQ differences across model seeds are expected from LLM server batching nondeterminism (temperature 0 on a shared vLLM server). **Proceed regardless.**
2. **Keep all model seeds everywhere** (Fig. 4 Methods: pool five model seeds × five eval seeds).
   - Exp 4: pool all rows.
   - Exp 5: unit = `eval_seed`; for each eval_seed pool all 5 model seeds (do **not** restrict to model_seed=1). Bootstrap Report_IDs within model, keeping all of a report's rows (all model seeds) together.
3. Correspondence on **unique Question_Text**; concept tracking and consistency (b)/(c) on **instances**.
4. Outputs under this worktree `results/revision_r2/`. Commit code, README, `summary.md`, figures; **do not** commit large CSVs, `.npy`, or per-K assignment files (gitignore rules).
5. **Report only (no code changes):** Fig. 3 / Appendix Fig. A5 seed usage — see §15.
6. Stop only if gate **(a)** or **(b)** fails.

## 15. Fig. 3 / Appendix Fig. A5 seed usage (report only)

| Figure | Script / notebook | model_seed | eval_seed |
|------|-------------------|------------|-----------|
| **Fig. 3** (IU X-ray model comparison; `plot_iuxray_quantitative.py`) | `MODEL_SEED = "model_seed_1"`; ICARE averaged across `EVAL_SEEDS` | **1 only** | **all five**: 101, 123, 202, 456, 789 |
| **Appendix Fig. A5** (seed stability / variation analysis) | `results_compilation.ipynb` cell building `variation_analysis` under `outputs/IU_xray/_summary/variation_analysis/` | **all five**: 1–5 | **all five**: 123, 456, 789, 101, 202 |

Fig. 3 baselines (BLEU, GREEN, etc.) also use `model_seed_1` only. A5 uses shuffled_ans_choices_data report-level ICARE across the full seed grid.
