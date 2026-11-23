# Exp 5 assessment (eval-seed consistency) — honest read

**Setting:** stored K=20 labels/names; Fig. 4 diagonal labels; bootstrap CI / `low_n` reported but do not change labels.  
**Primary mode:** `model_seeds_all` (pool 5 model seeds per eval seed). Secondary: `model_seed_1`.  
**sklearn:** 1.3.2; full run job **30103005** (B=2000).

Numbers below are from the full run unless noted.

---

## a. Label stability

### `model_seeds_all` (primary)

| Model | % same label all 5 seeds | Excl. low_n | # low_n clusters |
|---|---:|---:|---:|
| CheX+MIMIC | **85%** (17/20) | 85% | 0 |
| MAIRA-2 | **90%** (18/20) | 90% | 0 |
| MIMIC | **90%** (18/20) | 89.5% (17/19) | 1 |

**Flips (7 cluster×model cells; all have \|Δ\| < 0.05; none low_n):**  
Thoracic Spine Degeneration, Pulmonary Vascular Condition, Pneumothorax Diagnosis (CheX); Pulmonary Vascular Condition, Focal Airspace Opacity (MAIRA); Pleural Effusion Presence, Osseous Structure Condition (MIMIC). All sit near the diagonal, so sign(GT−GEN) flips are expected under seed noise.

### `model_seed_1`

| Model | % same label | Excl. low_n | # low_n |
|---|---:|---:|---:|
| CheX+MIMIC | **70%** | 70.6% | 3 |
| MAIRA-2 | **60%** | 63.2% | 1 |
| MIMIC | **90%** | 87.5% | 4 |

More flips (16), of which 10 near-diagonal and 2 low_n — noisier because each seed has ~1/5 the rows.

**Verdict (a):** **supports stability** for the pooled mode (85–90% identical labels; flips are near-diagonal). **Mixed** for `model_seed_1` (60–70% for CheX/MAIRA).

---

## b. Score stability vs between-cluster spread

Across-seed SD of delta (median / max) vs SD of cluster mean deltas (between-cluster):

| Model | Mode | median seed-SD(Δ) | max seed-SD(Δ) | between-cluster SD(mean Δ) | ratio median/between |
|---|---|---:|---:|---:|---:|
| CheX+MIMIC | all | 0.016 | 0.048 | 0.127 | **0.13** |
| MAIRA-2 | all | 0.027 | 0.118 | 0.101 | **0.27** |
| MIMIC | all | 0.023 | 0.085 | 0.237 | **0.10** |
| CheX+MIMIC | 1 | 0.029 | 0.089 | (similar order) | higher |
| MAIRA-2 | 1 | 0.039 | 0.166 | | higher |

Median GT/GEN seed-SDs are ~0.01–0.02 (pooled). Seed noise is generally small relative to between-cluster Δ differences in the pooled mode; MAIRA’s max seed-SD(Δ)=0.118 is an outlier cluster and approaches the between-cluster scale.

**Verdict (b):** **supports stability** (pooled); seed noise ≪ between-cluster spread for medians. Caveat: a few clusters (esp. MAIRA) have large seed-SD.

---

## c. Concept coverage (report-level Jaccard)

Same-report across seeds vs different-report reference; gap = same − diff (GT-source):

| Model | Mode | same mean | diff mean | gap [95% CI] |
|---|---|---:|---:|---|
| CheX+MIMIC | all | 0.797 | 0.444 | **0.353 [0.343, 0.364]** |
| MAIRA-2 | all | 0.790 | 0.447 | **0.342 [0.331, 0.353]** |
| MIMIC | all | 0.792 | 0.447 | **0.345 [0.334, 0.356]** |
| (all three) | 1 | ~0.74–0.76 | ~0.48 | **~0.26–0.27** |

Gaps are clearly positive with tight CIs: the same report revisits more similar clusters across seeds than two random reports do. GEN-source gaps are smaller (MAIRA GEN gap only ~0.09 pooled) — expected if GEN questions are fewer/more variable.

**Verdict (c):** **supports stability** of topical coverage across seeds (GT-source especially).

---

## d. Tracked concepts

### Lung Lobe Abnormality (`model_seeds_all`)

| Model | Δ mean ± SD | Labels across 5 seeds |
|---|---|---|
| CheX+MIMIC | −0.353 ± 0.048 | omission **5/5** |
| MIMIC | −0.330 ± 0.024 | omission **5/5** |
| MAIRA-2 | −0.036 ± 0.020 | omission **5/5** (small \|Δ\|) |

### Calcified Granuloma Location (`model_seeds_all`)

| Model | Δ mean ± SD | Labels | low_n |
|---|---|---|---|
| CheX+MIMIC | +0.109 ± 0.024 | hallucination **5/5** | no |
| MAIRA-2 | −0.052 ± 0.025 | omission **5/5** | no |
| MIMIC | −0.855 ± 0.039 | omission **5/5** | **yes** (n_gen 11–18) |

Under `model_seed_1`, Lung Lobe stays omission for CheX/MIMIC (5/5) but MAIRA flips once (4 omission / 1 hallucination); Calcified CheX flips (1 omission / 4 hallucination) and is low_n.

**Verdict (d):** **supports stability** for Lung Lobe (all models, pooled). **Mixed** for Calcified: direction stable when pooled, but MIMIC is low_n on GEN, and `model_seed_1` is less stable for CheX/MAIRA.

---

## e. Model-level Fig. 4 conclusion

Claim tested: CheXpertPlus models are more omission-dominated than MAIRA-2 in every eval seed.

**Unweighted % omission clusters (`model_seeds_all`):** not always true for CheX vs MAIRA (e.g. seed 101: CheX 60% < MAIRA 65%; seed 456: CheX = MAIRA = 70%). MIMIC > MAIRA in 4/5 seeds.

**Size-weighted % omission (`model_seeds_all`):** CheX > MAIRA and MIMIC > MAIRA in **all 5 seeds** (CheX−MAIRA gaps ~0.7–19 pp; MIMIC−MAIRA larger).

**`model_seed_1`:** unweighted pattern is mixed; size-weighted CheX > MAIRA in 4/5 seeds (fails seed 202: 60.4 vs 62.9).

**Verdict (e):** **mixed** if stated as “every seed, unweighted cluster count.” **Supports** a softer claim: *when pooling model seeds, size-weighted omission share is higher for both CheXpertPlus models than for MAIRA-2 in every eval seed.*

---

## f. Fixed (`model_seed_1`) vs pooled (`model_seeds_all`)

- Label consistency drops when fixing model seed (CheX 85→70%, MAIRA 90→60%); MIMIC stays ~90%.
- Concept deltas keep the same sign when pooled vs seed-1 means, but seed-1 has more label flips and more low_n.
- Jaccard gaps shrink (~0.35 → ~0.27) but remain clearly positive.
- Question overlap: pooled intersection **9569/78708**; seed-1 only **3331/78708** (expected — fewer runs).

**Verdict (f):** **mixed** — qualitative story agrees, but stability metrics are materially weaker with a single model seed. Prefer reporting **pooled** as primary (matches Fig. 4 Methods).

---

## g. Summary table

| Question | Verdict | One-line reason |
|---|---|---|
| a Label stability | **supports** (pooled); mixed (seed-1) | 85–90% identical labels; flips near diagonal |
| b Score stability | **supports** | median seed-SD(Δ) ≪ between-cluster SD |
| c Jaccard coverage | **supports** | same-report − diff-report gap ≈ 0.34–0.35 (GT), CI excludes 0 |
| d Tracked concepts | **supports** / **mixed** | Lung Lobe rock-solid; Calcified MIMIC low_n; seed-1 flakier |
| e Fig. 4 model ordering | **mixed** (strict); **supports** (size-weighted pooled) | unweighted % not monotonic every seed; size-weighted is |
| f Pooled vs seed-1 | **mixed** | same directions, worse precision/stability if not pooled |

### Weak spots a reviewer could raise

1. Fig. 4’s strict diagonal label makes near-zero Δ clusters “flip” even when scientifically uninteresting (all 7 pooled flips have \|Δ\|<0.05).  
2. Calcified Granuloma / MIMIC has tiny GEN counts → huge \|Δ\| but statistically fragile; `low_n` must be disclosed.  
3. MAIRA Lung Lobe is omission in all seeds but with tiny Δ (−0.04): calling it “omission-dominated” overstates effect size relative to CheXpertPlus (~−0.33).  
4. Unweighted omission-share does **not** put both CheXpertPlus models above MAIRA in every seed.  
5. `model_seed_1` alone would look less reproducible — don’t lead with it.

### Honest paper wording (suggested)

> Across five evaluation seeds (pooling the five model seeds as in Fig. 4), 85–90% of clusters keep the same omission/hallucination label; the few flips lie near the diagonal. Seed-to-seed SD of cluster Δ is small relative to between-cluster differences. The same report revisits substantially more similar clusters across seeds than pairs of different reports (Jaccard gap ≈ 0.34). **Lung Lobe Abnormality** remains omission-dominated for all three models in every seed, with a much larger Δ for CheXpertPlus models than for MAIRA-2. Size-weighted omission share is higher for both CheXpertPlus models than for MAIRA-2 in every seed; an unweighted cluster count does not show that ordering every seed. Results with a single model seed are directionally similar but less stable and should not be the primary analysis.

