# Gate results (Exp 4/5)

- sklearn: `1.3.2`
- **(a) ARI vs stored K=20 labels:** 1.0 (exact match 1.0) — **PASS**
- **(b) max |diff| vs stored agreement CSVs:** GT=1.1102230246251565e-16, GEN=1.1102230246251565e-16 — **PASS**

## (c) Record-only (do not stop)

MCQ differences across model seeds are expected from LLM server batching nondeterminism (temperature 0 on a shared vLLM server). Proceed regardless.

### (i) Question Jaccard across model seeds
- Exact-text Jaccard mean/median: 0.22637897999577414/0.2 (n_pairs=2400)
- K=20-cluster Jaccard mean/median: 0.739567679393047/0.75

### (ii) Letter vs option-text answer comparison
- Correct letter match: 0.2699549733933688
- Correct option-text match: 0.716741711011052
- Pred-GT letter match: 0.2693409742120344
- Pred-GT option-text match: 0.7112157183790422
- n overlap rows: 4886
- note: If option-text match >> letter match, letter diffs are largely shuffle artifacts. If option-text still differs, content differs beyond shuffle.

### (iii) Generated report identity across model_seed 1–5
```json
{
  "maira-2": {
    "n_reports": 590,
    "pct_identical_across_seeds_1_5": 100.0,
    "pairwise_seed1_vs_2_pct": 100.0
  },
  "chexpert-mimic-cxr-findings-baseline": {
    "n_reports": 590,
    "pct_identical_across_seeds_1_5": 100.0,
    "pairwise_seed1_vs_2_pct": 100.0
  },
  "mimic-cxr-findings-baseline": {
    "n_reports": 590,
    "pct_identical_across_seeds_1_5": 100.0,
    "pairwise_seed1_vs_2_pct": 100.0
  }
}
```
