# Exp 5 summary — eval-seed consistency

- sklearn: `1.3.2`
- Unit: eval_seed; each seed pools all 5 model seeds (Fig. 4 Methods).
- Stored K=20 labels; no re-clustering.

## Label consistency per model

- **chexpert-mimic-cxr-findings-baseline**: 85.0% clusters same label in all 5 seeds (size-weighted 84.4%). Lung Lobe Abnormality same=1; Calcified Granuloma Location same=1.
- **maira-2**: 50.0% clusters same label in all 5 seeds (size-weighted 50.1%). Lung Lobe Abnormality same=1; Calcified Granuloma Location same=1.
- **mimic-cxr-findings-baseline**: 70.0% clusters same label in all 5 seeds (size-weighted 64.9%). Lung Lobe Abnormality same=1; Calcified Granuloma Location same=1.

## Tracked concepts across seeds

- Lung Lobe Abnormality / chexpert-mimic-cxr-findings-baseline: delta mean±SD=-0.3529±0.0479; labels omission/hall/bal=5/0/0
- Lung Lobe Abnormality / maira-2: delta mean±SD=-0.0364±0.0202; labels omission/hall/bal=0/0/5
- Lung Lobe Abnormality / mimic-cxr-findings-baseline: delta mean±SD=-0.3302±0.0237; labels omission/hall/bal=5/0/0
- Calcified Granuloma Location / chexpert-mimic-cxr-findings-baseline: delta mean±SD=0.1090±0.0239; labels omission/hall/bal=0/0/5
- Calcified Granuloma Location / maira-2: delta mean±SD=-0.0524±0.0248; labels omission/hall/bal=0/0/5
- Calcified Granuloma Location / mimic-cxr-findings-baseline: delta mean±SD=-0.8548±0.0394; labels omission/hall/bal=0/0/5

## Across-seed SD (delta)
- range: 0.0026 – 0.1177 (median 0.0241)

## Report-level Jaccard vs reference

- chexpert-mimic-cxr-findings-baseline / gt_reports_as_ref: same-report mean/median/IQR=0.797/0.800/[0.714,0.889]; diff-report mean=0.444
- chexpert-mimic-cxr-findings-baseline / gen_reports_as_ref: same-report mean/median/IQR=0.776/0.800/[0.667,0.857]; diff-report mean=0.559
- maira-2 / gt_reports_as_ref: same-report mean/median/IQR=0.790/0.800/[0.700,0.889]; diff-report mean=0.447
- maira-2 / gen_reports_as_ref: same-report mean/median/IQR=0.729/0.714/[0.600,0.833]; diff-report mean=0.638
- mimic-cxr-findings-baseline / gt_reports_as_ref: same-report mean/median/IQR=0.792/0.800/[0.700,0.889]; diff-report mean=0.447
- mimic-cxr-findings-baseline / gen_reports_as_ref: same-report mean/median/IQR=0.742/0.750/[0.625,0.857]; diff-report mean=0.480

## Question overlap
- Unique Q per eval_seed: {123: 32156, 456: 32333, 789: 34248, 202: 33861, 101: 33265}
- Intersection all 5 seeds: **9569** / 78708 (expected 9569)

## Unexpected / notes

- See ../exp4/gate.md for model-seed MCQ nondeterminism (gate c).

- **Calcified Granuloma Location / mimic-cxr-findings-baseline:** large negative delta (mean ≈ −0.85) but all 5 seeds labeled balanced because GEN instance counts are tiny (11–18 per seed) → bootstrap CI is NaN under report resampling. Treat label as unreliable; flag via instance_counts (`flag_lt_30_any_seed`).
- CheXpert and MAIRA on that concept also stay balanced across seeds (CheXpert mean delta ≈ +0.11; MAIRA ≈ −0.05).
- Lung Lobe Abnormality labels are stable: omission for CheXpert/MIMIC (5/5 seeds), balanced for MAIRA (5/5).
