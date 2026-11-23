# Exp 5 summary — eval-seed consistency (final)

- sklearn: `1.3.2`
- Stored K=20 labels/names; no re-clustering.
- Labels: Fig. 4 rule (GT < GEN → omission; GT > GEN → hallucination).
- Modes: `model_seeds_all` (primary) and `model_seed_1`.

## Side-by-side: both model-seed modes

| Model | Mode | % same label (all) | % same excl low_n | median Δ SD | max Δ SD | # low_n clusters | Jaccard gap GT (same−diff) [CI] |
|---|---|---:|---:|---:|---:|---:|---|
| chexpert-mimic-cxr-findings-baseline | all | 85.0% | 85.0% | 0.0165 | 0.0479 | 0 | 0.353 [0.343,0.364] |
| maira-2 | all | 90.0% | 90.0% | 0.0271 | 0.1177 | 0 | 0.342 [0.331,0.353] |
| mimic-cxr-findings-baseline | all | 90.0% | 89.5% | 0.0233 | 0.0849 | 1 | 0.345 [0.334,0.356] |
| chexpert-mimic-cxr-findings-baseline | 1 | 70.0% | 70.6% | 0.0292 | 0.0886 | 3 | 0.269 [0.258,0.280] |
| maira-2 | 1 | 60.0% | 63.2% | 0.0387 | 0.1660 | 1 | 0.265 [0.253,0.276] |
| mimic-cxr-findings-baseline | 1 | 90.0% | 87.5% | 0.0357 | 0.0781 | 4 | 0.269 [0.257,0.279] |

## Tracked concepts (across-seed mean ± SD; label counts)

### mode=all
- **Lung Lobe Abnormality** / chexpert-mimic-cxr-findings-baseline: delta=-0.3529±0.0479; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Lung Lobe Abnormality** / maira-2: delta=-0.0364±0.0202; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Lung Lobe Abnormality** / mimic-cxr-findings-baseline: delta=-0.3302±0.0237; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Calcified Granuloma Location** / chexpert-mimic-cxr-findings-baseline: delta=0.1090±0.0239; labels omission/hall/bal=0.0/5.0/0.0; low_n=False
- **Calcified Granuloma Location** / maira-2: delta=-0.0524±0.0248; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Calcified Granuloma Location** / mimic-cxr-findings-baseline: delta=-0.8548±0.0394; labels omission/hall/bal=5.0/0.0/0.0; low_n=True

### mode=1
- **Lung Lobe Abnormality** / chexpert-mimic-cxr-findings-baseline: delta=-0.3129±0.0296; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Lung Lobe Abnormality** / maira-2: delta=-0.0294±0.0644; labels omission/hall/bal=4.0/1.0/0.0; low_n=False
- **Lung Lobe Abnormality** / mimic-cxr-findings-baseline: delta=-0.2928±0.0707; labels omission/hall/bal=5.0/0.0/0.0; low_n=False
- **Calcified Granuloma Location** / chexpert-mimic-cxr-findings-baseline: delta=0.0956±0.0886; labels omission/hall/bal=1.0/4.0/0.0; low_n=True
- **Calcified Granuloma Location** / maira-2: delta=-0.0521±0.0576; labels omission/hall/bal=4.0/1.0/0.0; low_n=False
- **Calcified Granuloma Location** / mimic-cxr-findings-baseline: delta=-0.8679±0.0294; labels omission/hall/bal=5.0/0.0/0.0; low_n=True

## Question overlap

- mode=all: intersection **9569** / 78708 (expected 9569); per-seed {'123': 32156, '456': 32333, '789': 34248, '202': 33861, '101': 33265}
- mode=1: intersection **3331** / 78708 (expected 9569); per-seed {123: 12901, 456: 13933, 789: 14082, 202: 14230, 101: 13945}

## Figure paths

Under `results/revision_r2/exp5/model_seeds_all/` and `.../model_seed_1/`:
- `fig4_style_seed{123,456,789,202,101}.png/.pdf`
- `combined_seed_grid.png`
- `scatter_mean_with_errorbars.png`
- `jaccard_boxplot.png`
