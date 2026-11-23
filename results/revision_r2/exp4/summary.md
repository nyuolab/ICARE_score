# Exp 4 summary — cluster K sensitivity

- sklearn: `1.3.2`
- Gate (a) PASS=True ARI=1.0
- Gate (b) PASS=True max|diff| GT/GEN=1.1102230246251565e-16/1.1102230246251565e-16
- Gate (c): record-only (see gate.md)

## Matched clusters for tracked concepts

### Lung Lobe Abnormality
- K=10: matched cluster **5** (share=0.882); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: balanced; mimic-cxr-findings-baseline: omission-dominated
- K=15: matched cluster **9** (share=0.983); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: balanced; mimic-cxr-findings-baseline: omission-dominated
- K=20: matched cluster **12** (share=1.000); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: balanced; mimic-cxr-findings-baseline: omission-dominated
- K=25: matched cluster **14** (share=0.787); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: balanced; mimic-cxr-findings-baseline: omission-dominated
- K=30: matched cluster **14** (share=0.729); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: balanced; mimic-cxr-findings-baseline: omission-dominated

### Calcified Granuloma Location
- K=10: matched cluster **3** (share=0.621); labels: chexpert-mimic-cxr-findings-baseline: omission-dominated; maira-2: omission-dominated; mimic-cxr-findings-baseline: omission-dominated
- K=15: matched cluster **13** (share=0.894); labels: chexpert-mimic-cxr-findings-baseline: balanced; maira-2: balanced; mimic-cxr-findings-baseline: balanced
- K=20: matched cluster **13** (share=1.000); labels: chexpert-mimic-cxr-findings-baseline: balanced; maira-2: balanced; mimic-cxr-findings-baseline: balanced
- K=25: matched cluster **4** (share=0.898); labels: chexpert-mimic-cxr-findings-baseline: balanced; maira-2: balanced; mimic-cxr-findings-baseline: balanced
- K=30: matched cluster **10** (share=0.863); labels: chexpert-mimic-cxr-findings-baseline: balanced; maira-2: balanced; mimic-cxr-findings-baseline: balanced

## Consistency (a)–(c) per K

- K=10 chexpert-mimic-cxr-findings-baseline: (a) label-match=50.0% (b) Spearman=0.4842105263157894 (c) omission=57.8% hallucination=25.1%
- K=10 maira-2: (a) label-match=50.0% (b) Spearman=0.5804511278195488 (c) omission=41.2% hallucination=21.4%
- K=10 mimic-cxr-findings-baseline: (a) label-match=55.0% (b) Spearman=0.6541353383458646 (c) omission=81.2% hallucination=3.6%
- K=15 chexpert-mimic-cxr-findings-baseline: (a) label-match=65.0% (b) Spearman=0.6977443609022556 (c) omission=42.7% hallucination=27.6%
- K=15 maira-2: (a) label-match=55.0% (b) Spearman=0.6947368421052631 (c) omission=38.6% hallucination=40.9%
- K=15 mimic-cxr-findings-baseline: (a) label-match=60.0% (b) Spearman=0.7894736842105263 (c) omission=59.1% hallucination=12.3%
- K=20 chexpert-mimic-cxr-findings-baseline: (a) label-match=100.0% (b) Spearman=1.0 (c) omission=23.5% hallucination=19.8%
- K=20 maira-2: (a) label-match=100.0% (b) Spearman=1.0 (c) omission=15.9% hallucination=25.0%
- K=20 mimic-cxr-findings-baseline: (a) label-match=100.0% (b) Spearman=1.0 (c) omission=24.3% hallucination=6.0%
- K=25 chexpert-mimic-cxr-findings-baseline: (a) label-match=65.0% (b) Spearman=0.7609022556390977 (c) omission=29.3% hallucination=20.4%
- K=25 maira-2: (a) label-match=45.0% (b) Spearman=0.8646616541353382 (c) omission=35.9% hallucination=39.9%
- K=25 mimic-cxr-findings-baseline: (a) label-match=70.0% (b) Spearman=0.7503759398496239 (c) omission=39.7% hallucination=9.0%
- K=30 chexpert-mimic-cxr-findings-baseline: (a) label-match=85.0% (b) Spearman=nan (c) omission=16.6% hallucination=19.8%
- K=30 maira-2: (a) label-match=55.0% (b) Spearman=nan (c) omission=11.2% hallucination=36.1%
- K=30 mimic-cxr-findings-baseline: (a) label-match=95.0% (b) Spearman=nan (c) omission=22.8% hallucination=15.0%

## Correspondence vs K=20

- K=10: ARI=0.5281; mean best-match share=0.8466
- K=15: ARI=0.7173; mean best-match share=0.8404
- K=20: ARI=1.0000; mean best-match share=1.0000
- K=25: ARI=0.7279; mean best-match share=0.7807
- K=30: ARI=0.7372; mean best-match share=0.7635

## Unexpected / notes

- See gate.md for model-seed MCQ nondeterminism.

- At K=30, Spearman (b) is NaN for all models because some new clusters have `n_gen=0` (no gen_reports_as_ref instances), so mapped deltas contain NaN.
- Some clusters at finer K have NaN bootstrap CIs (sparse GEN rows under report resampling) and are labeled **balanced** by the CI rule fallback.
