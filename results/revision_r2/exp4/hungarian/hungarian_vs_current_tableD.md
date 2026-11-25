# Hungarian vs current Table D

Current Table D uses greedy best-match (largest forward share), not Hungarian.

## Side differences (Below/Above ignoring */†/§ marks)

- id 2 (Pleural Fluid Presence) | CheXpertPlus_MIMIC | K=30: TableD=Below* vs Hungarian=Above§ (main_id=2)
- id 2 (Pleural Fluid Presence) | CheXpertPlus_Chex_MIMIC | K=30: TableD=Above* vs Hungarian=Below§ (main_id=2)
- id 11 (Surgical Clip Location) | MAIRA2 | K=25: TableD=Below* vs Hungarian=Above§ (main_id=23)
- id 17 (Lung Clearness and Normalcy) | MAIRA2 | K=15: TableD=Below*† vs Hungarian=Above§ (main_id=8)
- id 17 (Lung Clearness and Normalcy) | MAIRA2 | K=25: TableD=Below*† vs Hungarian=Above§ (main_id=17)

## Mapping: greedy best-match vs Hungarian main id

- id 2 K=30: greedy_best=1 vs hungarian_main=2 (assigned=[2])
- id 11 K=25: greedy_best=2 vs hungarian_main=23 (assigned=[23])
- id 17 K=15: greedy_best=5 vs hungarian_main=8 (assigned=[8])
- id 17 K=25: greedy_best=0 vs hungarian_main=17 (assigned=[17])

## 3-line plain summary

1. Hungarian % same-side as K=20 — CheXpertPlus_MIMIC: K10=75%, K15=85%, K20=100%, K25=80%, K30=95%; CheXpertPlus_Chex_MIMIC: K10=80%, K15=75%, K20=100%, K25=70%, K30=80%; MAIRA2: K10=90%, K15=95%, K20=100%, K25=90%, K30=90%.
2. Side cells differing from current Table D: 5; main-id differing from greedy best-match: 4.
3. networkx.min_cost_flow cross-check matches Hungarian for all K: True.
