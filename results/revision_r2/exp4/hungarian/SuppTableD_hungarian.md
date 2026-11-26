# Supp Table D (Hungarian assignment)

Sides from pooled Exp4 `scores_K*.csv` (Fig. 4 pooling). Cell marks † / ‡ / § are defined in **Footnotes** below.

Finer K (25, 30): side of the **main** assigned cluster (largest overlap); `pct_same_side` = % of the K=20 concept’s unique questions in assigned K-clusters on the same side as that main cluster (not recomputed from pooled pieces).

Coarser K (10, 15): side of the assigned parent; siblings listed below.

| ID | Concept | CheXpertPlus_MIMIC@10 | CheXpertPlus_MIMIC@15 | CheXpertPlus_MIMIC@20 | CheXpertPlus_MIMIC@25 | CheXpertPlus_MIMIC@30 | CheXpertPlus_Chex_MIMIC@10 | CheXpertPlus_Chex_MIMIC@15 | CheXpertPlus_Chex_MIMIC@20 | CheXpertPlus_Chex_MIMIC@25 | CheXpertPlus_Chex_MIMIC@30 | MAIRA2@10 | MAIRA2@15 | MAIRA2@20 | MAIRA2@25 | MAIRA2@30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Lung Condition Assessment | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below | Below |
| 1 | Thoracic Spine Degeneration | Below‡ | Above‡ | Above‡ | Above‡ | Above‡ | Above‡ | Above‡ | Above‡ | Below‡ | Below‡ | Below | Below | Below | Below | Below‡ |
| 2 | Pleural Fluid Presence | Below§ | Below§ | Below | Below | Above§ | Above§ | Above§ | Below | Below | Below§ | Above§ | Above§ | Above | Above | Above§ |
| 3 | Diaphragm Position Abnormality | Below§ | Below | Below | Above‡§ | Below§ | Below§ | Below | Below | Below§ | Below§ | Below§ | Below | Below | Below§ | Above‡§ |
| 4 | Pulmonary Vascular Condition | Below§ | Below§ | Below | Above§ | Below | Below§ | Below§ | Below | Above§ | Above | Below§ | Below§ | Below | Above§ | Below |
| 5 | Heart Size Normalcy | Above§ | Above | Above | Above | Above | Below§ | Below | Below | Below | Below | Above§ | Above | Above | Above | Above |
| 6 | Lung Opacity and Infiltrates | Below§ | Below§ | Below | Below‡§ | Below§ | Below§ | Below§ | Above | Below‡§ | Above§ | Below§ | Below§ | Below | Below§ | Below§ |
| 7 | Pleural Effusion Presence | Below | Below | Below | Below | Below | Above | Above | Above | Above | Above | Above | Above | Above | Above | Above |
| 8 | Additional Radiologic Findings | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below |
| 9 | Aortic Condition | Below | Below | Below | Below | Below | Below‡ | Below‡ | Below‡ | Below‡ | Below‡ | Below | Below | Below | Below | Below |
| 10 | Pneumothorax Diagnosis | Below | Below | Below | Below | Below | Below | Above | Below | Above | Above | Above | Above | Above | Above | Above |
| 11 | Surgical Clip Location | Below§ | Below§ | Above‡ | Above§ | Above‡ | Below§ | Below§ | Above‡ | Below§ | Above‡ | Below§ | Below§ | Below | Above§ | Below |
| 12 | Lung Lobe Abnormality | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Above |
| 13 | Calcified Granuloma Location | Below§ | Below | Below | Below† | Below | Below§ | Above‡ | Above‡ | Above‡ | Above‡ | Below§ | Below | Below | Below | Below |
| 14 | Cardiac and Mediastinal Contours | Above | Above§ | Above | Above | Above | Above | Above§ | Above | Above | Above | Above | Above§ | Above | Above | Above |
| 15 | Osseous Structure Condition | Below§ | Above | Above | Below | Above | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below |
| 16 | Focal Airspace Opacity | Below | Below | Below | Below | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below | Below§ |
| 17 | Lung Clearness and Normalcy | Below§ | Below§ | Above | Below§ | Above | Below§ | Below§ | Below | Below§ | Below | Below§ | Above§ | Above | Above§ | Above |
| 18 | Lung Volume Condition | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below | Below§ | Below | Below | Below | Below |
| 19 | Heart Abnormality Presence | Above | Above§ | Below | Below | Below | Below | Above§ | Below | Above | Above | Above | Above§ | Below | Below | Below |
|  | % same side as K=20 | 75% | 85% | 100% | 80% | 95% | 80% | 75% | 100% | 70% | 80% | 90% | 95% | 100% | 90% | 90% |

## Clusters 12 & 13

### 12: Lung Lobe Abnormality

| Model | K | side | main_id | assigned | overlap | Jaccard | weak | pct_same_side | siblings |
|---|---:|---|---:|---|---:|---:|---|---:|---|
| CheXpertPlus_MIMIC | 10 | Below§ | 5 | 5 | 2988 | 0.4474 | True |  | 3,12 |
| CheXpertPlus_MIMIC | 15 | Below | 9 | 9 | 3429 | 0.8613 | False |  | 12 |
| CheXpertPlus_MIMIC | 20 | Below | 12 | 12 | 3496 | 1.0000 | False |  |  |
| CheXpertPlus_MIMIC | 25 | Below | 14 | 14 | 2781 | 0.7600 | False | 79.5 |  |
| CheXpertPlus_MIMIC | 30 | Below | 5 | 5 | 2797 | 0.7883 | False | 80.0 |  |
| CheXpertPlus_Chex_MIMIC | 10 | Below§ | 5 | 5 | 2988 | 0.4474 | True |  | 3,12 |
| CheXpertPlus_Chex_MIMIC | 15 | Below | 9 | 9 | 3429 | 0.8613 | False |  | 12 |
| CheXpertPlus_Chex_MIMIC | 20 | Below | 12 | 12 | 3496 | 1.0000 | False |  |  |
| CheXpertPlus_Chex_MIMIC | 25 | Below | 14 | 14 | 2781 | 0.7600 | False | 79.5 |  |
| CheXpertPlus_Chex_MIMIC | 30 | Below | 5 | 5 | 2797 | 0.7883 | False | 80.0 |  |
| MAIRA2 | 10 | Below§ | 5 | 5 | 2988 | 0.4474 | True |  | 3,12 |
| MAIRA2 | 15 | Below | 9 | 9 | 3429 | 0.8613 | False |  | 12 |
| MAIRA2 | 20 | Below | 12 | 12 | 3496 | 1.0000 | False |  |  |
| MAIRA2 | 25 | Below | 14 | 14 | 2781 | 0.7600 | False | 79.5 |  |
| MAIRA2 | 30 | Above | 5 | 5 | 2797 | 0.7883 | False | 80.0 |  |

### 13: Calcified Granuloma Location

| Model | K | side | main_id | assigned | overlap | Jaccard | weak | pct_same_side | siblings |
|---|---:|---|---:|---|---:|---:|---|---:|---|
| CheXpertPlus_MIMIC | 10 | Below§ | 3 | 3 | 1035 | 0.0633 | True |  | 6,8,11,13,15 |
| CheXpertPlus_MIMIC | 15 | Below | 13 | 13 | 1600 | 0.6700 | False |  | 13 |
| CheXpertPlus_MIMIC | 20 | Below | 13 | 13 | 1758 | 1.0000 | False |  |  |
| CheXpertPlus_MIMIC | 25 | Below† | 4 | 4 | 1590 | 0.7076 | False | 90.4 |  |
| CheXpertPlus_MIMIC | 30 | Below | 13 | 13 | 1308 | 0.7247 | False | 74.4 |  |
| CheXpertPlus_Chex_MIMIC | 10 | Below§ | 3 | 3 | 1035 | 0.0633 | True |  | 6,8,11,13,15 |
| CheXpertPlus_Chex_MIMIC | 15 | Above‡ | 13 | 13 | 1600 | 0.6700 | False |  | 13 |
| CheXpertPlus_Chex_MIMIC | 20 | Above‡ | 13 | 13 | 1758 | 1.0000 | False |  |  |
| CheXpertPlus_Chex_MIMIC | 25 | Above‡ | 4 | 4 | 1590 | 0.7076 | False | 90.4 |  |
| CheXpertPlus_Chex_MIMIC | 30 | Above‡ | 13 | 13 | 1308 | 0.7247 | False | 74.4 |  |
| MAIRA2 | 10 | Below§ | 3 | 3 | 1035 | 0.0633 | True |  | 6,8,11,13,15 |
| MAIRA2 | 15 | Below | 13 | 13 | 1600 | 0.6700 | False |  | 13 |
| MAIRA2 | 20 | Below | 13 | 13 | 1758 | 1.0000 | False |  |  |
| MAIRA2 | 25 | Below | 4 | 4 | 1590 | 0.7076 | False | 90.4 |  |
| MAIRA2 | 30 | Below | 13 | 13 | 1308 | 0.7247 | False | 74.4 |  |

## Coarser K: K=20 clusters sharing each parent

### K=10
- parent 0 ← 9:Aortic Condition
- parent 1 ← 0:Lung Condition Assessment, 4:Pulmonary Vascular Condition, 17:Lung Clearness and Normalcy, 18:Lung Volume Condition
- parent 2 ← 10:Pneumothorax Diagnosis
- parent 3 ← 6:Lung Opacity and Infiltrates, 8:Additional Radiologic Findings, 11:Surgical Clip Location, 13:Calcified Granuloma Location, 15:Osseous Structure Condition
- parent 4 ← 14:Cardiac and Mediastinal Contours
- parent 5 ← 3:Diaphragm Position Abnormality, 12:Lung Lobe Abnormality
- parent 6 ← 5:Heart Size Normalcy, 19:Heart Abnormality Presence
- parent 7 ← 16:Focal Airspace Opacity
- parent 8 ← 2:Pleural Fluid Presence, 7:Pleural Effusion Presence
- parent 9 ← 1:Thoracic Spine Degeneration

### K=15
- parent 0 ← 3:Diaphragm Position Abnormality, 11:Surgical Clip Location
- parent 1 ← 5:Heart Size Normalcy
- parent 2 ← 10:Pneumothorax Diagnosis
- parent 3 ← 4:Pulmonary Vascular Condition, 18:Lung Volume Condition
- parent 4 ← 6:Lung Opacity and Infiltrates, 8:Additional Radiologic Findings
- parent 5 ← 0:Lung Condition Assessment
- parent 6 ← 16:Focal Airspace Opacity
- parent 7 ← 1:Thoracic Spine Degeneration
- parent 8 ← 17:Lung Clearness and Normalcy
- parent 9 ← 12:Lung Lobe Abnormality
- parent 10 ← 2:Pleural Fluid Presence, 7:Pleural Effusion Presence
- parent 11 ← 14:Cardiac and Mediastinal Contours, 19:Heart Abnormality Presence
- parent 12 ← 15:Osseous Structure Condition
- parent 13 ← 13:Calcified Granuloma Location
- parent 14 ← 9:Aortic Condition

## Footnotes (symbol definitions)

Every mark that can appear after a Below/Above cell in this table:

| Symbol | Meaning |
|---|---|
| † | **n < 30**: the matched cluster has `n_gt < 30` or `n_gen < 30` (or the existing Exp4 `low_n` flag is true). |
| ‡ | **both-low**: ICARE-GT **and** ICARE-GEN are both **< 0.3** for that matched cluster (same flag as Supp Table A ‡). |
| § | **weak-match**: Jaccard overlap of the reported assignment edge (K=20 concept ↔ matched cluster) is **< 0.5**. |
| * | **Not used** in this Hungarian table. (In the greedy Supp Table D, `*` meant merged / reverse share < 0.5; that mark is not applied here—weak matches use § instead.) |

Side labels themselves: **Below** = ICARE-GT < ICARE-GEN (omission-dominated); **Above** = ICARE-GT > ICARE-GEN (hallucination-dominated).
