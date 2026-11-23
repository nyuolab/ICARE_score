# Full cluster report — Exp 4 (K sensitivity) & Exp 5 (eval-seed consistency)

Nature Communications R2 / ICARE. Worktree outputs under `results/revision_r2/`. Primary mode: **`model_seeds_all`** (pool model seeds 1–5). `model_seed_1` appears only in Part C.

Dominance labels use the manuscript Fig. 4 rule on point estimates (omission if GT < GEN; hallucination if GT > GEN). Bootstrap 95% CIs and `low_n` are reported alongside labels and do not redefine them. `both_low` = GT < 0.3 and GEN < 0.3.

Exp 4 Spearman mapped Δ: instance-weighted mean over new clusters with **defined** Δ only (re-normalised); drop an original cluster only if **>50%** of its instances fall in undefined new clusters.

## Part A — Exp 5 (eval-seed consistency), `model_seeds_all`

Labels follow the Fig. 4 rule (omission if GT < GEN, hallucination if GT > GEN). `both_low` marks clusters with GT < 0.3 and GEN < 0.3 (near the origin; a side-of-diagonal label alone can hide these).

### A1. All 20 clusters — CheX+MIMIC

| Cluster | n_gt | n_gen | GT mean±SD | GEN mean±SD | Δ mean±SD | labels (seeds) | #same | low_n | both_low |
|---|---|---|---|---|---|---|---|---|---|
| Lung Condition Assessment | 4709–4816 | 4029–4691 | 0.821±0.004 | 0.837±0.007 | -0.016±0.008 | O/O/O/O/O | 5/5 |  |  |
| Thoracic Spine Degeneration | 1376–1464 | 117–135 | 0.179±0.004 | 0.174±0.017 | 0.005±0.016 | O/H/H/O/H | 3/5 |  | Y |
| Pleural Fluid Presence | 540–592 | 2882–3340 | 0.900±0.007 | 0.923±0.005 | -0.023±0.006 | O/O/O/O/O | 5/5 |  |  |
| Diaphragm Position Abnormality | 896–985 | 296–338 | 0.258±0.015 | 0.396±0.037 | -0.138±0.037 | O/O/O/O/O | 5/5 |  |  |
| Pulmonary Vascular Condition | 1286–1389 | 579–614 | 0.457±0.018 | 0.474±0.032 | -0.018±0.040 | O/O/O/H/H | 3/5 |  |  |
| Heart Size Normalcy | 2867–2986 | 2178–2525 | 0.731±0.008 | 0.807±0.006 | -0.076±0.010 | O/O/O/O/O | 5/5 |  |  |
| Lung Opacity and Infiltrates | 1501–1585 | 461–490 | 0.485±0.009 | 0.462±0.032 | 0.023±0.025 | H/H/H/H/H | 5/5 |  |  |
| Pleural Effusion Presence | 4007–4197 | 5975–6144 | 0.886±0.004 | 0.781±0.008 | 0.105±0.005 | H/H/H/H/H | 5/5 |  |  |
| Additional Radiologic Findings | 1641–1712 | 326–385 | 0.386±0.011 | 0.553±0.030 | -0.167±0.027 | O/O/O/O/O | 5/5 |  |  |
| Aortic Condition | 635–685 | 179–211 | 0.150±0.011 | 0.255±0.007 | -0.106±0.015 | O/O/O/O/O | 5/5 |  | Y |
| Pneumothorax Diagnosis | 3905–4112 | 4315–4652 | 0.928±0.004 | 0.928±0.004 | -0.000±0.006 | H/O/O/H/H | 3/5 |  |  |
| Surgical Clip Location | 84–102 | 79–103 | 0.216±0.040 | 0.055±0.042 | 0.160±0.031 | H/H/H/H/H | 5/5 |  | Y |
| Lung Lobe Abnormality | 859–946 | 247–279 | 0.329±0.018 | 0.682±0.031 | -0.353±0.048 | O/O/O/O/O | 5/5 |  |  |
| Calcified Granuloma Location | 611–649 | 47–65 | 0.139±0.013 | 0.030±0.029 | 0.109±0.024 | H/H/H/H/H | 5/5 |  | Y |
| Cardiac and Mediastinal Contours | 1236–1301 | 1136–1207 | 0.898±0.015 | 0.657±0.029 | 0.241±0.042 | H/H/H/H/H | 5/5 |  |  |
| Osseous Structure Condition | 1539–1630 | 1657–1837 | 0.418±0.015 | 0.465±0.012 | -0.047±0.015 | O/O/O/O/O | 5/5 |  |  |
| Focal Airspace Opacity | 2609–2766 | 1414–1703 | 0.714±0.013 | 0.800±0.010 | -0.086±0.017 | O/O/O/O/O | 5/5 |  |  |
| Lung Clearness and Normalcy | 414–465 | 182–252 | 0.751±0.024 | 0.798±0.025 | -0.047±0.014 | O/O/O/O/O | 5/5 |  |  |
| Lung Volume Condition | 1649–1698 | 750–827 | 0.344±0.012 | 0.468±0.012 | -0.124±0.012 | O/O/O/O/O | 5/5 |  |  |
| Heart Abnormality Presence | 2118–2205 | 926–1081 | 0.768±0.009 | 0.804±0.023 | -0.036±0.019 | O/O/O/O/O | 5/5 |  |  |

### A1. All 20 clusters — MIMIC

| Cluster | n_gt | n_gen | GT mean±SD | GEN mean±SD | Δ mean±SD | labels (seeds) | #same | low_n | both_low |
|---|---|---|---|---|---|---|---|---|---|
| Lung Condition Assessment | 4624–4789 | 4127–4907 | 0.839±0.006 | 0.860±0.006 | -0.022±0.009 | O/O/O/O/O | 5/5 |  |  |
| Thoracic Spine Degeneration | 1417–1474 | 109–132 | 0.165±0.011 | 0.064±0.018 | 0.101±0.019 | H/H/H/H/H | 5/5 |  | Y |
| Pleural Fluid Presence | 540–615 | 1721–1996 | 0.847±0.013 | 0.927±0.006 | -0.080±0.016 | O/O/O/O/O | 5/5 |  |  |
| Diaphragm Position Abnormality | 926–985 | 276–297 | 0.270±0.022 | 0.350±0.046 | -0.080±0.037 | O/O/O/O/O | 5/5 |  |  |
| Pulmonary Vascular Condition | 1256–1408 | 500–544 | 0.467±0.018 | 0.536±0.008 | -0.069±0.022 | O/O/O/O/O | 5/5 |  |  |
| Heart Size Normalcy | 2854–2907 | 1582–1719 | 0.774±0.010 | 0.745±0.005 | 0.030±0.007 | H/H/H/H/H | 5/5 |  |  |
| Lung Opacity and Infiltrates | 1471–1608 | 388–623 | 0.471±0.010 | 0.588±0.039 | -0.117±0.035 | O/O/O/O/O | 5/5 |  |  |
| Pleural Effusion Presence | 4007–4183 | 3082–3347 | 0.789±0.007 | 0.792±0.008 | -0.003±0.004 | O/H/O/O/O | 4/5 |  |  |
| Additional Radiologic Findings | 1617–1694 | 464–542 | 0.418±0.008 | 0.576±0.027 | -0.158±0.031 | O/O/O/O/O | 5/5 |  |  |
| Aortic Condition | 651–714 | 153–170 | 0.190±0.015 | 0.439±0.033 | -0.249±0.036 | O/O/O/O/O | 5/5 |  |  |
| Pneumothorax Diagnosis | 3889–3985 | 3266–3658 | 0.899±0.006 | 0.924±0.012 | -0.025±0.011 | O/O/O/O/O | 5/5 |  |  |
| Surgical Clip Location | 79–108 | 195–225 | 0.217±0.029 | 0.085±0.031 | 0.132±0.024 | H/H/H/H/H | 5/5 |  | Y |
| Lung Lobe Abnormality | 870–992 | 295–321 | 0.304±0.010 | 0.634±0.032 | -0.330±0.024 | O/O/O/O/O | 5/5 |  |  |
| Calcified Granuloma Location | 618–648 | 11–18 | 0.132±0.012 | 0.987±0.030 | -0.855±0.039 | O/O/O/O/O | 5/5 | Y |  |
| Cardiac and Mediastinal Contours | 1160–1310 | 806–866 | 0.886±0.012 | 0.627±0.020 | 0.258±0.030 | H/H/H/H/H | 5/5 |  |  |
| Osseous Structure Condition | 1557–1600 | 3229–4095 | 0.580±0.019 | 0.559±0.020 | 0.021±0.034 | O/H/O/H/H | 3/5 |  |  |
| Focal Airspace Opacity | 2627–2765 | 754–926 | 0.723±0.007 | 0.813±0.010 | -0.091±0.011 | O/O/O/O/O | 5/5 |  |  |
| Lung Clearness and Normalcy | 445–514 | 759–1428 | 0.779±0.016 | 0.489±0.070 | 0.290±0.085 | H/H/H/H/H | 5/5 |  |  |
| Lung Volume Condition | 1610–1695 | 901–971 | 0.352±0.004 | 0.502±0.022 | -0.150±0.023 | O/O/O/O/O | 5/5 |  |  |
| Heart Abnormality Presence | 2098–2184 | 940–1161 | 0.781±0.004 | 0.821±0.011 | -0.040±0.011 | O/O/O/O/O | 5/5 |  |  |

### A1. All 20 clusters — MAIRA-2

| Cluster | n_gt | n_gen | GT mean±SD | GEN mean±SD | Δ mean±SD | labels (seeds) | #same | low_n | both_low |
|---|---|---|---|---|---|---|---|---|---|
| Lung Condition Assessment | 4705–4876 | 8758–9285 | 0.843±0.005 | 0.859±0.003 | -0.017±0.003 | O/O/O/O/O | 5/5 |  |  |
| Thoracic Spine Degeneration | 1425–1465 | 522–575 | 0.187±0.014 | 0.307±0.028 | -0.119±0.025 | O/O/O/O/O | 5/5 |  |  |
| Pleural Fluid Presence | 517–610 | 300–777 | 0.926±0.005 | 0.883±0.029 | 0.043±0.025 | H/H/H/H/H | 5/5 |  |  |
| Diaphragm Position Abnormality | 899–986 | 794–869 | 0.264±0.019 | 0.318±0.015 | -0.054±0.028 | O/O/O/O/O | 5/5 |  |  |
| Pulmonary Vascular Condition | 1279–1405 | 222–256 | 0.473±0.024 | 0.496±0.024 | -0.023±0.047 | O/O/O/O/H | 4/5 |  |  |
| Heart Size Normalcy | 2875–2913 | 711–802 | 0.747±0.007 | 0.616±0.031 | 0.132±0.029 | H/H/H/H/H | 5/5 |  |  |
| Lung Opacity and Infiltrates | 1469–1576 | 430–513 | 0.458±0.020 | 0.582±0.035 | -0.124±0.039 | O/O/O/O/O | 5/5 |  |  |
| Pleural Effusion Presence | 3994–4186 | 8687–9104 | 0.925±0.005 | 0.863±0.018 | 0.061±0.021 | H/H/H/H/H | 5/5 |  |  |
| Additional Radiologic Findings | 1646–1709 | 1097–1390 | 0.438±0.014 | 0.507±0.029 | -0.070±0.039 | O/O/O/O/O | 5/5 |  |  |
| Aortic Condition | 646–691 | 231–294 | 0.208±0.009 | 0.354±0.022 | -0.146±0.030 | O/O/O/O/O | 5/5 |  |  |
| Pneumothorax Diagnosis | 3844–4015 | 7951–9063 | 0.945±0.006 | 0.926±0.010 | 0.019±0.008 | H/H/H/H/H | 5/5 |  |  |
| Surgical Clip Location | 76–110 | 67–91 | 0.219±0.048 | 0.366±0.088 | -0.146±0.118 | O/O/O/O/O | 5/5 |  |  |
| Lung Lobe Abnormality | 849–972 | 534–604 | 0.406±0.010 | 0.442±0.020 | -0.036±0.020 | O/O/O/O/O | 5/5 |  |  |
| Calcified Granuloma Location | 626–673 | 455–516 | 0.276±0.026 | 0.328±0.011 | -0.052±0.025 | O/O/O/O/O | 5/5 |  |  |
| Cardiac and Mediastinal Contours | 1219–1306 | 318–364 | 0.852±0.013 | 0.600±0.015 | 0.251±0.024 | H/H/H/H/H | 5/5 |  |  |
| Osseous Structure Condition | 1529–1636 | 635–700 | 0.351±0.014 | 0.485±0.006 | -0.133±0.015 | O/O/O/O/O | 5/5 |  |  |
| Focal Airspace Opacity | 2591–2725 | 862–959 | 0.695±0.011 | 0.705±0.025 | -0.009±0.027 | O/O/H/H/O | 3/5 |  |  |
| Lung Clearness and Normalcy | 424–485 | 362–557 | 0.787±0.006 | 0.712±0.039 | 0.075±0.035 | H/H/H/H/H | 5/5 |  |  |
| Lung Volume Condition | 1637–1694 | 996–1136 | 0.366±0.014 | 0.464±0.019 | -0.098±0.028 | O/O/O/O/O | 5/5 |  |  |
| Heart Abnormality Presence | 2153–2210 | 1249–1689 | 0.737±0.009 | 0.801±0.036 | -0.063±0.041 | O/O/O/O/O | 5/5 |  |  |

### A2. Stability groups per model

**CheX+MIMIC**: stable omission 12/20; stable hallucination 5/20; flipping 3/20.
- Stable omission: Lung Condition Assessment; Pleural Fluid Presence; Diaphragm Position Abnormality; Heart Size Normalcy; Additional Radiologic Findings; Aortic Condition; Lung Lobe Abnormality; Osseous Structure Condition; Focal Airspace Opacity; Lung Clearness and Normalcy; Lung Volume Condition; Heart Abnormality Presence
- Stable hallucination: Lung Opacity and Infiltrates; Pleural Effusion Presence; Surgical Clip Location; Calcified Granuloma Location; Cardiac and Mediastinal Contours
- Flipping: Thoracic Spine Degeneration (Δ=0.005, |Δ|=0.005, O2/H3, both_low); Pulmonary Vascular Condition (Δ=-0.018, |Δ|=0.018, O3/H2); Pneumothorax Diagnosis (Δ=-0.000, |Δ|=0.000, O2/H3)

**MIMIC**: stable omission 13/20; stable hallucination 5/20; flipping 2/20.
- Stable omission: Lung Condition Assessment; Pleural Fluid Presence; Diaphragm Position Abnormality; Pulmonary Vascular Condition; Lung Opacity and Infiltrates; Additional Radiologic Findings; Aortic Condition; Pneumothorax Diagnosis; Lung Lobe Abnormality; Calcified Granuloma Location; Focal Airspace Opacity; Lung Volume Condition; Heart Abnormality Presence
- Stable hallucination: Thoracic Spine Degeneration; Heart Size Normalcy; Surgical Clip Location; Cardiac and Mediastinal Contours; Lung Clearness and Normalcy
- Flipping: Pleural Effusion Presence (Δ=-0.003, |Δ|=0.003, O4/H1); Osseous Structure Condition (Δ=0.021, |Δ|=0.021, O2/H3)

**MAIRA-2**: stable omission 12/20; stable hallucination 6/20; flipping 2/20.
- Stable omission: Lung Condition Assessment; Thoracic Spine Degeneration; Diaphragm Position Abnormality; Lung Opacity and Infiltrates; Additional Radiologic Findings; Aortic Condition; Surgical Clip Location; Lung Lobe Abnormality; Calcified Granuloma Location; Osseous Structure Condition; Lung Volume Condition; Heart Abnormality Presence
- Stable hallucination: Pleural Fluid Presence; Heart Size Normalcy; Pleural Effusion Presence; Pneumothorax Diagnosis; Cardiac and Mediastinal Contours; Lung Clearness and Normalcy
- Flipping: Pulmonary Vascular Condition (Δ=-0.023, |Δ|=0.023, O4/H1); Focal Airspace Opacity (Δ=-0.009, |Δ|=0.009, O3/H2)

### A3. Omission vs hallucination share by seed

| Model | seed | % clusters O | % clusters H | size-weighted % O | size-weighted % H |
|---|---|---|---|---|---|
| CheX+MIMIC | 101 | 60 | 40 | 58.9 | 41.1 |
| CheX+MIMIC | 123 | 70 | 30 | 67.6 | 32.4 |
| CheX+MIMIC | 202 | 65 | 35 | 63.7 | 36.3 |
| CheX+MIMIC | 456 | 70 | 30 | 74.6 | 25.4 |
| CheX+MIMIC | 789 | 70 | 30 | 74.0 | 26.0 |
| MAIRA-2 | 101 | 65 | 35 | 58.2 | 41.8 |
| MAIRA-2 | 123 | 70 | 30 | 62.8 | 37.2 |
| MAIRA-2 | 202 | 65 | 35 | 55.3 | 44.7 |
| MAIRA-2 | 456 | 70 | 30 | 62.5 | 37.5 |
| MAIRA-2 | 789 | 65 | 35 | 55.0 | 45.0 |
| MIMIC | 101 | 70 | 30 | 78.3 | 21.7 |
| MIMIC | 123 | 75 | 25 | 82.6 | 17.4 |
| MIMIC | 202 | 70 | 30 | 78.0 | 22.0 |
| MIMIC | 456 | 65 | 35 | 66.4 | 33.6 |
| MIMIC | 789 | 75 | 25 | 82.7 | 17.3 |

### A4. Overall

- **CheX+MIMIC**: 85% clusters stable (80.9% size-weighted); median/max seed-SD(Δ)=0.016/0.048; between-cluster SD of mean Δ=0.127; n_low_n clusters=0.
- **MAIRA-2**: 90% clusters stable (88.7% size-weighted); median/max seed-SD(Δ)=0.027/0.118; between-cluster SD of mean Δ=0.101; n_low_n clusters=0.
- **MIMIC**: 90% clusters stable (83.9% size-weighted); median/max seed-SD(Δ)=0.023/0.085; between-cluster SD of mean Δ=0.237; n_low_n clusters=1.

Report-level Jaccard (same model_seed vs different; gap = same − diff):

| Model | metric | same mean | diff mean | gap | 95% CI |
|---|---|---|---|---|---|
| CheX+MIMIC | gt_reports_as_ref | 0.797 | 0.444 | 0.353 | [0.343, 0.364] |
| CheX+MIMIC | gen_reports_as_ref | 0.776 | 0.559 | 0.217 | [0.204, 0.230] |
| MAIRA-2 | gt_reports_as_ref | 0.790 | 0.447 | 0.342 | [0.331, 0.353] |
| MAIRA-2 | gen_reports_as_ref | 0.729 | 0.638 | 0.091 | [0.076, 0.105] |
| MIMIC | gt_reports_as_ref | 0.792 | 0.447 | 0.345 | [0.334, 0.356] |
| MIMIC | gen_reports_as_ref | 0.742 | 0.480 | 0.261 | [0.247, 0.275] |

Question overlap across the 5 eval seeds: **9,569 / 78,708** unique questions appear in all five seeds (intersection / union of unique Question_Text).

## Part B — Exp 4 (cluster K sensitivity), `model_seeds_all`

For K≠20, clusters are referred to by their best-matching K=20 name (unique-question correspondence share). Fresh KMeans every K including 20 (ARI=1.0 vs stored).

### B1. Per original K=20 cluster across K

#### Lung Condition Assessment (K=20 id 0)

- Matches: K=10→c1 (share=0.99, intact); K=15→c5 (share=0.97, intact); K=25→c0 (share=0.78, intact); K=30→c6 (share=0.85, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c1 | 0.99 | CheX+MIMIC | 0.673 | 0.757 | -0.084 | [-0.123, -0.043] | 40598 | 30017 | O |  |
| 10 | c1 | 0.99 | MIMIC | 0.684 | 0.777 | -0.093 | [-0.134, -0.051] | 40431 | 29981 | O |  |
| 10 | c1 | 0.99 | MAIRA-2 | 0.696 | 0.812 | -0.116 | [-0.147, -0.085] | 40817 | 55063 | O |  |
| 15 | c5 | 0.97 | CheX+MIMIC | 0.744 | 0.816 | -0.072 | [-0.107, -0.033] | 30406 | 23144 | O |  |
| 15 | c5 | 0.97 | MIMIC | 0.761 | 0.839 | -0.078 | [-0.112, -0.042] | 30288 | 23538 | O |  |
| 15 | c5 | 0.97 | MAIRA-2 | 0.771 | 0.846 | -0.074 | [-0.104, -0.044] | 30696 | 48180 | O |  |
| 20 | c0 | 1.00 | CheX+MIMIC | 0.821 | 0.837 | -0.016 | [-0.051, 0.017] | 23753 | 21774 | O |  |
| 20 | c0 | 1.00 | MIMIC | 0.839 | 0.860 | -0.021 | [-0.050, 0.008] | 23581 | 22206 | O |  |
| 20 | c0 | 1.00 | MAIRA-2 | 0.843 | 0.859 | -0.017 | [-0.042, 0.010] | 23998 | 45365 | O |  |
| 25 | c0 | 0.78 | CheX+MIMIC | 0.739 | 0.821 | -0.082 | [-0.120, -0.041] | 25979 | 19661 | O |  |
| 25 | c0 | 0.78 | MIMIC | 0.756 | 0.824 | -0.068 | [-0.108, -0.026] | 26080 | 15261 | O |  |
| 25 | c0 | 0.78 | MAIRA-2 | 0.767 | 0.849 | -0.082 | [-0.115, -0.049] | 26348 | 40949 | O |  |
| 30 | c6 | 0.85 | CheX+MIMIC | 0.838 | 0.840 | -0.002 | [-0.037, 0.034] | 22247 | 20761 | O |  |
| 30 | c6 | 0.85 | MIMIC | 0.855 | 0.865 | -0.010 | [-0.040, 0.021] | 22081 | 20676 | O |  |
| 30 | c6 | 0.85 | MAIRA-2 | 0.861 | 0.868 | -0.007 | [-0.033, 0.019] | 22465 | 44574 | O |  |

#### Thoracic Spine Degeneration (K=20 id 1)

- Matches: K=10→c9 (share=0.99, intact); K=15→c7 (share=0.99, intact); K=25→c1 (share=0.98, intact); K=30→c16 (share=0.66, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c9 | 0.99 | CheX+MIMIC | 0.183 | 0.181 | 0.002 | [-0.124, 0.126] | 7216 | 646 | H | both_low |
| 10 | c9 | 0.99 | MIMIC | 0.170 | 0.173 | -0.002 | [-0.154, 0.080] | 7362 | 897 | O | both_low |
| 10 | c9 | 0.99 | MAIRA-2 | 0.192 | 0.311 | -0.119 | [-0.215, -0.022] | 7359 | 2855 | O |  |
| 15 | c7 | 0.99 | CheX+MIMIC | 0.180 | 0.176 | 0.004 | [-0.122, 0.136] | 7132 | 631 | H | both_low |
| 15 | c7 | 0.99 | MIMIC | 0.165 | 0.064 | 0.101 | [-0.015, 0.166] | 7276 | 590 | H | both_low |
| 15 | c7 | 0.99 | MAIRA-2 | 0.188 | 0.305 | -0.117 | [-0.214, -0.017] | 7269 | 2767 | O |  |
| 20 | c1 | 1.00 | CheX+MIMIC | 0.179 | 0.174 | 0.005 | [-0.119, 0.135] | 7086 | 639 | H | both_low |
| 20 | c1 | 1.00 | MIMIC | 0.165 | 0.065 | 0.100 | [-0.016, 0.165] | 7227 | 589 | H | both_low |
| 20 | c1 | 1.00 | MAIRA-2 | 0.187 | 0.307 | -0.120 | [-0.216, -0.020] | 7224 | 2759 | O |  |
| 25 | c1 | 0.98 | CheX+MIMIC | 0.171 | 0.176 | -0.005 | [-0.141, 0.124] | 6929 | 630 | O | both_low |
| 25 | c1 | 0.98 | MIMIC | 0.164 | 0.054 | 0.110 | [-0.001, 0.177] | 7085 | 570 | H | both_low |
| 25 | c1 | 0.98 | MAIRA-2 | 0.186 | 0.309 | -0.123 | [-0.218, -0.022] | 7077 | 2703 | O |  |
| 30 | c16 | 0.66 | CheX+MIMIC | 0.136 | 0.212 | -0.076 | [-0.209, 0.062] | 4718 | 519 | O | both_low |
| 30 | c16 | 0.66 | MIMIC | 0.133 | 0.054 | 0.079 | [-0.033, 0.147] | 4869 | 569 | H | both_low |
| 30 | c16 | 0.66 | MAIRA-2 | 0.147 | 0.210 | -0.063 | [-0.179, 0.032] | 4863 | 1362 | O | both_low |
- Label changes (CheX+MIMIC): K10:H, K15:H, K20:H, K25:O, K30:O
- Label changes (MIMIC): K10:O, K15:H, K20:H, K25:H, K30:H

#### Pleural Fluid Presence (K=20 id 2)

- Matches: K=10→c8 (share=0.82, intact); K=15→c10 (share=0.81, intact); K=25→c9 (share=0.96, intact); K=30→c1 (share=0.80, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c8 | 0.82 | CheX+MIMIC | 0.886 | 0.829 | 0.057 | [0.027, 0.088] | 23244 | 45430 | H |  |
| 10 | c8 | 0.82 | MIMIC | 0.794 | 0.839 | -0.045 | [-0.084, -0.007] | 23307 | 24928 | O |  |
| 10 | c8 | 0.82 | MAIRA-2 | 0.924 | 0.865 | 0.059 | [0.032, 0.086] | 23166 | 46200 | H |  |
| 15 | c10 | 0.81 | CheX+MIMIC | 0.889 | 0.829 | 0.060 | [0.030, 0.090] | 23148 | 45247 | H |  |
| 15 | c10 | 0.81 | MIMIC | 0.796 | 0.839 | -0.043 | [-0.082, -0.006] | 23192 | 24834 | O |  |
| 15 | c10 | 0.81 | MAIRA-2 | 0.926 | 0.865 | 0.061 | [0.035, 0.087] | 23085 | 45915 | H |  |
| 20 | c2 | 1.00 | CheX+MIMIC | 0.900 | 0.923 | -0.023 | [-0.068, 0.014] | 2855 | 15821 | O |  |
| 20 | c2 | 1.00 | MIMIC | 0.847 | 0.927 | -0.080 | [-0.142, -0.022] | 2896 | 9140 | O |  |
| 20 | c2 | 1.00 | MAIRA-2 | 0.926 | 0.885 | 0.040 | [-0.027, 0.097] | 2787 | 2801 | H |  |
| 25 | c9 | 0.96 | CheX+MIMIC | 0.899 | 0.922 | -0.023 | [-0.071, 0.014] | 2790 | 15919 | O |  |
| 25 | c9 | 0.96 | MIMIC | 0.857 | 0.927 | -0.069 | [-0.121, -0.018] | 2835 | 9151 | O |  |
| 25 | c9 | 0.96 | MAIRA-2 | 0.933 | 0.881 | 0.052 | [0.001, 0.101] | 2737 | 2741 | H |  |
| 30 | c1 | 0.80 | CheX+MIMIC | 0.885 | 0.829 | 0.057 | [0.027, 0.087] | 22881 | 45088 | H |  |
| 30 | c1 | 0.80 | MIMIC | 0.794 | 0.839 | -0.045 | [-0.084, -0.007] | 22914 | 24749 | O |  |
| 30 | c1 | 0.80 | MAIRA-2 | 0.924 | 0.866 | 0.058 | [0.031, 0.085] | 22849 | 45977 | H |  |
- Label changes (CheX+MIMIC): K10:H, K15:H, K20:O, K25:O, K30:H

#### Diaphragm Position Abnormality (K=20 id 3)

- Matches: K=10→c5 (share=0.53, intact); K=15→c0 (share=0.86, intact); K=25→c2 (share=0.45, split; ≥20%: c2:0.45, c6:0.22); K=30→c7 (share=0.55, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c5 | 0.53 | CheX+MIMIC | 0.302 | 0.469 | -0.167 | [-0.259, -0.067] | 8060 | 3000 | O |  |
| 10 | c5 | 0.53 | MIMIC | 0.293 | 0.449 | -0.156 | [-0.218, -0.091] | 8306 | 7429 | O |  |
| 10 | c5 | 0.53 | MAIRA-2 | 0.340 | 0.422 | -0.082 | [-0.165, 0.001] | 8166 | 4722 | O |  |
| 15 | c0 | 0.86 | CheX+MIMIC | 0.261 | 0.332 | -0.071 | [-0.187, 0.038] | 6396 | 3189 | O |  |
| 15 | c0 | 0.86 | MIMIC | 0.277 | 0.366 | -0.089 | [-0.161, -0.015] | 6393 | 8830 | O |  |
| 15 | c0 | 0.86 | MAIRA-2 | 0.264 | 0.308 | -0.044 | [-0.127, 0.039] | 6377 | 5737 | O |  |
| 20 | c3 | 1.00 | CheX+MIMIC | 0.258 | 0.397 | -0.139 | [-0.331, 0.029] | 4728 | 1582 | O |  |
| 20 | c3 | 1.00 | MIMIC | 0.270 | 0.351 | -0.081 | [-0.246, 0.043] | 4775 | 1443 | O |  |
| 20 | c3 | 1.00 | MAIRA-2 | 0.264 | 0.319 | -0.054 | [-0.154, 0.045] | 4739 | 4122 | O |  |
| 25 | c2 | 0.45 | CheX+MIMIC | 0.251 | 0.352 | -0.101 | [-0.319, 0.072] | 2453 | 1577 | O |  |
| 25 | c2 | 0.45 | MIMIC | 0.216 | 0.193 | 0.023 | [-0.078, 0.117] | 2433 | 1972 | H | both_low |
| 25 | c2 | 0.45 | MAIRA-2 | 0.215 | 0.371 | -0.156 | [-0.278, -0.032] | 2422 | 1201 | O |  |
| 30 | c7 | 0.55 | CheX+MIMIC | 0.273 | 0.311 | -0.037 | [-0.294, 0.164] | 3504 | 840 | O |  |
| 30 | c7 | 0.55 | MIMIC | 0.274 | 0.543 | -0.269 | [-0.419, -0.116] | 3519 | 693 | O |  |
| 30 | c7 | 0.55 | MAIRA-2 | 0.285 | 0.284 | 0.001 | [-0.121, 0.127] | 3509 | 4058 | H | both_low |
- Label changes (MIMIC): K10:O, K15:O, K20:O, K25:H, K30:O
- Label changes (MAIRA-2): K10:O, K15:O, K20:O, K25:O, K30:H

#### Pulmonary Vascular Condition (K=20 id 4)

- Matches: K=10→c1 (share=0.90, intact); K=15→c3 (share=0.49, split; ≥20%: c3:0.49, c5:0.43); K=25→c10 (share=0.48, split; ≥20%: c10:0.48, c0:0.44); K=30→c15 (share=0.57, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c1 | 0.90 | CheX+MIMIC | 0.673 | 0.757 | -0.084 | [-0.123, -0.043] | 40598 | 30017 | O |  |
| 10 | c1 | 0.90 | MIMIC | 0.684 | 0.777 | -0.093 | [-0.134, -0.051] | 40431 | 29981 | O |  |
| 10 | c1 | 0.90 | MAIRA-2 | 0.696 | 0.812 | -0.116 | [-0.147, -0.085] | 40817 | 55063 | O |  |
| 15 | c3 | 0.49 | CheX+MIMIC | 0.427 | 0.550 | -0.123 | [-0.217, -0.041] | 10058 | 6924 | O |  |
| 15 | c3 | 0.49 | MIMIC | 0.431 | 0.555 | -0.124 | [-0.225, -0.026] | 10014 | 6481 | O |  |
| 15 | c3 | 0.49 | MAIRA-2 | 0.430 | 0.529 | -0.099 | [-0.166, -0.031] | 9959 | 6290 | O |  |
| 20 | c4 | 1.00 | CheX+MIMIC | 0.457 | 0.474 | -0.018 | [-0.132, 0.088] | 6771 | 2972 | O |  |
| 20 | c4 | 1.00 | MIMIC | 0.467 | 0.537 | -0.070 | [-0.200, 0.059] | 6723 | 2589 | O |  |
| 20 | c4 | 1.00 | MAIRA-2 | 0.474 | 0.496 | -0.023 | [-0.143, 0.085] | 6722 | 1181 | O |  |
| 25 | c10 | 0.48 | CheX+MIMIC | 0.761 | 0.539 | 0.222 | [0.094, 0.345] | 1534 | 2167 | H |  |
| 25 | c10 | 0.48 | MIMIC | 0.749 | 0.618 | 0.131 | [-0.048, 0.306] | 1567 | 1512 | H |  |
| 25 | c10 | 0.48 | MAIRA-2 | 0.732 | 0.602 | 0.130 | [-0.101, 0.307] | 1491 | 294 | H |  |
| 30 | c15 | 0.57 | CheX+MIMIC | 0.414 | 0.345 | 0.069 | [-0.111, 0.221] | 6106 | 1153 | H |  |
| 30 | c15 | 0.57 | MIMIC | 0.427 | 0.478 | -0.051 | [-0.232, 0.134] | 5985 | 1458 | O |  |
| 30 | c15 | 0.57 | MAIRA-2 | 0.437 | 0.447 | -0.009 | [-0.126, 0.110] | 6077 | 1048 | O |  |
- Label changes (CheX+MIMIC): K10:O, K15:O, K20:O, K25:H, K30:H
- Label changes (MIMIC): K10:O, K15:O, K20:O, K25:H, K30:O
- Label changes (MAIRA-2): K10:O, K15:O, K20:O, K25:H, K30:O

#### Heart Size Normalcy (K=20 id 5)

- Matches: K=10→c6 (share=0.98, intact); K=15→c1 (share=0.85, intact); K=25→c5 (share=0.81, intact); K=30→c21 (share=0.79, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c6 | 0.98 | CheX+MIMIC | 0.747 | 0.804 | -0.057 | [-0.103, -0.005] | 24912 | 16954 | O |  |
| 10 | c6 | 0.98 | MIMIC | 0.779 | 0.775 | 0.004 | [-0.056, 0.060] | 24661 | 13455 | H |  |
| 10 | c6 | 0.98 | MAIRA-2 | 0.746 | 0.729 | 0.016 | [-0.046, 0.080] | 24845 | 10696 | H |  |
| 15 | c1 | 0.85 | CheX+MIMIC | 0.721 | 0.797 | -0.077 | [-0.143, -0.008] | 13596 | 9350 | O |  |
| 15 | c1 | 0.85 | MIMIC | 0.767 | 0.618 | 0.149 | [0.037, 0.263] | 13404 | 5088 | H |  |
| 15 | c1 | 0.85 | MAIRA-2 | 0.742 | 0.623 | 0.119 | [-0.049, 0.314] | 13499 | 1085 | H |  |
| 20 | c5 | 1.00 | CheX+MIMIC | 0.731 | 0.808 | -0.076 | [-0.139, -0.012] | 14561 | 11966 | O |  |
| 20 | c5 | 1.00 | MIMIC | 0.774 | 0.745 | 0.030 | [-0.055, 0.109] | 14375 | 8215 | H |  |
| 20 | c5 | 1.00 | MAIRA-2 | 0.747 | 0.616 | 0.131 | [0.015, 0.238] | 14449 | 3816 | H |  |
| 25 | c5 | 0.81 | CheX+MIMIC | 0.726 | 0.797 | -0.071 | [-0.138, 0.001] | 12871 | 9264 | O |  |
| 25 | c5 | 0.81 | MIMIC | 0.769 | 0.650 | 0.120 | [0.003, 0.240] | 12698 | 4666 | H |  |
| 25 | c5 | 0.81 | MAIRA-2 | 0.740 | 0.733 | 0.006 | [-0.126, 0.193] | 12811 | 738 | H |  |
| 30 | c21 | 0.79 | CheX+MIMIC | 0.730 | 0.798 | -0.068 | [-0.137, 0.005] | 12274 | 9252 | O |  |
| 30 | c21 | 0.79 | MIMIC | 0.772 | 0.652 | 0.120 | [0.002, 0.239] | 12139 | 4644 | H |  |
| 30 | c21 | 0.79 | MAIRA-2 | 0.744 | 0.740 | 0.004 | [-0.129, 0.197] | 12231 | 719 | H |  |

#### Lung Opacity and Infiltrates (K=20 id 6)

- Matches: K=10→c3 (share=0.46, split; ≥20%: c3:0.46, c7:0.43); K=15→c4 (share=0.43, split; ≥20%: c4:0.43, c6:0.30); K=25→c11 (share=0.46, split; ≥20%: c11:0.46, c18:0.39); K=30→c28 (share=0.37, split; ≥20%: c28:0.37, c17:0.31)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c3 | 0.46 | CheX+MIMIC | 0.358 | 0.453 | -0.095 | [-0.152, -0.041] | 23064 | 12254 | O |  |
| 10 | c3 | 0.46 | MIMIC | 0.422 | 0.548 | -0.126 | [-0.174, -0.080] | 22765 | 23849 | O |  |
| 10 | c3 | 0.46 | MAIRA-2 | 0.370 | 0.459 | -0.089 | [-0.148, -0.035] | 22918 | 15216 | O |  |
| 15 | c4 | 0.43 | CheX+MIMIC | 0.407 | 0.521 | -0.114 | [-0.212, -0.013] | 10996 | 2008 | O |  |
| 15 | c4 | 0.43 | MIMIC | 0.423 | 0.676 | -0.253 | [-0.322, -0.184] | 10966 | 3116 | O |  |
| 15 | c4 | 0.43 | MAIRA-2 | 0.449 | 0.628 | -0.179 | [-0.249, -0.110] | 11000 | 7760 | O |  |
| 20 | c6 | 1.00 | CheX+MIMIC | 0.485 | 0.462 | 0.023 | [-0.083, 0.126] | 7647 | 2361 | H |  |
| 20 | c6 | 1.00 | MIMIC | 0.472 | 0.595 | -0.123 | [-0.212, -0.040] | 7699 | 2676 | O |  |
| 20 | c6 | 1.00 | MAIRA-2 | 0.458 | 0.583 | -0.125 | [-0.254, 0.004] | 7586 | 2391 | O |  |
| 25 | c11 | 0.46 | CheX+MIMIC | 0.281 | 0.288 | -0.007 | [-0.107, 0.095] | 5081 | 1819 | O | both_low |
| 25 | c11 | 0.46 | MIMIC | 0.257 | 0.264 | -0.007 | [-0.106, 0.098] | 5101 | 1113 | O | both_low |
| 25 | c11 | 0.46 | MAIRA-2 | 0.308 | 0.450 | -0.142 | [-0.284, -0.002] | 5067 | 1993 | O |  |
| 30 | c28 | 0.37 | CheX+MIMIC | 0.380 | 0.253 | 0.127 | [-0.440, 0.348] | 3393 | 190 | H |  |
| 30 | c28 | 0.37 | MIMIC | 0.375 | 0.663 | -0.289 | [-0.607, -0.059] | 3356 | 95 | O |  |
| 30 | c28 | 0.37 | MAIRA-2 | 0.381 | 0.537 | -0.156 | [-0.281, -0.039] | 3368 | 2322 | O |  |
- Label changes (CheX+MIMIC): K10:O, K15:O, K20:H, K25:O, K30:H

#### Pleural Effusion Presence (K=20 id 7)

- Matches: K=10→c8 (share=1.00, intact); K=15→c10 (share=1.00, intact); K=25→c20 (share=0.97, intact); K=30→c1 (share=0.99, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c8 | 1.00 | CheX+MIMIC | 0.886 | 0.829 | 0.057 | [0.027, 0.088] | 23244 | 45430 | H |  |
| 10 | c8 | 1.00 | MIMIC | 0.794 | 0.839 | -0.045 | [-0.084, -0.007] | 23307 | 24928 | O |  |
| 10 | c8 | 1.00 | MAIRA-2 | 0.924 | 0.865 | 0.059 | [0.032, 0.086] | 23166 | 46200 | H |  |
| 15 | c10 | 1.00 | CheX+MIMIC | 0.889 | 0.829 | 0.060 | [0.030, 0.090] | 23148 | 45247 | H |  |
| 15 | c10 | 1.00 | MIMIC | 0.796 | 0.839 | -0.043 | [-0.082, -0.006] | 23192 | 24834 | O |  |
| 15 | c10 | 1.00 | MAIRA-2 | 0.926 | 0.865 | 0.061 | [0.035, 0.087] | 23085 | 45915 | H |  |
| 20 | c7 | 1.00 | CheX+MIMIC | 0.886 | 0.781 | 0.105 | [0.069, 0.141] | 20558 | 30311 | H |  |
| 20 | c7 | 1.00 | MIMIC | 0.789 | 0.792 | -0.003 | [-0.049, 0.042] | 20562 | 16187 | O |  |
| 20 | c7 | 1.00 | MAIRA-2 | 0.925 | 0.863 | 0.061 | [0.034, 0.089] | 20591 | 44212 | H |  |
| 25 | c20 | 0.97 | CheX+MIMIC | 0.886 | 0.781 | 0.104 | [0.068, 0.140] | 20298 | 29976 | H |  |
| 25 | c20 | 0.97 | MIMIC | 0.788 | 0.795 | -0.007 | [-0.052, 0.037] | 20335 | 16096 | O |  |
| 25 | c20 | 0.97 | MAIRA-2 | 0.924 | 0.863 | 0.061 | [0.034, 0.089] | 20350 | 44407 | H |  |
| 30 | c1 | 0.99 | CheX+MIMIC | 0.885 | 0.829 | 0.057 | [0.027, 0.087] | 22881 | 45088 | H |  |
| 30 | c1 | 0.99 | MIMIC | 0.794 | 0.839 | -0.045 | [-0.084, -0.007] | 22914 | 24749 | O |  |
| 30 | c1 | 0.99 | MAIRA-2 | 0.924 | 0.866 | 0.058 | [0.031, 0.085] | 22849 | 45977 | H |  |

#### Additional Radiologic Findings (K=20 id 8)

- Matches: K=10→c3 (share=0.97, intact); K=15→c4 (share=0.85, intact); K=25→c3 (share=0.80, intact); K=30→c23 (share=0.77, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c3 | 0.97 | CheX+MIMIC | 0.358 | 0.453 | -0.095 | [-0.152, -0.041] | 23064 | 12254 | O |  |
| 10 | c3 | 0.97 | MIMIC | 0.422 | 0.548 | -0.126 | [-0.174, -0.080] | 22765 | 23849 | O |  |
| 10 | c3 | 0.97 | MAIRA-2 | 0.370 | 0.459 | -0.089 | [-0.148, -0.035] | 22918 | 15216 | O |  |
| 15 | c4 | 0.85 | CheX+MIMIC | 0.407 | 0.521 | -0.114 | [-0.212, -0.013] | 10996 | 2008 | O |  |
| 15 | c4 | 0.85 | MIMIC | 0.423 | 0.676 | -0.253 | [-0.322, -0.184] | 10966 | 3116 | O |  |
| 15 | c4 | 0.85 | MAIRA-2 | 0.449 | 0.628 | -0.179 | [-0.249, -0.110] | 11000 | 7760 | O |  |
| 20 | c8 | 1.00 | CheX+MIMIC | 0.386 | 0.553 | -0.166 | [-0.262, -0.066] | 8374 | 1776 | O |  |
| 20 | c8 | 1.00 | MIMIC | 0.418 | 0.576 | -0.158 | [-0.246, -0.069] | 8258 | 2530 | O |  |
| 20 | c8 | 1.00 | MAIRA-2 | 0.438 | 0.509 | -0.071 | [-0.147, 0.001] | 8427 | 6323 | O |  |
| 25 | c3 | 0.80 | CheX+MIMIC | 0.382 | 0.502 | -0.119 | [-0.239, -0.004] | 7556 | 1658 | O |  |
| 25 | c3 | 0.80 | MIMIC | 0.407 | 0.502 | -0.095 | [-0.192, 0.008] | 7476 | 1712 | O |  |
| 25 | c3 | 0.80 | MAIRA-2 | 0.430 | 0.625 | -0.195 | [-0.277, -0.116] | 7587 | 4595 | O |  |
| 30 | c23 | 0.77 | CheX+MIMIC | 0.400 | 0.539 | -0.139 | [-0.253, -0.026] | 7006 | 1538 | O |  |
| 30 | c23 | 0.77 | MIMIC | 0.429 | 0.519 | -0.090 | [-0.179, 0.001] | 6867 | 1618 | O |  |
| 30 | c23 | 0.77 | MAIRA-2 | 0.444 | 0.629 | -0.185 | [-0.280, -0.097] | 7034 | 4445 | O |  |

#### Aortic Condition (K=20 id 9)

- Matches: K=10→c0 (share=1.00, intact); K=15→c14 (share=0.99, intact); K=25→c8 (share=0.99, intact); K=30→c11 (share=0.98, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c0 | 1.00 | CheX+MIMIC | 0.174 | 0.273 | -0.099 | [-0.317, 0.063] | 3634 | 1070 | O | both_low |
| 10 | c0 | 1.00 | MIMIC | 0.197 | 0.474 | -0.277 | [-0.499, -0.069] | 3687 | 858 | O |  |
| 10 | c0 | 1.00 | MAIRA-2 | 0.219 | 0.338 | -0.119 | [-0.241, -0.010] | 3713 | 2089 | O |  |
| 15 | c14 | 0.99 | CheX+MIMIC | 0.166 | 0.246 | -0.080 | [-0.299, 0.080] | 3514 | 1021 | O | both_low |
| 15 | c14 | 0.99 | MIMIC | 0.199 | 0.475 | -0.276 | [-0.501, -0.065] | 3568 | 842 | O |  |
| 15 | c14 | 0.99 | MAIRA-2 | 0.222 | 0.341 | -0.119 | [-0.243, -0.007] | 3593 | 2025 | O |  |
| 20 | c9 | 1.00 | CheX+MIMIC | 0.150 | 0.255 | -0.105 | [-0.331, 0.065] | 3312 | 976 | O | both_low |
| 20 | c9 | 1.00 | MIMIC | 0.190 | 0.439 | -0.249 | [-0.474, -0.051] | 3383 | 798 | O |  |
| 20 | c9 | 1.00 | MAIRA-2 | 0.208 | 0.354 | -0.146 | [-0.306, -0.009] | 3367 | 1300 | O |  |
| 25 | c8 | 0.99 | CheX+MIMIC | 0.165 | 0.246 | -0.081 | [-0.299, 0.078] | 3424 | 1021 | O | both_low |
| 25 | c8 | 0.99 | MIMIC | 0.197 | 0.472 | -0.275 | [-0.496, -0.063] | 3491 | 848 | O |  |
| 25 | c8 | 0.99 | MAIRA-2 | 0.223 | 0.339 | -0.116 | [-0.240, -0.003] | 3500 | 2022 | O |  |
| 30 | c11 | 0.98 | CheX+MIMIC | 0.152 | 0.257 | -0.104 | [-0.328, 0.065] | 3299 | 978 | O | both_low |
| 30 | c11 | 0.98 | MIMIC | 0.192 | 0.428 | -0.236 | [-0.461, -0.044] | 3367 | 776 | O |  |
| 30 | c11 | 0.98 | MAIRA-2 | 0.209 | 0.354 | -0.144 | [-0.307, -0.002] | 3348 | 1252 | O |  |

#### Pneumothorax Diagnosis (K=20 id 10)

- Matches: K=10→c2 (share=0.99, intact); K=15→c2 (share=0.98, intact); K=25→c12 (share=0.98, intact); K=30→c9 (share=0.98, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c2 | 0.99 | CheX+MIMIC | 0.915 | 0.925 | -0.010 | [-0.040, 0.017] | 20616 | 22155 | O |  |
| 10 | c2 | 0.99 | MIMIC | 0.887 | 0.919 | -0.031 | [-0.061, -0.004] | 20351 | 17493 | O |  |
| 10 | c2 | 0.99 | MAIRA-2 | 0.927 | 0.926 | 0.001 | [-0.025, 0.026] | 20551 | 43019 | H |  |
| 15 | c2 | 0.98 | CheX+MIMIC | 0.934 | 0.928 | 0.006 | [-0.019, 0.030] | 19805 | 22142 | H |  |
| 15 | c2 | 0.98 | MIMIC | 0.905 | 0.924 | -0.019 | [-0.046, 0.006] | 19536 | 17321 | O |  |
| 15 | c2 | 0.98 | MAIRA-2 | 0.952 | 0.926 | 0.026 | [0.005, 0.046] | 19720 | 42913 | H |  |
| 20 | c10 | 1.00 | CheX+MIMIC | 0.928 | 0.928 | -0.000 | [-0.026, 0.024] | 19872 | 22202 | O |  |
| 20 | c10 | 1.00 | MIMIC | 0.899 | 0.923 | -0.025 | [-0.052, 0.001] | 19593 | 17326 | O |  |
| 20 | c10 | 1.00 | MAIRA-2 | 0.945 | 0.926 | 0.019 | [-0.003, 0.040] | 19772 | 42859 | H |  |
| 25 | c12 | 0.98 | CheX+MIMIC | 0.934 | 0.929 | 0.005 | [-0.020, 0.029] | 19767 | 22174 | H |  |
| 25 | c12 | 0.98 | MIMIC | 0.905 | 0.923 | -0.019 | [-0.046, 0.006] | 19505 | 17320 | O |  |
| 25 | c12 | 0.98 | MAIRA-2 | 0.952 | 0.926 | 0.026 | [0.005, 0.046] | 19689 | 42900 | H |  |
| 30 | c9 | 0.98 | CheX+MIMIC | 0.930 | 0.929 | 0.001 | [-0.024, 0.024] | 19833 | 22080 | H |  |
| 30 | c9 | 0.98 | MIMIC | 0.900 | 0.924 | -0.023 | [-0.051, 0.002] | 19556 | 17304 | O |  |
| 30 | c9 | 0.98 | MAIRA-2 | 0.947 | 0.926 | 0.021 | [-0.000, 0.042] | 19757 | 42843 | H |  |
- Label changes (CheX+MIMIC): K10:O, K15:H, K20:O, K25:H, K30:H

#### Surgical Clip Location (K=20 id 11)

- Matches: K=10→c3 (share=0.73, intact); K=15→c0 (share=0.97, intact); K=25→c2 (share=0.95, intact); K=30→c18 (share=1.00, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c3 | 0.73 | CheX+MIMIC | 0.358 | 0.453 | -0.095 | [-0.152, -0.041] | 23064 | 12254 | O |  |
| 10 | c3 | 0.73 | MIMIC | 0.422 | 0.548 | -0.126 | [-0.174, -0.080] | 22765 | 23849 | O |  |
| 10 | c3 | 0.73 | MAIRA-2 | 0.370 | 0.459 | -0.089 | [-0.148, -0.035] | 22918 | 15216 | O |  |
| 15 | c0 | 0.97 | CheX+MIMIC | 0.261 | 0.332 | -0.071 | [-0.187, 0.038] | 6396 | 3189 | O |  |
| 15 | c0 | 0.97 | MIMIC | 0.277 | 0.366 | -0.089 | [-0.161, -0.015] | 6393 | 8830 | O |  |
| 15 | c0 | 0.97 | MAIRA-2 | 0.264 | 0.308 | -0.044 | [-0.127, 0.039] | 6377 | 5737 | O |  |
| 20 | c11 | 1.00 | CheX+MIMIC | 0.216 | 0.054 | 0.162 | [0.024, 0.271] | 472 | 443 | H | both_low |
| 20 | c11 | 1.00 | MIMIC | 0.218 | 0.084 | 0.134 | [-0.001, 0.261] | 482 | 1068 | H | both_low |
| 20 | c11 | 1.00 | MAIRA-2 | 0.221 | 0.370 | -0.149 | [-0.389, 0.081] | 438 | 386 | O |  |
| 25 | c2 | 0.95 | CheX+MIMIC | 0.251 | 0.352 | -0.101 | [-0.319, 0.072] | 2453 | 1577 | O |  |
| 25 | c2 | 0.95 | MIMIC | 0.216 | 0.193 | 0.023 | [-0.078, 0.117] | 2433 | 1972 | H | both_low |
| 25 | c2 | 0.95 | MAIRA-2 | 0.215 | 0.371 | -0.156 | [-0.278, -0.032] | 2422 | 1201 | O |  |
| 30 | c18 | 1.00 | CheX+MIMIC | 0.216 | 0.054 | 0.162 | [0.024, 0.271] | 472 | 443 | H | both_low |
| 30 | c18 | 1.00 | MIMIC | 0.218 | 0.084 | 0.134 | [-0.001, 0.261] | 482 | 1068 | H | both_low |
| 30 | c18 | 1.00 | MAIRA-2 | 0.221 | 0.370 | -0.149 | [-0.389, 0.081] | 438 | 386 | O |  |
- Label changes (CheX+MIMIC): K10:O, K15:O, K20:H, K25:O, K30:H
- Label changes (MIMIC): K10:O, K15:O, K20:H, K25:H, K30:H

#### Lung Lobe Abnormality (K=20 id 12)

- Matches: K=10→c5 (share=0.85, intact); K=15→c9 (share=0.98, intact); K=25→c14 (share=0.80, intact); K=30→c5 (share=0.80, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c5 | 0.85 | CheX+MIMIC | 0.302 | 0.469 | -0.167 | [-0.259, -0.067] | 8060 | 3000 | O |  |
| 10 | c5 | 0.85 | MIMIC | 0.293 | 0.449 | -0.156 | [-0.218, -0.091] | 8306 | 7429 | O |  |
| 10 | c5 | 0.85 | MAIRA-2 | 0.340 | 0.422 | -0.082 | [-0.165, 0.001] | 8166 | 4722 | O |  |
| 15 | c9 | 0.98 | CheX+MIMIC | 0.324 | 0.656 | -0.332 | [-0.451, -0.182] | 5068 | 1439 | O |  |
| 15 | c9 | 0.98 | MIMIC | 0.313 | 0.637 | -0.324 | [-0.437, -0.186] | 5230 | 1636 | O |  |
| 15 | c9 | 0.98 | MAIRA-2 | 0.403 | 0.451 | -0.048 | [-0.150, 0.047] | 5147 | 3360 | O |  |
| 20 | c12 | 1.00 | CheX+MIMIC | 0.330 | 0.681 | -0.352 | [-0.477, -0.199] | 4530 | 1321 | O |  |
| 20 | c12 | 1.00 | MIMIC | 0.304 | 0.635 | -0.331 | [-0.441, -0.198] | 4679 | 1541 | O |  |
| 20 | c12 | 1.00 | MAIRA-2 | 0.406 | 0.442 | -0.036 | [-0.136, 0.060] | 4616 | 2916 | O |  |
| 25 | c14 | 0.80 | CheX+MIMIC | 0.336 | 0.698 | -0.363 | [-0.488, -0.208] | 3689 | 1114 | O |  |
| 25 | c14 | 0.80 | MIMIC | 0.317 | 0.607 | -0.290 | [-0.401, -0.142] | 3806 | 1089 | O |  |
| 25 | c14 | 0.80 | MAIRA-2 | 0.418 | 0.464 | -0.046 | [-0.147, 0.059] | 3753 | 2746 | O |  |
| 30 | c5 | 0.80 | CheX+MIMIC | 0.364 | 0.690 | -0.326 | [-0.463, -0.162] | 3507 | 1254 | O |  |
| 30 | c5 | 0.80 | MIMIC | 0.346 | 0.608 | -0.262 | [-0.384, -0.109] | 3626 | 1332 | O |  |
| 30 | c5 | 0.80 | MAIRA-2 | 0.439 | 0.433 | 0.006 | [-0.094, 0.100] | 3591 | 2787 | H |  |
- Label changes (MAIRA-2): K10:O, K15:O, K20:O, K25:O, K30:H

#### Calcified Granuloma Location (K=20 id 13)

- Matches: K=10→c3 (share=0.59, intact); K=15→c13 (share=0.91, intact); K=25→c4 (share=0.90, intact); K=30→c13 (share=0.74, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c3 | 0.59 | CheX+MIMIC | 0.358 | 0.453 | -0.095 | [-0.152, -0.041] | 23064 | 12254 | O |  |
| 10 | c3 | 0.59 | MIMIC | 0.422 | 0.548 | -0.126 | [-0.174, -0.080] | 22765 | 23849 | O |  |
| 10 | c3 | 0.59 | MAIRA-2 | 0.370 | 0.459 | -0.089 | [-0.148, -0.035] | 22918 | 15216 | O |  |
| 15 | c13 | 0.91 | CheX+MIMIC | 0.162 | 0.057 | 0.105 | [-0.837, 0.173] | 4008 | 228 | H | both_low |
| 15 | c13 | 0.91 | MIMIC | 0.154 | 0.969 | -0.815 | [-0.877, -0.537] | 4019 | 32 | O |  |
| 15 | c13 | 0.91 | MAIRA-2 | 0.277 | 0.364 | -0.087 | [-0.210, 0.030] | 4057 | 2212 | O |  |
| 20 | c13 | 1.00 | CheX+MIMIC | 0.140 | 0.030 | 0.110 | [0.042, 0.175] | 3131 | 267 | H | both_low |
| 20 | c13 | 1.00 | MIMIC | 0.132 | 0.986 | -0.854 | [-0.892, -0.788] | 3159 | 74 | O |  |
| 20 | c13 | 1.00 | MAIRA-2 | 0.275 | 0.328 | -0.053 | [-0.165, 0.058] | 3251 | 2393 | O |  |
| 25 | c4 | 0.90 | CheX+MIMIC | 0.157 | 0.027 | 0.130 | [0.011, 0.184] | 3787 | 221 | H | both_low |
| 25 | c4 | 0.90 | MIMIC | 0.146 | 1.000 | -0.854 | [-0.890, -0.803] | 3825 | 23 | O | low_n |
| 25 | c4 | 0.90 | MAIRA-2 | 0.280 | 0.364 | -0.085 | [-0.213, 0.037] | 3856 | 2079 | O |  |
| 30 | c13 | 0.74 | CheX+MIMIC | 0.130 | 0.040 | 0.090 | [0.030, 0.160] | 2483 | 199 | H | both_low |
| 30 | c13 | 0.74 | MIMIC | 0.119 | 0.989 | -0.870 | [-0.922, -0.799] | 2500 | 90 | O |  |
| 30 | c13 | 0.74 | MAIRA-2 | 0.282 | 0.337 | -0.055 | [-0.177, 0.070] | 2578 | 2164 | O |  |
- Label changes (CheX+MIMIC): K10:O, K15:H, K20:H, K25:H, K30:H

#### Cardiac and Mediastinal Contours (K=20 id 14)

- Matches: K=10→c4 (share=0.99, intact); K=15→c11 (share=0.86, intact); K=25→c19 (share=0.95, intact); K=30→c19 (share=0.84, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c4 | 0.99 | CheX+MIMIC | 0.873 | 0.662 | 0.212 | [0.145, 0.279] | 6719 | 6005 | H |  |
| 10 | c4 | 0.99 | MIMIC | 0.861 | 0.613 | 0.248 | [0.166, 0.334] | 6562 | 4394 | H |  |
| 10 | c4 | 0.99 | MAIRA-2 | 0.826 | 0.552 | 0.274 | [0.140, 0.413] | 6630 | 2090 | H |  |
| 15 | c11 | 0.86 | CheX+MIMIC | 0.824 | 0.661 | 0.164 | [0.108, 0.222] | 13876 | 7026 | H |  |
| 15 | c11 | 0.86 | MIMIC | 0.826 | 0.680 | 0.146 | [0.090, 0.206] | 13702 | 5268 | H |  |
| 15 | c11 | 0.86 | MAIRA-2 | 0.779 | 0.657 | 0.121 | [0.032, 0.227] | 13900 | 3433 | H |  |
| 20 | c14 | 1.00 | CheX+MIMIC | 0.898 | 0.657 | 0.241 | [0.176, 0.309] | 6364 | 5860 | H |  |
| 20 | c14 | 1.00 | MIMIC | 0.886 | 0.628 | 0.258 | [0.176, 0.343] | 6184 | 4172 | H |  |
| 20 | c14 | 1.00 | MAIRA-2 | 0.852 | 0.600 | 0.252 | [0.114, 0.395] | 6264 | 1688 | H |  |
| 25 | c19 | 0.95 | CheX+MIMIC | 0.891 | 0.653 | 0.238 | [0.172, 0.308] | 6179 | 5833 | H |  |
| 25 | c19 | 0.95 | MIMIC | 0.881 | 0.622 | 0.258 | [0.174, 0.346] | 6003 | 4083 | H |  |
| 25 | c19 | 0.95 | MAIRA-2 | 0.845 | 0.611 | 0.234 | [0.076, 0.394] | 6072 | 1631 | H |  |
| 30 | c19 | 0.84 | CheX+MIMIC | 0.897 | 0.676 | 0.222 | [0.153, 0.294] | 6036 | 3612 | H |  |
| 30 | c19 | 0.84 | MIMIC | 0.889 | 0.643 | 0.246 | [0.158, 0.339] | 5856 | 3362 | H |  |
| 30 | c19 | 0.84 | MAIRA-2 | 0.853 | 0.631 | 0.222 | [0.070, 0.386] | 5927 | 1332 | H |  |

#### Osseous Structure Condition (K=20 id 15)

- Matches: K=10→c3 (share=0.90, intact); K=15→c12 (share=0.99, intact); K=25→c21 (share=0.61, intact); K=30→c25 (share=0.65, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c3 | 0.90 | CheX+MIMIC | 0.358 | 0.453 | -0.095 | [-0.152, -0.041] | 23064 | 12254 | O |  |
| 10 | c3 | 0.90 | MIMIC | 0.422 | 0.548 | -0.126 | [-0.174, -0.080] | 22765 | 23849 | O |  |
| 10 | c3 | 0.90 | MAIRA-2 | 0.370 | 0.459 | -0.089 | [-0.148, -0.035] | 22918 | 15216 | O |  |
| 15 | c12 | 0.99 | CheX+MIMIC | 0.417 | 0.465 | -0.048 | [-0.121, 0.026] | 8002 | 8714 | O |  |
| 15 | c12 | 0.99 | MIMIC | 0.578 | 0.560 | 0.017 | [-0.055, 0.095] | 7915 | 18542 | H |  |
| 15 | c12 | 0.99 | MAIRA-2 | 0.349 | 0.487 | -0.138 | [-0.244, -0.031] | 7936 | 3378 | O |  |
| 20 | c15 | 1.00 | CheX+MIMIC | 0.419 | 0.465 | -0.047 | [-0.120, 0.028] | 7988 | 8745 | O |  |
| 20 | c15 | 1.00 | MIMIC | 0.581 | 0.561 | 0.020 | [-0.052, 0.099] | 7891 | 18546 | H |  |
| 20 | c15 | 1.00 | MAIRA-2 | 0.352 | 0.485 | -0.133 | [-0.237, -0.027] | 7924 | 3319 | O |  |
| 25 | c21 | 0.61 | CheX+MIMIC | 0.359 | 0.446 | -0.087 | [-0.178, -0.003] | 5370 | 6783 | O |  |
| 25 | c21 | 0.61 | MIMIC | 0.557 | 0.558 | -0.001 | [-0.098, 0.108] | 5315 | 8074 | O |  |
| 25 | c21 | 0.61 | MAIRA-2 | 0.311 | 0.403 | -0.092 | [-0.227, 0.029] | 5311 | 3406 | O |  |
| 30 | c25 | 0.65 | CheX+MIMIC | 0.370 | 0.442 | -0.072 | [-0.162, 0.014] | 5674 | 6965 | O |  |
| 30 | c25 | 0.65 | MIMIC | 0.565 | 0.561 | 0.005 | [-0.085, 0.109] | 5634 | 8296 | H |  |
| 30 | c25 | 0.65 | MAIRA-2 | 0.315 | 0.432 | -0.116 | [-0.246, 0.002] | 5644 | 3305 | O |  |
- Label changes (MIMIC): K10:O, K15:H, K20:H, K25:O, K30:H

#### Focal Airspace Opacity (K=20 id 16)

- Matches: K=10→c7 (share=0.95, intact); K=15→c6 (share=0.91, intact); K=25→c22 (share=0.90, intact); K=30→c8 (share=0.42, split; ≥20%: c8:0.42, c27:0.33, c22:0.22)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c7 | 0.95 | CheX+MIMIC | 0.640 | 0.776 | -0.136 | [-0.189, -0.084] | 18835 | 9510 | O |  |
| 10 | c7 | 0.95 | MIMIC | 0.652 | 0.792 | -0.141 | [-0.191, -0.090] | 18915 | 5355 | O |  |
| 10 | c7 | 0.95 | MAIRA-2 | 0.626 | 0.650 | -0.024 | [-0.099, 0.051] | 18778 | 5585 | O |  |
| 15 | c6 | 0.91 | CheX+MIMIC | 0.708 | 0.795 | -0.087 | [-0.136, -0.040] | 16082 | 9148 | O |  |
| 15 | c6 | 0.91 | MIMIC | 0.715 | 0.792 | -0.077 | [-0.128, -0.027] | 16109 | 5090 | O |  |
| 15 | c6 | 0.91 | MAIRA-2 | 0.677 | 0.705 | -0.028 | [-0.098, 0.044] | 15973 | 4275 | O |  |
| 20 | c16 | 1.00 | CheX+MIMIC | 0.714 | 0.800 | -0.086 | [-0.137, -0.033] | 13458 | 7917 | O |  |
| 20 | c16 | 1.00 | MIMIC | 0.723 | 0.813 | -0.090 | [-0.145, -0.038] | 13452 | 4117 | O |  |
| 20 | c16 | 1.00 | MAIRA-2 | 0.696 | 0.705 | -0.009 | [-0.084, 0.069] | 13335 | 4570 | O |  |
| 25 | c22 | 0.90 | CheX+MIMIC | 0.739 | 0.815 | -0.076 | [-0.128, -0.021] | 12925 | 8024 | O |  |
| 25 | c22 | 0.90 | MIMIC | 0.747 | 0.814 | -0.067 | [-0.126, -0.011] | 12965 | 4028 | O |  |
| 25 | c22 | 0.90 | MAIRA-2 | 0.716 | 0.719 | -0.003 | [-0.080, 0.079] | 12799 | 4263 | O |  |
| 30 | c8 | 0.42 | CheX+MIMIC | 0.545 | 0.761 | -0.216 | [-0.298, -0.132] | 5874 | 2783 | O |  |
| 30 | c8 | 0.42 | MIMIC | 0.566 | 0.748 | -0.181 | [-0.289, -0.084] | 5958 | 884 | O |  |
| 30 | c8 | 0.42 | MAIRA-2 | 0.562 | 0.610 | -0.048 | [-0.143, 0.043] | 5873 | 2976 | O |  |

#### Lung Clearness and Normalcy (K=20 id 17)

- Matches: K=10→c1 (share=0.47, merged; ≥20%: c1:0.47); K=15→c5 (share=0.41, split; ≥20%: c5:0.41, c0:0.26); K=25→c0 (share=0.37, split; ≥20%: c0:0.37, c6:0.29); K=30→c29 (share=0.98, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c1 | 0.47 | CheX+MIMIC | 0.673 | 0.757 | -0.084 | [-0.123, -0.043] | 40598 | 30017 | O |  |
| 10 | c1 | 0.47 | MIMIC | 0.684 | 0.777 | -0.093 | [-0.134, -0.051] | 40431 | 29981 | O |  |
| 10 | c1 | 0.47 | MAIRA-2 | 0.696 | 0.812 | -0.116 | [-0.147, -0.085] | 40817 | 55063 | O |  |
| 15 | c5 | 0.41 | CheX+MIMIC | 0.744 | 0.816 | -0.072 | [-0.107, -0.033] | 30406 | 23144 | O |  |
| 15 | c5 | 0.41 | MIMIC | 0.761 | 0.839 | -0.078 | [-0.112, -0.042] | 30288 | 23538 | O |  |
| 15 | c5 | 0.41 | MAIRA-2 | 0.771 | 0.846 | -0.074 | [-0.104, -0.044] | 30696 | 48180 | O |  |
| 20 | c17 | 1.00 | CheX+MIMIC | 0.751 | 0.798 | -0.047 | [-0.151, 0.060] | 2229 | 1075 | O |  |
| 20 | c17 | 1.00 | MIMIC | 0.779 | 0.474 | 0.305 | [0.209, 0.397] | 2421 | 5479 | H |  |
| 20 | c17 | 1.00 | MAIRA-2 | 0.787 | 0.716 | 0.071 | [-0.026, 0.165] | 2289 | 2339 | H |  |
| 25 | c0 | 0.37 | CheX+MIMIC | 0.739 | 0.821 | -0.082 | [-0.120, -0.041] | 25979 | 19661 | O |  |
| 25 | c0 | 0.37 | MIMIC | 0.756 | 0.824 | -0.068 | [-0.108, -0.026] | 26080 | 15261 | O |  |
| 25 | c0 | 0.37 | MAIRA-2 | 0.767 | 0.849 | -0.082 | [-0.115, -0.049] | 26348 | 40949 | O |  |
| 30 | c29 | 0.98 | CheX+MIMIC | 0.751 | 0.795 | -0.044 | [-0.148, 0.061] | 2213 | 1024 | O |  |
| 30 | c29 | 0.98 | MIMIC | 0.777 | 0.474 | 0.303 | [0.206, 0.396] | 2393 | 5468 | H |  |
| 30 | c29 | 0.98 | MAIRA-2 | 0.786 | 0.723 | 0.063 | [-0.034, 0.157] | 2270 | 2263 | H |  |
- Label changes (MIMIC): K10:O, K15:O, K20:H, K25:O, K30:H
- Label changes (MAIRA-2): K10:O, K15:O, K20:H, K25:O, K30:H

#### Lung Volume Condition (K=20 id 18)

- Matches: K=10→c1 (share=0.89, intact); K=15→c3 (share=0.89, intact); K=25→c16 (share=0.76, intact); K=30→c3 (share=0.91, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c1 | 0.89 | CheX+MIMIC | 0.673 | 0.757 | -0.084 | [-0.123, -0.043] | 40598 | 30017 | O |  |
| 10 | c1 | 0.89 | MIMIC | 0.684 | 0.777 | -0.093 | [-0.134, -0.051] | 40431 | 29981 | O |  |
| 10 | c1 | 0.89 | MAIRA-2 | 0.696 | 0.812 | -0.116 | [-0.147, -0.085] | 40817 | 55063 | O |  |
| 15 | c3 | 0.89 | CheX+MIMIC | 0.427 | 0.550 | -0.123 | [-0.217, -0.041] | 10058 | 6924 | O |  |
| 15 | c3 | 0.89 | MIMIC | 0.431 | 0.555 | -0.124 | [-0.225, -0.026] | 10014 | 6481 | O |  |
| 15 | c3 | 0.89 | MAIRA-2 | 0.430 | 0.529 | -0.099 | [-0.166, -0.031] | 9959 | 6290 | O |  |
| 20 | c18 | 1.00 | CheX+MIMIC | 0.344 | 0.468 | -0.124 | [-0.247, -0.017] | 8355 | 3903 | O |  |
| 20 | c18 | 1.00 | MIMIC | 0.352 | 0.502 | -0.150 | [-0.271, -0.038] | 8244 | 4677 | O |  |
| 20 | c18 | 1.00 | MAIRA-2 | 0.366 | 0.464 | -0.098 | [-0.168, -0.029] | 8296 | 5328 | O |  |
| 25 | c16 | 0.76 | CheX+MIMIC | 0.364 | 0.475 | -0.111 | [-0.239, 0.003] | 7105 | 3452 | O |  |
| 25 | c16 | 0.76 | MIMIC | 0.358 | 0.480 | -0.122 | [-0.258, 0.002] | 6961 | 4057 | O |  |
| 25 | c16 | 0.76 | MAIRA-2 | 0.376 | 0.467 | -0.091 | [-0.166, -0.018] | 7006 | 4714 | O |  |
| 30 | c3 | 0.91 | CheX+MIMIC | 0.356 | 0.555 | -0.199 | [-0.318, -0.099] | 8236 | 4694 | O |  |
| 30 | c3 | 0.91 | MIMIC | 0.363 | 0.538 | -0.175 | [-0.293, -0.070] | 8103 | 5020 | O |  |
| 30 | c3 | 0.91 | MAIRA-2 | 0.377 | 0.478 | -0.101 | [-0.175, -0.031] | 8184 | 5266 | O |  |

#### Heart Abnormality Presence (K=20 id 19)

- Matches: K=10→c6 (share=0.92, intact); K=15→c11 (share=0.69, intact); K=25→c13 (share=0.72, intact); K=30→c26 (share=0.71, intact)

| K | match | share | Model | GT | GEN | Δ | 95% CI | n_gt | n_gen | label | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | c6 | 0.92 | CheX+MIMIC | 0.747 | 0.804 | -0.057 | [-0.103, -0.005] | 24912 | 16954 | O |  |
| 10 | c6 | 0.92 | MIMIC | 0.779 | 0.775 | 0.004 | [-0.056, 0.060] | 24661 | 13455 | H |  |
| 10 | c6 | 0.92 | MAIRA-2 | 0.746 | 0.729 | 0.016 | [-0.046, 0.080] | 24845 | 10696 | H |  |
| 15 | c11 | 0.69 | CheX+MIMIC | 0.824 | 0.661 | 0.164 | [0.108, 0.222] | 13876 | 7026 | H |  |
| 15 | c11 | 0.69 | MIMIC | 0.826 | 0.680 | 0.146 | [0.090, 0.206] | 13702 | 5268 | H |  |
| 15 | c11 | 0.69 | MAIRA-2 | 0.779 | 0.657 | 0.121 | [0.032, 0.227] | 13900 | 3433 | H |  |
| 20 | c19 | 1.00 | CheX+MIMIC | 0.769 | 0.805 | -0.036 | [-0.098, 0.030] | 10854 | 5130 | O |  |
| 20 | c19 | 1.00 | MIMIC | 0.781 | 0.820 | -0.039 | [-0.102, 0.019] | 10763 | 5266 | O |  |
| 20 | c19 | 1.00 | MAIRA-2 | 0.737 | 0.798 | -0.060 | [-0.115, -0.008] | 10868 | 7467 | O |  |
| 25 | c13 | 0.72 | CheX+MIMIC | 0.778 | 0.686 | 0.092 | [-0.031, 0.224] | 8287 | 1551 | H |  |
| 25 | c13 | 0.72 | MIMIC | 0.792 | 0.795 | -0.003 | [-0.087, 0.082] | 8256 | 1741 | O |  |
| 25 | c13 | 0.72 | MAIRA-2 | 0.742 | 0.792 | -0.050 | [-0.123, 0.027] | 8330 | 2351 | O |  |
| 30 | c26 | 0.71 | CheX+MIMIC | 0.777 | 0.673 | 0.103 | [-0.028, 0.239] | 8310 | 1594 | H |  |
| 30 | c26 | 0.71 | MIMIC | 0.790 | 0.793 | -0.003 | [-0.089, 0.082] | 8241 | 1739 | O |  |
| 30 | c26 | 0.71 | MAIRA-2 | 0.740 | 0.788 | -0.048 | [-0.121, 0.030] | 8336 | 2292 | O |  |
- Label changes (CheX+MIMIC): K10:O, K15:H, K20:O, K25:H, K30:H
- Label changes (MIMIC): K10:H, K15:H, K20:O, K25:O, K30:O
- Label changes (MAIRA-2): K10:H, K15:H, K20:O, K25:O, K30:O

### B2. Label stability groups across K (best-match)

**CheX+MIMIC**: same label all K=12; changes only at K=10=1; changes elsewhere=7.
- Same all K: Lung Condition Assessment; Diaphragm Position Abnormality; Heart Size Normalcy; Pleural Effusion Presence; Additional Radiologic Findings; Aortic Condition; Lung Lobe Abnormality; Cardiac and Mediastinal Contours; Osseous Structure Condition; Focal Airspace Opacity; Lung Clearness and Normalcy; Lung Volume Condition
- Only K=10: Calcified Granuloma Location
- Elsewhere: Thoracic Spine Degeneration (K10:H, K15:H, K20:H, K25:O, K30:O); Pleural Fluid Presence (K10:H, K15:H, K20:O, K25:O, K30:H); Pulmonary Vascular Condition (K10:O, K15:O, K20:O, K25:H, K30:H); Lung Opacity and Infiltrates (K10:O, K15:O, K20:H, K25:O, K30:H); Pneumothorax Diagnosis (K10:O, K15:H, K20:O, K25:H, K30:H); Surgical Clip Location (K10:O, K15:O, K20:H, K25:O, K30:H); Heart Abnormality Presence (K10:O, K15:H, K20:O, K25:H, K30:H)

**MIMIC**: same label all K=13; changes only at K=10=1; changes elsewhere=6.
- Same all K: Lung Condition Assessment; Pleural Fluid Presence; Heart Size Normalcy; Lung Opacity and Infiltrates; Pleural Effusion Presence; Additional Radiologic Findings; Aortic Condition; Pneumothorax Diagnosis; Lung Lobe Abnormality; Calcified Granuloma Location; Cardiac and Mediastinal Contours; Focal Airspace Opacity; Lung Volume Condition
- Only K=10: Thoracic Spine Degeneration
- Elsewhere: Diaphragm Position Abnormality (K10:O, K15:O, K20:O, K25:H, K30:O); Pulmonary Vascular Condition (K10:O, K15:O, K20:O, K25:H, K30:O); Surgical Clip Location (K10:O, K15:O, K20:H, K25:H, K30:H); Osseous Structure Condition (K10:O, K15:H, K20:H, K25:O, K30:H); Lung Clearness and Normalcy (K10:O, K15:O, K20:H, K25:O, K30:H); Heart Abnormality Presence (K10:H, K15:H, K20:O, K25:O, K30:O)

**MAIRA-2**: same label all K=15; changes only at K=10=0; changes elsewhere=5.
- Same all K: Lung Condition Assessment; Thoracic Spine Degeneration; Pleural Fluid Presence; Heart Size Normalcy; Lung Opacity and Infiltrates; Pleural Effusion Presence; Additional Radiologic Findings; Aortic Condition; Pneumothorax Diagnosis; Surgical Clip Location; Calcified Granuloma Location; Cardiac and Mediastinal Contours; Osseous Structure Condition; Focal Airspace Opacity; Lung Volume Condition
- Elsewhere: Diaphragm Position Abnormality (K10:O, K15:O, K20:O, K25:O, K30:H); Pulmonary Vascular Condition (K10:O, K15:O, K20:O, K25:H, K30:O); Lung Lobe Abnormality (K10:O, K15:O, K20:O, K25:O, K30:H); Lung Clearness and Normalcy (K10:O, K15:O, K20:H, K25:O, K30:H); Heart Abnormality Presence (K10:H, K15:H, K20:O, K25:O, K30:O)

### B3. Omission share, label-match %, Spearman vs K=20

| Model | K | omit% | hall% | label-match% | Spearman | n used | dropped |
|---|---|---|---|---|---|---|---|
| CheX+MIMIC | 10 | 72.4 | 27.6 | 80 | 0.484 | 20 | none |
| CheX+MIMIC | 15 | 55.8 | 44.2 | 75 | 0.698 | 20 | none |
| CheX+MIMIC | 20 | 73.7 | 26.3 | 100 | 1.000 | 20 | none |
| CheX+MIMIC | 25 | 62.4 | 37.6 | 70 | 0.761 | 20 | none |
| CheX+MIMIC | 30 | 47.0 | 53.0 | 75 | 0.839 | 20 | none |
| MAIRA-2 | 10 | 51.4 | 48.6 | 90 | 0.580 | 20 | none |
| MAIRA-2 | 15 | 51.6 | 48.4 | 90 | 0.695 | 20 | none |
| MAIRA-2 | 20 | 55.0 | 45.0 | 100 | 1.000 | 20 | none |
| MAIRA-2 | 25 | 51.6 | 48.4 | 90 | 0.865 | 20 | none |
| MAIRA-2 | 30 | 49.6 | 50.4 | 90 | 0.916 | 20 | none |
| MIMIC | 10 | 83.9 | 16.1 | 75 | 0.654 | 20 | none |
| MIMIC | 15 | 76.5 | 23.5 | 85 | 0.789 | 20 | none |
| MIMIC | 20 | 74.9 | 25.1 | 100 | 1.000 | 20 | none |
| MIMIC | 25 | 81.3 | 18.7 | 80 | 0.750 | 20 | none |
| MIMIC | 30 | 72.8 | 26.2 | 100 | 0.961 | 20 | none |

### B4. Model ordering of omission share at each K

Question: are both CheXpertPlus models (CheX+MIMIC and MIMIC) higher-omission than MAIRA-2? Is MIMIC highest?

- K=10: MIMIC=83.9%, CheX+MIMIC=72.4%, MAIRA-2=51.4% — both CheXpertPlus > MAIRA-2: yes; MIMIC highest: yes.
- K=15: MIMIC=76.5%, CheX+MIMIC=55.8%, MAIRA-2=51.6% — both CheXpertPlus > MAIRA-2: yes; MIMIC highest: yes.
- K=20: MIMIC=74.9%, CheX+MIMIC=73.7%, MAIRA-2=55.0% — both CheXpertPlus > MAIRA-2: yes; MIMIC highest: yes.
- K=25: MIMIC=81.3%, CheX+MIMIC=62.4%, MAIRA-2=51.6% — both CheXpertPlus > MAIRA-2: yes; MIMIC highest: yes.
- K=30: MIMIC=72.8%, MAIRA-2=49.6%, CheX+MIMIC=47.0% — both CheXpertPlus > MAIRA-2: no; MIMIC highest: yes.

### B5. K-specific special clusters

- **K=30 cluster 10**: GEN-empty for **MIMIC only** (`n_gen=0`, `n_gt=2755`, undefined Δ). CheX+MIMIC and MAIRA-2 still have GEN rows in this cluster. Unique Q=1453; mostly spine / degenerative-change questions (XXXX in only 19/1453). Because no original K=20 cluster has >50% of its instances here, Spearman still uses all 20 clusters under the NaN-aware rule.
  - e.g. What type of changes are observed in the spine?
  - e.g. Are mild degenerative changes limited to the spine?
  - e.g. Are there severe degenerative changes in the spine?
- **Prior / alternate KMeans realizations** can isolate a different GEN-empty neighbourhood (previously observed: a K=30 cluster of GT-only de-identified `XXXX` questions with `n_gen=0` for all models). The NaN-aware mapped-Δ rule is meant to cover both cases without dropping original clusters that only lightly touch such regions.

## Part C — Mode comparison (`model_seeds_all` vs `model_seed_1`)

### Exp 5

| Model | % stable (all) | % stable (seed1) | size-wt all | size-wt seed1 | n_low_n all | n_low_n seed1 |
|---|---|---|---|---|---|---|
| CheX+MIMIC | 85 | 70 | 80.9 | 73.0 | 0 | 3 |
| MIMIC | 90 | 90 | 83.9 | 83.7 | 1 | 4 |
| MAIRA-2 | 90 | 60 | 88.7 | 63.0 | 0 | 1 |

- Flipping clusters (all): 7; (seed1): 16. Seed-1 generally shows lower stability and more `low_n`, as expected from ~1/5 the instances; directions of stable omission/hallucination largely agree where sample size is adequate.

### Exp 4

| Model | K | omit% all | omit% seed1 | Spearman all (n) | Spearman seed1 (n) |
|---|---|---|---|---|---|
| CheX+MIMIC | 10 | 72.4 | 72.5 | 0.484 (20) | 0.493 (20) |
| CheX+MIMIC | 15 | 55.8 | 56.1 | 0.698 (20) | 0.648 (20) |
| CheX+MIMIC | 20 | 73.7 | 70.9 | 1.000 (20) | 1.000 (20) |
| CheX+MIMIC | 25 | 62.4 | 57.0 | 0.761 (20) | 0.777 (20) |
| CheX+MIMIC | 30 | 47.0 | 60.0 | 0.839 (20) | 0.890 (20) |
| MIMIC | 10 | 83.9 | 84.1 | 0.654 (20) | 0.708 (20) |
| MIMIC | 15 | 76.5 | 76.5 | 0.789 (20) | 0.762 (20) |
| MIMIC | 20 | 74.9 | 62.8 | 1.000 (20) | 1.000 (20) |
| MIMIC | 25 | 81.3 | 79.2 | 0.750 (20) | 0.744 (20) |
| MIMIC | 30 | 72.8 | 77.4 | 0.961 (20) | 0.961 (20) |
| MAIRA-2 | 10 | 51.4 | 51.3 | 0.580 (20) | 0.583 (20) |
| MAIRA-2 | 15 | 51.6 | 51.5 | 0.695 (20) | 0.758 (20) |
| MAIRA-2 | 20 | 55.0 | 52.7 | 1.000 (20) | 1.000 (20) |
| MAIRA-2 | 25 | 51.6 | 51.5 | 0.865 (20) | 0.920 (20) |
| MAIRA-2 | 30 | 49.6 | 48.8 | 0.916 (20) | 0.937 (20) |

Omission-share ordering across models is usually preserved under seed1; absolute percentages move more for sparse GEN clusters (often `low_n` under seed1).

## Part D — Plain-language conclusions

- Across the 20 clinical question clusters and 5 eval seeds, most clusters keep a stable Fig. 4 dominance label (about 88% on average across models in `model_seeds_all`); between-cluster differences in Δ are much larger than seed-to-seed SD within a cluster.
- Flipping clusters sit near the GT=GEN diagonal (small |Δ|) rather than reflecting wholesale changes in clinical behaviour; several also sit in the `both_low` corner (GT and GEN both < 0.3), where a binary side-of-diagonal label is least informative.
- Report-level question Jaccard is higher for same model_seed than for different seeds, with a positive gap and CI excluding zero for GT-as-ref (and usually for GEN-as-ref); only 9,569 / 78,708 unique questions appear in all five eval seeds.
- Varying K (10–30) preserves the broad omission vs hallucination landscape for most clusters under best-match correspondence; label-match vs K=20 is often high, and Spearman of mapped Δ (NaN-aware, >50% undefined drop rule) remains substantial even when a few new clusters are GEN- or GT-empty.
- Model ordering of size-weighted omission share is usually MIMIC highest, with CheX+MIMIC above MAIRA-2 at K=10–25; at K=30 in `model_seeds_all`, CheX+MIMIC falls slightly below MAIRA-2 (47% vs 50%). This is about mass relative to the diagonal under this metric, not clinical superiority.
- Unstable cases to treat carefully in text: near-diagonal flips across seeds; `low_n` GEN (especially Calcified Granuloma Location under MIMIC); concept merges at small K (share < 0.5) and splits at large K (two clusters ≥20%); `both_low` clusters whose label can look decisive despite low accuracy on both axes.
- Examples (not headlines): Lung Lobe Abnormality is a stable omission-leaning concept for the CheXpertPlus models across seeds and most K; Calcified Granuloma Location is more fragile (often hallucination for CheX+MIMIC, omission with very small n_gen for MIMIC)—useful for illustrating limits, not as the main result.
- For R1-C7 / R4 responses: clustering K and eval seed change point estimates and can move a minority of near-diagonal / low-n clusters, but they do not overturn the overall pattern that many clusters are stably omission- or hallucination-dominated under the Fig. 4 rule when n is adequate.
- Honest wording: do not claim label invariance for every cluster; report CIs, n_gt/n_gen, `low_n`, and `both_low` alongside labels; describe Spearman/label-match as graded consistency, not identity.
