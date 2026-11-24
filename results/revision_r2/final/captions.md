# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Markers are uniform in size (no size encoding); question counts are reported in Supplementary Tables A and B. Cluster point labels are IDs; names appear in key panels and tables.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Colour encodes side of the diagonal (red: below / omission-dominated; blue: above / hallucination-dominated). Points are labelled by K=20 cluster ID. The dashed line is the identity. (d) Cluster ID → name key (same as Fig. 4d). Question counts are given in Supplementary Table A.

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡). Summary rows report the percentage of clusters on the same side of the diagonal in all five seeds; median across-seed SD of ICARE-GT and of ICARE-GEN; report-level Jaccard of clusters probed for the same report across seeds versus different reports (mean with 95% CI); and unique questions common to all five seeds (9,569 / 78,708).

## Supp Fig. B
**Supplementary Figure B. ICARE cluster scores across the number of clusters K (Fig. 4–style grid).**  
Each panel is a Fig. 4–style scatterplot of ICARE-GEN (x) versus ICARE-GT (y) for one model at one K. Rows: K = 10, 15, 20, 25, 30 (the K=20 row is the Fig. 4 setting, shown on a light grey background); columns: CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC, MAIRA2. Markers are uniform in size; colour encodes side of the diagonal; points are labelled by cluster ID. Question counts are given in Supplementary Table B. Cluster ID → name keys for K ≠ 20 are in Supplementary Table C; the K=20 key is Fig. 4d / Supplementary Figure A(d).

Best-matching cluster IDs for the tracked concepts:
- K = 10: Lung Lobe Abnormality = cluster 5; Calcified Granuloma Location = cluster 3.
- K = 15: Lung Lobe Abnormality = cluster 9; Calcified Granuloma Location = cluster 13.
- K = 20: Lung Lobe Abnormality = cluster 12; Calcified Granuloma Location = cluster 13.
- K = 25: Lung Lobe Abnormality = cluster 14; Calcified Granuloma Location = cluster 4.
- K = 30: Lung Lobe Abnormality = cluster 5; Calcified Granuloma Location = cluster 13.

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN (n = 20). Part 2: Lung Lobe Abnormality and Calcified Granuloma Location — best-matching cluster at each K (with share); per model, ICARE-GT and ICARE-GEN with 95% CIs, n_gt / n_gen, and side of the diagonal; † marks < 30 questions.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. Local names are filled when `cluster_names_K{K}.json` is present (LLM naming can be applied later without recomputing scores). The K=20 name key is Fig. 4d.
