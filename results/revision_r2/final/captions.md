# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Point style matches submitted Fig. 4 (`cluster_analysis.create_cluster_summary_table_and_scatterplots`): marker area = GT question count / 30, black edges, ID labels offset (5, 5) pt. Colour encodes side of the diagonal (the only intentional style difference from Fig. 4). Question counts are also reported in Supplementary Tables A and B.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Point size uses pooled GT question count / 30 (same formula as Fig. 4); colour encodes side of the diagonal. Points are labelled by K=20 cluster ID as in Fig. 4. The dashed line is the identity. (d) Cluster ID → name key.

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡).

## Supp Fig. B
**Supplementary Figure B. ICARE cluster scores across the number of clusters K (Fig. 4–style grid).**  
Each panel matches the submitted Fig. 4 scatter style. Rows: K = 10, 15, 20, 25, 30 (K=20 shaded and labelled Fig. 4); columns: CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC, MAIRA2. Point size = GT question count / 30; colour encodes side of the diagonal.

Best-matching cluster IDs for the tracked concepts:
- K = 10: Lung Lobe Abnormality = cluster 5; Calcified Granuloma Location = cluster 3.
- K = 15: Lung Lobe Abnormality = cluster 9; Calcified Granuloma Location = cluster 13.
- K = 20: Lung Lobe Abnormality = cluster 12; Calcified Granuloma Location = cluster 13.
- K = 25: Lung Lobe Abnormality = cluster 14; Calcified Granuloma Location = cluster 4.
- K = 30: Lung Lobe Abnormality = cluster 5; Calcified Granuloma Location = cluster 13.

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN. Part 2: Lung Lobe Abnormality and Calcified Granuloma Location tracking.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. The K=20 name key is Fig. 4d.
