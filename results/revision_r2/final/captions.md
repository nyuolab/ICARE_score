# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Cluster point labels are IDs; names appear in key panels and tables.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Point size scales with mean question count; colour encodes side of the diagonal (red: below / omission-dominated; blue: above / hallucination-dominated). Points are labelled by K=20 cluster ID. The dashed line is the identity. (d) Cluster ID → name key.

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡). Summary rows report the percentage of clusters on the same side of the diagonal in all five seeds; median across-seed SD of ICARE-GT and of ICARE-GEN; report-level Jaccard of clusters probed for the same report across seeds versus different reports (mean with 95% CI); and unique questions common to all five seeds (9,569 / 78,708).

## Supp Fig. B
**Supplementary Figure B. Sensitivity of ICARE cluster structure and scores to the number of clusters K.**  
(a) Percentage of clusters below the diagonal (ICARE-GT < ICARE-GEN) versus K for each model. (b) Label agreement with K=20: percentage of K=20 clusters whose best-matching cluster at the new K lies on the same side of the diagonal. (c) Spearman correlation with K=20 for ICARE-GT (solid) and ICARE-GEN (dashed), comparing each K=20 cluster to the question-weighted value of the clusters its questions map to at the new K. Dashed vertical line: K=20. (d) Fig. 4–style scatterplots for K = 10, 15, 25, 30 (rows) and the three models (columns); points labelled by cluster ID and coloured by side of the diagonal; size encodes question count.

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN (n = 20). Part 2: Lung Lobe Abnormality and Calcified Granuloma Location — best-matching cluster at each K (with share); per model, ICARE-GT and ICARE-GEN with 95% CIs, n_gt / n_gen, and side of the diagonal; † marks < 30 questions.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. Local names are filled when `cluster_names_K{K}.json` is present (LLM naming can be applied later without recomputing scores).
