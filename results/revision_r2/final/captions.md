# Captions — R2 supplementary figures and tables (Exp 4–5)

Data: IU X-ray; 590 reports per model. Evaluation uses five evaluation seeds with model seeds pooled as in Fig. 4. Side of the diagonal follows Fig. 4: below the diagonal = omission-dominated (ICARE-GT < ICARE-GEN); above the diagonal = hallucination-dominated (ICARE-GT > ICARE-GEN). K=20 was chosen with radiologists. Point style matches submitted Fig. 4 (`cluster_analysis.create_cluster_summary_table_and_scatterplots`): marker area = GT question count / 30, ID labels offset (5, 5) pt. Colour encodes side of the diagonal (an intentional style difference from Fig. 4). Supp Fig. A retains black marker edges; Supp Fig. B uses no marker edges. Question counts are also reported in Supplementary Tables A and B.

## Supp Fig. A
**Supplementary Figure A. Evaluation-seed consistency of ICARE cluster scores (K=20).**  
(a–c) Across-seed mean ICARE-GEN (x) versus mean ICARE-GT (y) for CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC and MAIRA2. Error bars show the across-seed standard deviation over five evaluation seeds. Point size uses pooled GT question count / 30 (same formula as Fig. 4); colour encodes side of the diagonal. Points are labelled by K=20 cluster ID as in Fig. 4. The dashed line is the identity. (d) Cluster ID → name key.

## Supp Table A
**Supplementary Table A. Evaluation-seed consistency at K=20.**  
For each cluster and model: ICARE-GT and ICARE-GEN as across-seed mean ± SD, and the number of seeds on the modal side of the diagonal (out of 5). Footnotes mark clusters with fewer than 30 questions on either side in any seed (†) and clusters with ICARE-GT and ICARE-GEN both < 0.3 (‡).

## Supp Fig. B
**Supplementary Figure B. ICARE cluster scores across the number of clusters K (Fig. 4–style grid).**  
Each panel matches the submitted Fig. 4 scatter style (marker area = GT question count / 30; ID labels offset (5, 5) pt) without marker edge lines. Rows: K = 10, 15, 20, 25, 30 (K=20 labelled Fig. 4); columns: CheXpertPlus_MIMIC, CheXpertPlus_Chex_MIMIC, MAIRA2. Colour encodes side of the diagonal. Cluster IDs are specific to each K.

## Supp Table B
**Supplementary Table B. Sensitivity to K (Exp 4).**  
Part 1: for each K and model, counts and percentages of clusters below / above the diagonal, label agreement with K=20 (%), and Spearman correlations for ICARE-GT and ICARE-GEN. Part 2: Lung Lobe Abnormality and Calcified Granuloma Location tracking.

## Supp Table D
**Supplementary Table D. Concept-level consistency across K.**  
For each of the 20 K=20 concepts (Fig. 4d) and each model, cells give the side of the diagonal (Below / Above) of the cluster at K ∈ (10, 15, 20, 25, 30) that contains the largest share of that concept’s K=20 questions. Light red = Below (ICARE-GT < ICARE-GEN); light blue = Above. * merged into a broader cluster (concept < 50% of the matched cluster); † split (matched cluster holds < 50% of the concept’s questions). Bottom row: % of the 20 concepts on the same side of the diagonal as at K=20. Concepts matched across K by shared questions.

## Supp Table C
**Supplementary Table C. Cluster keys for K = 10, 15, 25, 30.**  
Cluster ID → best-matching K=20 cluster name and share. The K=20 name key is Fig. 4d.

Cluster names for K ≠ 20 were generated with Llama 3.3 70B using the same prompt and nearest-centroid representatives as Fig. 4; K = 20 names are those of Fig. 4.
