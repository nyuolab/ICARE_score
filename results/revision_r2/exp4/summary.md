# Exp 4 summary — cluster K sensitivity (final)

- sklearn: `1.3.2`
- Label rule: Fig. 4 point estimate (omission if GT < GEN; hallucination if GT > GEN).
- Fresh `cluster_questions` for every K including 20; stored names used for K=20 display only.
- Modes: `model_seeds_all` (primary) and `model_seed_1`.

## K=20 reproduction (fresh vs stored)

- **model_seeds=all**: ARI=1.0, exact-label match=1.0
  - max |diff| vs stored Fig. 4 CSVs (from scores): GT=1.1102230246251565e-16, GEN=1.1102230246251565e-16
  - max |diff| via calculate_agreement: GT=1.1102230246251565e-16, GEN=1.1102230246251565e-16
- **model_seeds=1**: ARI=1.0, exact-label match=1.0

## Consistency side-by-side (all vs seed_1)

| K | Model | mode | label-match% vs K=20 | Spearman (n) | omission% | hallucination% | n_low_n |
|---|---|---|---|---|---|---|---|
| 10 | CheX+MIMIC | 1 | 75.0 | 0.493 (20) | 72.5 | 27.5 | 0 |
| 10 | CheX+MIMIC | all | 80.0 | 0.484 (20) | 72.4 | 27.6 | 0 |
| 10 | MAIRA-2 | 1 | 85.0 | 0.583 (20) | 51.3 | 48.7 | 0 |
| 10 | MAIRA-2 | all | 90.0 | 0.580 (20) | 51.4 | 48.6 | 0 |
| 10 | MIMIC | 1 | 70.0 | 0.708 (20) | 84.1 | 15.9 | 0 |
| 10 | MIMIC | all | 75.0 | 0.654 (20) | 83.9 | 16.1 | 0 |
| 15 | CheX+MIMIC | 1 | 70.0 | 0.648 (20) | 56.1 | 43.9 | 0 |
| 15 | CheX+MIMIC | all | 75.0 | 0.698 (20) | 55.8 | 44.2 | 0 |
| 15 | MAIRA-2 | 1 | 85.0 | 0.758 (20) | 51.5 | 48.5 | 0 |
| 15 | MAIRA-2 | all | 90.0 | 0.695 (20) | 51.6 | 48.4 | 0 |
| 15 | MIMIC | 1 | 80.0 | 0.762 (20) | 76.5 | 23.5 | 1 |
| 15 | MIMIC | all | 85.0 | 0.789 (20) | 76.5 | 23.5 | 0 |
| 20 | CheX+MIMIC | 1 | 100.0 | 1.000 (20) | 70.9 | 29.1 | 0 |
| 20 | CheX+MIMIC | all | 100.0 | 1.000 (20) | 73.7 | 26.3 | 0 |
| 20 | MAIRA-2 | 1 | 100.0 | 1.000 (20) | 52.7 | 47.3 | 0 |
| 20 | MAIRA-2 | all | 100.0 | 1.000 (20) | 55.0 | 45.0 | 0 |
| 20 | MIMIC | 1 | 100.0 | 1.000 (20) | 62.8 | 37.2 | 1 |
| 20 | MIMIC | all | 100.0 | 1.000 (20) | 74.9 | 25.1 | 0 |
| 25 | CheX+MIMIC | 1 | 85.0 | 0.777 (20) | 57.0 | 43.0 | 1 |
| 25 | CheX+MIMIC | all | 70.0 | 0.761 (20) | 62.4 | 37.6 | 0 |
| 25 | MAIRA-2 | 1 | 95.0 | 0.920 (20) | 51.5 | 48.5 | 0 |
| 25 | MAIRA-2 | all | 90.0 | 0.865 (20) | 51.6 | 48.4 | 0 |
| 25 | MIMIC | 1 | 70.0 | 0.744 (20) | 79.2 | 20.8 | 1 |
| 25 | MIMIC | all | 80.0 | 0.750 (20) | 81.3 | 18.7 | 1 |
| 30 | CheX+MIMIC | 1 | 80.0 | 0.745 (10) | 54.2 | 44.9 | 1 |
| 30 | CheX+MIMIC | all | 70.0 | 0.429 (8) | 55.9 | 43.2 | 1 |
| 30 | MAIRA-2 | 1 | 90.0 | 0.967 (13) | 47.1 | 52.1 | 1 |
| 30 | MAIRA-2 | all | 95.0 | 0.983 (9) | 47.2 | 52.0 | 1 |
| 30 | MIMIC | 1 | 85.0 | 0.964 (11) | 79.4 | 19.6 | 2 |
| 30 | MIMIC | all | 90.0 | 1.000 (8) | 79.3 | 19.7 | 2 |

## Matched clusters for tracked concepts

### Lung Lobe Abnormality
#### model_seeds=all
- K=10: matched cluster **5** (share=0.882); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the nature of the findings in the right lower lobe according to the radiology report? | Is there any mention of the right middle lobe being more collapsed than in the prior study? | Is there any mention of a right lower lobe abnormality? | Is there any mention of the right middle lobe being more collapsed than before? | Is the right lung clear or does it have any abnormalities mentioned?
- K=15: matched cluster **9** (share=0.983); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | Is there any mention of a right lower lobe abnormality? | What is the nature of the finding in the left lower lobe of the lung? | What can be said about the presence of a large opacity in the right lower lobe? | Is the right lower lung opacity due to a significant pathological condition?
- K=20: matched cluster **12** (share=1.000); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
- K=25: matched cluster **14** (share=0.787); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | What is the nature of the finding in the left lower lobe of the lung? | What is the appearance of the lungs except for the right upper lobe? | What is noted in the right lower lung area? | Is there any mention of a right lower lobe abnormality?
- K=30: matched cluster **14** (share=0.729); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | What is the nature of the finding in the left lower lobe of the lung? | What is the appearance of the lungs except for the right upper lobe? | Is there any mention of a right lower lobe abnormality? | What is noted in the right lower lung area?

#### model_seeds=1
- K=10: matched cluster **5** (share=0.887); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the nature of the findings in the right lower lobe according to the radiology report? | Is there any mention of the right middle lobe being more collapsed than in the prior study? | Is there any mention of a right lower lobe abnormality? | Is there any mention of the right middle lobe being more collapsed than before? | Is the right lung clear or does it have any abnormalities mentioned?
- K=15: matched cluster **9** (share=0.983); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | Is there any mention of a right lower lobe abnormality? | What is the nature of the finding in the left lower lobe of the lung? | What can be said about the presence of a large opacity in the right lower lobe? | Is the right lower lung opacity due to a significant pathological condition?
- K=20: matched cluster **12** (share=1.000); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
- K=25: matched cluster **14** (share=0.791); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | What is the nature of the finding in the left lower lobe of the lung? | What is the appearance of the lungs except for the right upper lobe? | What is noted in the right lower lung area? | Is there any mention of a right lower lobe abnormality?
- K=30: matched cluster **14** (share=0.730); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: What is the condition of the lower lobe of the right lung according to the report? | What is the nature of the finding in the left lower lobe of the lung? | What is the appearance of the lungs except for the right upper lobe? | Is there any mention of a right lower lobe abnormality? | What is noted in the right lower lung area?


### Calcified Granuloma Location
#### model_seeds=all
- K=10: matched cluster **3** (share=0.621); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: Are there any osseous abnormalities or findings mentioned in the report? | Is the condition of the bony structures relevant to the diagnosis of any condition mentioned in the report? | Which of the following is mentioned as present in the radiology report? | Are there any other relevant findings mentioned in the radiology report? | Which of the following conditions is mentioned in the radiology report?
- K=15: matched cluster **13** (share=0.894); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where are the calcified granulomas specifically mentioned to be located? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Is there any evidence of a calcified granuloma in the radiology report?
- K=20: matched cluster **13** (share=1.000); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
- K=25: matched cluster **4** (share=0.898); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Where are the calcified granulomas specifically mentioned to be located? | Where are the calcified granulomas specifically mentioned to be?
- K=30: matched cluster **10** (share=0.863); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Where are the calcified granulomas specifically mentioned to be located? | Where are calcified granulomas found, according to the report?

#### model_seeds=1
- K=10: matched cluster **3** (share=0.627); CheX+MIMIC: omission-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated
  - representatives: Are there any osseous abnormalities or findings mentioned in the report? | Is the condition of the bony structures relevant to the diagnosis of any condition mentioned in the report? | Which of the following is mentioned as present in the radiology report? | Are there any other relevant findings mentioned in the radiology report? | Which of the following conditions is mentioned in the radiology report?
- K=15: matched cluster **13** (share=0.900); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where are the calcified granulomas specifically mentioned to be located? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Is there any evidence of a calcified granuloma in the radiology report?
- K=20: matched cluster **13** (share=1.000); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
- K=25: matched cluster **4** (share=0.903); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Where are the calcified granulomas specifically mentioned to be located? | Where are the calcified granulomas specifically mentioned to be?
- K=30: matched cluster **10** (share=0.869); CheX+MIMIC: hallucination-dominated; MAIRA-2: omission-dominated; MIMIC: omission-dominated [low_n]
  - representatives: Where are the calcified granulomas specifically located, as stated in the report? | Where is the calcified granuloma found, according to the report? | Where are the calcified granulomas found according to the report? | Where are the calcified granulomas specifically mentioned to be located? | Where are calcified granulomas found, according to the report?


## Figure paths

Per mode under `results/revision_r2/exp4/model_seeds_all/` and `.../model_seed_1/`:
- `fig4_style_K{10,15,20,25,30}.png/.pdf`
- `scatter_grid.png`, `consistency_share_vs_K.png`
- `correspondence_heatmap_K{K}.png`

## Notes

- `--name` is implemented but was not run in this final pass; K≠20 points use cluster IDs.
