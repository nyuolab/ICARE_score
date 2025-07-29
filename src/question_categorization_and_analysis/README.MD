# Question Categorization and Analysis

This step organizes and analyzes questions generated during the evaluation of radiology reports.

## What it does

1. **Collect questions**  
   - Extracts questions from ground truth and model-generated reports  
   - Uses 3 models: `maira2`, `chexpertplus_chexpertplus_mimic`, `chexpertplus_mimic`  
   - Includes 5 model seeds and 5 evaluation seeds  
   - Based on ~600 reports from the IU X-Ray test dataset  
   - Total: ~993,403 questions  
     (600 reports × 2 [GT and generated] × 5 model seeds × 5 eval seeds × 3 models × ~40 questions per report)

2. **Remove duplicates**  
   - Extracts only **unique questions** before clustering (~78,708 questions)  
   - Prevents redundant questions from dominating cluster samples  
   - Ensures that the 5 representative questions per cluster are meaningful and diverse

3. **Generate embeddings**  
   - Uses a language model to create vector embeddings of each question  
   - Currently using: `MedCPT`

4. **Cluster the questions**  
   - Uses K-Means to group question embeddings into up to 20 clusters  
   - Selects 5 representative questions from each cluster (closest to cluster center)

5. **Name clusters using LLaMA**  
   - Prompts LLaMA to generate a category name for each cluster  
   - Assigns the category name to all questions in the cluster

6. **Analyze**  
   - Calculates the average agreement score for each category  
   - Counts how many questions fall into each category across all models

## Setup

Before running the scripts, set up the environment as follows:

```bash
source ~/.bashrc
conda create -n medcpt_embed python=3.10
conda activate medcpt_embed
git clone https://huggingface.co/ncbi/MedCPT-Query-Encoder
```

## When to run

Run this step **after** completing evaluation for all models, model seeds, and evaluation seeds.

## How to run

Execute the following scripts in order:

```bash
sbatch scripts/iuxray_data/question_categorization_and_analysis/create_combined_questions.sh
sbatch scripts/iuxray_data/question_categorization_and_analysis/question_embedding_and_clustering.sh
sbatch scripts/iuxray_data/question_categorization_and_analysis/cluster_analysis.sh
```

Modify the OUTPUT_DIR and BASE_DIR or other information in these scripts before running them. 
The results can be seen at the ${OUTPUT_DIR}/analysis/all_models_gt_vs_gen_agreement.png

