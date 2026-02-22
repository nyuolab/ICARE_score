# Example Test

Quick-test scripts and sample data for the ICARE evaluation pipeline.

## Sample Data

Uses `test_data/sample_iuxray_reports.csv` — a 10-row sample from the IU X-Ray dataset.

### Columns

| Column | Description |
|--------|-------------|
| `id` | Unique study identifier |
| `image_paths` | Image file paths (not used by pipeline) |
| `ground_truth_report` | Ground truth radiology report |
| `generated_report` | Model-generated report |

### Report Mix

- **Normal** (rows 0–1): Clear lungs, normal cardiac silhouette
- **Mild findings** (rows 2–4): Degenerative spine, hiatal hernia
- **GT–Gen divergence** (rows 5–6): Model adds or misses findings
- **Abnormal** (rows 7–9): Opacities, cardiomegaly, effusions, atelectasis

## End-to-End Pipeline

```
INPUT: test_data/sample_iuxray_reports.csv
       (ground_truth_report, generated_report columns)
    │
    ├── Step 1: MCQ Generation (src/mcq_generation.py)
    │   Reads: ground_truth_report OR generated_report
    │   Produces: test_data/output/{orig_data,shuffled_ans_choices_data}/{gt,gen}_reports_as_ref/mcqa_data.json
    │
    ├── Step 2: MCQ Filtering (src/mcq_filtering.py)
    │   Reads: mcqa_data.json from Step 1
    │   Produces: mcqa_filtering/filtered_questions_shuffled.csv (report-dependent questions only)
    │
    ├── Step 3: MCQA Evaluation (src/mcqa_evaluation.py)
    │   Reads: filtered_questions_shuffled.csv + original CSV (both report columns)
    │   Produces: mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv, report_level_stats, etc.
    │
    └── Step 4 (optional): Question Categorization
        Reads: filtered_questions_shuffled.csv + mcqa_eval_answer_predictions.csv from Step 2 & 3
        Produces: question_categorization/combined_mcqa_data.csv, clustered_questions_with_names.csv,
                  cluster_names.json, analysis/all_models_gt_vs_gen_agreement.png
```

**Scripts:** `run_eval.sh` → Steps 1–3 | `run_question_categorization.sh` → Step 4

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval.sh` | Runs steps 1–3 (MCQ gen → filtering → MCQA eval) on the sample data |
| `run_question_categorization.sh` | Runs step 4 on the eval output. Run `run_eval.sh` first. |

## Usage

### 1. Create the conda environment

Run on a compute node with sufficient memory:

```bash
srun --pty --cpus-per-task=8 --gpus=2 --mem=128G --partition=oermannlab bash

cd /path/to/ICARE_score
conda create -n rrg-eval-clean python=3.8 -y
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1
wget -O build-constraints.txt https://raw.githubusercontent.com/explosion/thinc/master/build-constraints.txt
PIP_CONSTRAINT=./build-constraints.txt pip install -r requirements.txt "pytz" "python-dateutil" "huggingface-hub>=0.14.1" "bottleneck>=1.3.6" --no-cache-dir
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and set `RRGEVAL_API_KEY`, `RRGEVAL_API_URL`, etc. (see root README).

### 3. Run the eval

```bash
cd ICARE_score
sbatch scripts/example_test/run_eval.sh
# or: bash scripts/example_test/run_eval.sh
```

Results in `test_data/output/`. Uses 5 MCQs per report (instead of 40) for a fast run.

### 4. (Optional) Question categorization

```bash
sbatch scripts/example_test/run_question_categorization.sh
```

Results in `test_data/output/question_categorization/`.

**If compute nodes lack Hugging Face access**, pre-download MedCPT and set in `.env`:
```bash
git clone https://huggingface.co/ncbi/MedCPT-Query-Encoder /path/to/MedCPT-Query-Encoder
# In .env: MEDCPT_MODEL_PATH=/path/to/MedCPT-Query-Encoder
```

> **Note:** Requires LLM API access configured in `.env`.
