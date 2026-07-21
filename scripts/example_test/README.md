# Example Test

Quick-test scripts and sample data for the ICARE evaluation pipeline.

## Prerequisites

Before running the pipeline, do the following in order:

**1. Clone the repository:**

```bash
git clone https://github.com/nyuolab/ICARE_score.git
cd ICARE_score
```

**2. Create the conda environment and install dependencies** (run on a compute node with sufficient memory):

```bash

cd ICARE_score  # ensure you're in the cloned repo
conda create -n rrg-eval-clean python=3.8 -y
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1
wget -O build-constraints.txt https://raw.githubusercontent.com/explosion/thinc/master/build-constraints.txt
PIP_CONSTRAINT=./build-constraints.txt pip install -r requirements.txt "pytz" "python-dateutil" "huggingface-hub>=0.14.1" "bottleneck>=1.3.6" --no-cache-dir
```

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
    ├── Step 4: Compile Results (src/compile_results.py)  [run_eval_final_without_orig.sh only]
    │   Reads: Step 1–3 outputs under shuffled_ans_choices_data/
    │   Produces: icare_results.json, icare_results_summary.csv, pipeline_timing.json
    │
    └── Step 5 (optional): Question Categorization
        Reads: filtered_questions_shuffled.csv + mcqa_eval_answer_predictions.csv from Step 2 & 3
        Produces: question_categorization/combined_mcqa_data.csv, clustered_questions_with_names.csv,
                  cluster_names.json, analysis/all_models_gt_vs_gen_agreement.png
```

**Default script:** `run_eval_final_without_orig.sh` → Steps 1–4 on `shuffled_ans_choices_data`

**Ablation script:** `run_eval.sh` → Steps 1–3 on **both** `orig_data` and `shuffled_ans_choices_data` (useful for comparing unshuffled vs shuffled answer choices)

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval_final_without_orig.sh` | End-to-end pipeline (Steps 1–4): same as `run_eval.sh` but Steps 2–3 on `shuffled_ans_choices_data` only; writes `icare_results.json`, `icare_results_summary.csv`, `pipeline_timing.json` |
| `run_eval.sh` | Steps 1–3 on **both** `orig_data` and `shuffled_ans_choices_data` (includes orig_data ablation steps) |
| `run_question_categorization.sh` | Optional question categorization. Run an eval script first. |

## Usage

### 1. Configure `.env`

- **Local LLM (no API key):** Follow the "Local LLM Setup" section in the root README, then `cp .env.local_example .env`.
- **Private/hosted API:** `cp .env.example .env` and fill in `RRGEVAL_API_KEY`, `RRGEVAL_API_URL`.

### 2. Run the end-to-end pipeline

From the repo root:

```bash
bash scripts/example_test/run_eval_final_without_orig.sh
# or: sbatch scripts/example_test/run_eval_final_without_orig.sh  (SLURM)
```

Results in `test_data/output/`:
- `icare_results.json` — per-sample detail + step timing
- `icare_results_summary.csv` — one row per sample (agreement %, question counts)
- `pipeline_timing.json` — step1/2/3 durations (seconds)

To also run filtering and evaluation on `orig_data` (answer-order ablation):

```bash
bash scripts/example_test/run_eval.sh
```

### 3. (Optional) Question categorization

Run after Steps 1–3 (either eval script):

```bash
bash scripts/example_test/run_question_categorization.sh
```

Results in `test_data/output/question_categorization/`.

**If compute nodes lack Hugging Face access**, pre-download MedCPT and set in `.env`:
```bash
git clone https://huggingface.co/ncbi/MedCPT-Query-Encoder /path/to/MedCPT-Query-Encoder
# In .env: MEDCPT_MODEL_PATH=/path/to/MedCPT-Query-Encoder
```
