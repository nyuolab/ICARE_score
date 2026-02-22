# Radiology Report Generation and Evaluation

## 📌 Table of Contents

| Section | Description |
|---------|-------------|
| [📂 IU-Xray Dataset](#📂-iu-xray-dataset) | Quick test, full evaluation, perturbation experiments, visualization |

## 📌 Generate Radiology Reports

First, generate radiology reports using different RRG models.

#### ✅ Supported RRG Models:
- **MAIRA2**
- **Chexpertplus trained on Chexpertplus + MIMIC data**
- **Chexpertplus trained on MIMIC data**

---

## 📂 IU-Xray Dataset

### 🏆 Run Evaluation of Our Evaluation Approach

📌 **First, clone the repository, install the conda environment, and navigate into the repo:**

```bash
git clone https://github.com/nyuolab/RRGEval.git
cd RRGEval
```

**Create the conda environment** (run on a compute node with sufficient memory, e.g. via `srun`):

```bash
srun --pty --cpus-per-task=8 --gpus=2 --mem=128G --partition=oermannlab bash

cd /path/to/ICARE_score  # or your repo path
conda create -n rrg-eval-clean python=3.8 -y
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1
wget -O build-constraints.txt https://raw.githubusercontent.com/explosion/thinc/master/build-constraints.txt
PIP_CONSTRAINT=./build-constraints.txt pip install -r requirements.txt "pytz" "python-dateutil" "huggingface-hub>=0.14.1" "bottleneck>=1.3.6" --no-cache-dir
```

📌 **Next, create and configure your `.env` file before running any scripts:**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set the following:
RRGEVAL_BASE_DATA_PATH="/path/to/your/base/data/directory"  # Parent dir of RRG_models/, RRG_evaluation/, etc.
RRGEVAL_API_KEY="your_api_key_here"
RRGEVAL_API_URL="your_api_url_here"
```
> **Note:** All scripts derive their data paths from `RRGEVAL_BASE_DATA_PATH`. Set this to the directory that contains your `RRG_models/`, `RRG_evaluation/`, and `cxr_report_datasets/` folders.

📌 **Quick test:** You can quickly run the pipeline on our sample dataset (`test_data/sample_iuxray_reports.csv`). For running on your own test set, provide a CSV in the same format: columns `ground_truth_report` and `generated_report` (and optionally `id`).  
→ Full details: [scripts/example_test/README.md](scripts/example_test/README.md)

📌 **Evaluate reports generated from different RRG models:**

##### 🔹 MAIRA2:
```bash
sbatch scripts/iuxray_data/maira2.sh
```

##### 🔹 Chexpertplus model trained on MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_mimic.sh
```

##### 🔹 Chexpertplus model trained on Chexpertplus + MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic.sh
```

📌 **Modify the following variables in each script as needed:**
- `EVAL_SEED`
- `MODEL_SEED`
- `INPUT_CSV` (Path to output file containing generated reports from the RRG model)
- `OUTPUT_DIR` (Path to store results)
- 
📌 **Results Structure**
The results are stored in `${OUTPUT_DIR}/shuffled_ans_choices_data/`. Within this directory:

- `gen_reports_as_ref/` and `gt_reports_as_ref/`: Contain all ICARE_GEN and ICARE_GT evaluation results. Each of these directories includes a `mcqa_eval/` subdirectory with the complete set of evaluation scores.
- `mcq_eval_dataset_level_agreement_stats.csv`: Contains dataset-level agreement scores.
- `mcq_eval_report_level_stats.csv`: Contains agreement scores for individual reports.
- `mcq_eval_report_level_stats_aggregated.csv`: Provides aggregated report-level results across the dataset.

### Question Categorization and Analysis: follow the steps in the readme here [src/question_categorization_and_analysis/](src/question_categorization_and_analysis/)
---

### 🔄 Perturbation Experiments (Word Level)

Evaluate our approach on reports generated from different RRG models:

##### 🔹 MAIRA2:
```bash
sbatch scripts/iuxray_data/maira2_perturbed_word_level.sh
```

##### 🔹 Chexpertplus model trained on MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_mimic_perturbed_word_level.sh
```

##### 🔹 Chexpertplus model trained on Chexpertplus + MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic_perturbed_word_level.sh
```

---

### 🔄 Perturbation Experiments (Character Level)

Evaluate our approach on reports generated from different RRG models:

##### 🔹 MAIRA2:
```bash
sbatch scripts/iuxray_data/maira2_perturbed.sh
```

##### 🔹 Chexpertplus model trained on MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_mimic_perturbed.sh
```

##### 🔹 Chexpertplus model trained on Chexpertplus + MIMIC:
```bash
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic_perturbed.sh
```

📌 **Modify the following variables in each script as needed:**
- `EVAL_SEED`
- `MODEL_SEED`
- `INPUT_CSV` (Path to output from RRG model)
- `OUTPUT_DIR` (Path to store results)

---

#### 📊 Visualization & Agreement Analysis

To generate plots showing agreement percentage as a function of perturbation intensity:
```bash
sbatch scripts/iuxray_data/plot_agreement_with_perturbation_stats.sh
```

📂 **Results will be stored in:**
- `INPUT_DIR/plots/perturbation_char_level`
- `INPUT_DIR/plots/perturbation_word_level`

---

#### 📑 Results Compilation

Run the following notebook to compile all results:
```bash
jupyter notebook src/results_compilation.ipynb
```


