# Radiology Report Generation and Evaluation

## 📌 Table of Contents

| Section | Description |
|---------|-------------|
| [📂 IU-Xray Dataset](#📂-iu-xray-dataset) | Evaluation, Perturbation Experiments, Visualization, and Results Compilation |

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
conda env create -f environment.yml
```

📌 **Next, update the .env file with your API credentials before running any scripts:**
```bash
# In the .env file, set the following:
RRGEVAL_API_KEY="your_api_key_here"
RRGEVAL_API_URL="your_api_url_here"
```

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


