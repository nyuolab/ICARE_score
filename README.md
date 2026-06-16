# ICARE: Clinically Grounded Agent-based Report Evaluation

ICARE is an interpretable LLM-based metric for radiology report generation. Given a ground-truth and a generated report, it generates multiple-choice questions, filters them for report-dependency, then measures how consistently an LLM answers those questions when given each report. Higher agreement = higher fidelity to the ground-truth clinical content.

**Paper:** [arXiv:2508.02808](https://arxiv.org/abs/2508.02808)  
**Code:** This repository

---

## Quickstart

Three steps: host the LLM locally → install the pipeline → run on sample data.

### Step 1 — Host a local LLM

ICARE queries Llama 3.3 70B. Use [vLLM](https://github.com/vllm-project/vllm) to host it locally — no private API key needed.

**GPU requirements:** 2× A100/H100 80 GB (full precision) or 1× A100/H100 40 GB (4-bit AWQ).

**a) Request HuggingFace access** (free):
[https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

**b) Install vLLM** in a Python 3.10 environment:
```bash
conda create -n vllm-server python=3.10 -y
conda activate vllm-server
pip install -r scripts/local_llm/requirements_local_llm.txt
huggingface-cli login
```

**c) Launch the server** (keep this terminal open):
```bash
bash scripts/local_llm/launch_vllm_server.sh
# Wait for: "Application startup complete."
```

> **Single-GPU (40 GB) option:** Use a 4-bit quantized model:
> ```bash
> ICARE_LOCAL_MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
> ICARE_TENSOR_PARALLEL=1 bash scripts/local_llm/launch_vllm_server.sh
> ```

### Step 2 — Install the ICARE pipeline

In a new terminal:

```bash
git clone https://github.com/nyuolab/ICARE_score.git
cd ICARE_score

conda create -n rrg-eval-clean python=3.8 -y
conda activate rrg-eval-clean
export PYTHONNOUSERSITE=1
wget -O build-constraints.txt https://raw.githubusercontent.com/explosion/thinc/master/build-constraints.txt
PIP_CONSTRAINT=./build-constraints.txt pip install -r requirements.txt \
    "pytz" "python-dateutil" "huggingface-hub>=0.14.1" "bottleneck>=1.3.6" --no-cache-dir

cp .env.local_example .env   # pre-configured to point at the local vLLM server
```

If using a private hosted API instead, copy `.env.example` and fill in `RRGEVAL_API_KEY`, `RRGEVAL_API_URL`, and `RRGEVAL_MODEL_NAME`.

### Step 3 — Run the end-to-end example

`test_data/sample_iuxray_reports.csv` contains 10 IU-Xray report pairs. Run the full ICARE pipeline with plain Python (no SLURM required):

```bash
conda activate rrg-eval-clean

# Generate MCQs from ground-truth and generated reports
python src/mcq_generation.py \
    --input_csv test_data/sample_iuxray_reports.csv \
    --output_dir test_data/output \
    --reference gt --num_questions 5 --seed 123

python src/mcq_generation.py \
    --input_csv test_data/sample_iuxray_reports.csv \
    --output_dir test_data/output \
    --reference gen --num_questions 5 --seed 123

# Filter to report-dependent questions (on shuffled answer choices)
python src/mcq_filtering.py \
    --input-json test_data/output/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_data.json \
    --output-dir test_data/output/shuffled_ans_choices_data/gt_reports_as_ref/mcqa_filtering \
    --seed 123

python src/mcq_filtering.py \
    --input-json test_data/output/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_data.json \
    --output-dir test_data/output/shuffled_ans_choices_data/gen_reports_as_ref/mcqa_filtering \
    --seed 123

# Compute ICARE agreement scores
python src/mcqa_evaluation.py \
    --base_dir test_data/output \
    --data_type shuffled_ans_choices_data \
    --seed 123 \
    --gen_report_csv_file test_data/sample_iuxray_reports.csv \
    --gt_report_csv_file  test_data/sample_iuxray_reports.csv
```

These steps are also wrapped in `bash scripts/example_test/run_eval.sh`.

**Output files:**
```
test_data/output/shuffled_ans_choices_data/
├── gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv   ← ICARE-GT
└── gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv  ← ICARE-GEN
```

`agreement_percentage` in these files is the ICARE score (0–100). **ICARE-AVG** is the mean of ICARE-GT and ICARE-GEN.

---

## Running on your own data

Provide a CSV with columns `ground_truth_report` and `generated_report` (and optionally `id`). Replace the paths in the commands above with your own. Use `--num_questions 40` for production runs (the sample test uses 5 for speed). For ReXVal and RadPref we used `--num_questions 60`.

---

## Full experiments from the paper (SLURM)

All scripts support both `sbatch` (SLURM) and plain `bash`.

### Datasets

| Dataset | Access |
|---------|--------|
| IU X-ray (590 test studies) | [NLM Open-i](https://openi.nlm.nih.gov/) |
| ReXVal | [PhysioNet](https://doi.org/10.13026/2fp8-qr71) |
| RadPref | [CRIMSON GitHub](https://github.com/CRIMSON-radiology/crimson) |

### RRG models (IU-Xray experiments)

| Model | HuggingFace |
|-------|-------------|
| MAIRA-2 | [microsoft/maira-2](https://huggingface.co/microsoft/maira-2) |
| CheXpertPlus MIMIC | [IAMJB/mimic-cxr-findings-baseline](https://huggingface.co/IAMJB/mimic-cxr-findings-baseline) |
| CheXpertPlus CheX+MIMIC | [IAMJB/chexpert-mimic-cxr-findings-baseline](https://huggingface.co/IAMJB/chexpert-mimic-cxr-findings-baseline) |

All three models use greedy decoding, so generated reports are identical across model seeds. The 5 model seeds exist to support clustering robustness analysis.

### ICARE — dynamic questions

We ran 5 evaluation seeds (123, 456, 789, 202, 101) with 40 questions/report for IU-Xray and 60 for ReXVal/RadPref.

```bash
# IU-Xray (one script per RRG model)
sbatch scripts/iuxray_data/maira2.sh
sbatch scripts/iuxray_data/chexpertplus_mimic.sh
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic.sh

# RexVal
sbatch scripts/rexval_data/icare_rexval.sh

# RadPref
sbatch scripts/radpref_data/icare_radpref.sh
```

Edit `EVAL_SEED`, `MODEL_SEED`, `INPUT_CSV`, and `OUTPUT_DIR` at the top of each script as needed.

### ICARE — predefined question bank

Generate 45 questions with Claude ([example prompt](https://claude.ai/share/b11b3168-212d-4916-baf2-c7fb8b348da8)) or write your own in the same JSON format. Save to `outputs/predefined_ques_list/mcqa_data.json`. The evaluation scripts convert it to CSV automatically.

```bash
sbatch scripts/iuxray_data/icare_predefined.sh
sbatch scripts/rexval_data/icare_rexval_predefined.sh
sbatch scripts/radpref_data/icare_radpref_predefined.sh
```

### Analysis and plots

```bash
# IU-Xray quantitative results (bar chart + Bradley-Terry ranking)
conda activate rrg-eval-clean && python plot_iuxray_quantitative.py

# RexVal correlation analysis
conda activate rrg-eval-clean && python scripts/rexval_data/plot_rexval_correlation.py

# RadPref correlation analysis
conda activate rrg-eval-clean && python scripts/radpref_data/plot_radpref_correlation.py
```

Output locations:
- IU-Xray plots and CSVs → `outputs/IU_xray/plots/`
- ReXVal plots and CSVs → `outputs/rexval/plots/`
- RadPref plots and CSVs → `outputs/radpref/`

Full results compilation: [src/results_compilation.ipynb](src/results_compilation.ipynb)  
Human evaluation (NYU Langone): [Google Colab](https://colab.research.google.com/drive/1Dtzzek2cYjro8Gm8jb0CjfcH94mpR6Rl)

### Question categorization and cluster analysis

```bash
bash scripts/example_test/run_question_categorization.sh   # quick test on sample data
```

For the full experiment across all models and seeds, see [scripts/iuxray_data/question_categorization_and_analysis/](scripts/iuxray_data/question_categorization_and_analysis/).

> If your compute nodes lack HuggingFace access, pre-download MedCPT-Query-Encoder:
> ```bash
> git clone https://huggingface.co/ncbi/MedCPT-Query-Encoder /path/to/MedCPT
> # Add to .env: MEDCPT_MODEL_PATH=/path/to/MedCPT
> ```

---

## Auxiliary experiments

### Baseline metrics

Baseline scores were computed using the following open-source implementations. Refer to each project's own setup instructions to install and run them.

| Metric | Reference |
|--------|-----------|
| BLEU, BERTScore, SembScore, RadGraph, RadCliQ-v0 | [rajpurkarlab/CXR-Report-Metric](https://github.com/rajpurkarlab/CXR-Report-Metric) |
| GREEN | [StanfordAIMI/GREEN](https://github.com/StanfordAIMI/GREEN) |
| AlignScore | [yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore) |
| CRIMSON | [CRIMSON GitHub](https://github.com/CRIMSON-radiology/crimson) |

> **Note on RadCliQ-v0:** Lower is better, so it is negated in all plots (`1/avg_score` in the quantitative bar chart; negated value for correlation analyses).

### Perturbation experiments

**Random word/character deletion** — tests that ICARE degrades proportionally as more of the report is randomly deleted:

```bash
# IU-Xray — word level
sbatch scripts/iuxray_data/maira2_perturbed_word_level.sh
sbatch scripts/iuxray_data/chexpertplus_mimic_perturbed_word_level.sh
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic_perturbed_word_level.sh

# IU-Xray — character level
sbatch scripts/iuxray_data/maira2_perturbed.sh
sbatch scripts/iuxray_data/chexpertplus_mimic_perturbed.sh
sbatch scripts/iuxray_data/chexpertplus_chexpertplus_mimic_perturbed.sh
```

Perturbation agreement plots:
```bash
sbatch scripts/iuxray_data/plot_agreement_with_perturbation_stats.sh
```

**Controlled clinical deletion** — tests that deleting clinically relevant words degrades ICARE more than deleting non-clinical or randomly chosen words (all conditions matched on total words deleted):

```bash
# Step 1 — Generate perturbed report CSVs (CPU job, ~1 hr)
sbatch scripts/iuxray_data/controlled_perturbation_generate.sh

# Step 2 — ICARE evaluation (GPU array, 13 tasks; run after Step 1)
sbatch scripts/iuxray_data/controlled_perturbation_icare.sh

# Step 3 — Plot results (run after Step 2)
conda activate rrg-eval-clean && python src/plot_controlled_perturbation.py
```
