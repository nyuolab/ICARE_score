#!/usr/bin/env python3
"""
Generate controlled perturbed GT reports for the matched deletion experiment.

Three conditions, each matched on TOTAL WORDS DELETED per report:

  clinical_ctrl   Delete N% of clinical tokens (radiology vocabulary).
                  This fixes the per-report deletion count used by the
                  other two conditions.

  nonclinical     Delete the SAME COUNT from the non-clinical token pool.
                  Clinical terms AND negation words (no, not, without, …)
                  are protected — sentence polarity is never reversed.

  random_ctrl     Delete the SAME COUNT of randomly chosen tokens.

Rates are expressed as a fraction of the CLINICAL token pool per report,
so 10/20/30/40 % = fraction of clinical vocabulary removed.

Output CSVs (structure expected by mcqa_evaluation.py):
  generated_report    = perturbed ground_truth_report   (candidate)
  ground_truth_report = original                         (reference, unchanged)

Directory layout (lives inside the existing model base_dir):
  {output_dir}/perturbed_reports_clinical_ctrl_level/perturbed_{N}percent.csv
  {output_dir}/perturbed_reports_nonclinical_level/perturbed_{N}percent.csv
  {output_dir}/perturbed_reports_random_ctrl_level/perturbed_{N}percent.csv
"""

import argparse
import os
import random
import re
import logging
from typing import List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Clinical vocabulary (RadLex-consistent) ────────────────────────────────────
CLINICAL_TERMS = {
    "heart", "cardiac", "cardiomediastinal",
    "aorta", "aortic", "pericardium", "pericardial",
    "ventricle", "ventricular", "atrium", "atrial",
    "lung", "lungs", "pulmonary", "lobe", "lobes",
    "bronchial", "bronchus", "airway",
    "pleura", "pleural", "hemithorax", "costophrenic",
    "mediastinum", "mediastinal", "hilar", "hilum",
    "trachea", "tracheal", "carina",
    "diaphragm", "diaphragmatic", "hemidiaphragm",
    "rib", "ribs", "clavicle", "clavicular",
    "sternum", "sternal", "scapula", "scapular",
    "spine", "vertebra", "vertebrae", "vertebral", "thoracic",
    "humerus", "humeral", "shoulder", "chest", "thorax",
    "opacity", "opacities", "consolidation", "consolidations",
    "infiltrate", "infiltrates", "infiltration",
    "effusion", "effusions", "atelectasis", "collapse",
    "pneumothorax", "pneumothoraces", "pneumomediastinum", "pneumonia",
    "edema", "congestion", "cardiomegaly", "enlarged", "enlargement",
    "mass", "masses", "nodule", "nodules",
    "lesion", "lesions", "density", "densities",
    "calcification", "calcifications", "calcified",
    "thickening", "subpleural", "haziness", "airspace",
    "emphysema", "hyperinflation", "hyperexpansion",
    "fibrosis", "fibrotic", "adenopathy", "lymphadenopathy",
    "elevation", "elevated", "displacement", "displaced",
    "deformity", "deformities", "fracture", "fractures",
    "lucency", "lucencies", "blunting", "hernia",
    "widening", "narrowing", "prominence", "prominent",
    "silhouette", "effacement", "discoid", "subsegmental",
    "subcutaneous", "vascular",
    "bilateral", "bibasilar", "unilateral", "left", "right",
    "upper", "lower", "middle",
    "apical", "basal", "basilar",
    "peripheral", "central", "perihilar",
    "retrocardiac", "retrosternal", "paramediastinal", "parahilar",
    "anterior", "posterior", "lateral", "medial", "superior", "inferior",
    "mild", "moderate", "severe", "minimal", "small", "large", "subtle",
    "increased", "decreased",
    "stable", "unchanged", "resolved", "resolving", "worsening", "improving",
    "new", "acute", "chronic", "subacute", "normal", "abnormal",
    "catheter", "tube", "pacemaker", "lead", "leads",
    "wire", "line", "port", "stent",
    "prosthesis", "implant", "device", "electrode",
}

# ── Negation / polarity words: protect these in nonclinical condition ──────────
# These are not in CLINICAL_TERMS but reverse clinical meaning if deleted.
NEGATION_TERMS = {
    "no", "not", "without", "nor", "negative", "absent", "free", "clear",
}

_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _strip(token: str) -> str:
    return _PUNCT_RE.sub("", token).lower()


def _is_clinical(token: str) -> bool:
    return _strip(token) in CLINICAL_TERMS


def _is_nonclinical(token: str) -> bool:
    """Safe to delete in the nonclinical condition (not clinical, not negation)."""
    s = _strip(token)
    return s not in CLINICAL_TERMS and s not in NEGATION_TERMS


def _delete(tokens: List[str], pool: List[int],
            n: int, rng: random.Random) -> str:
    chosen = set(rng.sample(pool, min(n, len(pool))))
    return " ".join(t for i, t in enumerate(tokens) if i not in chosen).strip()


def perturb_clinical(text: str, rate: float,
                     seed: Optional[int] = None) -> Tuple[str, int]:
    """Delete `rate` fraction of clinical tokens. Returns (text, n_deleted)."""
    rng = random.Random(seed)
    tokens = text.split()
    pool = [i for i, t in enumerate(tokens) if _is_clinical(t)]
    n = round(len(pool) * rate)
    if n == 0:
        return text, 0
    return _delete(tokens, pool, n, rng), min(n, len(pool))


def perturb_nonclinical(text: str, n: int,
                         seed: Optional[int] = None) -> str:
    """Delete exactly n tokens from the non-clinical / non-negation pool."""
    rng = random.Random(seed)
    tokens = text.split()
    pool = [i for i, t in enumerate(tokens) if _is_nonclinical(t)]
    return _delete(tokens, pool, n, rng)


def perturb_random_ctrl(text: str, n: int,
                         seed: Optional[int] = None) -> str:
    """Delete exactly n randomly chosen tokens."""
    rng = random.Random(seed)
    tokens = text.split()
    return _delete(tokens, list(range(len(tokens))), n, rng)


def generate_all(input_file: str, output_dir: str,
                 rates: List[float], seed: int = 123) -> None:
    df = pd.read_csv(input_file)
    if "ground_truth_report" not in df.columns:
        raise ValueError("CSV must contain 'ground_truth_report' column")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Loaded {len(df)} reports from {input_file}")

    dirs = {
        "clinical_ctrl": os.path.join(output_dir, "perturbed_reports_clinical_ctrl_level"),
        "nonclinical":   os.path.join(output_dir, "perturbed_reports_nonclinical_level"),
        "random_ctrl":   os.path.join(output_dir, "perturbed_reports_random_ctrl_level"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    gt_texts = df["ground_truth_report"].astype(str).tolist()

    # Log clinical token coverage
    clin_counts = [sum(_is_clinical(t) for t in r.split()) for r in gt_texts]
    total_counts = [len(r.split()) for r in gt_texts]
    mean_c = sum(clin_counts) / len(clin_counts)
    mean_t = sum(total_counts) / len(total_counts)
    logging.info(f"GT clinical coverage: {mean_c:.1f}/{mean_t:.1f} tokens ({100*mean_c/mean_t:.1f}%)")
    logging.info(f"Reports with zero clinical tokens: {clin_counts.count(0)}/{len(df)}")

    for rate in sorted(set(rates)):
        rate_pct = int(round(rate * 100))
        fname = f"perturbed_{rate_pct}percent.csv"

        if rate == 0:
            # 0% baseline: gen = original GT (identical across all three conditions)
            out = df.copy()
            out["generated_report"] = out["ground_truth_report"]
            for d in dirs.values():
                out.to_csv(os.path.join(d, fname), index=False)
            logging.info(f"Rate  0%: saved baseline (gen = original GT) — all 3 conditions identical")
            continue

        clin_texts, nonclin_texts, rand_texts, n_list = [], [], [], []
        for i, gt in enumerate(gt_texts):
            rs = seed + i * 100          # unique per-report seed
            c_text, n_del = perturb_clinical(gt, rate, seed=rs)
            nc_text = perturb_nonclinical(gt, n_del, seed=rs + 1)
            r_text  = perturb_random_ctrl(gt, n_del, seed=rs + 2)
            clin_texts.append(c_text)
            nonclin_texts.append(nc_text)
            rand_texts.append(r_text)
            n_list.append(n_del)

        mean_del = sum(n_list) / len(n_list)
        logging.info(f"Rate {rate_pct:2d}%: mean words deleted = {mean_del:.1f}")

        for cond, texts in [("clinical_ctrl", clin_texts),
                             ("nonclinical",   nonclin_texts),
                             ("random_ctrl",   rand_texts)]:
            out = df.copy()
            out["generated_report"] = texts
            out.to_csv(os.path.join(dirs[cond], fname), index=False)

        logging.info(f"       Saved {fname} for all 3 conditions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate controlled perturbed GT reports (clinical_ctrl / nonclinical / random_ctrl). "
            "All three conditions delete the same number of words per report, determined by the "
            "clinical deletion count at the given rate."
        )
    )
    parser.add_argument("--input_csv", required=True,
                        help="Input CSV with ground_truth_report column.")
    parser.add_argument("--output_dir", required=True,
                        help="Base output dir (model's eval base_dir). "
                             "Subdirs perturbed_reports_*_level/ are created here.")
    parser.add_argument("--rates", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help="Fraction of clinical tokens to delete (default: 0 0.1 0.2 0.3 0.4).")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    generate_all(args.input_csv, args.output_dir, args.rates, args.seed)
