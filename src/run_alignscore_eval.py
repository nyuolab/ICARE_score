import argparse
import os

import numpy as np
import pandas as pd
from alignscore import AlignScore


def main():
    parser = argparse.ArgumentParser(description="Run AlignScore and save sample-level + aggregate results.")
    parser.add_argument("--input-csv",  required=True,  help="Input CSV with generated_report and ground_truth_report.")
    parser.add_argument("--output-dir", required=True,  help="Directory for AlignScore outputs.")
    parser.add_argument("--ckpt-path",  required=True,  help="Path to AlignScore checkpoint (.ckpt).")
    parser.add_argument("--model",      default="roberta-base", choices=["roberta-base", "roberta-large"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--eval-mode",  default="nli_sp",
                        choices=["nli_sp", "nli", "bin_sp", "bin"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    refs = df["ground_truth_report"].astype(str).tolist()
    hyps = df["generated_report"].astype(str).tolist()

    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt_path,
        evaluation_mode=args.eval_mode,
    )
    scores = scorer.score(contexts=refs, claims=hyps)

    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]

    sample_df = pd.DataFrame({
        "index":           range(len(df)),
        "study_id":        df["id"] if "id" in df.columns else df.index,
        "alignscore":      scores,
    })
    sample_path = os.path.join(args.output_dir, f"{input_basename}_alignscore_results.csv")
    sample_df.to_csv(sample_path, index=False)

    summary_df = pd.DataFrame([{
        "report_name":       input_basename,
        "alignscore_mean":   float(np.mean(scores)),
        "alignscore_std":    float(np.std(scores)),
        "n_samples":         len(scores),
    }])
    summary_path = os.path.join(args.output_dir, "summary_of_averages.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Sample results  → {sample_path}")
    print(f"Summary         → {summary_path}")
    print(f"AlignScore mean : {np.mean(scores):.4f}  std: {np.std(scores):.4f}  n={len(scores)}")


if __name__ == "__main__":
    main()
