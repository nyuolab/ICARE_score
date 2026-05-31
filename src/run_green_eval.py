import argparse
import os
import sys

import pandas as pd

GREEN_ROOT = "/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/GREEN"
if GREEN_ROOT not in sys.path:
    sys.path.insert(0, GREEN_ROOT)

from green_score import GREEN


def main():
    parser = argparse.ArgumentParser(description="Run GREEN and save sample-level + aggregate results.")
    parser.add_argument("--input-csv", required=True, help="Input CSV with generated_report and ground_truth_report.")
    parser.add_argument("--output-dir", required=True, help="Directory for GREEN outputs.")
    parser.add_argument("--model-name", default="StanfordAIMI/GREEN-radllama2-7b")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    hyps = df["generated_report"]
    refs = df["ground_truth_report"]

    green_scorer = GREEN(args.model_name, output_dir=args.output_dir)
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]

    sample_df = pd.DataFrame(
        {
            "index": range(len(df)),
            "study_id": df["id"] if "id" in df.columns else df.index,
            "green_score": green_score_list,
        }
    )
    for col in result_df.columns:
        if col in {"reference", "predictions", "green_analysis", "green_score"}:
            continue
        sample_df[col] = result_df[col].values

    sample_path = os.path.join(args.output_dir, f"{input_basename}_green_results.csv")
    sample_df.to_csv(sample_path, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "report_name": input_basename,
                "green_score_mean": mean,
                "green_score_std": std,
                "n_samples": len(green_score_list),
            }
        ]
    )
    summary_path = os.path.join(args.output_dir, "summary_of_averages.csv")
    summary_df.to_csv(summary_path, index=False)

    summary_txt_path = os.path.join(args.output_dir, f"{input_basename}_green_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Sample results saved to: {sample_path}")
    print(f"Summary saved to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
