import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import argparse
import ast

def parse_arguments():
    """
    Parse command line arguments for configuration
    """
    parser = argparse.ArgumentParser(description='Create combined MCQ questions data')
    
    parser.add_argument('--model_seeds', type=str, default='[1, 2, 3, 4, 5]',
                        help='List of model training seeds (as string representation of list)')
    parser.add_argument('--eval_seeds', type=str, default='[123, 456, 789, 202, 101]',
                        help='List of evaluation seeds (as string representation of list)')
    parser.add_argument('--datasets', type=str, default='["IU_xray"]',
                        help='List of datasets (as string representation of list)')
    parser.add_argument('--models', type=str, default='["chexpert-mimic-cxr-findings-baseline", "mimic-cxr-findings-baseline", "maira-2"]',
                        help='List of models (as string representation of list)')
    parser.add_argument('--metrics', type=str, default='["gt_reports_as_ref", "gen_reports_as_ref"]',
                        help='List of metrics (as string representation of list)')
    parser.add_argument('--base_dir', type=str, 
                        default='/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_seed',
                        help='Base directory for data')
    parser.add_argument('--output_dir', type=str,
                        default='/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_summarized_results/IU_xray/question_categorization_and_analysis',
                        help='Output directory for combined data')
    
    return parser.parse_args()

def load_and_combine_data(model_seeds, eval_seeds, datasets, models, metrics, base_dir):
    """
    Load and combine data from multiple files into a single dataframe,
    keeping only essential columns
    """
    combined_data = []
    
    for dataset in datasets:
        for model in models:
            for model_seed in model_seeds:
                for eval_seed in eval_seeds:
                    for metric in metrics:  # Added loop for metrics
                        # Path to filtered questions
                        filtered_questions_path = os.path.join(
                            base_dir + str(eval_seed),
                            dataset,
                            model,
                            f"seed_{model_seed}",
                            f"shuffled_ans_choices_data/{metric}/mcqa_filtering/filtered_questions_shuffled.csv"
                        )
                        
                        # Path to answer predictions
                        predictions_path = os.path.join(
                            base_dir + str(eval_seed),
                            dataset,
                            model,
                            f"seed_{model_seed}",
                            f"shuffled_ans_choices_data/{metric}/mcqa_eval/mcqa_eval_answer_predictions.csv"
                        )
                        
                        # Check if both files exist
                        if not (os.path.exists(filtered_questions_path) and os.path.exists(predictions_path)):
                            print(f"Missing files for {dataset}, {model}, model_seed={model_seed}, eval_seed={eval_seed}, metric={metric}")
                            continue
                        
                        try:
                            # Load filtered questions
                            questions_df = pd.read_csv(filtered_questions_path)
                            
                            # Keep only essential columns from questions
                            if 'Question_Text' in questions_df.columns:
                                questions_df = questions_df[['Report_ID', 'Question_ID', 'Question_Text']]
                            else:
                                print(f"Warning: 'Question_Text' not found in {filtered_questions_path}")
                                continue
                            
                            # Load predictions
                            predictions_df = pd.read_csv(predictions_path)
                            
                            # Keep only essential columns from predictions
                            essential_pred_cols = ['Report_ID', 'Question_ID', 'Correct_Answer', 
                                                  'Predicted_Answer_Using_Gen', 'Predicted_Answer_Using_GT']
                            available_pred_cols = [col for col in essential_pred_cols if col in predictions_df.columns]
                            
                            if len(available_pred_cols) < 3:  # Need at least ID columns and one prediction
                                print(f"Warning: Not enough essential columns in {predictions_path}")
                                continue
                                
                            predictions_df = predictions_df[available_pred_cols]
                            
                            # Merge the dataframes
                            merged_df = pd.merge(
                                questions_df, 
                                predictions_df, 
                                on=["Report_ID", "Question_ID"], 
                                how="inner"
                            )
                            
                            # Add metadata columns
                            merged_df["dataset"] = dataset
                            merged_df["model_name"] = model
                            merged_df["model_seed"] = model_seed
                            merged_df["eval_seed"] = eval_seed
                            merged_df["metric"] = metric  # Added metric column
                            
                            combined_data.append(merged_df)
                        except Exception as e:
                            print(f"Error processing {dataset}, {model}, model_seed={model_seed}, eval_seed={eval_seed}, metric={metric}: {e}")
    
    # Combine all dataframes
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return combined_df
    else:
        print("No data was loaded successfully.")
        return None

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert string representations to actual lists
    model_seeds = ast.literal_eval(args.model_seeds)
    eval_seeds = ast.literal_eval(args.eval_seeds)
    datasets = ast.literal_eval(args.datasets)
    models = ast.literal_eval(args.models)
    metrics = ast.literal_eval(args.metrics)
    
    print("Configuration:")
    print(f"Model seeds: {model_seeds}")
    print(f"Evaluation seeds: {eval_seeds}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Metrics: {metrics}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load the combined data
    print("Loading and combining data...")
    combined_df = load_and_combine_data(model_seeds, eval_seeds, datasets, models, metrics, args.base_dir)

    if combined_df is not None:
        print(f"Combined data shape: {combined_df.shape}")
        print("\nSample of combined data:")
        print(combined_df.head())
        
        # Save the combined data
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "combined_mcqa_data.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")
    else:
        print("No data was loaded successfully.")

if __name__ == "__main__":
    main()