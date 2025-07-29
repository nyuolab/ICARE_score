import pandas as pd
import json
import os
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import argparse
import random
from utils import make_llama_request, ensure_dir
from config import Config
import csv
import ast  # Add this import at the top


def get_model_prediction(
    report: str, 
    question: str,
    options: Dict[str, str],
    pred_using_report_setting: str,
    url: str = Config.API_URL,
    api_key: str = Config.API_KEY,
    timeout: int = Config.DEFAULT_TIMEOUT,
    max_tokens: int = Config.FILTERING_MAX_TOKENS,
    temperature: float = Config.DEFAULT_TEMPERATURE,
    top_p: float = Config.DEFAULT_TOP_P,
    n: int = Config.DEFAULT_N,
    seed: int = Config.DEFAULT_SEED
) -> Optional[str]:
    """Get model's prediction for a question."""

    if pred_using_report_setting == "using_report":
        prompt = f"""Given the following radiology report:
        "{report}"
    
        Answer the following question:
        {question}
    
        Options:
        A) {options['A']}
        B) {options['B']}
        C) {options['C']}
        D) {options['D']}
        
        Your life depends on providing ONLY a single letter (A, B, C, or D) as your answer. 
        Do not include any other text, punctuation, or explanation.
        Format: Just the letter.
        Example correct format: A
        Example incorrect formats: A., The answer is A, Option A"""
    else:
        prompt = f"""Answer the following question:
        {question}
    
        Options:
        A) {options['A']}
        B) {options['B']}
        C) {options['C']}
        D) {options['D']}
    
        Your life depends on providing ONLY a single letter (A, B, C, or D) as your answer. 
        Do not include any other text, punctuation, or explanation.
        Format: Just the letter.
        Example correct format: A
        Example incorrect formats: A., The answer is A, Option A"""

    
    return make_llama_request(
        prompt=prompt, 
        url=url,
        api_key=api_key,
        max_tokens=max_tokens, 
        temperature=temperature, 
        timeout=timeout,
        model="llama3-3-70b-chat",
        seed=seed,
        top_p=top_p,
        n=n,
        stream=False,
        stop=None, 
        frequency_penalty=0
    )


def predict_answers_for_mcqs(
    input_json: str,
    output_dir: str,
    pred_setting: str,
    seed: int = 123
) -> pd.DataFrame:
    """Generate predictions for MCQs."""
    try:
        with open(input_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_json} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {input_json}")

    # Create CSV file and writer
    output_file = f"{output_dir}/for_filtering_predictions_{pred_setting}.csv"
    csv_headers = ['Index', 'Report_ID', 'Question_ID', 'Question_Text', 'Options', 'Predicted_Answer', 'Correct_Answer']
    
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        csv_writer.writeheader()
        
        results = []
        index = 0
        for report_id, report_data in enumerate(tqdm(data['mcq_data'])):
            report = report_data['report']
            for question in report_data['questions']:
                # print("okbye",type(question['options']))
                pred_answer = get_model_prediction(
                    report,
                    question['question_text'],
                    question['options'],
                    pred_setting,
                    seed=seed
                )
                
                result = {
                    'Index': index,
                    'Report_ID': report_id,
                    'Question_ID': question['question_id'],
                    'Question_Text': question['question_text'],
                    # 'Options': json.dumps(question['options']),  # Convert dict to JSON string
                    'Options': question['options'],
                    'Predicted_Answer': pred_answer,
                    'Correct_Answer': question['correct_answer']
                }
                
                results.append(result)
                csv_writer.writerow(result)
                index += 1

    df = pd.DataFrame(results)
    # df['Options'] = df['Options'].apply(json.loads) 
    df.to_csv(output_file, index=False)
    return df

def shuffle_filtered_answers(filtered_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """
    Shuffle answer choices for filtered questions while maintaining correct answer mapping.
    
    Args:
        filtered_df: DataFrame containing questions with their options and correct answers
        seed: Random seed for reproducible shuffling (default: 123)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    shuffled_df = filtered_df.copy()
    for idx in shuffled_df.index:
        # Get current options and correct answer
        options = shuffled_df.at[idx, 'Options']
        if isinstance(options, str):
            # Just use eval() - it handles Python dict format correctly
            options = eval(options)
            
        current_correct = shuffled_df.at[idx, 'Correct_Answer']
        
        # Create list of option letters and their corresponding texts
        option_items = list(options.items())
        
        # Shuffle the option texts while keeping track of original letters
        random.shuffle(option_items)
        
        # Create new options dictionary with shuffled texts
        new_options = {}
        for i, (old_letter, text) in enumerate(option_items):
            new_letter = chr(65 + i)  # A, B, C, D
            new_options[new_letter] = text
            # Update correct answer if this was the correct option
            if old_letter == current_correct:
                shuffled_df.at[idx, 'Correct_Answer'] = new_letter
        
        shuffled_df.at[idx, 'Options'] = str(new_options)  # Store back as string with single quotes
    
    return shuffled_df

def calculate_statistics(df: pd.DataFrame, output_dir: str) -> Dict:
    """Calculate and save statistics about questions answered with and without report."""
    
    # Dataset-level statistics
    dataset_stats = {
        'Total_Questions': len(df),
        'Dataset_Accuracy_With_Report': (df['Predicted_Answer_with_report'] == df['Correct_Answer']).mean() * 100,
        'Dataset_Accuracy_Without_Report': (df['Predicted_Answer_without_report'] == df['Correct_Answer']).mean() * 100
    }
    
    # Calculate questions answered correctly only with report
    correct_with_report = df['Predicted_Answer_with_report'] == df['Correct_Answer']
    incorrect_without_report = df['Predicted_Answer_without_report'] != df['Correct_Answer']
    report_dependent_questions = df[correct_with_report & incorrect_without_report].copy()
    
    dataset_stats['Report_Dependent_Questions_count'] = len(report_dependent_questions)
    dataset_stats['Report_Dependent_Questions_percentage'] = (len(report_dependent_questions) / len(df)) * 100
    
    # Save report-dependent questions
    report_dependent_questions.to_csv(f"{output_dir}/filtered_questions.csv", index=False)
    
    # Report-level statistics
    report_stats = []
    report_dependent_counts = []
    
    for report_id in df['Report_ID'].unique():
        report_df = df[df['Report_ID'] == report_id]
        
        correct_with_report = (report_df['Predicted_Answer_with_report'] == report_df['Correct_Answer']).mean() * 100
        correct_without_report = (report_df['Predicted_Answer_without_report'] == report_df['Correct_Answer']).mean() * 100
        
        # Get report-dependent questions for this report
        report_dependent_mask = (
            (report_df['Predicted_Answer_with_report'] == report_df['Correct_Answer']) &
            (report_df['Predicted_Answer_without_report'] != report_df['Correct_Answer'])
        )
        report_dependent_count = report_dependent_mask.sum()
        report_dependent_counts.append(report_dependent_count)
        
        report_stats.append({
            'Index': report_df['Index'].tolist(),  # Add Index
            'Report_ID': report_id,
            'Question_Count': len(report_df),
            'Correct_With_Report': correct_with_report,
            'Correct_Without_Report': correct_without_report,
            'Difference': correct_with_report - correct_without_report,
            'Report_Dependent_Count': report_dependent_count,
            'Report_Dependent_Percentage': (report_dependent_count / len(report_df)) * 100,
            'Report_Dependent_Indices': report_df[report_dependent_mask]['Index'].tolist()
        })
    
    # Calculate report-level aggregate statistics
    report_level_stats = {
        'Total_Reports': len(report_stats),
        'Questions_Per_Report': {
            'Mean': np.mean([stat['Question_Count'] for stat in report_stats]),
            'Std': np.std([stat['Question_Count'] for stat in report_stats])
        },
        'Report_Dependent_Questions': {
            'Mean_Count': np.mean(report_dependent_counts),
            'Std_Count': np.std(report_dependent_counts),
            'Mean_Percentage': np.mean([stat['Report_Dependent_Percentage'] for stat in report_stats]),
            'Std_Percentage': np.std([stat['Report_Dependent_Percentage'] for stat in report_stats])
        },
        'Accuracy_Stats': {
            'With_Report': {
                'Mean': np.mean([stat['Correct_With_Report'] for stat in report_stats]),
                'Std': np.std([stat['Correct_With_Report'] for stat in report_stats])
            },
            'Without_Report': {
                'Mean': np.mean([stat['Correct_Without_Report'] for stat in report_stats]),
                'Std': np.std([stat['Correct_Without_Report'] for stat in report_stats])
            },
            'Difference': {
                'Mean': np.mean([stat['Difference'] for stat in report_stats]),
                'Std': np.std([stat['Difference'] for stat in report_stats])
            }
        }
    }
    
    # Save statistics
    
    stats_df = pd.DataFrame(report_stats)
    stats_df.to_csv(f"{output_dir}/per_report_statistics.csv", index=False)

    with open(f"{output_dir}/dataset_statistics.json", 'w') as f:
        json.dump(dataset_stats, f, indent=2)
        
    with open(f"{output_dir}/report_level_aggregate_statistics.json", 'w') as f:
        json.dump(report_level_stats, f, indent=2)
    
    # Print comprehensive summary
    print("\nDataset Statistics:")
    print(f"Total Questions: {dataset_stats['Total_Questions']}")
    print(f"Accuracy with Report: {dataset_stats['Dataset_Accuracy_With_Report']:.2f}%")
    print(f"Accuracy without Report: {dataset_stats['Dataset_Accuracy_Without_Report']:.2f}%")
    
    print("\nReport-Level Statistics:")
    print(f"Total Reports: {report_level_stats['Total_Reports']}")
    print(f"Questions per Report: {report_level_stats['Questions_Per_Report']['Mean']:.1f} ± {report_level_stats['Questions_Per_Report']['Std']:.1f}")
    print("\nReport-Dependent Questions:")
    print(f"Mean Count per Report: {report_level_stats['Report_Dependent_Questions']['Mean_Count']:.1f} ± {report_level_stats['Report_Dependent_Questions']['Std_Count']:.1f}")
    print(f"Mean Percentage per Report: {report_level_stats['Report_Dependent_Questions']['Mean_Percentage']:.1f}% ± {report_level_stats['Report_Dependent_Questions']['Std_Percentage']:.1f}%")
    print("\nAccuracy Statistics (Mean ± Std):")
    print(f"With Report: {report_level_stats['Accuracy_Stats']['With_Report']['Mean']:.1f}% ± {report_level_stats['Accuracy_Stats']['With_Report']['Std']:.1f}%")
    print(f"Without Report: {report_level_stats['Accuracy_Stats']['Without_Report']['Mean']:.1f}% ± {report_level_stats['Accuracy_Stats']['Without_Report']['Std']:.1f}%")
    print(f"Difference: {report_level_stats['Accuracy_Stats']['Difference']['Mean']:.1f}% ± {report_level_stats['Accuracy_Stats']['Difference']['Std']:.1f}%")
    
    return report_dependent_questions

def main():
    parser = argparse.ArgumentParser(description='Filter and Shuffle MCQs')
    parser.add_argument('--input-json', required=True, help='Input JSON file with MCQs')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--seed', default=123, help='Random seed')
    
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    
    # Generate predictions with and without report
    results = {}
    for setting in ["using_report", "without_using_report"]:
        results[setting] = predict_answers_for_mcqs(
            args.input_json, 
            args.output_dir,
            setting,
            seed=args.seed
        )
    
    # Merge predictions and calculate statistics
    # If Options is a dictionary column
    results["using_report"]["Options"] = results["using_report"]["Options"].astype(str)
    results["without_using_report"]["Options"] = results["without_using_report"]["Options"].astype(str)
    merged_predictions = results["using_report"].merge(
        results["without_using_report"],
        on=['Index', 'Report_ID', 'Question_ID', 'Question_Text', 'Options', 'Correct_Answer'],
        suffixes=('_with_report', '_without_report')
    )
    
    # Calculate statistics and get report-dependent questions
    report_dependent_questions = calculate_statistics(merged_predictions, args.output_dir)
    
    # Shuffle report-dependent questions
    shuffled_questions = shuffle_filtered_answers(report_dependent_questions)
    
    # Save filtered and shuffled questions
    shuffled_questions.to_csv(f"{args.output_dir}/filtered_questions_shuffled.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}/")
    print(f"Total questions processed: {len(merged_predictions)}")
    print(f"Questions kept after filtering: {len(report_dependent_questions)}")
    print(f"Filtering rate: {(len(report_dependent_questions)/len(merged_predictions))*100:.2f}%")

if __name__ == "__main__":
    main()