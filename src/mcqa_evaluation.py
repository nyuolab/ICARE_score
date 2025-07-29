import os
import json
import csv
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from utils import make_llama_request, ensure_dir
from config import Config
import argparse


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


def predict_answers_for_mcq_data(
    ques_csv_file: str,
    gen_report_csv_file: str,
    gt_report_csv_file: str,
    output_csv_file: str,
    seed: int = 123
) -> List[Dict]:
    """
    Process MCQ data and generate predictions using both GT and generated reports.
    """
    # Read question IDs from CSV
    ques_ids_csv = pd.read_csv(ques_csv_file)

    # Load csv data to obtain generated and ground truth reports

    gen_report_csv_data = pd.read_csv(gen_report_csv_file)
    
    gt_report_csv_data = pd.read_csv(gt_report_csv_file)

    # Prepare CSV output
    csv_headers = ['Index', 'Report_ID', 'Question_ID', 'Correct_Answer', 
                  'Predicted_Answer_Using_GT', 'Predicted_Answer_Using_Gen', 'Options']
    csvfile = open(output_csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    csv_writer.writeheader()

    results = []
    idx = 0

    # Process each question
    for _, row in tqdm(ques_ids_csv.iterrows(), total=len(ques_ids_csv)):
        report_id = int(row['Report_ID'])
        question_id = int(row['Question_ID'])
        question_text = row['Question_Text']
        options = eval(row['Options'])
        correct_answer = row['Correct_Answer']
        if(idx == 1):
            print(options, type(options))
        
        # Get appropriate reports based on reference
        gt_report = gt_report_csv_data['ground_truth_report'][report_id]
        gen_report = gen_report_csv_data['generated_report'][report_id]

        # Get model predictions
        predicted_answer_using_gt = get_model_prediction(gt_report, question_text, options, "using_report", seed=seed)
        predicted_answer_using_gen = get_model_prediction(gen_report, question_text, options, "using_report", seed=seed)

        result = {
            'Index': idx,
            'Report_ID': report_id,
            'Question_ID': question_id,
            'Correct_Answer': correct_answer,
            'Predicted_Answer_Using_GT': predicted_answer_using_gt,
            'Predicted_Answer_Using_Gen': predicted_answer_using_gen,
            'Options': options
        }
        results.append(result)
        csv_writer.writerow(result)
        csvfile.flush()
        idx += 1

    csvfile.close()
    return results

def calculate_dataset_level_agreement(csv_file: str) -> Dict[str, float]:
    """
    Calculate agreement and disagreement percentages at the dataset level.
    
    Args:
        csv_file: Path to the CSV file containing predictions
        
    Returns:
        Dictionary containing agreement and disagreement percentages
    """
    try:
        df = pd.read_csv(csv_file)
        total_questions = len(df)
        
        # Calculate agreement
        agreements = df['Predicted_Answer_Using_GT'] == df['Predicted_Answer_Using_Gen']
        agreement_count = agreements.sum()
        disagreement_count = total_questions - agreement_count
        
        agreement_pct = (agreement_count / total_questions) * 100
        disagreement_pct = (disagreement_count / total_questions) * 100
        
        return {
            'total_questions': total_questions,
            'agreement_count': int(agreement_count),
            'disagreement_count': int(disagreement_count),
            'agreement_percentage': round(agreement_pct, 2),
            'disagreement_percentage': round(disagreement_pct, 2)
        }
    except Exception as e:
        print(f"Error calculating dataset agreement: {e}")
        return None

def plot_report_level_agreement(csv_file: str, output_dir: str, reference: str):
    """
    Plot report-level agreement statistics.
    
    Args:
        csv_file: Path to the CSV file containing predictions
        output_dir: Directory to save the plot
        reference: Reference type ('gt' or 'gen')
    """
    try:
        df = pd.read_csv(csv_file)
        report_agreements = []
        
        # Calculate agreement percentage for each report
        for report_id in df['Report_ID'].unique():
            report_df = df[df['Report_ID'] == report_id]
            total = len(report_df)
            matches = len(report_df[report_df['Predicted_Answer_Using_GT'] == 
                                  report_df['Predicted_Answer_Using_Gen']])
            agreement_pct = (matches / total) * 100 if total > 0 else 0
            report_agreements.append(agreement_pct)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(report_agreements, bins=20, edgecolor='black')
        plt.xlabel('Agreement Percentage', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Number of Reports', fontsize=24, fontweight='bold', labelpad=15)
        # plt.title(f'Distribution of GT-GEN Agreement Across Reports\n({reference.upper()} Reference)', fontsize=24)
        plt.xticks(fontsize=22, weight='bold')
        plt.yticks(fontsize=22, weight='bold')
        
        # Add mean and std dev lines
        mean_agreement = np.mean(report_agreements)
        std_agreement = np.std(report_agreements)
        plt.axvline(mean_agreement, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_agreement:.1f}%')
        plt.axvline(mean_agreement + std_agreement, color='g', linestyle=':', linewidth=2,
                   label=f'SD: {std_agreement:.1f}%')
        plt.axvline(mean_agreement - std_agreement, color='g', linestyle=':', linewidth=2)
        
        plt.legend(fontsize=24, prop={'weight': 'bold', 'size': 16})
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join(output_dir, f'mcq_eval_report_level_agreement_hist.png')
        plt.savefig(plot_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        # Save report-level statistics
        report_stats = pd.DataFrame({
            'Report_ID': df['Report_ID'].unique(),
            'Agreement_Percentage': report_agreements
        })
        stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats.csv')
        report_stats.to_csv(stats_file, index=False)

        aggregated_stats_file = os.path.join(output_dir, f'mcq_eval_report_level_stats_aggregated.csv')
        aggregated_stats = pd.DataFrame({
            'Mean_Agreement': [mean_agreement],
            'Std_Deviation': [std_agreement]
        })
        aggregated_stats.to_csv(aggregated_stats_file, index=False)
        
        print(f"\nReport-level statistics:")
        print(f"Mean agreement: {mean_agreement:.1f}%")
        print(f"Standard deviation: {std_agreement:.1f}%")
        print(f"Plot saved to {plot_file}")
        print(f"Report-level statistics saved to {stats_file}")
        
    except Exception as e:
        print(f"Error plotting report-level agreement: {e}")

def main():
    parser = argparse.ArgumentParser(description='MCQ Evaluation Script')
    parser.add_argument('--base_dir', type=str, default='../MCQ_gen_data/RexValTest',
                      help='Base directory path (default: ../MCQ_gen_data/RexValTest)')
    parser.add_argument('--data_type', type=str, default='orig_data',
                      help='Type of data (default: orig_data)')
    parser.add_argument('--seed', type=int, default=123,
                      help='Random seed (default: 123)')
    parser.add_argument('--gen_report_csv_file', type=str, default='',
                      help='Generated report CSV file (default: )')
    parser.add_argument('--gt_report_csv_file', type=str, default='',
                      help='Ground truth report CSV file (default: )')
    parser.add_argument('--perturbation', type=str, default='',
                      help='Perturbation (default: )')
    parser.add_argument('--perturbation_degree', type=int, default=0,
                      help='Perturbation degree (default: 0)')
    parser.add_argument('--perturbation_type', type=str, default='char',
                      help='Perturbation type (default: char)')
    
    args = parser.parse_args()
    
    base_directory = args.base_dir
    data_type = args.data_type
    seed = args.seed
    
    os.makedirs(base_directory, exist_ok=True)
    
    # Process both gt and gen references
    for ques_reference in ['gen', 'gt']:
        if args.perturbation == "perturbed":
            if ques_reference == "gt":
                gt_report_csv_file = args.gt_report_csv_file
                gen_report_csv_file = os.path.join(base_directory, f"perturbed_reports_{args.perturbation_type}_level", f'perturbed_{args.perturbation_degree}percent.csv')
                output_dir = os.path.join(base_directory, data_type, f'{ques_reference}_reports_as_ref', f'mcqa_eval_perturbed_gen_reports_{args.perturbation_type}_level', f'perturbation_degree{args.perturbation_degree}')
            else:
                gt_report_csv_file = os.path.join(base_directory, f"perturbed_reports_{args.perturbation_type}_level", f'perturbed_{args.perturbation_degree}percent.csv')
                gen_report_csv_file = args.gen_report_csv_file
                output_dir = os.path.join(base_directory, data_type, f'{ques_reference}_reports_as_ref', f'mcqa_eval_perturbed_gt_reports_{args.perturbation_type}_level', f'perturbation_degree{args.perturbation_degree}')
        else:
            gen_report_csv_file = args.gen_report_csv_file
            gt_report_csv_file = args.gt_report_csv_file
            output_dir = os.path.join(base_directory, data_type, f'{ques_reference}_reports_as_ref', 'mcqa_eval')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Input files
        ques_csv_file = os.path.join(base_directory, data_type,
                                   f'{ques_reference}_reports_as_ref/mcqa_filtering/filtered_questions_shuffled.csv')
        
        # Output file
        mcqa_eval_ans_predictions_output_csv_file = os.path.join(output_dir, f"mcqa_eval_answer_predictions.csv")
        
        print(f"\nProcessing MCQ evaluation for {ques_reference} reference...")
        results = predict_answers_for_mcq_data( 
            ques_csv_file,
            gen_report_csv_file,
            gt_report_csv_file,
            mcqa_eval_ans_predictions_output_csv_file,
            seed=seed
        )
        print(f"Results saved to {mcqa_eval_ans_predictions_output_csv_file}")
        
        # Calculate dataset-level agreement
        agreement_stats = calculate_dataset_level_agreement(mcqa_eval_ans_predictions_output_csv_file)
        if agreement_stats:
            print(f"\nDataset-level statistics:")
            print(f"Total questions: {agreement_stats['total_questions']}")
            print(f"Agreement: {agreement_stats['agreement_percentage']}%")
            print(f"Disagreement: {agreement_stats['disagreement_percentage']}%")
            
            # Save agreement stats
            stats_file = os.path.join(output_dir, f'mcq_eval_dataset_level_agreement_stats.csv')
            pd.DataFrame([agreement_stats]).to_csv(stats_file, index=False)
            print(f"Agreement statistics saved to {stats_file}")
        
        # Generate report-level analysis
        plot_report_level_agreement(mcqa_eval_ans_predictions_output_csv_file, output_dir, ques_reference)

if __name__ == "__main__":
    main() 