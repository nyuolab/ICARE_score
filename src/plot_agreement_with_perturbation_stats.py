import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def read_agreement_data(file_path):
    """
    Read agreement percentage from a CSV file
    """
    try:
        df = pd.read_csv(file_path)
        return df['agreement_percentage'].iloc[0]
    except:
        print(f"Could not read file: {file_path}")
        return None

def collect_agreement_data(base_path, zero_perturbation_path):
    """
    Collect agreement percentages for all perturbation intensities
    """
    perturbation_intensities = []
    agreement_percentages = []
    
    # Read 0% perturbation data
    zero_agreement = read_agreement_data(zero_perturbation_path)
    if zero_agreement is not None:
        perturbation_intensities.append(0)
        agreement_percentages.append(zero_agreement)
    
    # Read data for perturbation intensities from 5 to 100
    for intensity in range(5, 105, 5):
        file_path = os.path.join(base_path, f"perturbation_degree{intensity}", 
                                "mcq_eval_dataset_level_agreement_stats.csv")
        agreement = read_agreement_data(file_path)
        
        if agreement is not None:
            perturbation_intensities.append(intensity)
            agreement_percentages.append(agreement)
    
    return perturbation_intensities, agreement_percentages

def create_separate_plots(data_dict, output_dir='plots'):
    """
    Create separate plots for each model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model_data in data_dict.items():
        plt.figure(figsize=(10, 6))
        
        # Plot GT questions line
        gt_intensities, gt_agreements = model_data['gt_ques']
        plt.plot(gt_intensities, gt_agreements, 'b-o', 
                linewidth=2, markersize=8, label='Questions from GT Reports')
        
        # Plot Generated questions line
        gen_intensities, gen_agreements = model_data['gen_ques']
        plt.plot(gen_intensities, gen_agreements, 'r-o', 
                linewidth=2, markersize=8, label='Questions from Generated Reports')
        
        plt.xlabel('Perturbation Intensity (%)', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Agreement (%)', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylim(0, 100)
        plt.xlim(0, 105)
        plt.xticks(fontsize=22, weight='bold')
        plt.yticks(fontsize=22, weight='bold')
        # plt.title(f'Agreement Percentage vs Perturbation Intensity\n{model_name}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=24, prop={'weight': 'bold', 'size': 16})
        
        # Add value labels
        # for x, y in zip(gt_intensities, gt_agreements):
        #     plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
        #                 xytext=(0,10), ha='center', color='blue')
        # for x, y in zip(gen_intensities, gen_agreements):
        #     plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
        #                 xytext=(0,-15), ha='center', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'agreement_vs_perturbation_{model_name}.png'), dpi=600)
        plt.close()

def create_combined_plot(data_dict, output_dir='plots'):
    """
    Create a single plot combining all models
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for (model_name, model_data), color in zip(data_dict.items(), colors):
        # Plot GT questions
        gt_intensities, gt_agreements = model_data['gt_ques']
        plt.plot(gt_intensities, gt_agreements, 
                color=color, linestyle='-', marker='o',
                linewidth=2, markersize=8, 
                label=f'{model_name} (GT Questions)')
        
        # Plot Generated questions
        gen_intensities, gen_agreements = model_data['gen_ques']
        plt.plot(gen_intensities, gen_agreements, 
                color=color, linestyle='--', marker='s',
                linewidth=2, markersize=8, 
                label=f'{model_name} (Generated Questions)')
    
    plt.ylim(0, 100)
    plt.xlim(0, 105)
    plt.xticks(fontsize=22, weight='bold')
    plt.yticks(fontsize=22, weight='bold')
    plt.xlabel('Perturbation Intensity (%)', fontsize=24, fontweight='bold')
    plt.ylabel('Agreement(%)', fontsize=24, fontweight='bold')
    plt.title('Agreement Percentage vs Perturbation Intensity\nAll Models', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24, prop={'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plot.png'), bbox_inches='tight', dpi=600)
    plt.close()

def read_report_level_data(file_path):
    """
    Read mean agreement and standard deviation from a CSV file
    """
    try:
        df = pd.read_csv(file_path)
        return df['Mean_Agreement'].iloc[0], df['Std_Deviation'].iloc[0]
    except:
        print(f"Could not read file: {file_path}")
        return None, None

def collect_report_level_data(base_path, zero_perturbation_path):
    """
    Collect mean agreement and std deviation for all perturbation intensities
    """
    perturbation_intensities = []
    mean_agreements = []
    std_deviations = []
    
    # # Read 0% perturbation data
    # zero_mean, zero_std = read_report_level_data(zero_perturbation_path)
    # if zero_mean is not None:
    #     perturbation_intensities.append(0)
    #     mean_agreements.append(zero_mean)
    #     std_deviations.append(zero_std)
    
    # Read data for perturbation intensities from 5 to 50
    for intensity in range(5, 105, 5):
        file_path = os.path.join(base_path, f"perturbation_degree{intensity}", 
                                "mcq_eval_report_level_stats_aggregated.csv")
        mean, std = read_report_level_data(file_path)
        
        if mean is not None:
            perturbation_intensities.append(intensity)
            mean_agreements.append(mean)
            std_deviations.append(std)
    
    return perturbation_intensities, mean_agreements, std_deviations

def create_separate_report_level_plots(data_dict, output_dir='plots'):
    """
    Create separate plots for each model with error bars
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model_data in data_dict.items():
        plt.figure(figsize=(10, 6))
        
        # Plot GT questions line with error bars
        gt_intensities, gt_means, gt_stds = model_data['gt_ques_report']
        plt.errorbar(gt_intensities, gt_means, yerr=gt_stds, 
                    fmt='b-o', linewidth=2, markersize=8, 
                    capsize=5, capthick=2,
                    label='Questions from GT Reports')
        
        # Plot Generated questions line with error bars
        gen_intensities, gen_means, gen_stds = model_data['gen_ques_report']
        plt.errorbar(gen_intensities, gen_means, yerr=gen_stds, 
                    fmt='r-o', linewidth=2, markersize=8,
                    capsize=5, capthick=2,
                    label='Questions from Generated Reports')
        
        plt.xlabel('Perturbation Intensity (%)', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Mean Agreement (%)', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylim(0, 105)
        plt.xlim(0, 105)
        plt.xticks(fontsize=22, weight='bold')
        plt.yticks(fontsize=22, weight='bold')
        # plt.title(f'Report-Level Agreement vs Perturbation Intensity\n{model_name}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=24, prop={'weight': 'bold', 'size': 16})
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'report_level_agreement_{model_name}.png'), bbox_inches='tight', dpi=600)
        plt.close()

def create_combined_report_level_plot(data_dict, output_dir='plots'):
    """
    Create a single plot combining all models with error bars
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for (model_name, model_data), color in zip(data_dict.items(), colors):
        # Plot GT questions
        gt_intensities, gt_means, gt_stds = model_data['gt_ques_report']
        plt.errorbar(gt_intensities, gt_means, yerr=gt_stds,
                    color=color, linestyle='-', marker='o',
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label=f'{model_name} (GT Questions)')
        
        # Plot Generated questions
        gen_intensities, gen_means, gen_stds = model_data['gen_ques_report']
        plt.errorbar(gen_intensities, gen_means, yerr=gen_stds,
                    color=color, linestyle='--', marker='s',
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label=f'{model_name} (Generated Questions)')
    
    plt.ylim(0, 105)
    plt.xlim(0, 105)
    plt.xlabel('Perturbation Intensity (%)', fontsize=24, fontweight='bold')
    plt.ylabel('Mean Agreement(%)', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=22, weight='bold')
    plt.yticks(fontsize=22, weight='bold')
    plt.title('Report-Level Agreement vs Perturbation Intensity\nAll Models', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24, prop={'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_report_level_plot.png'), dpi=600)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot agreement with perturbation stats')
    parser.add_argument('--eval_seed', type=int, default=123,
                      help='Evaluation seed')
    parser.add_argument('--model_seed', type=int, default=1,
                      help='Model seed')
    parser.add_argument('--perturbation_types', type=list, default=["word", "char"],
                      help='Perturbation types to plot')
    parser.add_argument('--base_dir', type=str, 
                      default="/gpfs/data/oermannlab/users/rd3571/RRG_evaluation/MCQ_generation/MCQ_gen_data_our_eval_seed123/RexValTest",
                      help='Base directory')
    parser.add_argument('--dataset', type=str, choices=['iuxray', 'rexval'], default='rexval',
                      help='Dataset name to process')
    
    args = parser.parse_args()

    eval_seed = args.eval_seed
    model_seed = args.model_seed
    perturbation_types = args.perturbation_types
    base_dir = args.base_dir
    dataset = args.dataset
    
    for perturbation_type in perturbation_types:
        data_dict = {}
        
        if dataset == 'rexval':
            # For RexVal, process single path without model/seed structure
            model_name = 'rexval'
            
            # Paths for GT ref questions and perturbed gen reports
            gt_base_path = os.path.join(base_dir, 
                                    f"shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval_perturbed_gen_reports_{perturbation_type}_level")
            gt_zero_path = os.path.join(base_dir,
                                    "shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv")
            
            # Paths for Generated ref questions and perturbed gt reports
            gen_base_path = os.path.join(base_dir,
                                    f"shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval_perturbed_gt_reports_{perturbation_type}_level")
            gen_zero_path = os.path.join(base_dir,
                                    "shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv")
            
            # Collect data for both question types
            gt_data = collect_agreement_data(gt_base_path, gt_zero_path)
            gen_data = collect_agreement_data(gen_base_path, gen_zero_path)
            
            # Collect data for report level
            gt_report_data = collect_report_level_data(gt_base_path, gt_zero_path.replace('dataset_level_agreement_stats', 'report_level_stats_aggregated'))
            gen_report_data = collect_report_level_data(gen_base_path, gen_zero_path.replace('dataset_level_agreement_stats', 'report_level_stats_aggregated'))
            
            data_dict[model_name] = {
                'gt_ques': gt_data,
                'gen_ques': gen_data,
                'gt_ques_report': gt_report_data,
                'gen_ques_report': gen_report_data
            }

        else:  # dataset == 'iuxray'
            # Original code for IU-Xray with multiple models and seeds
            models = {
                'maira-2': [f'seed_{model_seed}'],
                'chexpert-mimic-cxr-findings-baseline': [f'seed_{model_seed}'],
                'mimic-cxr-findings-baseline': [f'seed_{model_seed}'],
            }
            
            for model, seeds in models.items():
                for seed in seeds:
                    model_name = f"{model}-{seed}"
                    
                    # Original paths and data collection code
                    # Paths for GT ref questions and perturbed gen reports
                    gt_base_path = os.path.join(base_dir, model, seed, 
                                            f"shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval_perturbed_gen_reports_{perturbation_type}_level")
                    gt_zero_path = os.path.join(base_dir, model, seed,
                                            "shuffled_ans_choices_data/gt_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv")
                    
                    # Paths for Generated ref questions and perturbed gt reports
                    gen_base_path = os.path.join(base_dir, model, seed,
                                            f"shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval_perturbed_gt_reports_{perturbation_type}_level")
                    gen_zero_path = os.path.join(base_dir, model, seed,
                                            "shuffled_ans_choices_data/gen_reports_as_ref/mcqa_eval/mcq_eval_dataset_level_agreement_stats.csv")
                    
                    # Collect data for both question types
                    gt_data = collect_agreement_data(gt_base_path, gt_zero_path)
                    gen_data = collect_agreement_data(gen_base_path, gen_zero_path)
                    
                    # Collect data for report level
                    gt_report_data = collect_report_level_data(gt_base_path, gt_zero_path.replace('dataset_level_agreement_stats', 'report_level_stats_aggregated'))
                    gen_report_data = collect_report_level_data(gen_base_path, gen_zero_path.replace('dataset_level_agreement_stats', 'report_level_stats_aggregated'))
                    
                    data_dict[model_name] = {
                        'gt_ques': gt_data,
                        'gen_ques': gen_data,
                        'gt_ques_report': gt_report_data,
                        'gen_ques_report': gen_report_data
                    }
        
        output_path = os.path.join(base_dir, f"plots/perturbation_{perturbation_type}_level")
        if dataset == 'iuxray':
            output_path = os.path.join(output_path, f"seed_{model_seed}")
            
        # Create all plots
        create_separate_plots(data_dict, output_dir=output_path)
        create_combined_plot(data_dict, output_dir=output_path)
        create_separate_report_level_plots(data_dict, output_dir=output_path)
        create_combined_report_level_plot(data_dict, output_dir=output_path)

if __name__ == "__main__":
    main() 