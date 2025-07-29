# Analysis of question clusters
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from typing import Dict, Tuple, List, Optional, Union
from matplotlib.gridspec import GridSpec
import argparse


def calculate_agreement(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Calculates the agreement rate between predictions made using generated reports and 
    ground truth reports across specified groupings of questions.
    
    Args:
        df: DataFrame containing question data with columns:
            - Question_ID: Unique identifier for each question
            - Predicted_Answer_Using_Gen: Answer predicted using generated reports
            - Predicted_Answer_Using_GT: Answer predicted using ground truth reports
            - Additional columns used for grouping (e.g., cluster_id, model_name)
        group_cols: List of column names to group by (e.g., ['cluster_id'] or ['model_name', 'cluster_id'])
        
    Returns:
        DataFrame with columns for each grouping variable plus:
            - Agreement: Proportion of questions where predictions using different reference types match
            - Count: Number of questions in each group
    """
    return df.groupby(group_cols).agg(
        Agreement=pd.NamedAgg(
            column='Question_ID',
            aggfunc=lambda x: (
                df.loc[x.index, 'Predicted_Answer_Using_Gen'] == 
                df.loc[x.index, 'Predicted_Answer_Using_GT']
            ).mean()
        ),
        Count=pd.NamedAgg(column='Question_ID', aggfunc='count')
    ).reset_index()


def analyze_clusters(combined_df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze clusters by agreement metrics for both reference types.
    
    Args:
        combined_df: DataFrame containing all question data with cluster assignments
        output_dir: Directory to save analysis results
        
    Returns:
        Tuple of (cluster_results, gt_model_cluster_results, gen_model_cluster_results)
    """
    print("Analyzing clusters...")
    
    # Create separate dataframes for each metric type
    gt_as_ref_df = combined_df[combined_df['metric'] == 'gt_reports_as_ref']
    gen_as_ref_df = combined_df[combined_df['metric'] == 'gen_reports_as_ref']
    
    # Calculate agreement by cluster for GT as reference
    gt_cluster_results = calculate_agreement(gt_as_ref_df, ['cluster_id'])
    gt_cluster_results = gt_cluster_results.rename(columns={
        'cluster_id': 'Cluster',
        'Agreement': 'Agreement_GT_Ref',
        'Count': 'Question_Count'
    })
    
    # Calculate agreement by cluster for Gen as reference
    gen_cluster_results = calculate_agreement(gen_as_ref_df, ['cluster_id'])
    gen_cluster_results = gen_cluster_results.rename(columns={
        'cluster_id': 'Cluster',
        'Agreement': 'Agreement_Gen_Ref',
        'Count': 'Question_Count'
    })
    
    # Merge the results
    cluster_results = pd.merge(gt_cluster_results, gen_cluster_results, on='Cluster', suffixes=('_gt_ref', '_gen_ref'))
    
    # Calculate agreement by model and cluster for GT as reference
    gt_model_cluster_results = calculate_agreement(gt_as_ref_df, ['model_name', 'cluster_id'])
    gt_model_cluster_results = gt_model_cluster_results.rename(columns={
        'model_name': 'Model', 
        'cluster_id': 'Cluster'
    })
    
    # Calculate agreement by model and cluster for Gen as reference
    gen_model_cluster_results = calculate_agreement(gen_as_ref_df, ['model_name', 'cluster_id'])
    gen_model_cluster_results = gen_model_cluster_results.rename(columns={
        'model_name': 'Model', 
        'cluster_id': 'Cluster'
    })
    
    # Add cluster names if available
    if 'cluster_name' in combined_df.columns:
        # Create a mapping from cluster_id to cluster_name
        cluster_name_mapping = combined_df[['cluster_id', 'cluster_name']].drop_duplicates()
        cluster_name_mapping = dict(zip(cluster_name_mapping['cluster_id'], cluster_name_mapping['cluster_name']))
        
        # Apply the mapping to the results
        for df in [gt_cluster_results, gen_cluster_results, gt_model_cluster_results, gen_model_cluster_results]:
            df['Cluster_Name'] = df['Cluster'].map(cluster_name_mapping)
    
    # Save results
    cluster_results.to_csv(os.path.join(output_dir, "cluster_agreement.csv"), index=False)
    gt_model_cluster_results.to_csv(os.path.join(output_dir, "model_cluster_agreement_gt_ref.csv"), index=False)
    gen_model_cluster_results.to_csv(os.path.join(output_dir, "model_cluster_agreement_gen_ref.csv"), index=False)
    
    print(f"Analysis results saved to {output_dir}")
    
    return cluster_results, gt_model_cluster_results, gen_model_cluster_results



def calculate_cluster_performance_summary(model_cluster_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate min, max, and average performance for each cluster.
    
    Args:
        model_cluster_results: DataFrame with model and cluster results
        
    Returns:
        DataFrame with performance summary for each cluster
    """
    cluster_min = model_cluster_results.groupby('Cluster')['Agreement'].min().reset_index()
    cluster_max = model_cluster_results.groupby('Cluster')['Agreement'].max().reset_index()
    cluster_avg = model_cluster_results.groupby('Cluster')['Agreement'].mean().reset_index()
    
    cluster_performance = pd.merge(cluster_min, cluster_max, on='Cluster', suffixes=('_Min', '_Max'))
    cluster_performance = pd.merge(cluster_performance, cluster_avg, on='Cluster')
    cluster_performance = cluster_performance.rename(columns={'Agreement': 'Agreement_Avg'})
    
    # Add cluster names if available
    if 'Cluster_Name' in model_cluster_results.columns:
        cluster_name_mapping = dict(zip(
            model_cluster_results['Cluster'], 
            model_cluster_results['Cluster_Name']
        ))
        cluster_performance['Cluster_Name'] = cluster_performance['Cluster'].map(cluster_name_mapping)
    
    # Sort by average performance to identify problematic clusters
    return cluster_performance.sort_values('Agreement_Avg')



def create_cluster_summary_table_and_scatterplots(gt_model_cluster_results: pd.DataFrame,
                                                   gen_model_cluster_results: pd.DataFrame,
                                                   output_dir: str,
                                                   use_cluster_names: bool = False,
                                                   models_to_plot: list = None) -> pd.DataFrame:
    """
    Generate a summary table and scatterplots comparing GT and Gen agreement per cluster across models.

    Args:
        gt_model_cluster_results (pd.DataFrame): DataFrame with GT reference agreement and question counts.
        gen_model_cluster_results (pd.DataFrame): DataFrame with Gen reference agreement and question counts.
        output_dir (str): Directory to save scatterplots.
        use_cluster_names (bool): Whether to use cluster names in the plot.
        models_to_plot (list): Optional list of models to include. If None, include all.
    
    Returns:
        pd.DataFrame: Combined summary table.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.gridspec import GridSpec

    os.makedirs(output_dir, exist_ok=True)

    # Model name mapping
    model_display_names = {
        'mimic-cxr-findings-baseline': 'CheXpertPlus_MIMIC',
        'chexpert-mimic-cxr-findings-baseline': 'CheXpertPlus_Chex_MIMIC',
        'maira-2': 'MAIRA2'
    }

    # Define the desired model order
    model_order = [
        'mimic-cxr-findings-baseline',
        'chexpert-mimic-cxr-findings-baseline',
        'maira-2'
    ]

    summary_rows = []
    all_models = sorted(gt_model_cluster_results['Model'].unique())
    models = models_to_plot if models_to_plot else model_order  # Use ordered models instead of all_models

    # Get cluster name mapping if available
    cluster_name_mapping = {}
    if use_cluster_names and 'Cluster_Name' in gt_model_cluster_results.columns:
        cluster_name_mapping = dict(zip(
            gt_model_cluster_results['Cluster'], 
            gt_model_cluster_results['Cluster_Name']
        ))

    # Create a figure with a grid layout
    plt.rcParams.update({
        'font.size': 20,
        'font.weight': 'bold'  # Make all text bold by default
    })
    fig = plt.figure(figsize=(24, 28))
    gs = GridSpec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1], figure=fig)  # Adjusted width ratio to give more space to table
    gs.update(hspace=0.3)  # Slightly increase vertical space between subplots

    # List of subplot labels
    subplot_labels = ['a', 'b', 'c']

    # Process data and create subplots
    for idx, model in enumerate(models):
        gt_data = gt_model_cluster_results[gt_model_cluster_results['Model'] == model]
        gen_data = gen_model_cluster_results[gen_model_cluster_results['Model'] == model]

        all_clusters = sorted(set(gt_data['Cluster']).union(gen_data['Cluster']))

        for cluster in all_clusters:
            row = {
                'Model': model,
                'Cluster': cluster
            }
            gt_row = gt_data[gt_data['Cluster'] == cluster]
            gen_row = gen_data[gen_data['Cluster'] == cluster]

            row['GT_Count'] = int(gt_row['Count'].values[0]) if not gt_row.empty else 0
            row['Gen_Count'] = int(gen_row['Count'].values[0]) if not gen_row.empty else 0
            row['GT_Agree'] = float(gt_row['Agreement'].values[0]) if not gt_row.empty else 0.0
            row['Gen_Agree'] = float(gen_row['Agreement'].values[0]) if not gen_row.empty else 0.0

            summary_rows.append(row)

        # Create subplot for this model
        df_model = pd.DataFrame([r for r in summary_rows if r['Model'] == model])
        ax = fig.add_subplot(gs[idx, 0])
        
        # Add subplot label
        ax.text(-0.15, 1.05, f'({subplot_labels[idx]})', transform=ax.transAxes, 
                fontsize=30, fontweight='bold')  # Increased font size from 24 to 28
        
        scatter = ax.scatter(df_model['Gen_Agree'], df_model['GT_Agree'],
                           s=np.array(df_model['GT_Count']) / 30, alpha=0.7, 
                           c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add cluster ID labels to points with larger font
        for _, r in df_model.iterrows():
            label = str(int(r['Cluster']))
            ax.annotate(label, (r['Gen_Agree'], r['GT_Agree']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=18, weight='bold', ha='left', alpha=0.8)
        
        # Add diagonal line for reference
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Perfect Agreement')
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ICARE-GEN", fontsize=22, weight='bold', labelpad=15)  # Increased labelpad
        ax.set_ylabel("ICARE-GT", fontsize=22, weight='bold', labelpad=15)  # Increased labelpad
        ax.set_title(f"{model_display_names[model]}", 
                    fontsize=24, weight='bold', pad=20)
        ax.grid(True)
        
        # Make tick labels bold and remove redundant zero
        ax.tick_params(axis='both', which='major', labelsize=20)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight('bold')
            label.set_fontsize(20)
        
        # Remove redundant zero at origin
        y_ticks = ax.get_yticks()
        ax.set_yticks([ytick for ytick in y_ticks if ytick != 0])
        
        # Add correlation information with larger font
        correlation = df_model[['GT_Agree', 'Gen_Agree']].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=20, weight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')

    # Create table on the right that spans all subplots
    ax_table = fig.add_subplot(gs[:, 1])  # Span all rows
    ax_table.axis('off')

    # Add table label
    ax_table.text(-0.15, 1.0, '(d)', fontsize=30, fontweight='bold')  # Increased font size from 24 to 28

    # Prepare table data
    table_data = [[str(int(cluster)), name] for cluster, name in cluster_name_mapping.items()]
    table_data.sort(key=lambda x: int(x[0]))  # Sort by cluster ID
    
    # Create table with larger cells and font
    table = ax_table.table(cellText=table_data,
                          colLabels=['ID', 'Cluster Name'],  # Updated column headers
                          loc='center',
                          cellLoc='left',
                          bbox=[0.02, 0.02, 1.3, 0.98],  # Increased width by extending beyond 1
                          colWidths=[0.24, 0.76])  # Adjusted column widths: increased ID width, decreased Name width
    
    # Style the table with larger font
    table.auto_set_font_size(False)
    table.set_fontsize(24)  # Increase base font size for table
    table.scale(1.2, 1.8)  # Slightly reduce cell height to fit more rows
    table.fontweight = 'bold'
    
    # Style header and cells - make all text bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', fontsize=24)  # Slightly smaller header font
            cell.set_facecolor('#E6E6E6')
            # Center align headers
            cell._text.set_horizontalalignment('center')
            if col == 0:
                cell.get_text().set_text('Cluster ID')  # Update header text
        else:
            cell.set_text_props(weight='bold', fontsize=22)  # Slightly smaller font for content
            # Align cluster IDs to center in first column
            if col == 0:
                cell._text.set_horizontalalignment('center')
            else:
                cell._text.set_horizontalalignment('left')
                # Wrap text for cluster names
                cell.get_text().set_wrap(True)
                current_text = cell.get_text().get_text()
                # Limit text length and add ellipsis if needed
                if len(current_text) > 50:
                    cell.get_text().set_text(current_text[:47] + '...')
        
        cell.PAD = 0.1  # Add padding to cells

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_models_gt_vs_gen_agreement.png"), 
                dpi=600, bbox_inches='tight')
    plt.close()

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['Model', 'Cluster'])
    
    # Create summary tables
    cluster_counts_by_model = summary_df[['Model', 'Cluster', 'GT_Count', 'Gen_Count']].copy()
    cluster_counts_total = summary_df.groupby(['Cluster']).agg({
        'GT_Count': 'sum',
        'Gen_Count': 'sum'
    }).reset_index()
    cluster_counts_total = cluster_counts_total.sort_values('Cluster')
    
    # Save summary tables
    summary_df.to_csv(os.path.join(output_dir, "cluster_agreement_summary_detailed.csv"), index=False)
    cluster_counts_by_model.to_csv(os.path.join(output_dir, "cluster_counts_by_model.csv"), index=False)
    cluster_counts_total.to_csv(os.path.join(output_dir, "cluster_counts_total.csv"), index=False)

    print(f"Cluster summary table and scatterplots saved to {output_dir}")
    print(f"- Detailed summary: cluster_agreement_summary_detailed.csv")
    print(f"- Counts by model: cluster_counts_by_model.csv")
    print(f"- Counts total: cluster_counts_total.csv")
    print(f"- Combined plot with table: all_models_gt_vs_gen_agreement.png")

    return summary_df


def create_visualizations(gt_model_cluster_results: pd.DataFrame, 
                          gen_model_cluster_results: pd.DataFrame, 
                          cluster_results: pd.DataFrame, 
                          output_dir: str) -> pd.DataFrame:
    """
    Create visualizations for cluster performance analysis.
    
    Args:
        gt_model_cluster_results: DataFrame with GT reference results by model and cluster
        gen_model_cluster_results: DataFrame with Gen reference results by model and cluster
        cluster_results: DataFrame with overall cluster results
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with combined cluster data for both reference types
    """
    # Check if cluster names are available
    use_cluster_names = 'Cluster_Name' in gt_model_cluster_results.columns
    
    # 1. Prepare data for combined bar chart
    gt_cluster_stats = gt_model_cluster_results.groupby('Cluster').agg(
        Agreement_GT=('Agreement', 'mean'),
        Agreement_GT_Std=('Agreement', 'std')
    ).reset_index()

    gen_cluster_stats = gen_model_cluster_results.groupby('Cluster').agg(
        Agreement_Gen=('Agreement', 'mean'),
        Agreement_Gen_Std=('Agreement', 'std')
    ).reset_index()

    # Add cluster names if available
    if use_cluster_names:
        cluster_name_mapping = dict(zip(
            gt_model_cluster_results['Cluster'], 
            gt_model_cluster_results['Cluster_Name']
        ))
        gt_cluster_stats['Cluster_Name'] = gt_cluster_stats['Cluster'].map(cluster_name_mapping)
        gen_cluster_stats['Cluster_Name'] = gen_cluster_stats['Cluster'].map(cluster_name_mapping)

    # Merge for plotting
    if use_cluster_names:
        combined_data = pd.merge(gt_cluster_stats, gen_cluster_stats, on=['Cluster', 'Cluster_Name'])
    else:
        combined_data = pd.merge(gt_cluster_stats, gen_cluster_stats, on='Cluster')

    # Sort by GT agreement for better visualization
    combined_data = combined_data.sort_values('Agreement_GT')
    
    # Calculate performance summaries (moved here to fix the missing variables)
    gt_cluster_performance = calculate_cluster_performance_summary(gt_model_cluster_results)
    gen_cluster_performance = calculate_cluster_performance_summary(gen_model_cluster_results)
    
    # 17. Create cluster summary table and scatterplots
    create_cluster_summary_table_and_scatterplots(gt_model_cluster_results, gen_model_cluster_results, output_dir, use_cluster_names)
    
    # 18. Save the performance summaries
    gt_cluster_performance.to_csv(os.path.join(output_dir, "cluster_performance_summary_gt_ref.csv"), index=False)
    gen_cluster_performance.to_csv(os.path.join(output_dir, "cluster_performance_summary_gen_ref.csv"), index=False)
    
    return combined_data


def load_cluster_names(cluster_names_path: str) -> Dict[int, str]:
    """
    Load cluster names from a JSON file.
    
    Args:
        cluster_names_path: Path to the JSON file containing cluster names
        
    Returns:
        Dictionary mapping cluster IDs to names
    """
    print(f"Loading cluster names from {cluster_names_path}")
    with open(cluster_names_path, 'r') as f:
        cluster_names = json.load(f)
    
    # Convert keys to integers if they're stored as strings in the JSON
    if all(isinstance(k, str) for k in cluster_names.keys()):
        cluster_names = {int(k): v for k, v in cluster_names.items()}
        
    return cluster_names


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cluster Analysis for MCQ Questions")
    parser.add_argument('--clustered_data_path', type=str, required=True, 
                        help='Path to clustered_questions.csv file')
    parser.add_argument('--cluster_names_path', type=str, required=True,
                        help='Path to cluster_names.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for analysis results')
    parser.add_argument('--analysis_folder', type=str, required=True,
                        help='Analysis folder for storing all results')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create analysis folder for storing all results
    os.makedirs(args.analysis_folder, exist_ok=True)
    print(f"Created analysis folder at {args.analysis_folder}")

    # Load the previously saved combined data with clusters
    print(f"Loading clustered data from {args.clustered_data_path}")
    combined_df = pd.read_csv(args.clustered_data_path)
    print(f"Loaded data shape: {combined_df.shape}")
    print("\nSample of loaded data:")
    print(combined_df.head())
  
    # Load cluster names from the specific JSON file path
    cluster_names = load_cluster_names(args.cluster_names_path)
    
    # Add cluster names to the dataframe
    combined_df['cluster_name'] = combined_df['cluster_id'].map(cluster_names)
    print("Added cluster names from JSON file.")
    
    # Analyze clusters - use the analysis folder for output
    cluster_results, gt_model_cluster_results, gen_model_cluster_results = analyze_clusters(combined_df, args.analysis_folder)
    
    # Create visualizations - use the analysis folder for output
    combined_data = create_visualizations(gt_model_cluster_results, gen_model_cluster_results, cluster_results, args.analysis_folder)

    print("\nAnalysis complete!")
    print(f"1. Review the agreement metrics in '{args.analysis_folder}/cluster_agreement.csv'")
    print(f"2. Check the visualizations in the '{args.analysis_folder}' directory to identify patterns across clusters and models")


if __name__ == "__main__":
    main() 