# src/clean_dataset.py
import pandas as pd
import argparse
from typing import Optional

def clean_rexval_dataset(
    input_path: str,
    output_path: str,
    unique_column: str = 'study_number'
) -> Optional[pd.DataFrame]:
    """Clean RexVal dataset by selecting rows with unique values."""
    try:
        # Read the dataset
        df = pd.read_csv(input_path)
        
        # Rename columns to use underscores
        column_mapping = {
            'ground truth reports': 'ground_truth_report',
            'generated reports': 'generated_report'
        }
        df = df.rename(columns=column_mapping)
        
        # Select unique rows
        unique_rows = df.drop_duplicates(subset=[unique_column])
        
        # Save cleaned dataset
        unique_rows.to_csv(output_path, index=False)
        
        print(f"Original dataframe had {len(df)} rows")
        print(f"Unique rows dataframe has {len(unique_rows)} rows")
        return unique_rows
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Clean RexVal Dataset')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--unique-column', default='study_number', help='Column to use for uniqueness')
    
    args = parser.parse_args()
    clean_rexval_dataset(args.input, args.output, args.unique_column)

if __name__ == "__main__":
    main()