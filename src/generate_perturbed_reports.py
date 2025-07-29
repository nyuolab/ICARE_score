import argparse
import pandas as pd
import random
import string
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def perturb_text(text: str, perturbation_rate: float, seed: Optional[int] = None) -> str:
    """
    Perturb text by randomly deleting characters at the specified perturbation rate.

    Args:
        text (str): The original text to perturb
        perturbation_rate (float): Percentage of characters to delete (0-1)
        seed (Optional[int]): Random seed for reproducible perturbations

    Returns:
        str: The perturbed text with characters deleted

    Raises:
        ValueError: If perturbation_rate is not between 0 and 1
    """
    if not isinstance(text, str):
        return str(text)
    
    if not 0 <= perturbation_rate <= 1:
        raise ValueError("Perturbation rate must be between 0 and 1")
        
    if not text or perturbation_rate <= 0:
        return text
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        
    chars = list(text)
    num_chars_to_perturb = int(len(chars) * perturbation_rate)
    
    # Keep track of deleted positions to avoid deleting the same position twice
    positions_to_delete = random.sample(range(len(chars)), min(num_chars_to_perturb, len(chars)-1))
    
    # Sort in reverse order to avoid index shifting when deleting
    for pos in sorted(positions_to_delete, reverse=True):
        chars.pop(pos)
            
    return ''.join(chars)

def perturb_text_word_level(text: str, perturbation_rate: float, seed: Optional[int] = None) -> str:
    """
    Perturb text by randomly deleting words at the specified perturbation rate.

    Args:
        text (str): The original text to perturb
        perturbation_rate (float): Percentage of words to delete (0-1)
        seed (Optional[int]): Random seed for reproducible perturbations

    Returns:
        str: The perturbed text with words deleted
    """
    if not isinstance(text, str):
        return str(text)
    
    if not 0 <= perturbation_rate <= 1:
        raise ValueError("Perturbation rate must be between 0 and 1")
        
    if not text or perturbation_rate <= 0:
        return text
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        
    # Split text into words
    words = text.split()
    if not words:
        return text
        
    num_words_to_perturb = int(len(words) * perturbation_rate)
    
    # Keep track of deleted positions to avoid deleting the same position twice
    positions_to_delete = random.sample(range(len(words)), min(num_words_to_perturb, len(words)-1))
    
    # Sort in reverse order to avoid index shifting when deleting
    for pos in sorted(positions_to_delete, reverse=True):
        words.pop(pos)
            
    return ' '.join(words)

def perturb_dataset(input_file: str, output_dir: str, perturbation_type: str = 'char', seed: Optional[int] = None) -> Optional[bool]:
    """
    Read input CSV and create perturbed versions at different percentages.

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save perturbed CSV files
        perturbation_type (str): Type of perturbation ('char' or 'word')
        seed (Optional[int]): Random seed for reproducible perturbations

    Returns:
        Optional[bool]: True if successful, None if failed

    Raises:
        FileNotFoundError: If input_file doesn't exist
        pd.errors.EmptyDataError: If input CSV is empty
    """
    try:
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read input CSV
        df = pd.read_csv(input_file)
        
        if df.empty:
            raise pd.errors.EmptyDataError("Input CSV file is empty")
            
        required_columns = ['generated_report', 'ground_truth_report']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Choose perturbation function based on type
        perturb_func = perturb_text if perturbation_type == 'char' else perturb_text_word_level
        
        # Perturbation percentages from 5% to 100% in steps of 5%
        perturbation_rates = [i/100 for i in range(0, 101, 5)]
        
        for rate in perturbation_rates:
            # Create copy of original dataframe
            perturbed_df = df.copy()
            
            # Perturb both report columns
            for column in required_columns:
                perturbed_df[column] = perturbed_df[column].apply(
                    lambda x: perturb_func(x, rate, seed)
                )
            
            # Save to new CSV file
            output_file = os.path.join(output_dir, f'perturbed_{int(rate*100)}percent.csv')
            perturbed_df.to_csv(output_file, index=False)
            logging.info(f"Created {perturbation_type}-level perturbed dataset at {rate*100}% rate: {output_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during dataset perturbation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perturb reports dataset')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save perturbed datasets')
    parser.add_argument('--perturbation_type', type=str, choices=['char', 'word'], default='char',
                      help='Type of perturbation to apply (character or word level)')
    parser.add_argument('--seed', type=int, default=123,
                      help='Random seed for reproducible perturbations')
    
    args = parser.parse_args()
    
    try:
        logging.info(f"Starting perturbation process...")
        logging.info(f"Input file: {args.input_csv}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Perturbation type: {args.perturbation_type}")
        if args.seed is not None:
            logging.info(f"Using random seed: {args.seed}")
        
        output_dir = f'{args.output_dir}/perturbed_reports_{args.perturbation_type}_level'
        success = perturb_dataset(args.input_csv, output_dir, args.perturbation_type, args.seed)
        
        if success:
            logging.info("Dataset perturbation completed successfully")
            
    except Exception as e:
        logging.error(f"Failed to complete perturbation process: {str(e)}")