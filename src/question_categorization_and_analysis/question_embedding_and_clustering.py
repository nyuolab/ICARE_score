# Question categorization using embeddings and clustering
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import requests
from typing import Dict, Any, Optional, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils import make_llama_request
import random
import argparse

def get_embeddings(questions):
    """
    Get embeddings for questions using MedCPT-Query-Encoder
    """
    print("Loading MedCPT-Query-Encoder model...")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    # Convert questions to list if it's not already
    if not isinstance(questions, list):
        questions = questions.tolist()
    
    start_time = time.time()
    for i in tqdm(range(0, len(questions), batch_size), desc="Computing embeddings"):
        batch = questions[i:i+batch_size]
        
        with torch.no_grad():
            # Tokenize the questions
            encoded = tokenizer(
                batch, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=64,
            )
            
            # Encode the questions (use the [CLS] last hidden states as the representations)
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    end_time = time.time()
    print(f"Embedding generation took {end_time - start_time:.2f} seconds")
    return embeddings

def cluster_questions(embeddings, unique_questions, n_clusters=20):
    """
    Cluster questions based on embeddings
    """
    print(f"Clustering questions into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Create a mapping from question to cluster
    question_to_cluster = {q: c for q, c in zip(unique_questions, cluster_labels)}
    
    # Get representative questions for each cluster
    cluster_representatives = {}
    cluster_sizes = {}
    
    for cluster_id in range(n_clusters):
        # Get questions in this cluster
        cluster_questions_indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        cluster_questions = [unique_questions[i] for i in cluster_questions_indices]
        
        cluster_sizes[cluster_id] = len(cluster_questions)
        print(f"Cluster {cluster_id}: {len(cluster_questions)} questions")
        
        # Get cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Find closest questions to centroid
        question_embeddings = embeddings[cluster_questions_indices]
        distances = np.linalg.norm(question_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:5]  # Get 5 closest
        
        representatives = [cluster_questions[i] for i in closest_indices]
        cluster_representatives[cluster_id] = representatives
    
    return cluster_labels, question_to_cluster, cluster_representatives, cluster_sizes

def visualize_clusters(embeddings, cluster_labels, unique_questions, output_dir):
    """
    Create t-SNE visualization of clusters
    """
    print("Creating t-SNE visualization...")
    # Use a fixed random_state for reproducibility
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': cluster_labels,
        'question': unique_questions
    })
    
    # Plot
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(plot_df['x'], plot_df['y'], c=plot_df['cluster'], cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Question Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_clusters_tsne.png"), dpi=300)
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'question_clusters_tsne.png')}")
    
    return plot_df

def prepare_llm_prompt(cluster_representatives):
    """
    Prepare a prompt for LLM to name each cluster
    """
    prompt = """You are an expert in radiology and medical natural language processing. Your task is to analyze clusters of medical questions (primarily from radiology reports) and generate a concise, medically accurate category name for each cluster.

Instructions:
For each cluster of questions, identify the common theme, considering aspects such as:
- Medical findings or observations
- Pathologies or conditions
- Anatomical structures
- Imaging techniques or modalities
- Clinical relevance or diagnostic considerations

For each cluster, generate a descriptive category name (2-6 words) that best captures the theme. The name should:
- Be specific enough to be meaningful
- Use appropriate medical terminology
- Accurately represent the questions in that cluster
- Be distinct from other cluster names

Please provide your answers in the following format:
Category for Cluster 1: [Generated Category Name]
Category for Cluster 2: [Generated Category Name]
And so on.

Here are the clusters of questions:
"""
    
    for cluster_id, representatives in cluster_representatives.items():
        prompt += f"\nQuestions from Cluster {cluster_id + 1}:\n"
        for i, question in enumerate(representatives):
            prompt += f"{i+1}. {question}\n"
    
    return prompt

def name_clusters_with_llama(cluster_representatives, output_dir):
    """
    Use LLaMA to name the clusters based on representative questions
    """
    print("Generating cluster names using LLaMA...")
    
    # Prepare the prompt
    prompt = prepare_llm_prompt(cluster_representatives)
    
    # Save the prompt for reference
    with open(os.path.join(output_dir, "llm_naming_prompt.txt"), 'w') as f:
        f.write(prompt)
    
    # Call LLaMA API with a fixed seed
    response = make_llama_request(prompt, seed=42)
    
    if response:
        # Save the raw response
        with open(os.path.join(output_dir, "llm_naming_responses.txt"), 'w') as f:
            f.write(response)
        
        # Parse the response to extract cluster names
        cluster_names = {}
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Category for Cluster"):
                try:
                    parts = line.split(":", 1)
                    cluster_id_str = parts[0].replace("Category for Cluster", "").strip()
                    cluster_id = int(cluster_id_str) - 1  # Convert to 0-indexed
                    cluster_name = parts[1].strip()
                    cluster_names[cluster_id] = cluster_name
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
        
        # Save the parsed cluster names
        with open(os.path.join(output_dir, "cluster_names.json"), 'w') as f:
            json.dump(cluster_names, f, indent=2)
        
        print(f"Cluster names generated and saved to {os.path.join(output_dir, 'cluster_names.json')}")
        return cluster_names
    else:
        print("Failed to generate cluster names using LLaMA")
        return None

def process_questions(combined_df, output_dir):
    """
    Process questions: generate embeddings, cluster, and name clusters
    """
    # Get unique questions
    unique_questions = combined_df['Question_Text'].unique()
    print(f"Found {len(unique_questions)} unique questions out of {len(combined_df)} total questions")
    
    # Check if embeddings already exist
    embeddings_file = os.path.join(output_dir, "question_embeddings.pkl")
    if os.path.exists(embeddings_file):
        print(f"Loading existing embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # Get embeddings
        print("Computing embeddings for unique questions...")
        embeddings = get_embeddings(unique_questions)
        # Save embeddings for future use
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {embeddings_file}")
    
    # Cluster questions
    cluster_labels, question_to_cluster, cluster_representatives, cluster_sizes = cluster_questions(embeddings, unique_questions)
    
    # Add cluster ID to the dataframe
    combined_df['cluster_id'] = combined_df['Question_Text'].map(question_to_cluster)
    
    # Save clustered data
    combined_df.to_csv(os.path.join(output_dir, "clustered_questions.csv"), index=False)
    print(f"Clustered questions saved to {os.path.join(output_dir, 'clustered_questions.csv')}")
    
    # Save cluster representatives
    with open(os.path.join(output_dir, "cluster_representatives.txt"), 'w') as f:
        for cluster_id, representatives in cluster_representatives.items():
            f.write(f"=== Cluster {cluster_id} ({cluster_sizes[cluster_id]} questions) ===\n\n")
            for i, question in enumerate(representatives):
                f.write(f"{i+1}. {question}\n")
            f.write("\n")
    
    print(f"Representative questions saved to {os.path.join(output_dir, 'cluster_representatives.txt')}")
    
    # Visualize clusters
    plot_df = visualize_clusters(embeddings, cluster_labels, unique_questions, output_dir)
    
    # Name clusters with LLaMA
    cluster_names = name_clusters_with_llama(cluster_representatives, output_dir)
    
    # If cluster names were successfully generated, add them to the dataframe
    if cluster_names:
        print("Adding cluster names to the dataframe...")
        # Create a mapping from cluster ID to name
        cluster_id_to_name = {cluster_id: name for cluster_id, name in cluster_names.items()}
        
        # Add cluster names to the dataframe
        combined_df['cluster_name'] = combined_df['cluster_id'].map(cluster_id_to_name)
        
        # Save the updated dataframe
        combined_df.to_csv(os.path.join(output_dir, "clustered_questions_with_names.csv"), index=False)
        print(f"Clustered questions with names saved to {os.path.join(output_dir, 'clustered_questions_with_names.csv')}")
    
    return combined_df, cluster_representatives, cluster_sizes 

def main():
    """
    Main function to run the embedding and clustering pipeline
    """
    parser = argparse.ArgumentParser(description="Question Embedding and Clustering")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--combined_data_path', type=str, required=True, help='Path to combined_mcqa_data.csv')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)

    output_dir = args.output_dir
    combined_data_path = args.combined_data_path

    if os.path.exists(combined_data_path):
        print(f"Loading combined data from {combined_data_path}")
        combined_df = pd.read_csv(combined_data_path)
        print(f"Loaded data shape: {combined_df.shape}")
        print("\nSample of loaded data:")
        print(combined_df.head())
    else:
        print(f"Combined data file not found: {combined_data_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process questions (embedding, clustering, naming)
    combined_df, cluster_representatives, cluster_sizes = process_questions(combined_df, output_dir)

    print("\nEmbedding and clustering complete!")
    print("Next steps:")
    print("1. Review the generated cluster names in 'cluster_names.json'")
    print("2. Run cluster_analysis.py to analyze agreement metrics by cluster")

if __name__ == "__main__":
    main() 