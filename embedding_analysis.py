#!/usr/bin/env python3
"""
embedding_analysis.py

This script performs an embedding analysis on the sample outputs from raw_data.csv.
It:
  1. Loads raw_data.csv containing benchmark results.
  2. Generates embeddings for the 'sample_output' texts using a SentenceTransformer.
  3. Saves embeddings back into a new CSV.
  4. Performs dimensionality reduction with PCA and t-SNE.
  5. Visualizes the reduced embeddings and a cosine similarity heatmap.
  6. Computes average intra-model cosine similarity.
  
All visualizations are saved in a folder called "ea_viz".
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sentence_trans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Create a folder for all visualizations
    viz_folder = "ea_viz"
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    # 1. Load the benchmark data
    df = pd.read_csv("raw_data.csv")
    if "sample_output" not in df.columns:
        raise ValueError("raw_data.csv must contain a 'sample_output' column.")

    # Fill missing values in sample_output
    df["sample_output"] = df["sample_output"].fillna("")

    # 2. Generate embeddings using a Sentence Transformer
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model '{model_name}'...")
    embedder = SentenceTransformer(model_name)
    texts = df["sample_output"].tolist()
    print("Generating embeddings for sample outputs...")
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    
    # Save embeddings in the dataframe (as a list of floats)
    df["embedding"] = [emb.tolist() for emb in embeddings]
    
    # Save an intermediate CSV with embeddings
    df.to_csv("raw_data_with_embeddings.csv", index=False)
    print("Saved raw_data_with_embeddings.csv.")

    # 3. Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    df["pca_x"] = pca_result[:, 0]
    df["pca_y"] = pca_result[:, 1]
    
    # Plot PCA results, colored by model_name
    plt.figure(figsize=(10, 8))
    unique_models = df["model_name"].unique()
    for model in unique_models:
        subset = df[df["model_name"] == model]
        plt.scatter(subset["pca_x"], subset["pca_y"], label=model, alpha=0.6)
    plt.title("PCA of Sample Outputs Embeddings (by model_name)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pca_viz_path = os.path.join(viz_folder, "pca_embeddings_by_model.png")
    plt.savefig(pca_viz_path, dpi=150)
    plt.show()
    print(f"Saved PCA visualization to {pca_viz_path}")
    
    # 4. Dimensionality Reduction with t-SNE
    n_samples = embeddings.shape[0]
    default_perplexity = 30
    # Ensure perplexity is less than the number of samples.
    if n_samples <= default_perplexity:
        perplexity = max(1, n_samples - 1)
        print(f"Number of samples ({n_samples}) is less than or equal to the default perplexity ({default_perplexity}). Setting TSNE perplexity to {perplexity}.")
    else:
        perplexity = default_perplexity

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    df["tsne_x"] = tsne_result[:, 0]
    df["tsne_y"] = tsne_result[:, 1]
    
    # Plot t-SNE results, colored by model_name
    plt.figure(figsize=(10, 8))
    for model in unique_models:
        subset = df[df["model_name"] == model]
        plt.scatter(subset["tsne_x"], subset["tsne_y"], label=model, alpha=0.6)
    plt.title("t-SNE of Sample Outputs Embeddings (by model_name)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    tsne_viz_path = os.path.join(viz_folder, "tsne_embeddings_by_model.png")
    plt.savefig(tsne_viz_path, dpi=150)
    plt.show()
    print(f"Saved t-SNE visualization to {tsne_viz_path}")


    # 5. Compute and visualize cosine similarity across all embeddings
    cos_sim = cosine_similarity(embeddings)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cos_sim, cmap="viridis")
    plt.title("Cosine Similarity Heatmap of Sample Outputs Embeddings")
    plt.tight_layout()
    heatmap_viz_path = os.path.join(viz_folder, "cosine_similarity_heatmap.png")
    plt.savefig(heatmap_viz_path, dpi=150)
    plt.show()
    print(f"Saved cosine similarity heatmap to {heatmap_viz_path}")

    # 6. Compute average cosine similarity within each model group
    similarity_results = {}
    for model in unique_models:
        indices = df[df["model_name"] == model].index
        if len(indices) > 1:
            group_embeddings = [embeddings[i] for i in indices]
            sim_matrix = cosine_similarity(group_embeddings)
            # Use only the upper triangle (excluding the diagonal)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            avg_sim = np.mean(sim_matrix[triu_indices])
            similarity_results[model] = avg_sim
        else:
            similarity_results[model] = None
    print("Average cosine similarity within each model group:")
    for model, avg_sim in similarity_results.items():
        if avg_sim is not None:
            print(f"  {model}: {avg_sim:.4f}")
        else:
            print(f"  {model}: Not enough samples for comparison.")

    # 7. Save the final dataframe with embeddings and reduction coordinates
    final_csv_path = "raw_data_with_embeddings_and_reduction.csv"
    df.to_csv(final_csv_path, index=False)
    print(f"Saved final data with embeddings and reduction coordinates to {final_csv_path}.")

if __name__ == "__main__":
    main()
