#!/usr/bin/env python3
"""
Test the embedding mapper on many common words from the corpus.
Generates statistics on performance across a large set of vocabulary.
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_generator import EmbeddingGenerator

# Define the TransformerEmbeddingMapper (from train_transformer_mapper.py)
class TransformerEmbeddingMapper(nn.Module):
    def __init__(self, source_dim, target_dim, hidden_dim=3072, num_layers=8, num_heads=12, dropout=0.05):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        
        # Project source embedding to hidden dimension
        self.source_proj = nn.Linear(source_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project hidden dimension back to target embedding
        self.target_proj = nn.Linear(hidden_dim, target_dim)
    
    def forward(self, x):
        # x shape: (batch_size, source_dim)
        x = self.source_proj(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim) - add sequence dimension
        x = self.transformer(x)  # (batch_size, 1, hidden_dim)
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        x = self.target_proj(x)  # (batch_size, target_dim)
        return x

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_squared_error(a, b):
    """Compute MSE between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.mean((a - b) ** 2)

def main():
    # Setup
    workspace_path = Path(__file__).parent.parent
    config_path = workspace_path / "config" / "transformer_mapper_config.json"
    model_path = workspace_path / "models" / "transformer_mapper_best.pth"
    corpus_path = workspace_path / "data" / "word_corpus.txt"
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load word corpus
    with open(corpus_path, 'r') as f:
        all_words = [line.strip() for line in f if line.strip()]
    
    # Test top N common words
    num_words = min(200, len(all_words))
    test_words = all_words[:num_words]
    
    print("=" * 70)
    print(f"EMBEDDING MAPPER TEST: {num_words} Most Common Words")
    print("=" * 70)
    print(f"Testing words from position 1 to {num_words}")
    print()
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEmbeddingMapper(
        source_dim=config['source_dim'],
        target_dim=config['target_dim'],
        hidden_dim=config.get('hidden_dim', 3072),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 12),
        dropout=config.get('dropout', 0.05)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded (device: {device})")
    print()
    
    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator()
    
    # Test each word
    print(f"Testing {num_words} words...")
    print("-" * 70)
    
    results = {
        'cosine_similarities': [],
        'mse_errors': [],
        'successful': 0,
        'failed': 0,
        'words': []
    }
    
    for i, word in enumerate(test_words, 1):
        try:
            # Get source embedding (nomic-embed-text)
            source_emb = embedding_gen.get_embedding(word, model_name="nomic-embed-text")
            
            # Get target embedding (embeddinggemma)
            target_emb = embedding_gen.get_embedding(word, model_name="embeddinggemma")
            
            if source_emb is None or target_emb is None:
                results['failed'] += 1
                continue
            
            # Convert to tensors
            source_tensor = torch.tensor([source_emb], dtype=torch.float32, device=device)
            target_tensor = torch.tensor([target_emb], dtype=torch.float32, device=device)
            
            # Get prediction
            with torch.no_grad():
                predicted = model(source_tensor)
            
            # Metrics
            pred_np = predicted[0].cpu().numpy()
            target_np = np.array(target_emb)
            
            cos_sim = cosine_similarity(pred_np, target_np)
            mse = mean_squared_error(pred_np, target_np)
            
            results['cosine_similarities'].append(cos_sim)
            results['mse_errors'].append(mse)
            results['successful'] += 1
            results['words'].append({
                'word': word,
                'cosine_similarity': cos_sim,
                'mse': mse
            })
            
            # Progress
            if i % 25 == 0:
                avg_cos = np.mean(results['cosine_similarities'])
                avg_mse = np.mean(results['mse_errors'])
                print(f"  [{i:3d}/{num_words}] Avg cosine: {avg_cos:.4f}, Avg MSE: {avg_mse:.6f}")
        
        except Exception as e:
            results['failed'] += 1
            if i % 25 == 0:
                print(f"  [{i:3d}/{num_words}] (error: {str(e)[:40]}...)")
    
    print("-" * 70)
    print()
    
    # Summary statistics
    if results['successful'] > 0:
        cos_sims = results['cosine_similarities']
        mses = results['mse_errors']
        
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Successfully tested: {results['successful']} / {num_words} words")
        print(f"Failed: {results['failed']} words")
        print()
        print(f"Cosine Similarity:")
        print(f"  Mean:   {np.mean(cos_sims):.4f}")
        print(f"  Median: {np.median(cos_sims):.4f}")
        print(f"  Std:    {np.std(cos_sims):.4f}")
        print(f"  Min:    {np.min(cos_sims):.4f}")
        print(f"  Max:    {np.max(cos_sims):.4f}")
        print()
        print(f"MSE Error:")
        print(f"  Mean:   {np.mean(mses):.6f}")
        print(f"  Median: {np.median(mses):.6f}")
        print(f"  Std:    {np.std(mses):.6f}")
        print(f"  Min:    {np.min(mses):.6f}")
        print(f"  Max:    {np.max(mses):.6f}")
        print()
        
        # Distribution
        print(f"Cosine Similarity Distribution:")
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        for low, high in bins:
            count = sum(1 for cs in cos_sims if low <= cs < high)
            pct = 100 * count / len(cos_sims)
            print(f"  [{low:.2f}-{high:.2f}): {count:3d} words ({pct:5.1f}%)")
        
        print()
        print("=" * 70)
        
        # Top and bottom performers
        print("\nTOP 10 Best Matches (Highest Cosine Similarity):")
        print("-" * 70)
        sorted_results = sorted(results['words'], key=lambda x: x['cosine_similarity'], reverse=True)
        for i, item in enumerate(sorted_results[:10], 1):
            print(f"  {i:2d}. {item['word']:15s} - cos_sim: {item['cosine_similarity']:.4f}, MSE: {item['mse']:.6f}")
        
        print("\nTOP 10 Worst Matches (Lowest Cosine Similarity):")
        print("-" * 70)
        for i, item in enumerate(sorted_results[-10:], 1):
            print(f"  {i:2d}. {item['word']:15s} - cos_sim: {item['cosine_similarity']:.4f}, MSE: {item['mse']:.6f}")
    
    print()

if __name__ == "__main__":
    main()
