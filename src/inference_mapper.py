import torch
import torch.nn as nn
import numpy as np
import sqlite3
import json
import ollama

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DB_PATH = 'embeddings.db'

class TransformerEmbeddingMapper(nn.Module):
    """Transformer-based mapper for embedding space translation."""
    
    def __init__(self, embedding_dim=768, num_heads=8, num_layers=4, hidden_dim=2048, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        x = self.output_projection(x)
        x = x.squeeze(1)
        return x

def load_model():
    """Load the trained model"""
    with open('transformer_mapper_config.json', 'r') as f:
        config = json.load(f)
    
    model = TransformerEmbeddingMapper(
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load('transformer_mapper_best.pth', map_location=DEVICE))
    model.eval()
    
    return model, config

def get_embedding_from_db(word, model_name):
    """Get embedding from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    table = 'nomic_embeddings' if model_name == 'nomic' else 'gemma_embeddings'
    
    cursor.execute(f'SELECT embedding FROM {table} WHERE word = ?', (word,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return np.frombuffer(result[0], dtype=np.float32)
    return None

def embed_word(word, model_name):
    """Get embedding for a word from ollama"""
    try:
        if model_name == 'nomic':
            model_id = 'nomic-embed-text'
        else:
            model_id = 'embeddinggemma'
        
        response = ollama.embed(model=model_id, input=word)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Error embedding '{word}': {e}")
        return None

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    return np.dot(vec1_norm, vec2_norm)

def calculate_mse(vec1, vec2):
    """Calculate mean squared error"""
    return np.mean((vec1 - vec2) ** 2)

def test_inference(model, test_words):
    """Test model on new unseen words"""
    model.eval()
    
    results = []
    
    print("\n" + "="*80)
    print("TESTING TRANSFORMER MAPPER ON UNSEEN WORDS")
    print("="*80)
    
    for word in test_words:
        print(f"\nTesting: {word}")
        print("-" * 80)
        
        # Get source embedding (nomic)
        source_emb = get_embedding_from_db(word, 'nomic')
        if source_emb is None:
            source_emb = embed_word(word, 'nomic')
            if source_emb is None:
                print(f"  Failed to get embedding")
                continue
        
        # Get target embedding (gemma) - ground truth
        target_emb = get_embedding_from_db(word, 'gemma')
        if target_emb is None:
            target_emb = embed_word(word, 'gemma')
            if target_emb is None:
                print(f"  Failed to get gemma embedding")
                continue
        
        # Map source to target using transformer
        source_tensor = torch.FloatTensor(source_emb.copy()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mapped_emb = model(source_tensor).cpu().numpy()[0]
        
        # Calculate metrics
        cosine_sim = calculate_cosine_similarity(mapped_emb, target_emb)
        mse = calculate_mse(mapped_emb, target_emb)
        
        results.append({
            'word': word,
            'cosine_similarity': cosine_sim,
            'mse': mse
        })
        
        print(f"  Cosine Similarity: {cosine_sim:.6f}")
        print(f"  MSE Error: {mse:.6f}")
    
    # Summary statistics
    if results:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        cosines = [r['cosine_similarity'] for r in results]
        mses = [r['mse'] for r in results]
        
        print(f"Average Cosine Similarity: {np.mean(cosines):.6f}")
        print(f"Min Cosine Similarity: {np.min(cosines):.6f}")
        print(f"Max Cosine Similarity: {np.max(cosines):.6f}")
        print(f"\nAverage MSE: {np.mean(mses):.6f}")
        print(f"Min MSE: {np.min(mses):.6f}")
        print(f"Max MSE: {np.max(mses):.6f}")
        
        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        avg_cosine = np.mean(cosines)
        if avg_cosine > 0.95:
            print("✓ EXCELLENT: Mapping is nearly perfect!")
        elif avg_cosine > 0.85:
            print("✓ VERY GOOD: Mapping quality is high")
        elif avg_cosine > 0.70:
            print("✓ GOOD: Mapping works reasonably well")
        elif avg_cosine > 0.50:
            print("~ FAIR: Mapping provides some value")
        else:
            print("✗ POOR: Mapping quality is low")

def main():
    print("="*80)
    print("TRANSFORMER EMBEDDING MAPPER - INFERENCE")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model()
    print(f"Model loaded from transformer_mapper_best.pth")
    print(f"Config: {config}")
    print(f"Device: {DEVICE}")
    
    # Test words (unseen during training)
    test_words = [
        'quantum',
        'symphony',
        'eclipse',
        'whisper',
        'thunder',
        'crystal',
        'labyrinth',
        'luminous',
        'odyssey',
        'serendipity'
    ]
    
    # Test inference
    test_inference(model, test_words)

if __name__ == '__main__':
    main()
