import torch
import torch.nn as nn
import numpy as np
import sqlite3
import json
from pathlib import Path
import ollama

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

DB_PATH = 'embeddings.db'
MODEL_PATH = 'transformer_mapper_best.pth'
CONFIG_PATH = 'transformer_mapper_config.json'

class TransformerEmbeddingMapper(nn.Module):
    """Transformer-based mapper for embedding space translation"""
    
    def __init__(self, source_dim=768, target_dim=768, num_heads=12, num_layers=8, hidden_dim=3072, dropout=0.05):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        self.input_projection = nn.Linear(source_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, target_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        x = x.squeeze(1)
        return x

def load_embeddings_from_db():
    """Load all embeddings from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT word, embedding FROM nomic_embeddings')
    nomic_rows = cursor.fetchall()
    
    cursor.execute('SELECT word, embedding FROM gemma_embeddings')
    gemma_rows = cursor.fetchall()
    
    conn.close()
    
    nomic_dict = {}
    for word, embedding_bytes in nomic_rows:
        nomic_dict[word] = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    gemma_dict = {}
    for word, embedding_bytes in gemma_rows:
        gemma_dict[word] = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    return nomic_dict, gemma_dict

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)

def generate_embeddings(word):
    """Generate embeddings for a word using ollama"""
    
    print(f"  Generating embeddings for '{word}'...")
    
    # Generate nomic embedding
    nomic_response = ollama.embed(
        model='nomic-embed-text',
        input=word,
    )
    nomic_emb = np.array(nomic_response['embeddings'][0], dtype=np.float32)
    
    # Generate gemma embedding
    gemma_response = ollama.embed(
        model='embeddinggemma',
        input=word,
    )
    gemma_emb = np.array(gemma_response['embeddings'][0], dtype=np.float32)
    
    return nomic_emb, gemma_emb

def mse_error(a, b):
    """Calculate MSE between two vectors (same dimension)"""
    # Only calculate if dimensions match
    if len(a) != len(b):
        return None
    return np.mean((a - b) ** 2)

def main():
    print("="*60)
    print("EMBEDDING MAPPER - USER INPUT TESTING")
    print("="*60)
    
    # Load config
    print("\nLoading model configuration...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Handle both old and new config formats
    if 'source_dim' in config:
        source_dim = config['source_dim']
        target_dim = config['target_dim']
    elif 'embedding_dim' in config:
        # Old format - load embeddings to detect dimensions
        nomic_dict_temp, gemma_dict_temp = load_embeddings_from_db()
        first_word = list(nomic_dict_temp.keys())[0]
        source_dim = len(nomic_dict_temp[first_word])
        target_dim = len(gemma_dict_temp[first_word])
        print("(Auto-detected dimensions from old config format)")
    else:
        raise ValueError("Config file missing dimension information. Please retrain the model.")
    
    print(f"Source dimension: {source_dim}")
    print(f"Target dimension: {target_dim}")
    print(f"Test metrics from training:")
    print(f"  MSE: {config['test_mse']:.6f}")
    print(f"  Cosine Similarity: {config['test_cosine_similarity']:.4f}")
    
    # Load model
    print("\nLoading trained model...")
    model = TransformerEmbeddingMapper(
        source_dim=source_dim,
        target_dim=target_dim,
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 8),
        hidden_dim=config.get('hidden_dim', 3072),
        dropout=config.get('dropout', 0.05)
    ).to(DEVICE)
    
    # Load state_dict and filter out old architecture keys
    state_dict = torch.load(MODEL_PATH)
    
    # Remove old keys that don't exist in new architecture
    old_keys = ['attention.in_proj_weight', 'attention.in_proj_bias', 
                'attention.out_proj.weight', 'attention.out_proj.bias',
                'layer_norm.weight', 'layer_norm.bias']
    
    for key in old_keys:
        if key in state_dict:
            del state_dict[key]
            print(f"Removed old architecture key: {key}")
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    
    # Load embeddings
    print("\nLoading embeddings database...")
    nomic_dict, gemma_dict = load_embeddings_from_db()
    print(f"Loaded {len(nomic_dict)} source embeddings")
    print(f"Loaded {len(gemma_dict)} target embeddings")
    
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("Enter a word to test (or 'quit' to exit)")
    print("If word is in database, will show comparison with actual target embedding\n")
    
    with torch.no_grad():
        while True:
            word = input("\nEnter word: ").strip().lower()
            
            if word == 'quit':
                print("Exiting...")
                break
            
            if not word:
                print("Please enter a valid word")
                continue
            
            # Check if word is in database or generate
            if word not in nomic_dict:
                print(f"[!] '{word}' not found in database, generating embeddings...")
                try:
                    source_emb, actual_target = generate_embeddings(word)
                    has_actual_target = True
                except Exception as e:
                    print(f"[ERROR] Failed to generate embeddings: {e}")
                    continue
            else:
                source_emb = nomic_dict[word]
                has_actual_target = word in gemma_dict
                if has_actual_target:
                    actual_target = gemma_dict[word]
            
            # Get source embedding tensor and map it
            source_tensor = torch.FloatTensor(source_emb).unsqueeze(0).to(DEVICE)
            
            # Map to target space
            mapped_emb = model(source_tensor).cpu().numpy()[0]
            
            # Results
            print(f"\n{'='*60}")
            print(f"Word: '{word}'")
            print(f"Source embedding dimension: {len(source_emb)}")
            print(f"Mapped embedding dimension: {len(mapped_emb)}")
            
            # If we have actual target, compare
            if has_actual_target:
                
                # Calculate metrics
                cosine_sim = cosine_similarity(mapped_emb, actual_target)
                mse = mse_error(mapped_emb, actual_target)
                
                print(f"\n[OK] Found actual target embedding for '{word}'")
                print(f"  Cosine Similarity: {cosine_sim:.6f}")
                print(f"  MSE Error: {mse:.6f}")
                
                # Comparison to training metrics
                print(f"\nComparison to training:")
                print(f"  Training avg cosine: {config.get('test_cosine_similarity', 0):.4f}")
                print(f"  This word cosine:    {cosine_sim:.4f}")
                print(f"  Training avg MSE:    {config.get('test_mse', 0):.6f}")
                print(f"  This word MSE:       {mse:.6f}")
                
            else:
                print(f"\n[!] No actual target embedding available for comparison")
                print(f"(Word not in target database)")
                print(f"Mapped embedding generated but cannot verify accuracy")
            
            print(f"{'='*60}")

if __name__ == '__main__':
    main()
