import ollama
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
import json

# Load anchor words
def load_anchor_words(filepath):
    """Extract just the numbered words from the anchor_words.txt file"""
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Look for lines that start with a number followed by a period
            if line and line[0].isdigit():
                # Split on the period and get the word after it
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    word = parts[1].strip()
                    words.append(word)
    return words

# Get embeddings from both models
def get_embeddings_batch(model, words):
    """Get embeddings for a list of words from a specific model"""
    embeddings = {}
    embedding_dim = None
    print(f"\nFetching embeddings from {model}...")
    for i, word in enumerate(words):
        try:
            response = ollama.embed(model=model, input=word)
            emb = np.array(response['embeddings'][0])
            
            # Check dimension on first embedding
            if embedding_dim is None:
                embedding_dim = len(emb)
                print(f"  Embedding dimension: {embedding_dim}")
            elif len(emb) != embedding_dim:
                print(f"  WARNING: '{word}' has dimension {len(emb)}, expected {embedding_dim}")
            
            embeddings[word] = emb
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(words)}")
        except Exception as e:
            print(f"Error embedding '{word}': {e}")
    print(f"Successfully embedded {len(embeddings)}/{len(words)} words")
    return embeddings, embedding_dim

# Calculate transformation matrix using Procrustes analysis
def calculate_transformation_matrix(source_embeddings, target_embeddings, words):
    """
    Calculate orthogonal transformation matrix that maps source space to target space
    Uses Procrustes analysis for optimal orthogonal mapping
    """
    # Prepare matrices
    source_matrix = np.array([source_embeddings[word] for word in words if word in source_embeddings and word in target_embeddings])
    target_matrix = np.array([target_embeddings[word] for word in words if word in source_embeddings and word in target_embeddings])
    
    # Standardize
    source_mean = source_matrix.mean(axis=0)
    target_mean = target_matrix.mean(axis=0)
    
    source_centered = source_matrix - source_mean
    target_centered = target_matrix - target_mean
    
    # SVD for Procrustes
    U, _, Vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation_matrix = (U @ Vt).T
    translation = target_mean - source_mean @ rotation_matrix.T
    
    print(f"\nTransformation matrix calculated from {len(source_matrix)} aligned words")
    print(f"Rotation matrix shape: {rotation_matrix.shape}")
    print(f"Translation vector shape: {translation.shape}")
    
    return rotation_matrix, translation, source_mean, target_mean

# Transform embeddings from source to target space
def transform_embedding(embedding, rotation_matrix, translation, source_mean):
    """Apply transformation to an embedding"""
    centered = embedding - source_mean
    transformed = centered @ rotation_matrix.T + translation
    return transformed

# Test mapping quality on anchor words
def test_anchor_mapping(source_embeddings, target_embeddings, rotation_matrix, translation, source_mean, words):
    """Test how well the mapping preserves anchor word relationships"""
    print("\n" + "="*60)
    print("TESTING MAPPING ON ANCHOR WORDS")
    print("="*60)
    
    errors = []
    for word in words[:10]:  # Test first 10 words
        if word in source_embeddings and word in target_embeddings:
            source_emb = source_embeddings[word]
            target_emb = target_embeddings[word]
            
            # Transform source to target space
            transformed = transform_embedding(source_emb, rotation_matrix, translation, source_mean)
            
            # Calculate error
            error = np.linalg.norm(transformed - target_emb)
            errors.append(error)
            print(f"{word:15} | Error: {error:.6f}")
    
    if errors:
        print(f"\nAverage error on anchor words: {np.mean(errors):.6f}")
        print(f"Max error: {np.max(errors):.6f}")
        print(f"Min error: {np.min(errors):.6f}")

# Test on unseen words
def test_unseen_words(rotation_matrix, translation, source_mean, test_words):
    """Embed unseen words and map them, showing the mapping in action"""
    print("\n" + "="*60)
    print("TESTING MAPPING ON UNSEEN WORDS")
    print("="*60)
    
    # Get embeddings from both models for test words
    source_embs, source_dim = get_embeddings_batch('nomic-embed-text', test_words)
    target_embs, target_dim = get_embeddings_batch('embeddinggemma', test_words)
    
    print("\n" + "-"*60)
    print("Mapping Quality (comparing direct vs. mapped):")
    print("-"*60)
    
    errors = []
    for word in test_words:
        if word in source_embs and word in target_embs:
            source_emb = source_embs[word]
            target_emb = target_embs[word]
            
            # Transform source to target space
            transformed = transform_embedding(source_emb, rotation_matrix, translation, source_mean)
            
            # Calculate error
            error = np.linalg.norm(transformed - target_emb)
            errors.append(error)
            
            # Calculate cosine similarity for context
            cos_sim = np.dot(transformed, target_emb) / (np.linalg.norm(transformed) * np.linalg.norm(target_emb) + 1e-8)
            
            print(f"{word:15} | Error: {error:.6f} | Cosine Sim: {cos_sim:.4f}")
    
    if errors:
        print(f"\nAverage error on unseen words: {np.mean(errors):.6f}")
        print(f"Max error: {np.max(errors):.6f}")
        print(f"Min error: {np.min(errors):.6f}")

def main():
    print("="*60)
    print("UNIVERSAL EMBEDDING SPACE MAPPER")
    print("="*60)
    
    # Load anchor words
    anchor_words = load_anchor_words('anchor_words.txt')
    print(f"\nLoaded {len(anchor_words)} anchor words")
    print(f"Sample: {anchor_words[:10]}")
    
    # Get embeddings from both models
    source_embeddings, source_dim = get_embeddings_batch('nomic-embed-text', anchor_words)
    target_embeddings, target_dim = get_embeddings_batch('embeddinggemma', anchor_words)
    
    # Check dimensions match
    print("\n" + "="*60)
    print("EMBEDDING DIMENSION CHECK")
    print("="*60)
    print(f"nomic-embed-text dimension: {source_dim}")
    print(f"embeddinggemma dimension: {target_dim}")
    
    if source_dim != target_dim:
        print(f"\nWARNING: Embedding dimensions don't match!")
        print(f"This will cause the mapping to fail.")
        print(f"Need to implement dimension matching first.")
        return
    else:
        print(f"✓ Dimensions match - mapping can proceed")
    
    # Calculate transformation matrix
    rotation_matrix, translation, source_mean, target_mean = calculate_transformation_matrix(
        source_embeddings, target_embeddings, anchor_words
    )
    
    # Test on anchor words
    test_anchor_mapping(source_embeddings, target_embeddings, rotation_matrix, translation, source_mean, anchor_words)
    
    # Test on unseen words
    test_words = ['ocean', 'mountain', 'sunset', 'dream', 'hope', 'danger', 'whisper', 'thunder']
    test_unseen_words(rotation_matrix, translation, source_mean, test_words)
    
    # Save transformation for later use
    save_transformation(rotation_matrix, translation, source_mean, target_mean, 'transformation.json')
    
    print("\n" + "="*60)
    print("MAPPING COMPLETE - Transformation saved to transformation.json")
    print("="*60)

def save_transformation(rotation_matrix, translation, source_mean, target_mean, filepath):
    """Save the transformation matrices for later use"""
    transformation = {
        'rotation_matrix': rotation_matrix.tolist(),
        'translation': translation.tolist(),
        'source_mean': source_mean.tolist(),
        'target_mean': target_mean.tolist(),
        'source_model': 'nomic-embed-text',
        'target_model': 'embeddinggemma'
    }
    with open(filepath, 'w') as f:
        json.dump(transformation, f, indent=2)
    print(f"Transformation saved to {filepath}")

if __name__ == '__main__':
    main()
