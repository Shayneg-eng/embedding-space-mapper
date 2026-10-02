import ollama
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import json
from pathlib import Path
import time

# Database setup
DB_PATH = 'embeddings.db'

def init_database():
    """Initialize SQLite database for storing embeddings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables for each model
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nomic_embeddings (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gemma_embeddings (
            id INTEGER PRIMARY KEY,
            word TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")

def load_anchor_words(filepath):
    """Load anchor words from file"""
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            if word:  # Skip empty lines
                words.append(word)
    return words

def embed_word(model, word):
    """Embed a single word with a model"""
    try:
        response = ollama.embed(model=model, input=word)
        embedding = np.array(response['embeddings'][0])
        return word, embedding, None
    except Exception as e:
        return word, None, str(e)

def embed_batch_parallel(model, words, num_workers=20):
    """
    Embed all words in parallel using ThreadPoolExecutor
    Returns dict of {word: embedding_array}
    """
    embeddings = {}
    errors = {}
    completed = 0
    
    print(f"\nEmbedding {len(words)} words with {model} using {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(embed_word, model, word): word for word in words}
        
        # Process results as they complete
        for future in as_completed(futures):
            word, embedding, error = future.result()
            completed += 1
            
            if error:
                errors[word] = error
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(words)} (ERROR: {word})")
            else:
                embeddings[word] = embedding
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(words)}")
    
    print(f"Successfully embedded {len(embeddings)}/{len(words)} words")
    if errors:
        print(f"Failed on {len(errors)} words")
    
    return embeddings, errors

def save_embeddings_to_db(model_name, embeddings):
    """Save embeddings to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    table_name = 'nomic_embeddings' if model_name == 'nomic-embed-text' else 'gemma_embeddings'
    
    saved_count = 0
    for word, embedding in embeddings.items():
        try:
            # Convert numpy array to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            embedding_dim = len(embedding)
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table_name} (word, embedding, embedding_dim)
                VALUES (?, ?, ?)
            ''', (word, embedding_bytes, embedding_dim))
            
            saved_count += 1
        except Exception as e:
            print(f"Error saving {word}: {e}")
    
    conn.commit()
    
    # Update metadata
    cursor.execute('INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)',
                   (f'{table_name}_count', str(saved_count)))
    conn.commit()
    conn.close()
    
    print(f"Saved {saved_count} embeddings to {table_name}")

def load_embeddings_from_db(model_name):
    """Load embeddings from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    table_name = 'nomic_embeddings' if model_name == 'nomic-embed-text' else 'gemma_embeddings'
    
    cursor.execute(f'SELECT word, embedding, embedding_dim FROM {table_name}')
    rows = cursor.fetchall()
    
    embeddings = {}
    for word, embedding_bytes, embedding_dim in rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)
        embeddings[word] = embedding
    
    conn.close()
    return embeddings

def get_embedding_stats(model_name):
    """Get statistics about stored embeddings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    table_name = 'nomic_embeddings' if model_name == 'nomic-embed-text' else 'gemma_embeddings'
    
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    count = cursor.fetchone()[0]
    
    cursor.execute(f'SELECT embedding_dim FROM {table_name} LIMIT 1')
    result = cursor.fetchone()
    dim = result[0] if result else None
    
    conn.close()
    
    return count, dim

def main():
    print("="*60)
    print("EMBEDDING GENERATOR - Batch Processing with Parallel Workers")
    print("="*60)
    
    # Initialize database
    init_database()
    
    # Load anchor words
    anchor_words = load_anchor_words('anchor_words.txt')
    print(f"\nLoaded {len(anchor_words)} anchor words")
    print(f"Sample: {anchor_words[:5]}")
    
    # Embed with both models
    models = ['nomic-embed-text', 'embeddinggemma']
    
    for model in models:
        print("\n" + "="*60)
        print(f"Processing: {model}")
        print("="*60)
        
        # Check if already embedded
        count, dim = get_embedding_stats(model)
        if count > 0:
            print(f"Found {count} existing embeddings (dimension: {dim})")
            if count == len(anchor_words):
                print(f"All {len(anchor_words)} words already embedded. Skipping.")
                continue
            else:
                print(f"Missing {len(anchor_words) - count} embeddings. Continuing...")
        
        # Embed in parallel
        start_time = time.time()
        embeddings, errors = embed_batch_parallel(model, anchor_words, num_workers=20)
        elapsed = time.time() - start_time
        
        # Save to database
        save_embeddings_to_db(model, embeddings)
        
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Speed: {len(anchor_words)/elapsed:.2f} words/second")
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING SUMMARY")
    print("="*60)
    
    for model in models:
        count, dim = get_embedding_stats(model)
        print(f"{model}: {count} embeddings (dim: {dim})")
    
    print(f"\nAll embeddings saved to: {DB_PATH}")

if __name__ == '__main__':
    main()
