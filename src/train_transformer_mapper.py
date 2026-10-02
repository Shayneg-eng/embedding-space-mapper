import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import sqlite3
from pathlib import Path
import time
import json

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

DB_PATH = 'embeddings.db'

class TransformerEmbeddingMapper(nn.Module):
    """
    Transformer-based mapper for embedding space translation.
    Uses self-attention to understand the structure of source embeddings
    and learns to transform them to target space.
    Supports different source and target embedding dimensions.
    """
    
    def __init__(self, source_dim=768, target_dim=768, num_heads=12, num_layers=8, hidden_dim=3072, dropout=0.05):
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        # Project source embeddings to hidden dimension
        self.input_projection = nn.Linear(source_dim, hidden_dim)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to target embedding dimension
        self.output_projection = nn.Linear(hidden_dim, target_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, embedding_dim)
        Returns:
            output: (batch_size, embedding_dim)
        """
        # Reshape for transformer: (batch_size, 1, embedding_dim)
        x = x.unsqueeze(1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Project back to embedding dimension
        x = self.output_projection(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        return x

def load_embeddings_from_db():
    """Load all embeddings from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Load nomic embeddings (source)
    cursor.execute('SELECT word, embedding FROM nomic_embeddings')
    nomic_rows = cursor.fetchall()
    
    # Load gemma embeddings (target)
    cursor.execute('SELECT word, embedding FROM gemma_embeddings')
    gemma_rows = cursor.fetchall()
    
    conn.close()
    
    # Create dictionaries
    nomic_dict = {}
    for word, embedding_bytes in nomic_rows:
        nomic_dict[word] = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    gemma_dict = {}
    for word, embedding_bytes in gemma_rows:
        gemma_dict[word] = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    return nomic_dict, gemma_dict

def prepare_training_data(nomic_dict, gemma_dict, test_split=0.1, val_split=0.1):
    """
    Prepare training, validation, and test datasets
    Only use words that are in both embeddings
    """
    common_words = set(nomic_dict.keys()) & set(gemma_dict.keys())
    common_words = sorted(list(common_words))
    
    print(f"\nFound {len(common_words)} words in both embeddings")
    
    # Create tensors
    source_embeddings = []
    target_embeddings = []
    word_list = []
    
    for word in common_words:
        source_embeddings.append(nomic_dict[word])
        target_embeddings.append(gemma_dict[word])
        word_list.append(word)
    
    source_tensor = torch.FloatTensor(np.array(source_embeddings)).to(DEVICE)
    target_tensor = torch.FloatTensor(np.array(target_embeddings)).to(DEVICE)
    
    # Calculate splits
    total = len(common_words)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - test_size - val_size
    
    # Create splits
    train_dataset = TensorDataset(source_tensor[:train_size], target_tensor[:train_size])
    val_dataset = TensorDataset(
        source_tensor[train_size:train_size+val_size],
        target_tensor[train_size:train_size+val_size]
    )
    test_dataset = TensorDataset(
        source_tensor[train_size+val_size:],
        target_tensor[train_size+val_size:]
    )
    
    test_words = word_list[train_size+val_size:]
    
    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset, test_words

def train_epoch(model, train_loader, optimizer, criterion):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for source, target in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(source)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for source, target in val_loader:
            output = model(source)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def calculate_cosine_similarity(output, target):
    """Calculate cosine similarity between output and target"""
    output_norm = output / (torch.norm(output, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    return torch.mean(torch.sum(output_norm * target_norm, dim=1)).item()

def test_model(model, test_loader, test_words):
    """Test model on test set"""
    model.eval()
    total_mse_loss = 0
    total_cosine_sim = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for source, target in test_loader:
            output = model(source)
            loss = criterion(output, target)
            cosine_sim = calculate_cosine_similarity(output, target)
            
            total_mse_loss += loss.item()
            total_cosine_sim += cosine_sim
    
    avg_mse = total_mse_loss / len(test_loader)
    avg_cosine = total_cosine_sim / len(test_loader)
    
    return avg_mse, avg_cosine

def main():
    print("="*60)
    print("TRANSFORMER-BASED EMBEDDING SPACE MAPPER")
    print("="*60)
    
    # Load embeddings first to detect dimensions
    print(f"\nLoading embeddings from {DB_PATH}...")
    nomic_dict, gemma_dict = load_embeddings_from_db()
    print(f"Loaded {len(nomic_dict)} nomic embeddings")
    print(f"Loaded {len(gemma_dict)} gemma embeddings")
    
    # Auto-detect embedding dimensions from first embedding
    first_word = list(nomic_dict.keys())[0]
    source_dim = len(nomic_dict[first_word])
    target_dim = len(gemma_dict[first_word])
    print(f"Detected source embedding dimension: {source_dim}")
    print(f"Detected target embedding dimension: {target_dim}")
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 5e-4
    num_heads = 12
    num_layers = 8
    hidden_dim = 3072
    dropout = 0.05
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Source dim: {source_dim} | Target dim: {target_dim}")
    print(f"  Heads: {num_heads} | Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, test_words = prepare_training_data(nomic_dict, gemma_dict)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TransformerEmbeddingMapper(
        source_dim=source_dim,
        target_dim=target_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(DEVICE)
    
    print(f"\nModel initialized on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'transformer_mapper_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Load best model
    model.load_state_dict(torch.load('transformer_mapper_best.pth'))
    
    # Test on test set
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    
    test_mse, test_cosine = test_model(model, test_loader, test_words)
    print(f"Test MSE Loss: {test_mse:.6f}")
    print(f"Test Cosine Similarity: {test_cosine:.4f}")
    
    # Save model and config
    model_config = {
        'source_dim': source_dim,
        'target_dim': target_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'test_mse': float(test_mse),
        'test_cosine_similarity': float(test_cosine),
        'source_model': 'nomic-embed-text',
        'target_model': 'embeddinggemma'
    }
    
    with open('transformer_mapper_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\nModel saved to: transformer_mapper_best.pth")
    print(f"Config saved to: transformer_mapper_config.json")

if __name__ == '__main__':
    main()
