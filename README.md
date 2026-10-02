# Embedding Space Mapper

A transformer-based tool for mapping embeddings between different models (e.g., Nomic to Gemma).

## Directory Structure

```
├── src/                          # Core training & inference code
│   ├── train_transformer_mapper.py    # Main training script
│   ├── inference_mapper.py            # Run inference on trained model
│   ├── embedding_generator.py         # Generate embeddings for words
│   └── embedding_mapper.py            # Map embeddings between spaces
│
├── scripts/                       # Utility & experimental scripts
│   ├── test_user_input.py             # Interactive testing on user input
│   ├── generate_anchor_words.py       # Auto-generate anchor word lists via DeepSeek
│   └── embedding_approximator.py      # Find words that approximate embeddings
│
├── models/                        # Trained model weights
│   └── transformer_mapper_best.pth    # Best trained model checkpoint
│
├── config/                        # Configuration files
│   ├── transformer_mapper_config.json # Model config & test metrics
│   └── transformation.json             # Transformation parameters
│
├── data/                          # Datasets & outputs
│   ├── anchor_words.txt                # List of test words
│   ├── embeddings.db                   # SQLite database of embeddings
│   └── *.txt                           # Output files from scripts
│
└── README.md                      # This file
```

## Quick Start

### 1. Train the Model
```bash
python src/train_transformer_mapper.py
```

This trains the transformer to map from source (Nomic) to target (Gemma) embeddings.
- Auto-detects embedding dimensions
- Uses 8 transformer layers, 12 attention heads
- Saves best model to `models/`

### 2. Test Interactively
```bash
python scripts/test_user_input.py
```

Enter words to test the mapping:
- If in database: shows actual vs mapped embeddings
- If not in database: generates embeddings on-the-fly with Ollama

### 3. Generate Anchor Words
```bash
python scripts/generate_anchor_words.py
```

Continuously generates new test words via DeepSeek API, avoiding duplicates.

## Requirements

- PyTorch
- Ollama (running locally with nomic-embed-text and embeddinggemma models)
- SQLite3
- DeepSeek API key (for script generation)
- OpenAI SDK

## Model Architecture

- **Input**: Source embedding (768 or variable dimensions)
- **Hidden**: 3072 dimensions
- **Layers**: 8 transformer encoder layers
- **Heads**: 12 multi-head attention
- **Output**: Target embedding (768 or variable dimensions)

## Hyperparameters

- Batch size: 64
- Learning rate: 5e-4
- Epochs: 100
- Dropout: 0.05
- Optimizer: Adam with learning rate scheduler

## Notes

- Supports variable source/target dimensions
- Automatic dimension detection from database
- Early stopping with validation monitoring
- Cosine similarity + MSE loss tracking
