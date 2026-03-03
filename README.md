# CyclopsHGT

Heterogeneous Graph Transformer (HGT) for link prediction on the ROBOKOP biomedical knowledge graph.

## Overview

This project implements HGT for link prediction on biomedical knowledge graphs, with a focus on evaluating performance on rare but important predicates like `treats`. The model learns embeddings for heterogeneous nodes (diseases, chemicals, genes, etc.) and predicts missing links in the graph.

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions, including platform-specific DGL setup.

**Quick start (Linux with GPU):**
```bash
uv sync
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.9/cu121/repo.html  # adjust CUDA version
```

## Input Data Format

The model expects a graph directory containing two TSV files:

**nodes.tsv**
```
id	type
MONDO:0005148	Disease
CHEBI:6801	Chemical
HGNC:1234	Gene
```

**edges.tsv**
```
subject	predicate	object
CHEBI:6801	treats	MONDO:0005148
HGNC:1234	associated_with	MONDO:0005148
```

Place your processed graph data in `input_graphs/<name>/`.

## Usage

### Basic Training

```bash
uv run python src/main.py \
  --graph_dir input_graphs/robokop_v1 \
  --output_dir output/experiment_1
```

### Advanced Options

```bash
uv run python src/main.py \
  --graph_dir input_graphs/robokop_v1 \
  --output_dir output/experiment_1 \
  --hidden_dim 256 \
  --num_layers 3 \
  --num_heads 8 \
  --epochs 200 \
  --lr 0.001 \
  --dropout 0.3 \
  --negative_ratio 5 \
  --focus_predicates treats affects \
  --eval_every 10 \
  --device cuda
```

### Key Arguments

**Model Architecture:**
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_layers`: Number of HGT layers (default: 2)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.2)

**Training:**
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--negative_ratio`: Negative samples per positive edge (default: 1)

**Evaluation:**
- `--focus_predicates`: Predicates to highlight (default: treats)
- `--eval_every`: Evaluate every N epochs (default: 5)

**Data Splitting:**
- `--train_ratio`: Training set ratio (default: 0.8)
- `--val_ratio`: Validation set ratio (default: 0.1)
- `--test_ratio`: Test set ratio (default: 0.1)
- `--seed`: Random seed (default: 42)

## Output

Results are saved to the output directory:

```
output/experiment_1/
├── config.json               # Training configuration
├── metadata.json             # Graph statistics
├── best_model.pt             # Best model checkpoint
├── training_log.tsv          # Epoch-by-epoch training progress
├── edge_metrics.tsv          # Per-edge-type test metrics
├── final_test_metrics.tsv    # Overall test performance
├── train_edges.tsv           # Training set edges
├── val_edges.tsv             # Validation set edges
└── test_edges.tsv            # Test set edges
```

## Evaluation Metrics

The model reports three metrics per edge type:

- **Accuracy**: Fraction of positive edges scoring higher than negative edges
- **MRR**: Mean Reciprocal Rank - average 1/rank of positive edges
- **AUC**: Area Under Curve approximation - probability positive scores higher than negative

Metrics are computed both overall and per predicate type, with special focus on rare predicates.

## Making Predictions

After training a model, you can generate predictions for specific entity pairs:

```bash
./predict_indications.sh
```

Or use the predict script directly:

```bash
uv run python src/predict.py \
  --model_dir output/experiment_1 \
  --graph_dir input_graphs/robokop_v1 \
  --pairs_file "medic/Indications List.csv" \
  --head_col "final normalized drug id" \
  --tail_col "final normalized disease id" \
  --predicate "treats" \
  --output_dir predictions/experiment_1_indications
```

This generates predictions for all drug-disease pairs, excluding training edges. See [docs/PREDICTION.md](docs/PREDICTION.md) for detailed documentation.

## Example Workflow

1. **Prepare your graph data:**
   ```bash
   mkdir -p input_graphs/robokop_v1
   # Copy nodes.tsv and edges.tsv to input_graphs/robokop_v1/
   ```

2. **Test data loading:**
   ```bash
   uv run python src/graph_loader.py input_graphs/robokop_v1
   ```

3. **Train model:**
   ```bash
   uv run python src/main.py \
     --graph_dir input_graphs/robokop_v1 \
     --output_dir output/baseline \
     --epochs 100
   ```

4. **Analyze results:**
   ```bash
   cat output/baseline/final_test_metrics.tsv
   cat output/baseline/edge_metrics.tsv
   ```

5. **Generate predictions:**
   ```bash
   uv run python src/predict.py \
     --model_dir output/baseline \
     --graph_dir input_graphs/robokop_v1 \
     --pairs_file "medic/Indications List.csv" \
     --head_col "final normalized drug id" \
     --tail_col "final normalized disease id" \
     --predicate "treats" \
     --output_dir predictions/baseline_indications
   ```

## Project Structure

```
.
├── src/
│   ├── graph_loader.py      # Load graph from TSV files
│   ├── data_split.py        # Train/val/test splitting and negative sampling
│   ├── model.py             # HGT model architecture
│   ├── train.py             # Training and evaluation utilities
│   ├── main.py              # Main training script
│   └── predict.py           # Generate predictions from trained model
├── docs/
│   └── PREDICTION.md        # Prediction documentation
├── tests/                   # Unit tests
├── input_graphs/            # Input graph data
├── output/                  # Training outputs
├── predictions/             # Prediction outputs
├── medic/                   # External evaluation data
├── predict_indications.sh   # Example prediction script
├── INSTALL.md               # Installation instructions
├── CLAUDE.md                # Project instructions for Claude
└── README.md                # This file
```

## Development

Run tests:
```bash
uv run pytest
```

## Future Work

- Sparse Autoencoder (SAE) analysis of learned embeddings
- Context-specific embedding analysis at prediction time
- Attention weight visualization
- Multi-hop path analysis

## License

MIT License - see LICENSE file
