# Making Predictions with Trained HGT Models

This guide explains how to use a trained HGT model to make predictions on specific entity pairs.

## Understanding the Process

### How Entity Mapping Works

The entities in your CSV (like `CHEBI:8327`, `MONDO:0005009`) are CURIEs - these **are** the node IDs. The script automatically detects each entity's type from the loaded graph:

1. **Your input**: `CHEBI:8327` (a CURIE from your CSV)
2. **Script looks up type**: Finds `CHEBI:8327` is type `Drug` in `nodes.tsv`
3. **Maps to index**: `CHEBI:8327` → index 42 within `Drug` nodes
4. **Model uses**: index 42 to look up the Drug embedding

**No manual type specification needed!** The script handles mixed types automatically. For example, if your CSV has both `Drug` and `SmallMolecule` entities, the script will:
- Auto-detect each entity's type
- Group pairs by edge type: `(Drug, treats, Disease)` and `(SmallMolecule, treats, Disease)`
- Make predictions for each edge type combination

## Overview

The `predict.py` script allows you to:
1. Load a trained HGT model
2. Read entity pairs from a CSV file
3. Generate prediction scores for all pairs
4. Filter out **training edges only** to avoid data leakage (validation/test edges are fair game)
5. Output results in a format compatible with external evaluation frameworks

**Key point**: Only edges from the training set are excluded. Validation and test edges were held-out during training, so predictions on them are valid for evaluation.

## Quick Start

```bash
./predict_indications.sh
```

This will generate predictions for drug-disease pairs from the Indications List using a trained model.

## Usage

### Basic Command

```bash
uv run python src/predict.py \
  --model_dir output/cyclops0 \
  --graph_dir input_graphs/mf1 \
  --pairs_file "medic/Indications List.csv" \
  --head_col "final normalized drug id" \
  --tail_col "final normalized disease id" \
  --predicate "treats" \
  --output_dir predictions/cyclops0_indications
```

### Required Arguments

- `--model_dir`: Directory containing the trained model (must have `best_model.pt` and `config.json`)
- `--graph_dir`: Directory with the original graph used for training (`nodes.tsv` and `edges.tsv`)
- `--pairs_file`: CSV file containing entity pairs to predict
- `--head_col`: Column name for head entities (e.g., drug IDs)
- `--tail_col`: Column name for tail entities (e.g., disease IDs)
- `--predicate`: Relation to predict (e.g., `"treats"`)
- `--output_dir`: Directory to save prediction results

### Optional Arguments

- `--exclude_training`: Exclude edges from training set (default: True)
- `--device`: Device to use for prediction (`cuda` or `cpu`, default: auto-detect)
- `--batch_size`: Batch size for predictions (default: 1024)

## Input Format

### Pairs File

The pairs file should be a CSV with at least two columns containing entity CURIEs:

```csv
final normalized drug id,final normalized drug label,final normalized disease id,final normalized disease label
CHEBI:8327,Polythiazide,MONDO:0005009,congestive heart failure
CHEBI:8327,Polythiazide,MONDO:0005155,liver cirrhosis
```

The script will:
1. Extract unique head and tail entities
2. Generate all possible head-tail pairs
3. Filter out pairs that were in the training set (if `--exclude_training` is enabled)
4. Make predictions for the remaining pairs

## Output Format

The script generates up to four files in the output directory:

### 1. predictions.tsv

Main predictions file (excludes training edges if `--exclude_training` is enabled):

```tsv
drug_id	disease_id	score
CHEBI:8327	MONDO:0005009	0.856234
CHEBI:8327	MONDO:0005155	0.742891
```

**Contains**: Val + Test + Other pairs (if `--exclude_training`), or all pairs otherwise.

### 2. training_predictions.tsv

Predictions on training data:

```tsv
drug_id	disease_id	score
CHEBI:6801	MONDO:0005015	0.923456
CHEBI:6801	MONDO:0011382	0.891234
```

**Contains**: Predictions for pairs that were in the training set. Useful for checking model fit.

### 3. holdout_predictions.tsv

Predictions on validation and test sets with split labels:

```tsv
drug_id	disease_id	score	split
CHEBI:8327	MONDO:0005009	0.856234	val
CHEBI:8327	MONDO:0005155	0.742891	test
CHEBI:64354	MONDO:0005044	0.812345	val
```

**Contains**: Only val/test pairs with an extra column indicating which split each pair belongs to. Perfect for evaluation on truly held-out data.

### 4. training_pairs.tsv

Training pairs file (for external evaluation integration):

```tsv
drug_id	disease_id	label
CHEBI:8327	MONDO:0005009	1
CHEBI:1234	MONDO:0005678	1
```

**Contains**: All training edges for the specified predicate. Used by external evaluation frameworks for data leakage prevention.

## Understanding Train/Val/Test Splits

During training, the model saves three edge split files in the model directory:

- **train_edges.tsv**: Edges used for training (model saw these during optimization)
- **val_edges.tsv**: Edges used for validation (used to select best model, but not for gradient updates)
- **test_edges.tsv**: Edges used for final testing (completely held-out)

The prediction script **only excludes train_edges.tsv** when making predictions. This means:
- ✅ You CAN make predictions on validation edges (fair game)
- ✅ You CAN make predictions on test edges (fair game)
- ❌ You CANNOT make predictions on training edges (would be data leakage)

This allows you to evaluate model performance on truly held-out data.

## Integration with External Evaluation

The output format is compatible with the external model integration framework described in `medic/README_external_models.md`.

To integrate your predictions:

```bash
python path/to/integrate_external_predictions.py \
  --predictions predictions/cyclops0_indications/predictions.tsv \
  --ground-truth "medic/Indications List.csv" \
  --training-pairs predictions/cyclops0_indications/training_pairs.tsv \
  --model-name "cyclops_hgt_v1"
```

## Node Type Discovery

If you're not sure what node types are available in your graph, run the script without correct types and it will show you:

```bash
uv run python src/predict.py \
  --model_dir output/cyclops0 \
  --graph_dir input_graphs/mf1 \
  --pairs_file "medic/Indications List.csv" \
  --head_col "final normalized drug id" \
  --tail_col "final normalized disease id" \
  --predicate "treats" \
  --output_dir predictions/test
```

The error message will list all available node types.

## Troubleshooting

### "No pairs could be mapped to graph nodes"

This means the entity CURIEs in your CSV don't match the node IDs in the graph.

Check:
1. The column names are correct (`--head_col` and `--tail_col`)
3. The entity IDs in the CSV exist in the graph's `nodes.tsv`

### "Edge type not found in graph"

The combination of (head_type, predicate, tail_type) doesn't exist in the training graph.

The script will still make predictions, but the model may not have been trained on this specific edge type. Check:
1. The predicate name matches what's in `edges.tsv`
2. The head/tail types are correct for this relation

### Out of Memory

If you run out of GPU memory:
1. Reduce `--batch_size` (try 512 or 256)
2. Use CPU instead: `--device cpu`

## Example Workflows

### Generate predictions for a specific model

```bash
uv run python src/predict.py \
  --model_dir output/cyclops2 \
  --graph_dir input_graphs/mf1 \
  --pairs_file "medic/Indications List.csv" \
  --head_col "final normalized drug id" \
  --tail_col "final normalized disease id" \
  --predicate "treats" \
  --output_dir predictions/cyclops2_indications \
  --batch_size 2048
```

### Compare multiple models

```bash
# Generate predictions for each model
for i in 0 1 2 3; do
  uv run python src/predict.py \
    --model_dir output/cyclops$i \
    --graph_dir input_graphs/mf1 \
    --pairs_file "medic/Indications List.csv" \
    --head_col "final normalized drug id" \
    --tail_col "final normalized disease id" \
    --predicate "treats" \
    --output_dir predictions/cyclops${i}_indications
done
```

Then integrate all models for comparison using the external evaluation framework.
