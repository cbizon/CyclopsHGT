#!/bin/bash
#
# Example script to generate predictions for drug-disease indications
#

# Set paths
MODEL_DIR="output/cyclops0"  # Directory with trained model
GRAPH_DIR="input_graphs/mf1"  # Original graph directory
PAIRS_FILE="medic/Indications List.csv"  # CSV with drug-disease pairs
OUTPUT_DIR="predictions/cyclops0_indications"  # Where to save predictions

# Run prediction
uv run python src/predict.py \
  --model_dir "$MODEL_DIR" \
  --graph_dir "$GRAPH_DIR" \
  --pairs_file "$PAIRS_FILE" \
  --head_col "final normalized drug id" \
  --tail_col "final normalized disease id" \
  --predicate "treats" \
  --output_dir "$OUTPUT_DIR" \
  --exclude_training \
  --device cuda \
  --batch_size 2048

echo "Done! Predictions saved to $OUTPUT_DIR"
