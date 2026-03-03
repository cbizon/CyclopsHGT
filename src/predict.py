"""
Generate predictions for a set of entity pairs using a trained HGT model.
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

from graph_loader import load_graph_from_tsv
from model import HGTLinkPredictor


def load_entity_pairs(csv_path, head_col, tail_col):
    """Load entity pairs from CSV file."""
    df = pd.read_csv(csv_path)

    # Get unique entities
    head_entities = df[head_col].unique()
    tail_entities = df[tail_col].unique()

    # Generate all pairs
    pairs = []
    for head in head_entities:
        for tail in tail_entities:
            pairs.append((head, tail))

    return pairs, set(zip(df[head_col], df[tail_col]))


def load_edge_splits(model_dir):
    """Load train/val/test edge splits from saved files."""
    splits = {}

    for split_name in ['train', 'val', 'test']:
        split_file = Path(model_dir) / f'{split_name}_edges.tsv'

        if not split_file.exists():
            raise FileNotFoundError(
                f"Edge split file not found: {split_file}\n"
                f"This file should be created during training. "
                f"Make sure you're using a model trained with the updated code."
            )

        df = pd.read_csv(split_file, sep='\t')
        # Create set of (subject, predicate, object) tuples
        splits[split_name] = set(zip(df['subject'], df['predicate'], df['object']))

    return splits['train'], splits['val'], splits['test']


def map_entities_to_ids(pairs, metadata):
    """
    Map entity CURIEs to internal node IDs, auto-detecting types.

    Returns dict mapping edge_type -> list of (head_curie, tail_curie, head_idx, tail_idx)
    """
    node_id_to_type_idx = metadata['node_id_to_type_idx']

    # Group pairs by edge type
    pairs_by_edge_type = {}
    unmapped_heads = set()
    unmapped_tails = set()

    for head_curie, tail_curie in pairs:
        # Look up types for each entity
        if head_curie not in node_id_to_type_idx:
            unmapped_heads.add(head_curie)
            continue
        if tail_curie not in node_id_to_type_idx:
            unmapped_tails.add(tail_curie)
            continue

        head_type, head_idx = node_id_to_type_idx[head_curie]
        tail_type, tail_idx = node_id_to_type_idx[tail_curie]

        # Create edge type tuple (will be combined with predicate later)
        edge_type_key = (head_type, tail_type)

        if edge_type_key not in pairs_by_edge_type:
            pairs_by_edge_type[edge_type_key] = []

        pairs_by_edge_type[edge_type_key].append((
            head_curie,
            tail_curie,
            head_idx,
            tail_idx
        ))

    if unmapped_heads or unmapped_tails:
        print(f"Warning: Could not map {len(unmapped_heads)} head entities and {len(unmapped_tails)} tail entities (not in graph)")
        if len(unmapped_heads) <= 10:
            print(f"  Unmapped heads: {list(unmapped_heads)[:10]}")
        if len(unmapped_tails) <= 10:
            print(f"  Unmapped tails: {list(unmapped_tails)[:10]}")

    return pairs_by_edge_type


def predict_scores_by_type(model, g_homog, g, pairs_by_type, predicate, device, batch_size=1024):
    """
    Generate prediction scores for entity pairs, grouped by edge type.

    Args:
        pairs_by_type: Dict mapping (head_type, tail_type) -> list of (head_curie, tail_curie, head_idx, tail_idx)
        predicate: The predicate to predict
    """
    model.eval()

    all_results = []

    with torch.no_grad():
        # Get embeddings once for the whole graph
        print("Computing node embeddings...")
        h = model.encoder(g_homog)

        # Process each edge type
        for (head_type, tail_type), pairs in pairs_by_type.items():
            edge_type = (head_type, predicate, tail_type)

            # Check if edge type exists in the graph
            if edge_type not in g.canonical_etypes:
                print(f"Warning: Edge type {edge_type} not in training graph. Predictions may be unreliable.")

            print(f"Predicting {len(pairs)} pairs for edge type {edge_type}...")

            # Process in batches
            for i in tqdm(range(0, len(pairs), batch_size), desc=f"  {head_type}->{tail_type}", leave=False):
                batch = pairs[i:i + batch_size]

                head_curies = [p[0] for p in batch]
                tail_curies = [p[1] for p in batch]
                head_ids = torch.tensor([p[2] for p in batch], dtype=torch.long, device=device)
                tail_ids = torch.tensor([p[3] for p in batch], dtype=torch.long, device=device)

                # Use the link predictor to compute scores
                scores = model.predictor(h, edge_type, head_ids, tail_ids)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(scores)

                # Store results
                for head_curie, tail_curie, prob in zip(head_curies, tail_curies, probs.cpu().numpy()):
                    all_results.append({
                        'head': head_curie,
                        'tail': tail_curie,
                        'score': float(prob)
                    })

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for entity pairs')

    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model (output from training)')
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Directory containing the original graph')

    # Input data arguments
    parser.add_argument('--pairs_file', type=str, required=True,
                        help='CSV file containing entity pairs')
    parser.add_argument('--head_col', type=str, required=True,
                        help='Column name for head entities')
    parser.add_argument('--tail_col', type=str, required=True,
                        help='Column name for tail entities')

    # Prediction arguments
    parser.add_argument('--predicate', type=str, required=True,
                        help='Predicate to predict (e.g., "treats")')

    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predictions')
    parser.add_argument('--exclude_training', action='store_true', default=True,
                        help='Exclude training edges from predictions')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for predictions')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print(f"Loading entity pairs from {args.pairs_file}...")
    all_pairs, known_positives = load_entity_pairs(args.pairs_file, args.head_col, args.tail_col)
    print(f"Found {len(all_pairs)} total pairs ({len(known_positives)} known positives)")

    print(f"\nLoading graph from {args.graph_dir}...")
    g, metadata = load_graph_from_tsv(Path(args.graph_dir))

    # Load edge splits
    print("Loading edge splits from model directory...")
    train_edges, val_edges, test_edges = load_edge_splits(model_dir)
    print(f"Found {len(train_edges)} training edges")
    print(f"Found {len(val_edges)} validation edges")
    print(f"Found {len(test_edges)} test edges")

    # Categorize all pairs into train/val/test/other
    predicate = args.predicate
    train_pairs = []
    val_pairs = []
    test_pairs = []
    other_pairs = []

    for head, tail in all_pairs:
        edge_tuple = (head, predicate, tail)
        if edge_tuple in train_edges:
            train_pairs.append((head, tail))
        elif edge_tuple in val_edges:
            val_pairs.append((head, tail))
        elif edge_tuple in test_edges:
            test_pairs.append((head, tail))
        else:
            other_pairs.append((head, tail))

    print(f"\nCategorized {len(all_pairs)} pairs from Indications List:")
    print(f"  Training: {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")
    print(f"  Test: {len(test_pairs)}")
    print(f"  Other (not in graph): {len(other_pairs)}")

    # Map entity pairs to node IDs for each category (auto-detecting types)
    print(f"\nMapping entities to node IDs (auto-detecting types)...")
    mapped_train_by_type = map_entities_to_ids(train_pairs, metadata)
    mapped_val_by_type = map_entities_to_ids(val_pairs, metadata)
    mapped_test_by_type = map_entities_to_ids(test_pairs, metadata)
    mapped_other_by_type = map_entities_to_ids(other_pairs, metadata)

    # Count total mapped pairs
    total_train = sum(len(pairs) for pairs in mapped_train_by_type.values())
    total_val = sum(len(pairs) for pairs in mapped_val_by_type.values())
    total_test = sum(len(pairs) for pairs in mapped_test_by_type.values())
    total_other = sum(len(pairs) for pairs in mapped_other_by_type.values())

    print(f"Successfully mapped:")
    print(f"  Training: {total_train} pairs")
    print(f"  Validation: {total_val} pairs")
    print(f"  Test: {total_test} pairs")
    print(f"  Other: {total_other} pairs")

    # Show edge type distribution
    all_edge_types = set()
    all_edge_types.update(mapped_train_by_type.keys())
    all_edge_types.update(mapped_val_by_type.keys())
    all_edge_types.update(mapped_test_by_type.keys())
    all_edge_types.update(mapped_other_by_type.keys())

    if all_edge_types:
        print(f"\nFound {len(all_edge_types)} edge type combinations:")
        for head_type, tail_type in sorted(all_edge_types):
            count = (len(mapped_train_by_type.get((head_type, tail_type), [])) +
                    len(mapped_val_by_type.get((head_type, tail_type), [])) +
                    len(mapped_test_by_type.get((head_type, tail_type), [])) +
                    len(mapped_other_by_type.get((head_type, tail_type), [])))
            print(f"  ({head_type}, {predicate}, {tail_type}): {count} pairs")

    if total_train == 0 and total_val == 0 and total_test == 0 and total_other == 0:
        print("ERROR: No pairs could be mapped to graph nodes!")
        print("All entities in the pairs file must exist in the graph.")
        return

    # Prepare graph for prediction
    print("\nPreparing graph...")
    ntype_to_id = {ntype: i for i, ntype in enumerate(g.ntypes)}
    etype_to_id = {etype: i for i, etype in enumerate(g.canonical_etypes)}

    g.ndata['_TYPE'] = {
        ntype: torch.full((g.num_nodes(ntype),), ntype_to_id[ntype], dtype=torch.long)
        for ntype in g.ntypes
    }
    g.edata['_TYPE'] = {
        etype: torch.full((g.num_edges(etype),), etype_to_id[etype], dtype=torch.long)
        for etype in g.canonical_etypes
    }

    import dgl
    g_homog = dgl.to_homogeneous(g, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)
    g_homog = g_homog.to(device)

    # Load model
    print("\nLoading trained model...")
    config_path = model_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    model = HGTLinkPredictor(
        g,
        n_hidden=config['hidden_dim'],
        n_layers=config['num_layers'],
        n_heads=config['num_heads'],
        dropout=config['dropout'],
        use_relation_transform=config['use_relation_transform']
    ).to(device)

    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Make predictions
    print(f"\nGenerating predictions...")

    # Generate predictions for each category
    print("\n1. Predictions on training data...")
    train_results = predict_scores_by_type(model, g_homog, g, mapped_train_by_type, predicate, device, args.batch_size) if mapped_train_by_type else []

    print("\n2. Predictions on validation data...")
    val_results = predict_scores_by_type(model, g_homog, g, mapped_val_by_type, predicate, device, args.batch_size) if mapped_val_by_type else []

    print("\n3. Predictions on test data...")
    test_results = predict_scores_by_type(model, g_homog, g, mapped_test_by_type, predicate, device, args.batch_size) if mapped_test_by_type else []

    print("\n4. Predictions on other pairs...")
    other_results = predict_scores_by_type(model, g_homog, g, mapped_other_by_type, predicate, device, args.batch_size) if mapped_other_by_type else []

    # Combine non-training results for main predictions file
    if args.exclude_training:
        main_results = val_results + test_results + other_results
    else:
        main_results = train_results + val_results + test_results + other_results

    # Write main predictions file (non-training or all, depending on exclude_training flag)
    predictions_path = output_dir / 'predictions.tsv'
    print(f"\nWriting main predictions to {predictions_path}...")
    with open(predictions_path, 'w') as f:
        f.write("drug_id\tdisease_id\tscore\n")
        for result in main_results:
            f.write(f"{result['head']}\t{result['tail']}\t{result['score']:.6f}\n")
    print(f"Wrote {len(main_results)} predictions")

    # Write training predictions file
    if train_results:
        training_preds_path = output_dir / 'training_predictions.tsv'
        print(f"\nWriting training predictions to {training_preds_path}...")
        with open(training_preds_path, 'w') as f:
            f.write("drug_id\tdisease_id\tscore\n")
            for result in train_results:
                f.write(f"{result['head']}\t{result['tail']}\t{result['score']:.6f}\n")
        print(f"Wrote {len(train_results)} training predictions")

    # Write holdout (val+test) predictions file with split labels
    if val_results or test_results:
        holdout_path = output_dir / 'holdout_predictions.tsv'
        print(f"\nWriting holdout predictions to {holdout_path}...")
        with open(holdout_path, 'w') as f:
            f.write("drug_id\tdisease_id\tscore\tsplit\n")
            for result in val_results:
                f.write(f"{result['head']}\t{result['tail']}\t{result['score']:.6f}\tval\n")
            for result in test_results:
                f.write(f"{result['head']}\t{result['tail']}\t{result['score']:.6f}\ttest\n")
        print(f"Wrote {len(val_results) + len(test_results)} holdout predictions ({len(val_results)} val, {len(test_results)} test)")

    # Write training pairs file (for integration with external models)
    if args.exclude_training:
        training_pairs_path = output_dir / 'training_pairs.tsv'
        print(f"\nWriting training pairs to {training_pairs_path}...")

        with open(training_pairs_path, 'w') as f:
            # Use standardized column names required by external evaluation
            f.write("drug_id\tdisease_id\tlabel\n")
            for head, pred, tail in train_edges:
                if pred == predicate:
                    f.write(f"{head}\t{tail}\t1\n")

        print(f"Wrote {sum(1 for h, p, t in train_edges if p == predicate)} training pairs")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
