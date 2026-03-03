"""
Main training script for HGT link prediction on ROBOKOP.
"""

import argparse
import torch
import dgl
from pathlib import Path
import json

from graph_loader import load_graph_from_tsv
from data_split import stratified_edge_split, create_train_graph, prepare_edge_batches
from model import HGTLinkPredictor
from train import train_epoch, evaluate, print_metrics


def main():
    parser = argparse.ArgumentParser(description='Train HGT for link prediction')

    # Data arguments
    parser.add_argument('--graph_dir', type=str, required=True,
                        help='Directory containing nodes.tsv and edges.tsv')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save model and results')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of HGT layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_relation_transform', action='store_true', default=True,
                        help='Use relation-specific transformations')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--negative_ratio', type=int, default=1,
                        help='Number of negative samples per positive edge')

    # Split arguments
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Evaluation arguments
    parser.add_argument('--focus_predicates', type=str, nargs='+', default=['treats'],
                        help='Predicates to highlight in evaluation')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Load graph
    print(f"\n{'='*80}")
    print("LOADING GRAPH")
    print(f"{'='*80}")
    graph_dir = Path(args.graph_dir)
    g, metadata = load_graph_from_tsv(graph_dir)

    # Save metadata
    # Remove non-serializable items
    serializable_metadata = {
        'node_type_counts': metadata['node_type_counts'],
        'edge_type_counts': metadata['edge_type_counts'],
        'canonical_edge_counts': {str(k): v for k, v in metadata['canonical_edge_counts'].items()},
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(serializable_metadata, f, indent=2)

    # Add type information required by HGT
    # Map node types to integers
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

    # Split edges
    print(f"\n{'='*80}")
    print("SPLITTING EDGES")
    print(f"{'='*80}")
    train_edges, val_edges, test_edges = stratified_edge_split(
        g, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    # Save edge splits for later use
    def save_edge_split(edges_dict, filepath, metadata):
        """Save edge split to TSV file."""
        with open(filepath, 'w') as f:
            f.write("subject\tpredicate\tobject\n")
            for edge_type, edge_indices in edges_dict.items():
                src_type, predicate, dst_type = edge_type

                # Get source and destination node indices for these edges
                src_nodes, dst_nodes = g.edges(etype=edge_type)

                # Filter to only the edges in this split
                src_nodes = src_nodes[edge_indices]
                dst_nodes = dst_nodes[edge_indices]

                # Get the node ID mappings
                src_type_ids = {idx: nid for nid, idx in metadata['node_type_to_ids'][src_type].items()}
                dst_type_ids = {idx: nid for nid, idx in metadata['node_type_to_ids'][dst_type].items()}

                # Write edges
                for src_idx, dst_idx in zip(src_nodes.tolist(), dst_nodes.tolist()):
                    src_id = src_type_ids[src_idx]
                    dst_id = dst_type_ids[dst_idx]
                    f.write(f"{src_id}\t{predicate}\t{dst_id}\n")

    print("\nSaving edge splits...")
    save_edge_split(train_edges, output_dir / 'train_edges.tsv', metadata)
    save_edge_split(val_edges, output_dir / 'val_edges.tsv', metadata)
    save_edge_split(test_edges, output_dir / 'test_edges.tsv', metadata)
    print(f"Saved train/val/test edge splits to {output_dir}")

    # Create training graph (only train edges)
    g_train = create_train_graph(g, train_edges)

    # Add type information to training graph
    g_train.ndata['_TYPE'] = {
        ntype: torch.full((g_train.num_nodes(ntype),), ntype_to_id[ntype], dtype=torch.long)
        for ntype in g_train.ntypes
    }
    g_train.edata['_TYPE'] = {
        etype: torch.full((g_train.num_edges(etype),), etype_to_id[etype], dtype=torch.long)
        for etype in g_train.canonical_etypes
    }

    print(f"\nTraining graph: {g_train.num_edges()} total edges")

    # Convert to homogeneous for HGTConv
    print("\nConverting training graph to homogeneous format...")
    g_train_homog = dgl.to_homogeneous(g_train, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)
    print(f"Homogeneous graph: {g_train_homog.num_nodes()} nodes, {g_train_homog.num_edges()} edges")

    # Prepare batches
    print(f"\n{'='*80}")
    print("PREPARING BATCHES")
    print(f"{'='*80}")
    print("Preparing training batches...")
    train_batches = prepare_edge_batches(g, train_edges, args.negative_ratio)

    print("Preparing validation batches...")
    val_batches = prepare_edge_batches(g, val_edges, args.negative_ratio)

    print("Preparing test batches...")
    test_batches = prepare_edge_batches(g, test_edges, args.negative_ratio)

    # Create model
    print(f"\n{'='*80}")
    print("CREATING MODEL")
    print(f"{'='*80}")
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = HGTLinkPredictor(
        g_train,  # Pass heterogeneous graph for schema
        n_hidden=args.hidden_dim,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        dropout=args.dropout,
        use_relation_transform=args.use_relation_transform
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Move graphs to device
    g_train_homog = g_train_homog.to(device)

    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}")

    best_val_mrr = 0
    best_epoch = 0
    training_log = []

    for epoch in range(1, args.epochs + 1):
        # Train
        avg_loss, losses_by_type = train_epoch(model, g_train_homog, train_batches, optimizer, device)

        # Initialize log entry
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_mrr': None,
            'is_best': False
        }

        # Evaluate validation set silently to track best model
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, val_mrr, val_auc = evaluate(model, g_train_homog, val_batches, device)

            # Check if best model
            avg_val_mrr = sum(val_mrr.values()) / len(val_mrr)
            log_entry['val_mrr'] = avg_val_mrr

            if avg_val_mrr > best_val_mrr:
                best_val_mrr = avg_val_mrr
                best_epoch = epoch
                log_entry['is_best'] = True

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mrr': avg_val_mrr,
                }, output_dir / 'best_model.pt')

                print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - Val MRR: {avg_val_mrr:.4f} [BEST]")
            else:
                print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - Val MRR: {avg_val_mrr:.4f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

        training_log.append(log_entry)

    # Write training log to TSV
    training_log_path = output_dir / 'training_log.tsv'
    with open(training_log_path, 'w') as f:
        f.write("epoch\ttrain_loss\tval_mrr\tis_best\n")
        for entry in training_log:
            val_mrr_str = f"{entry['val_mrr']:.4f}" if entry['val_mrr'] is not None else ""
            is_best_str = "TRUE" if entry['is_best'] else ""
            f.write(f"{entry['epoch']}\t{entry['train_loss']:.4f}\t{val_mrr_str}\t{is_best_str}\n")

    print(f"\nTraining log saved to {training_log_path}")

    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*80}")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_mrr, test_auc = evaluate(model, g_train_homog, test_batches, device)

    # Compute overall averages
    avg_acc = sum(test_metrics.values()) / len(test_metrics) if test_metrics else 0
    avg_mrr = sum(test_mrr.values()) / len(test_mrr) if test_mrr else 0
    avg_auc = sum(test_auc.values()) / len(test_auc) if test_auc else 0

    print(f"\nBest model from epoch {checkpoint['epoch']}")
    print(f"Overall Test Metrics:")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  MRR:      {avg_mrr:.4f}")
    print(f"  AUC:      {avg_auc:.4f}")

    # Write detailed metrics to TSV
    edge_metrics_path = output_dir / 'edge_metrics.tsv'
    with open(edge_metrics_path, 'w') as f:
        f.write("source_type\trelation\tdest_type\taccuracy\tmrr\tauc\n")
        for edge_type in sorted(test_metrics.keys(), key=lambda e: test_mrr[e], reverse=True):
            src_type, rel, dst_type = edge_type
            acc = test_metrics[edge_type]
            mrr_val = test_mrr[edge_type]
            auc_val = test_auc[edge_type]
            f.write(f"{src_type}\t{rel}\t{dst_type}\t{acc:.4f}\t{mrr_val:.4f}\t{auc_val:.4f}\n")

    print(f"\nEdge metrics saved to {edge_metrics_path}")

    # Write final test metrics to TSV
    final_test_path = output_dir / 'final_test_metrics.tsv'
    with open(final_test_path, 'w') as f:
        f.write("metric\tvalue\n")
        f.write(f"best_epoch\t{checkpoint['epoch']}\n")
        f.write(f"best_val_mrr\t{checkpoint['val_mrr']:.4f}\n")
        f.write(f"test_accuracy\t{avg_acc:.4f}\n")
        f.write(f"test_mrr\t{avg_mrr:.4f}\n")
        f.write(f"test_auc\t{avg_auc:.4f}\n")

    print(f"Final test metrics saved to {final_test_path}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
