"""
Load heterogeneous graph from TSV files into DGL format.

Expected input format:
- nodes.tsv: id, type
- edges.tsv: subject, predicate, object
"""

import pandas as pd
import dgl
import torch
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def load_graph_from_tsv(graph_dir: Path) -> Tuple[dgl.DGLGraph, Dict]:
    """
    Load heterogeneous graph from TSV files.

    Args:
        graph_dir: Directory containing nodes.tsv and edges.tsv

    Returns:
        graph: DGL heterogeneous graph
        metadata: Dictionary containing node/edge type mappings and statistics
    """
    nodes_file = graph_dir / "nodes.tsv"
    edges_file = graph_dir / "edges.tsv"

    # Load data
    print(f"Loading nodes from {nodes_file}")
    nodes_df = pd.read_csv(nodes_file, sep='\t')
    print(f"Loading edges from {edges_file}")
    edges_df = pd.read_csv(edges_file, sep='\t')

    # Create node ID to integer mapping
    node_id_to_int = {nid: i for i, nid in enumerate(nodes_df['id'])}
    int_to_node_id = {i: nid for nid, i in node_id_to_int.items()}

    # Group nodes by type
    nodes_by_type = nodes_df.groupby('type')['id'].apply(list).to_dict()

    # Create type-specific node mappings
    node_type_to_ids = {}
    node_id_to_type_idx = {}

    for ntype, node_ids in nodes_by_type.items():
        type_specific_ids = {nid: i for i, nid in enumerate(node_ids)}
        node_type_to_ids[ntype] = type_specific_ids
        for nid in node_ids:
            node_id_to_type_idx[nid] = (ntype, type_specific_ids[nid])

    # Build heterogeneous graph
    graph_data = defaultdict(lambda: ([], []))

    for _, row in edges_df.iterrows():
        src_id = row['subject']
        dst_id = row['object']
        rel_type = row['predicate']

        # Get node types and type-specific indices
        src_type, src_idx = node_id_to_type_idx[src_id]
        dst_type, dst_idx = node_id_to_type_idx[dst_id]

        # Canonical edge type: (src_type, rel_type, dst_type)
        canonical_etype = (src_type, rel_type, dst_type)

        graph_data[canonical_etype][0].append(src_idx)
        graph_data[canonical_etype][1].append(dst_idx)

    # Convert to DGL format
    dgl_graph_data = {
        etype: (torch.tensor(srcs), torch.tensor(dsts))
        for etype, (srcs, dsts) in graph_data.items()
    }

    # Create heterograph
    g = dgl.heterograph(dgl_graph_data)

    # Compute statistics
    node_type_counts = {ntype: len(ids) for ntype, ids in nodes_by_type.items()}
    edge_type_counts = edges_df['predicate'].value_counts().to_dict()
    canonical_edge_counts = {
        etype: len(srcs) for etype, (srcs, _) in graph_data.items()
    }

    metadata = {
        'node_id_to_int': node_id_to_int,
        'int_to_node_id': int_to_node_id,
        'node_type_to_ids': node_type_to_ids,
        'node_id_to_type_idx': node_id_to_type_idx,
        'node_type_counts': node_type_counts,
        'edge_type_counts': edge_type_counts,
        'canonical_edge_counts': canonical_edge_counts,
    }

    print(f"\nGraph loaded:")
    print(f"  Node types: {len(node_type_counts)}")
    print(f"  Total nodes: {sum(node_type_counts.values())}")
    print(f"  Edge types (predicates): {len(edge_type_counts)}")
    print(f"  Canonical edge types: {len(canonical_edge_counts)}")
    print(f"  Total edges: {sum(canonical_edge_counts.values())}")

    print(f"\nNode type distribution:")
    for ntype, count in sorted(node_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ntype}: {count:,}")

    print(f"\nTop edge types (by predicate):")
    for pred, count in sorted(edge_type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pred}: {count:,}")

    return g, metadata


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run python src/graph_loader.py <graph_dir>")
        sys.exit(1)

    graph_dir = Path(sys.argv[1])
    g, metadata = load_graph_from_tsv(graph_dir)

    print(f"\nDGL Graph structure:")
    print(g)
