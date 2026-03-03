"""
Stratified train/val/test splitting and negative sampling for link prediction.
"""

import dgl
import torch
import numpy as np
import time
from typing import Dict, Tuple, List
from collections import defaultdict


def stratified_edge_split(
    g: dgl.DGLGraph,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split edges into train/val/test sets with stratification by edge type.

    Args:
        g: DGL heterogeneous graph
        train_ratio: Proportion of edges for training
        val_ratio: Proportion of edges for validation
        test_ratio: Proportion of edges for testing
        seed: Random seed

    Returns:
        train_edges: Dictionary mapping edge types to edge indices
        val_edges: Dictionary mapping edge types to edge indices
        test_edges: Dictionary mapping edge types to edge indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    np.random.seed(seed)

    train_edges = {}
    val_edges = {}
    test_edges = {}

    print("\nSplitting edges by type:")

    for etype in g.canonical_etypes:
        num_edges = g.num_edges(etype)

        # Shuffle indices
        indices = np.random.permutation(num_edges)

        # For edge types with very few edges, ensure at least 1 goes to training
        if num_edges < 10:
            # Put all edges in training if too few to split meaningfully
            train_idx = indices
            val_idx = np.array([], dtype=np.int64)
            test_idx = np.array([], dtype=np.int64)
        else:
            # Calculate split points
            train_end = int(num_edges * train_ratio)
            val_end = train_end + int(num_edges * val_ratio)

            # Ensure at least 1 edge in training
            train_end = max(1, train_end)

            # Split indices
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

        train_edges[etype] = torch.from_numpy(train_idx)
        val_edges[etype] = torch.from_numpy(val_idx)
        test_edges[etype] = torch.from_numpy(test_idx)

        src_type, rel_type, dst_type = etype
        print(f"  {src_type} --[{rel_type}]--> {dst_type}:")
        print(f"    Total: {num_edges:,} | Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    return train_edges, val_edges, test_edges


def create_train_graph(g: dgl.DGLGraph, train_edges: Dict) -> dgl.DGLGraph:
    """
    Create a new graph containing only training edges.

    Args:
        g: Original graph
        train_edges: Dictionary of training edge indices per edge type

    Returns:
        Training graph with only training edges
    """
    edge_dict = {}

    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        train_idx = train_edges[etype]

        train_src = src[train_idx]
        train_dst = dst[train_idx]

        edge_dict[etype] = (train_src, train_dst)

    # Create new graph with same node types but only train edges
    g_train = dgl.heterograph(edge_dict, num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})

    return g_train


def sample_negative_edges_type_constrained(
    g: dgl.DGLGraph,
    edge_type: Tuple[str, str, str],
    num_samples: int,
    exclude_edges: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample negative edges that respect node type constraints.

    For edge type (src_type, rel_type, dst_type), negative samples are
    (src, dst') pairs where:
    - src is from an actual edge of this type
    - dst' is a random node of dst_type
    - (src, dst') does not exist in the graph

    Args:
        g: DGL heterogeneous graph
        edge_type: Canonical edge type (src_type, rel_type, dst_type)
        num_samples: Number of negative samples to generate
        exclude_edges: Edge indices to exclude (e.g., positive edges)

    Returns:
        neg_src: Source node indices
        neg_dst: Destination node indices
    """
    src_type, rel_type, dst_type = edge_type

    t0 = time.time()

    # Get positive edges
    pos_src, pos_dst = g.edges(etype=edge_type)

    if exclude_edges is not None:
        pos_src = pos_src[exclude_edges]
        pos_dst = pos_dst[exclude_edges]

    t1 = time.time()
    print(f"    - Get edges: {t1-t0:.2f}s")

    # Create set of positive edges for fast lookup
    pos_edges_set = set(zip(pos_src.tolist(), pos_dst.tolist()))

    t2 = time.time()
    print(f"    - Create edge set ({len(pos_edges_set):,} edges): {t2-t1:.2f}s")

    # Get total number of nodes of destination type
    num_dst_nodes = g.num_nodes(dst_type)

    neg_src_list = []
    neg_dst_list = []

    # Sample negatives by corrupting destination nodes
    # Strategy: for each positive edge, sample k negative destinations
    samples_per_edge = max(1, num_samples // len(pos_src))

    print(f"    - Main loop: sampling {samples_per_edge} negatives per source ({len(pos_src):,} sources)...")

    for src in pos_src:
        src_item = src.item()
        attempts = 0
        max_attempts = samples_per_edge * 10  # Prevent infinite loops

        while len(neg_dst_list) < num_samples and attempts < max_attempts:
            # Sample random destination node
            dst_neg = np.random.randint(0, num_dst_nodes)

            # Check if this is not a positive edge
            if (src_item, dst_neg) not in pos_edges_set:
                neg_src_list.append(src_item)
                neg_dst_list.append(dst_neg)

                if len(neg_dst_list) >= num_samples:
                    break

            attempts += 1

    t3 = time.time()
    print(f"    - Main loop completed: {t3-t2:.2f}s (generated {len(neg_dst_list):,}/{num_samples:,})")

    # If we didn't get enough samples, sample more randomly
    if len(neg_dst_list) < num_samples:
        remaining = num_samples - len(neg_dst_list)
        print(f"    - Fallback loop: need {remaining:,} more samples...")
        pos_src_list = pos_src.tolist()  # Convert once, not in every iteration
        while len(neg_dst_list) < num_samples:
            src = np.random.choice(pos_src_list)
            dst = np.random.randint(0, num_dst_nodes)

            if (src, dst) not in pos_edges_set:
                neg_src_list.append(src)
                neg_dst_list.append(dst)

        t4 = time.time()
        print(f"    - Fallback loop completed: {t4-t3:.2f}s")

    return torch.tensor(neg_src_list[:num_samples]), torch.tensor(neg_dst_list[:num_samples])


def prepare_edge_batches(
    g: dgl.DGLGraph,
    edge_dict: Dict,
    negative_ratio: int = 1
) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Prepare positive and negative edge batches for each edge type.

    Args:
        g: DGL graph
        edge_dict: Dictionary mapping edge types to edge indices
        negative_ratio: Number of negative samples per positive edge

    Returns:
        Dictionary mapping edge types to (pos_src, pos_dst, neg_src, neg_dst)
    """
    batches = {}

    total_etypes = len(edge_dict)
    print(f"\nProcessing {total_etypes} edge types...")

    for idx, (etype, edge_idx) in enumerate(edge_dict.items(), 1):
        # Skip edge types with no edges
        if len(edge_idx) == 0:
            continue

        src_type, rel_type, dst_type = etype
        num_edges = len(edge_idx)
        num_neg = num_edges * negative_ratio

        print(f"\n[{idx}/{total_etypes}] {src_type} --[{rel_type}]--> {dst_type}")
        print(f"  Positive edges: {num_edges:,} | Negative samples needed: {num_neg:,}")

        # Check if negative sampling is feasible
        # Get the actual training edges to determine unique sources
        src, dst = g.edges(etype=etype)
        train_src = src[edge_idx]
        train_dst = dst[edge_idx]

        # Ns = unique source nodes in training edges (we only sample from these)
        # Nd = all possible destination nodes (we sample from all of these)
        # Nr = number of training edges
        num_unique_train_sources = len(torch.unique(train_src))
        num_dst_nodes = g.num_nodes(dst_type)
        num_real_edges = g.num_edges(etype)  # Total real edges of this type in graph

        # Calculate available negatives vs needed negatives
        # Available = (Ns × Nd) - Nr
        # Needed = Nr × f (where f is negative_ratio)
        # We want: available >> needed
        # Check: (Nr × f) / (Ns × Nd - Nr) should be << 1

        max_possible_edges = num_unique_train_sources * num_dst_nodes
        available_negatives = max_possible_edges - num_edges
        needed_negatives = num_edges * negative_ratio

        # If no negatives available, skip immediately
        if available_negatives <= 0:
            print(f"  ⚠ SKIPPING: No negative edges available")
            print(f"     All {max_possible_edges:,} possible edges exist - keeping for training only")
            continue

        # Check ratio of needed to available
        ratio = needed_negatives / available_negatives
        threshold = 0.05  # Need at least 20x more available than needed

        if ratio > threshold:
            print(f"  ⚠ SKIPPING: Insufficient negatives available (ratio={ratio:.4f} > {threshold})")
            print(f"     Need {needed_negatives:,} negatives, only {available_negatives:,} available - keeping for training only")
            continue

        start_time = time.time()

        # Get positive edges
        src, dst = g.edges(etype=etype)
        pos_src = src[edge_idx]
        pos_dst = dst[edge_idx]

        # Sample negative edges
        neg_src, neg_dst = sample_negative_edges_type_constrained(
            g, etype, num_neg, exclude_edges=edge_idx
        )

        batches[etype] = (pos_src, pos_dst, neg_src, neg_dst)

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.2f}s")

    return batches


if __name__ == "__main__":
    # Test with a simple synthetic graph
    print("Testing stratified split with synthetic graph...")

    # Create a simple heterogeneous graph
    graph_data = {
        ('drug', 'treats', 'disease'): (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 0, 2, 1])),
        ('gene', 'associated_with', 'disease'): (torch.tensor([0, 1, 2, 3, 4, 5]), torch.tensor([0, 0, 1, 1, 2, 2])),
    }

    g = dgl.heterograph(graph_data)

    print(f"\nOriginal graph:")
    print(f"  Nodes: {dict(zip(g.ntypes, [g.num_nodes(nt) for nt in g.ntypes]))}")
    print(f"  Edges: {dict(zip(g.canonical_etypes, [g.num_edges(et) for et in g.canonical_etypes]))}")

    # Test split
    train_edges, val_edges, test_edges = stratified_edge_split(g)

    # Test negative sampling
    print("\nTesting negative sampling:")
    for etype in g.canonical_etypes:
        neg_src, neg_dst = sample_negative_edges_type_constrained(
            g, etype, num_samples=5, exclude_edges=train_edges[etype]
        )
        print(f"  {etype}: Generated {len(neg_src)} negative samples")
