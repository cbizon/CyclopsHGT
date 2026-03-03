"""
Test data splitting and negative sampling functions.
"""

import pytest
import torch
import dgl
import numpy as np
import sys
sys.path.insert(0, 'src')

from data_split import (
    stratified_edge_split,
    sample_negative_edges_type_constrained,
    prepare_edge_batches,
    create_train_graph
)


@pytest.fixture
def simple_hetero_graph():
    """Create a simple heterogeneous graph for testing."""
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                     torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        ('item', 'belongs_to', 'category'): (torch.tensor([0, 1, 2, 3, 4, 5]),
                                              torch.tensor([0, 0, 1, 1, 2, 2])),
    }
    g = dgl.heterograph(graph_data)
    return g


@pytest.fixture
def graph_with_few_edges():
    """Create graph with edge types having very few edges."""
    graph_data = {
        ('a', 'r1', 'b'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),  # 3 edges
        ('b', 'r2', 'c'): (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),  # 8 edges
        ('c', 'r3', 'd'): (torch.tensor([0]), torch.tensor([0])),  # 1 edge
    }
    g = dgl.heterograph(graph_data)
    return g


def test_stratified_edge_split_basic(simple_hetero_graph):
    """Test basic stratified edge split."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    # Check that dicts are returned
    assert isinstance(train_edges, dict)
    assert isinstance(val_edges, dict)
    assert isinstance(test_edges, dict)

    # Check all edge types are present
    for etype in simple_hetero_graph.canonical_etypes:
        assert etype in train_edges
        assert etype in val_edges
        assert etype in test_edges


def test_stratified_edge_split_ratios(simple_hetero_graph):
    """Test that split ratios are approximately correct."""
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    for etype in simple_hetero_graph.canonical_etypes:
        num_edges = simple_hetero_graph.num_edges(etype)

        # For edge types with enough edges, check ratios
        if num_edges >= 10:
            n_train = len(train_edges[etype])
            n_val = len(val_edges[etype])
            n_test = len(test_edges[etype])

            total = n_train + n_val + n_test
            assert total == num_edges

            # Check approximate ratios (within 15% tolerance due to rounding)
            assert abs(n_val / num_edges - val_ratio) < 0.15
            assert abs(n_test / num_edges - test_ratio) < 0.15


def test_stratified_edge_split_few_edges(graph_with_few_edges):
    """Test split with edge types having very few edges."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        graph_with_few_edges,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    # Edge type with 1 edge should have at least 1 edge in training
    etype_1_edge = ('c', 'r3', 'd')
    assert len(train_edges[etype_1_edge]) >= 1


def test_stratified_edge_split_no_overlap(simple_hetero_graph):
    """Test that train/val/test splits have no overlap."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    for etype in simple_hetero_graph.canonical_etypes:
        train_set = set(train_edges[etype].tolist())
        val_set = set(val_edges[etype].tolist())
        test_set = set(test_edges[etype].tolist())

        # Check no overlap
        assert len(train_set & val_set) == 0, f"Train and val overlap for {etype}"
        assert len(train_set & test_set) == 0, f"Train and test overlap for {etype}"
        assert len(val_set & test_set) == 0, f"Val and test overlap for {etype}"


def test_create_train_graph(simple_hetero_graph):
    """Test creating training graph from edge indices."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    g_train = create_train_graph(simple_hetero_graph, train_edges)

    # Should be a graph
    assert isinstance(g_train, dgl.DGLGraph)

    # Should have same node types
    assert set(g_train.ntypes) == set(simple_hetero_graph.ntypes)

    # Should have same edge types
    assert set(g_train.canonical_etypes) == set(simple_hetero_graph.canonical_etypes)

    # Number of edges should match train splits
    for etype in simple_hetero_graph.canonical_etypes:
        assert g_train.num_edges(etype) == len(train_edges[etype])


def test_sample_negative_edges_basic(simple_hetero_graph):
    """Test basic negative edge sampling."""
    edge_type = ('user', 'likes', 'item')

    neg_src, neg_dst = sample_negative_edges_type_constrained(
        simple_hetero_graph,
        edge_type,
        num_samples=20
    )

    # Should return requested number of samples
    assert len(neg_src) == 20
    assert len(neg_dst) == 20

    # Destinations should be valid node IDs
    num_dst_nodes = simple_hetero_graph.num_nodes(edge_type[2])
    assert all(0 <= d.item() < num_dst_nodes for d in neg_dst)


def test_sample_negative_edges_exclude(simple_hetero_graph):
    """Test negative sampling with edge exclusion."""
    edge_type = ('user', 'likes', 'item')

    # Exclude first 3 edges
    exclude_indices = torch.tensor([0, 1, 2])

    neg_src, neg_dst = sample_negative_edges_type_constrained(
        simple_hetero_graph,
        edge_type,
        num_samples=10,
        exclude_edges=exclude_indices
    )

    assert len(neg_src) == 10
    assert len(neg_dst) == 10


def test_sample_negative_edges_different_types(simple_hetero_graph):
    """Test negative sampling on different edge types."""
    test_cases = [
        ('user', 'likes', 'item'),
        ('item', 'belongs_to', 'category'),
        ('user', 'follows', 'user'),
    ]

    for edge_type in test_cases:
        neg_src, neg_dst = sample_negative_edges_type_constrained(
            simple_hetero_graph,
            edge_type,
            num_samples=5
        )

        assert len(neg_src) == 5
        assert len(neg_dst) == 5

        # Check destinations are valid for this edge type
        dst_type = edge_type[2]
        num_dst = simple_hetero_graph.num_nodes(dst_type)
        assert all(0 <= d.item() < num_dst for d in neg_dst)


def test_prepare_edge_batches_basic(simple_hetero_graph):
    """Test basic batch preparation."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    batches = prepare_edge_batches(
        simple_hetero_graph,  # Pass original graph, not training graph
        train_edges,
        negative_ratio=2
    )

    # Should return dict of batches
    assert isinstance(batches, dict)

    # Check batch structure
    for edge_type, (pos_src, pos_dst, neg_src, neg_dst) in batches.items():
        assert isinstance(pos_src, torch.Tensor)
        assert isinstance(pos_dst, torch.Tensor)
        assert isinstance(neg_src, torch.Tensor)
        assert isinstance(neg_dst, torch.Tensor)

        # Negative samples should be 2x positive samples
        if len(pos_src) > 0:
            assert len(neg_src) == len(pos_src) * 2
            assert len(neg_dst) == len(pos_dst) * 2


def test_prepare_edge_batches_skip_dense():
    """Test that dense edge types are skipped."""
    # Create a very dense edge type
    num_src = 10
    num_dst = 10
    # Create 95 out of 100 possible edges (very dense)
    edges = [(i, j) for i in range(num_src) for j in range(num_dst)][:95]
    src_nodes = torch.tensor([e[0] for e in edges])
    dst_nodes = torch.tensor([e[1] for e in edges])

    dense_graph_data = {
        ('a', 'dense', 'b'): (src_nodes, dst_nodes),
        ('b', 'sparse', 'c'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
    }
    dense_graph = dgl.heterograph(dense_graph_data)

    # Use all edges as training
    train_edges = {
        ('a', 'dense', 'b'): torch.arange(95),
        ('b', 'sparse', 'c'): torch.arange(3),
    }

    batches = prepare_edge_batches(
        dense_graph,
        train_edges,
        negative_ratio=2
    )

    # Dense edge type should be skipped (ratio > 0.05 threshold)
    assert ('a', 'dense', 'b') not in batches or len(batches[('a', 'dense', 'b')][0]) == 0


def test_prepare_edge_batches_multiple_edge_types():
    """Test batch preparation with multiple edge types."""
    # Create a larger, sparser graph to avoid density skip
    graph_data = {
        ('user', 'follows', 'user'): (torch.arange(20), torch.arange(20, 40) % 50),
        ('user', 'likes', 'item'): (torch.arange(30), torch.arange(0, 30)),
        ('item', 'belongs_to', 'category'): (torch.arange(15), torch.arange(0, 15) % 10),
    }
    g = dgl.heterograph(graph_data)

    train_edges, _, _ = stratified_edge_split(
        g,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    batches = prepare_edge_batches(
        g,
        train_edges,
        negative_ratio=2  # Use 2 instead of 3 to reduce density issues
    )

    # Should have at least one edge type with batches
    # (some may be skipped due to density)
    assert isinstance(batches, dict)

    # Check negative ratio for batches that were not skipped
    for edge_type, (pos_src, pos_dst, neg_src, neg_dst) in batches.items():
        if len(pos_src) > 0:
            assert len(neg_src) == len(pos_src) * 2


def test_prepare_edge_batches_empty_edge_type(graph_with_few_edges):
    """Test batch preparation when some edge types have no edges."""
    train_edges = {
        ('a', 'r1', 'b'): torch.arange(3),
        ('b', 'r2', 'c'): torch.arange(8),
        ('c', 'r3', 'd'): torch.tensor([], dtype=torch.long),  # No training edges
    }

    batches = prepare_edge_batches(
        graph_with_few_edges,  # Pass original graph
        train_edges,
        negative_ratio=2
    )

    # Edge type with no training edges should not be in batches
    assert ('c', 'r3', 'd') not in batches


def test_stratified_split_preserves_all_edges(simple_hetero_graph):
    """Test that all edges are preserved across splits."""
    train_edges, val_edges, test_edges = stratified_edge_split(
        simple_hetero_graph,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    for etype in simple_hetero_graph.canonical_etypes:
        total_split = len(train_edges[etype]) + len(val_edges[etype]) + len(test_edges[etype])
        total_orig = simple_hetero_graph.num_edges(etype)
        assert total_split == total_orig, f"{etype}: split={total_split}, orig={total_orig}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
