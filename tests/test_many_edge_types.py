"""
Test that model handles graphs with many edge types correctly.

This reproduces the CUDA bug where the model is initialized with a homogeneous
graph (1 edge type) but then receives edge type IDs from the original heterogeneous
graph (1845 edge types).
"""

import pytest
import torch
import dgl
import sys
sys.path.insert(0, 'src')

from model import HGTLinkPredictor


def test_model_with_many_edge_types():
    """
    Test that model can handle graphs with many edge types.

    This test reproduces the bug:
    1. Create heterogeneous graph with many edge types
    2. Convert to homogeneous
    3. Initialize model with homogeneous graph (BUG - should use hetero)
    4. Try to run forward pass - should fail because HGTConv expects 1 edge type
       but receives edge type IDs up to 99
    """
    # Create a heterogeneous graph with 100 edge types (similar to real data with 1845)
    graph_data = {}

    # Create 10 node types
    for i in range(10):
        # Create 10 edge types between each pair of node types
        for j in range(10):
            edge_type = (f'node_type_{i}', f'rel_{i}_{j}', f'node_type_{j}')
            # Create a few edges for each type
            src = torch.arange(5) + i * 10
            dst = torch.arange(5) + j * 10
            graph_data[edge_type] = (src, dst)

    g_hetero = dgl.heterograph(graph_data)

    print(f"Created heterogeneous graph with {len(g_hetero.canonical_etypes)} edge types")
    assert len(g_hetero.canonical_etypes) == 100

    # Add type information for conversion
    ntype_to_id = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
    etype_to_id = {etype: i for i, etype in enumerate(g_hetero.canonical_etypes)}

    g_hetero.ndata['_TYPE'] = {
        ntype: torch.full((g_hetero.num_nodes(ntype),), ntype_to_id[ntype], dtype=torch.long)
        for ntype in g_hetero.ntypes
    }
    g_hetero.edata['_TYPE'] = {
        etype: torch.full((g_hetero.num_edges(etype),), etype_to_id[etype], dtype=torch.long)
        for etype in g_hetero.canonical_etypes
    }

    # Convert to homogeneous
    g_homogeneous = dgl.to_homogeneous(g_hetero, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)

    print(f"Converted to homogeneous: {g_homogeneous.num_nodes()} nodes, {g_homogeneous.num_edges()} edges")
    print(f"Homogeneous graph has {len(g_homogeneous.canonical_etypes)} edge type(s)")
    print(f"But edge type IDs in edata['_TYPE'] range from {g_homogeneous.edata['_TYPE'].min()} to {g_homogeneous.edata['_TYPE'].max()}")

    # BUG: Initialize model with homogeneous graph
    # This will create HGTConv with num_etypes=1 (since homogeneous graph has 1 canonical edge type)
    model_wrong = HGTLinkPredictor(
        g_homogeneous,  # BUG: Should pass g_hetero instead
        n_hidden=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0
    )

    # Try to run forward pass - this should FAIL
    # Because HGTConv will try to access relation_pri[0][99] but it only has relation_pri[0][0]
    with pytest.raises((RuntimeError, IndexError)) as exc_info:
        h = model_wrong(g_homogeneous)

    print(f"Expected failure occurred: {exc_info.value}")
    assert "index" in str(exc_info.value).lower() or "out of" in str(exc_info.value).lower()


def test_model_with_many_edge_types_correct():
    """
    Test the CORRECT way: initialize model with heterogeneous graph.
    """
    # Create same graph as above
    graph_data = {}
    for i in range(10):
        for j in range(10):
            edge_type = (f'node_type_{i}', f'rel_{i}_{j}', f'node_type_{j}')
            src = torch.arange(5) + i * 10
            dst = torch.arange(5) + j * 10
            graph_data[edge_type] = (src, dst)

    g_hetero = dgl.heterograph(graph_data)

    # Add type information
    ntype_to_id = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
    etype_to_id = {etype: i for i, etype in enumerate(g_hetero.canonical_etypes)}

    g_hetero.ndata['_TYPE'] = {
        ntype: torch.full((g_hetero.num_nodes(ntype),), ntype_to_id[ntype], dtype=torch.long)
        for ntype in g_hetero.ntypes
    }
    g_hetero.edata['_TYPE'] = {
        etype: torch.full((g_hetero.num_edges(etype),), etype_to_id[etype], dtype=torch.long)
        for etype in g_hetero.canonical_etypes
    }

    # Convert to homogeneous
    g_homogeneous = dgl.to_homogeneous(g_hetero, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)

    # CORRECT: Initialize model with heterogeneous graph to get schema
    model_correct = HGTLinkPredictor(
        g_hetero,  # CORRECT: Pass heterogeneous graph for schema
        n_hidden=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0
    )

    # This should work - model knows about all 100 edge types
    h = model_correct(g_homogeneous)

    # Check output
    assert isinstance(h, dict)
    assert len(h) == len(g_hetero.ntypes)

    print("✓ Model initialized correctly with heterogeneous graph works!")


# NOTE: GPU version of bug test removed because even though pytest.raises()
# catches the Python exception, the GPU hardware retains error state and breaks
# all subsequent GPU tests in the same pytest session. The CPU version is
# sufficient to verify the bug.


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
