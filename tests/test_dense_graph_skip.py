"""Test that edge types with insufficient negatives are skipped during negative sampling."""

import pytest
import dgl
import torch
import sys
sys.path.insert(0, 'src')

from data_split import prepare_edge_batches


def test_single_destination_node_skipped():
    """
    Test the real problem case: SmallMolecule -> Device with 1 destination node.

    Real data: 150K SmallMolecule nodes, 1 Device node, 1 edge
    Available negatives = (150K × 1) - 1 = 150K - 1
    Needed negatives = 1
    Ratio = 1 / (150K - 1) ≈ 0.0000067 < 0.01 → should be PROCESSED

    BUT if we simulate just 1 source and 1 destination (minimal graph):
    Available = (1 × 1) - 1 = 0
    Ratio = undefined (division by zero) → should be SKIPPED
    """
    # Minimal case: 1 source, 1 destination, 1 edge = no negatives available
    graph_data = {
        ('SmallMolecule', 'contraindicated_in', 'Device'): (torch.tensor([0]), torch.tensor([0])),
    }

    g = dgl.heterograph(graph_data)

    edge_dict = {
        ('SmallMolecule', 'contraindicated_in', 'Device'): torch.tensor([0])
    }

    batches = prepare_edge_batches(g, edge_dict, negative_ratio=1)

    # Should be skipped because available_negatives = 0
    assert ('SmallMolecule', 'contraindicated_in', 'Device') not in batches, \
        "Edge type with no available negatives should be skipped"


def test_real_world_device_case():
    """
    Test approximation of real case: many sources exist, but only 1 has an edge, 1 destination, 1 edge.

    Ns = unique sources in training = 1 (only 1 SmallMolecule has this edge)
    Nd = all destinations = 1
    Nr = training edges = 1
    Available = (1 × 1) - 1 = 0
    Ratio = undefined → SKIP
    """
    graph_data = {
        ('SmallMolecule', 'contraindicated_in', 'Device'): (torch.tensor([0]), torch.tensor([0])),
    }

    # Even with 150K total SmallMolecules, only 1 has the edge
    g = dgl.heterograph(graph_data, num_nodes_dict={'SmallMolecule': 150000, 'Device': 1})

    edge_dict = {
        ('SmallMolecule', 'contraindicated_in', 'Device'): torch.tensor([0])
    }

    batches = prepare_edge_batches(g, edge_dict, negative_ratio=1)

    # Should be SKIPPED because available_negatives = (1×1) - 1 = 0
    assert ('SmallMolecule', 'contraindicated_in', 'Device') not in batches, \
        "Edge type with no available negatives should be skipped"


def test_high_ratio_skipped():
    """
    Test that high ratio of needed/available causes skip.

    10 sources, 10 destinations, 99 edges, need 99 negatives
    Available = (10 × 10) - 99 = 1
    Ratio = 99 / 1 = 99 > 0.01 → SKIP
    """
    # Create 99 edges out of 100 possible
    src_nodes = []
    dst_nodes = []
    for i in range(10):
        for j in range(10):
            if i == 0 and j == 0:
                continue  # Skip one edge to leave 1 available
            src_nodes.append(i)
            dst_nodes.append(j)

    graph_data = {
        ('A', 'rel', 'B'): (torch.tensor(src_nodes), torch.tensor(dst_nodes)),
    }

    g = dgl.heterograph(graph_data, num_nodes_dict={'A': 10, 'B': 10})

    edge_dict = {
        ('A', 'rel', 'B'): torch.arange(99)  # All 99 edges in training
    }

    batches = prepare_edge_batches(g, edge_dict, negative_ratio=1)

    # Should be skipped due to high ratio
    assert ('A', 'rel', 'B') not in batches, \
        "Edge type with insufficient negatives should be skipped"


def test_sparse_graph_processed():
    """Test that sparse graphs are processed normally."""
    # 3 edges out of 10,000 possible
    # Available = 10,000 - 3 = 9,997
    # Needed = 3
    # Ratio = 3 / 9,997 ≈ 0.0003 < 0.01 → PROCESSED
    graph_data = {
        ('Drug', 'treats', 'Disease'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
    }

    g = dgl.heterograph(graph_data, num_nodes_dict={'Drug': 100, 'Disease': 100})

    edge_dict = {
        ('Drug', 'treats', 'Disease'): torch.tensor([0, 1, 2])
    }

    batches = prepare_edge_batches(g, edge_dict, negative_ratio=1)

    # Should be processed
    assert ('Drug', 'treats', 'Disease') in batches, \
        "Sparse edge type should be processed"

    pos_src, pos_dst, neg_src, neg_dst = batches[('Drug', 'treats', 'Disease')]

    assert len(pos_src) == 3, "Should have 3 positive edges"
    assert len(neg_src) == 3, "Should have 3 negative samples"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
