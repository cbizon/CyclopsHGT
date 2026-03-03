"""
Test model prediction methods and all layers functionality.
"""

import pytest
import torch
import dgl
import sys
sys.path.insert(0, 'src')

from model import HGT, LinkPredictor, HGTLinkPredictor


@pytest.fixture
def hetero_graph():
    """Create heterogeneous graph."""
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        ('item', 'belongs_to', 'category'): (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])),
    }
    g = dgl.heterograph(graph_data)

    # Add type information
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

    return g


@pytest.fixture
def homog_graph(hetero_graph):
    """Convert to homogeneous graph."""
    return dgl.to_homogeneous(hetero_graph, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)


@pytest.fixture
def model(hetero_graph):
    """Create model."""
    return HGTLinkPredictor(
        hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    )


def test_predict_method(homog_graph, model):
    """Test HGTLinkPredictor.predict() method."""
    edge_type = ('user', 'likes', 'item')
    src_idx = torch.tensor([0, 1])
    dst_idx = torch.tensor([0, 1])

    scores = model.predict(homog_graph, edge_type, src_idx, dst_idx)

    # Check output
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)
    assert scores.dtype == torch.float32


def test_predict_single_edge(homog_graph, model):
    """Test prediction on single edge."""
    edge_type = ('user', 'likes', 'item')
    src_idx = torch.tensor([0])
    dst_idx = torch.tensor([1])

    scores = model.predict(homog_graph, edge_type, src_idx, dst_idx)

    assert scores.shape == (1,)


def test_predict_multiple_edge_types(homog_graph, model):
    """Test prediction on different edge types."""
    test_cases = [
        (('user', 'likes', 'item'), torch.tensor([0, 1]), torch.tensor([0, 1])),
        (('item', 'belongs_to', 'category'), torch.tensor([0, 1]), torch.tensor([0, 0])),
        (('user', 'follows', 'user'), torch.tensor([0, 1]), torch.tensor([1, 2])),
    ]

    for edge_type, src_idx, dst_idx in test_cases:
        scores = model.predict(homog_graph, edge_type, src_idx, dst_idx)
        assert scores.shape == (len(src_idx),)


def test_predict_consistency(homog_graph, model):
    """Test that predict gives same results for same inputs."""
    model.eval()  # Set to eval mode to disable dropout

    edge_type = ('user', 'likes', 'item')
    src_idx = torch.tensor([0, 1, 2])
    dst_idx = torch.tensor([0, 1, 2])

    scores1 = model.predict(homog_graph, edge_type, src_idx, dst_idx)
    scores2 = model.predict(homog_graph, edge_type, src_idx, dst_idx)

    assert torch.allclose(scores1, scores2)


def test_return_all_layers(hetero_graph, homog_graph):
    """Test HGT forward with return_all_layers=True."""
    hgt = HGT(
        g=hetero_graph,
        n_hidden=16,
        n_layers=3,  # Use 3 layers to test multiple layers
        n_heads=2,
        dropout=0.1
    )

    all_h = hgt(homog_graph, return_all_layers=True)

    # Should return list of embedding dicts (one per layer + initial)
    assert isinstance(all_h, list)
    assert len(all_h) == 4  # Initial embeddings + 3 layers

    # Each element should be a dict with embeddings for each node type
    for h_dict in all_h:
        assert isinstance(h_dict, dict)
        assert set(h_dict.keys()) == set(hetero_graph.ntypes)

        for ntype, emb in h_dict.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.shape == (hetero_graph.num_nodes(ntype), 16)


def test_return_all_layers_single_layer(hetero_graph, homog_graph):
    """Test return_all_layers with single layer."""
    hgt = HGT(
        g=hetero_graph,
        n_hidden=8,
        n_layers=1,
        n_heads=2,
        dropout=0.1
    )

    all_h = hgt(homog_graph, return_all_layers=True)

    # Should return initial embeddings + 1 layer = 2 dicts
    assert len(all_h) == 2


def test_return_all_layers_shapes(hetero_graph, homog_graph):
    """Test that all layers have correct shapes."""
    hgt = HGT(
        g=hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    )

    all_h = hgt(homog_graph, return_all_layers=True)

    # Check shapes for each layer
    for layer_idx, h_dict in enumerate(all_h):
        for ntype in hetero_graph.ntypes:
            expected_shape = (hetero_graph.num_nodes(ntype), 16)
            actual_shape = h_dict[ntype].shape
            assert actual_shape == expected_shape, \
                f"Layer {layer_idx}, {ntype}: expected {expected_shape}, got {actual_shape}"


def test_return_all_layers_evolution(hetero_graph, homog_graph):
    """Test that embeddings actually change across layers."""
    hgt = HGT(
        g=hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.0  # No dropout for deterministic behavior
    )
    hgt.eval()

    all_h = hgt(homog_graph, return_all_layers=True)

    # Check that embeddings change from layer to layer
    for ntype in hetero_graph.ntypes:
        layer0_emb = all_h[0][ntype]
        layer1_emb = all_h[1][ntype]
        layer2_emb = all_h[2][ntype]

        # Embeddings should be different across layers
        assert not torch.allclose(layer0_emb, layer1_emb), \
            f"Layer 0 and 1 embeddings should differ for {ntype}"
        assert not torch.allclose(layer1_emb, layer2_emb), \
            f"Layer 1 and 2 embeddings should differ for {ntype}"


def test_return_all_layers_default_false(hetero_graph, homog_graph):
    """Test that return_all_layers defaults to False."""
    hgt = HGT(
        g=hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    )

    # Call without return_all_layers parameter
    h = hgt(homog_graph)

    # Should return dict, not list
    assert isinstance(h, dict)
    assert not isinstance(h, list)


# GPU tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_predict_gpu(homog_graph, hetero_graph):
    """Test predict method on GPU."""
    device = torch.device('cuda')

    model = HGTLinkPredictor(
        hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    ).to(device)

    homog_graph_gpu = homog_graph.to(device)

    edge_type = ('user', 'likes', 'item')
    src_idx = torch.tensor([0, 1], device=device)
    dst_idx = torch.tensor([0, 1], device=device)

    scores = model.predict(homog_graph_gpu, edge_type, src_idx, dst_idx)

    assert scores.is_cuda
    assert scores.shape == (2,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_return_all_layers_gpu(hetero_graph, homog_graph):
    """Test return_all_layers on GPU."""
    device = torch.device('cuda')

    hgt = HGT(
        g=hetero_graph,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    ).to(device)

    homog_graph_gpu = homog_graph.to(device)

    all_h = hgt(homog_graph_gpu, return_all_layers=True)

    # Check all embeddings are on GPU
    for h_dict in all_h:
        for ntype, emb in h_dict.items():
            assert emb.is_cuda, f"Embeddings for {ntype} should be on GPU"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
