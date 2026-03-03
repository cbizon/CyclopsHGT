"""
Test HGT model with homogeneous graph conversion.
"""

import pytest
import torch
import dgl
import sys
sys.path.insert(0, 'src')

from model import HGT, LinkPredictor, HGTLinkPredictor


@pytest.fixture
def small_hetero_graph():
    """Create a small heterogeneous graph for testing."""
    graph_data = {
        ('user', 'follows', 'user'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
        ('user', 'likes', 'item'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        ('item', 'belongs_to', 'category'): (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])),
    }
    g = dgl.heterograph(graph_data)
    return g


@pytest.fixture
def small_homogg_graph(small_hetero_graph):
    """Convert to homogeneous graph."""
    g = small_hetero_graph

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

    # Convert to homogeneous
    g_homogg = dgl.to_homogeneous(g, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)

    return g_homogg, g


def test_homoggeneous_conversion(small_homogg_graph):
    """Test that homogeneous conversion creates expected structure."""
    g_homogg, g_hetero = small_homogg_graph

    # Check node counts
    total_nodes = sum(g_hetero.num_nodes(nt) for nt in g_hetero.ntypes)
    assert g_homogg.num_nodes() == total_nodes, f"Expected {total_nodes} nodes, got {g_homogg.num_nodes()}"

    # Check edge counts
    total_edges = sum(g_hetero.num_edges(et) for et in g_hetero.canonical_etypes)
    assert g_homogg.num_edges() == total_edges, f"Expected {total_edges} edges, got {g_homogg.num_edges()}"

    # Check that type information exists
    assert dgl.NTYPE in g_homogg.ndata, "Missing dgl.NTYPE in homogeneous graph"
    assert dgl.ETYPE in g_homogg.edata, "Missing dgl.ETYPE in homogeneous graph"

    # Check type ranges
    ntype_values = g_homogg.ndata[dgl.NTYPE]
    etype_values = g_homogg.edata[dgl.ETYPE]

    print(f"Node type range: {ntype_values.min()}-{ntype_values.max()}")
    print(f"Edge type range: {etype_values.min()}-{etype_values.max()}")
    print(f"Num node types: {len(g_hetero.ntypes)}")
    print(f"Num edge types: {len(g_hetero.canonical_etypes)}")

    assert ntype_values.min() >= 0, "Node types should be >= 0"
    assert ntype_values.max() < len(g_hetero.ntypes), f"Node types should be < {len(g_hetero.ntypes)}"

    assert etype_values.min() >= 0, "Edge types should be >= 0"
    assert etype_values.max() < len(g_hetero.canonical_etypes), f"Edge types should be < {len(g_hetero.canonical_etypes)}"


def test_hgt_initialization(small_homogg_graph):
    """Test HGT model initialization."""
    g_homogg, g_hetero = small_homogg_graph

    # Initialize HGT
    hgt = HGT(
        g=g_hetero,  # Pass hetero graph for initialization (to get node/edge types)
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    )

    # Check model structure
    assert len(hgt.layers) == 2, "Should have 2 HGT layers"
    assert len(hgt.node_types) == len(g_hetero.ntypes), "Should track all node types"
    assert len(hgt.edge_types) == len(g_hetero.canonical_etypes), "Should track all edge types"


def test_hgt_forward_pass(small_homogg_graph):
    """Test HGT forward pass with homogeneous graph."""
    g_homogg, g_hetero = small_homogg_graph

    # Initialize HGT
    hgt = HGT(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1
    )

    # Forward pass
    try:
        h_dict = hgt(g_homogg)
    except Exception as e:
        pytest.fail(f"HGT forward pass failed: {e}")

    # Check output format
    assert isinstance(h_dict, dict), "Output should be a dictionary"
    assert set(h_dict.keys()) == set(g_hetero.ntypes), "Should have embeddings for all node types"

    # Check embedding shapes
    for ntype in g_hetero.ntypes:
        expected_shape = (g_hetero.num_nodes(ntype), 16)
        actual_shape = h_dict[ntype].shape
        assert actual_shape == expected_shape, f"Wrong shape for {ntype}: {actual_shape} vs {expected_shape}"


def test_hgt_type_fields(small_homogg_graph):
    """Test that HGT uses the correct type fields."""
    g_homogg, g_hetero = small_homogg_graph

    # Check what type fields are available
    print(f"\nAvailable ndata keys: {g_homogg.ndata.keys()}")
    print(f"Available edata keys: {g_homogg.edata.keys()}")

    # Check if we have the expected type information
    if dgl.NTYPE in g_homogg.ndata:
        print(f"dgl.NTYPE range: {g_homogg.ndata[dgl.NTYPE].min()}-{g_homogg.ndata[dgl.NTYPE].max()}")
    if dgl.ETYPE in g_homogg.edata:
        print(f"dgl.ETYPE range: {g_homogg.edata[dgl.ETYPE].min()}-{g_homogg.edata[dgl.ETYPE].max()}")
    if '_TYPE' in g_homogg.ndata:
        print(f"_TYPE (node) range: {g_homogg.ndata['_TYPE'].min()}-{g_homogg.ndata['_TYPE'].max()}")
    if '_TYPE' in g_homogg.edata:
        print(f"_TYPE (edge) range: {g_homogg.edata['_TYPE'].min()}-{g_homogg.edata['_TYPE'].max()}")


def test_link_predictor(small_hetero_graph):
    """Test LinkPredictor with heterogeneous embeddings."""
    g = small_hetero_graph

    # Create fake embeddings
    h = {
        'user': torch.randn(g.num_nodes('user'), 16),
        'item': torch.randn(g.num_nodes('item'), 16),
        'category': torch.randn(g.num_nodes('category'), 16),
    }

    # Initialize LinkPredictor
    predictor = LinkPredictor(
        n_hidden=16,
        edge_types=g.canonical_etypes,
        use_relation_transform=True
    )

    # Test prediction
    edge_type = ('user', 'likes', 'item')
    src_idx = torch.tensor([0, 1])
    dst_idx = torch.tensor([0, 1])

    scores = predictor(h, edge_type, src_idx, dst_idx)

    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"


def test_full_model_forward(small_homogg_graph):
    """Test full HGTLinkPredictor forward pass."""
    g_homogg, g_hetero = small_homogg_graph

    # Initialize model
    model = HGTLinkPredictor(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    )

    # Forward pass
    h = model(g_homogg)

    assert isinstance(h, dict), "Output should be a dictionary"
    assert len(h) == len(g_hetero.ntypes), "Should have embeddings for all node types"


def test_full_model_loss(small_homogg_graph):
    """Test loss computation."""
    g_homogg, g_hetero = small_homogg_graph

    # Initialize model
    model = HGTLinkPredictor(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    )

    # Test loss computation
    edge_type = ('user', 'likes', 'item')
    pos_src = torch.tensor([0, 1])
    pos_dst = torch.tensor([0, 1])
    neg_src = torch.tensor([0, 1])
    neg_dst = torch.tensor([2, 0])

    try:
        loss = model.compute_loss(g_homogg, edge_type, pos_src, pos_dst, neg_src, neg_dst)
        assert loss.item() >= 0, "Loss should be non-negative"
    except Exception as e:
        pytest.fail(f"Loss computation failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
