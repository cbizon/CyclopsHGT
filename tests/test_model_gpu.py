"""
Test HGT model on GPU to isolate CUDA errors.
"""

import pytest
import torch
import dgl
import sys
sys.path.insert(0, 'src')

from model import HGTLinkPredictor


# Skip all tests if no GPU available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


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
def small_homogg_graph_cpu(small_hetero_graph):
    """Convert to homogeneous graph on CPU."""
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


def test_model_to_gpu(small_homogg_graph_cpu):
    """Test moving model to GPU."""
    g_homogg, g_hetero = small_homogg_graph_cpu

    # Initialize model on CPU
    model = HGTLinkPredictor(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    )

    # Move to GPU
    device = torch.device('cuda')
    model = model.to(device)

    # Check model is on GPU
    assert next(model.parameters()).is_cuda, "Model should be on GPU"


def test_graph_to_gpu(small_homogg_graph_cpu):
    """Test moving graph to GPU."""
    g_homogg, g_hetero = small_homogg_graph_cpu

    device = torch.device('cuda')
    g_homogg_gpu = g_homogg.to(device)

    # Check graph data is on GPU
    assert g_homogg_gpu.ndata['_TYPE'].is_cuda, "Node type data should be on GPU"
    assert g_homogg_gpu.edata['_TYPE'].is_cuda, "Edge type data should be on GPU"


def test_forward_pass_gpu(small_homogg_graph_cpu):
    """Test forward pass on GPU."""
    g_homogg, g_hetero = small_homogg_graph_cpu

    device = torch.device('cuda')

    # Initialize model and move to GPU
    model = HGTLinkPredictor(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    ).to(device)

    # Move graph to GPU
    g_homogg_gpu = g_homogg.to(device)

    # Forward pass
    print(f"\n=== Forward Pass Debug ===")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Graph ndata['_TYPE'] device: {g_homogg_gpu.ndata['_TYPE'].device}")
    print(f"Graph edata['_TYPE'] device: {g_homogg_gpu.edata['_TYPE'].device}")
    print(f"Graph ndata[dgl.NTYPE] range: {g_homogg_gpu.ndata[dgl.NTYPE].min()}-{g_homogg_gpu.ndata[dgl.NTYPE].max()}")
    print(f"Graph edata[dgl.ETYPE] range: {g_homogg_gpu.edata[dgl.ETYPE].min()}-{g_homogg_gpu.edata[dgl.ETYPE].max()}")
    print(f"Model num_ntypes: {len(model.encoder.node_types)}")
    print(f"Model num_etypes: {len(model.encoder.edge_types)}")

    try:
        h = model(g_homogg_gpu)
        print("✓ Forward pass succeeded")

        # Check output
        assert isinstance(h, dict), "Output should be a dictionary"
        for ntype, emb in h.items():
            assert emb.is_cuda, f"Embeddings for {ntype} should be on GPU"
        print("✓ All embeddings on GPU")

    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n✗ CUDA ERROR: {e}")
            # Print more debug info
            import traceback
            traceback.print_exc()
        raise


def test_loss_computation_gpu(small_homogg_graph_cpu):
    """Test loss computation on GPU."""
    g_homogg, g_hetero = small_homogg_graph_cpu

    device = torch.device('cuda')

    # Initialize model and move to GPU
    model = HGTLinkPredictor(
        g=g_hetero,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_relation_transform=True
    ).to(device)

    # Move graph to GPU
    g_homogg_gpu = g_homogg.to(device)

    # Move edge indices to GPU
    edge_type = ('user', 'likes', 'item')
    pos_src = torch.tensor([0, 1], device=device)
    pos_dst = torch.tensor([0, 1], device=device)
    neg_src = torch.tensor([0, 1], device=device)
    neg_dst = torch.tensor([2, 0], device=device)

    try:
        loss = model.compute_loss(g_homogg_gpu, edge_type, pos_src, pos_dst, neg_src, neg_dst)
        print(f"✓ Loss computation succeeded: {loss.item():.4f}")
        assert loss.item() >= 0, "Loss should be non-negative"
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n✗ CUDA ERROR in loss computation: {e}")
            import traceback
            traceback.print_exc()
        raise


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
