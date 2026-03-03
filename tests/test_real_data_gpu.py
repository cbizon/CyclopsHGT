"""
Test with actual ROBOKOP data to reproduce the CUDA error.
"""

import pytest
import torch
import dgl
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from graph_loader import load_graph_from_tsv
from data_split import stratified_edge_split, create_train_graph, prepare_edge_batches
from model import HGTLinkPredictor


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


@pytest.fixture
def real_graph():
    """Load the actual ROBOKOP graph."""
    graph_dir = Path("input_graphs/mf1")
    if not graph_dir.exists():
        pytest.skip("Real data not available")

    print("\nLoading real graph...")
    g, metadata = load_graph_from_tsv(graph_dir)
    print(f"Loaded: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Node types: {len(g.ntypes)}")
    print(f"Edge types: {len(g.canonical_etypes)}")

    return g


@pytest.fixture
def real_train_graph(real_graph):
    """Create training graph from real data."""
    g = real_graph

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

    # Split edges (use small ratios for testing)
    print("\nSplitting edges...")
    train_edges, val_edges, test_edges = stratified_edge_split(
        g, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # Create training graph
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

    print(f"Training graph: {g_train.num_edges()} edges")

    # Convert to homogeneous
    print("Converting to homogeneous...")
    g_train_homog = dgl.to_homogeneous(g_train, ndata=['_TYPE'], edata=['_TYPE'], store_type=True)
    print(f"Homogeneous: {g_train_homog.num_nodes()} nodes, {g_train_homog.num_edges()} edges")

    # Prepare small batches for testing
    print("Preparing batches (first 5 edge types only)...")
    limited_train_edges = {k: v for i, (k, v) in enumerate(train_edges.items()) if i < 5}
    train_batches = prepare_edge_batches(g, limited_train_edges, negative_ratio=1)

    return g_train_homog, g_train, train_batches, ntype_to_id, etype_to_id


def test_real_graph_conversion(real_train_graph):
    """Test that real graph converts correctly."""
    g_train_homog, g_train, train_batches, ntype_to_id, etype_to_id = real_train_graph

    # Check type ranges
    print(f"\n=== Type Information ===")
    print(f"Num node types: {len(ntype_to_id)}")
    print(f"Num edge types: {len(etype_to_id)}")

    ntype_values = g_train_homog.ndata[dgl.NTYPE]
    etype_values = g_train_homog.edata[dgl.ETYPE]

    print(f"Node type range in homogeneous graph: {ntype_values.min()}-{ntype_values.max()}")
    print(f"Edge type range in homogeneous graph: {etype_values.min()}-{etype_values.max()}")

    assert ntype_values.max() < len(ntype_to_id), f"Node types out of bounds: {ntype_values.max()} >= {len(ntype_to_id)}"
    assert etype_values.max() < len(etype_to_id), f"Edge types out of bounds: {etype_values.max()} >= {len(etype_to_id)}"


def test_real_model_init(real_train_graph):
    """Test model initialization with real data."""
    g_train_homog, g_train, train_batches, ntype_to_id, etype_to_id = real_train_graph

    print(f"\n=== Model Initialization ===")
    model = HGTLinkPredictor(
        g=g_train,  # Pass heterogeneous graph for init
        n_hidden=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_relation_transform=True
    )

    print(f"Model created with:")
    print(f"  - {len(model.encoder.node_types)} node types")
    print(f"  - {len(model.encoder.edge_types)} edge types")
    print(f"  - {sum(p.numel() for p in model.parameters()):,} parameters")


def test_real_forward_pass_gpu(real_train_graph):
    """Test forward pass on GPU with real data."""
    g_train_homog, g_train, train_batches, ntype_to_id, etype_to_id = real_train_graph

    device = torch.device('cuda')

    print(f"\n=== GPU Forward Pass ===")

    # Initialize model
    model = HGTLinkPredictor(
        g=g_train,
        n_hidden=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_relation_transform=True
    ).to(device)

    # Move graph to GPU
    g_train_homog_gpu = g_train_homog.to(device)

    print(f"Model on: {next(model.parameters()).device}")
    print(f"Graph on: {g_train_homog_gpu.ndata['_TYPE'].device}")

    # Try forward pass
    try:
        h = model(g_train_homog_gpu)
        print("✓ Forward pass succeeded!")
        print(f"  Output keys: {list(h.keys())}")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n✗ CUDA ERROR in forward pass:")
            print(f"  {e}")

            # Print detailed debug info
            print(f"\n=== Debug Info ===")
            print(f"Node type range: {g_train_homog_gpu.ndata[dgl.NTYPE].min()}-{g_train_homog_gpu.ndata[dgl.NTYPE].max()}")
            print(f"Edge type range: {g_train_homog_gpu.edata[dgl.ETYPE].min()}-{g_train_homog_gpu.edata[dgl.ETYPE].max()}")
            print(f"Model expects {len(model.encoder.node_types)} node types")
            print(f"Model expects {len(model.encoder.edge_types)} edge types")
            print(f"HGT layer num_ntypes: {model.encoder.layers[0].num_ntypes}")
            print(f"HGT layer num_etypes: {model.encoder.layers[0].num_etypes}")
        raise


def test_real_loss_computation_gpu(real_train_graph):
    """Test loss computation on GPU with real data."""
    g_train_homog, g_train, train_batches, ntype_to_id, etype_to_id = real_train_graph

    if len(train_batches) == 0:
        pytest.skip("No training batches available")

    device = torch.device('cuda')

    print(f"\n=== GPU Loss Computation ===")

    # Initialize model
    model = HGTLinkPredictor(
        g=g_train,
        n_hidden=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_relation_transform=True
    ).to(device)

    # Move graph to GPU
    g_train_homog_gpu = g_train_homog.to(device)

    # Get first batch
    edge_type = list(train_batches.keys())[0]
    pos_src, pos_dst, neg_src, neg_dst = train_batches[edge_type]

    print(f"Testing edge type: {edge_type}")
    print(f"Batch sizes: pos={len(pos_src)}, neg={len(neg_src)}")
    print(f"Source range: {pos_src.min()}-{pos_src.max()}")
    print(f"Dest range: {pos_dst.min()}-{pos_dst.max()}")

    # Move to GPU
    pos_src = pos_src.to(device)
    pos_dst = pos_dst.to(device)
    neg_src = neg_src.to(device)
    neg_dst = neg_dst.to(device)

    try:
        loss = model.compute_loss(g_train_homog_gpu, edge_type, pos_src, pos_dst, neg_src, neg_dst)
        print(f"✓ Loss computation succeeded: {loss.item():.4f}")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n✗ CUDA ERROR in loss computation:")
            print(f"  {e}")
        raise


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
