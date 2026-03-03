"""
Test training and evaluation loops.
"""

import pytest
import torch
import dgl
import sys
sys.path.insert(0, 'src')

from model import HGTLinkPredictor
from train import train_epoch, evaluate, print_metrics


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


@pytest.fixture
def train_batches():
    """Create training batches."""
    return {
        ('user', 'likes', 'item'): (
            torch.tensor([0, 1]),  # pos_src
            torch.tensor([0, 1]),  # pos_dst
            torch.tensor([0, 1]),  # neg_src
            torch.tensor([2, 0])   # neg_dst
        ),
        ('item', 'belongs_to', 'category'): (
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0])
        ),
    }


@pytest.fixture
def eval_batches():
    """Create evaluation batches."""
    return {
        ('user', 'likes', 'item'): (
            torch.tensor([2]),     # pos_src
            torch.tensor([2]),     # pos_dst
            torch.tensor([2, 2]),  # neg_src
            torch.tensor([0, 1])   # neg_dst
        ),
    }


def test_train_epoch_cpu(homog_graph, model, train_batches):
    """Test training epoch on CPU."""
    device = torch.device('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg_loss, losses_by_type = train_epoch(model, homog_graph, train_batches, optimizer, device)

    # Check outputs
    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
    assert isinstance(losses_by_type, dict)
    assert len(losses_by_type) == len(train_batches)

    # Check all edge types have losses
    for edge_type in train_batches.keys():
        assert edge_type in losses_by_type
        assert losses_by_type[edge_type] >= 0


def test_train_epoch_single_edge_type(homog_graph, hetero_graph, train_batches):
    """Test training with single edge type."""
    device = torch.device('cpu')

    # Create model and single batch
    model = HGTLinkPredictor(hetero_graph, n_hidden=8, n_layers=1, n_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    single_batch = {
        ('user', 'likes', 'item'): train_batches[('user', 'likes', 'item')]
    }

    avg_loss, losses_by_type = train_epoch(model, homog_graph, single_batch, optimizer, device)

    assert len(losses_by_type) == 1
    assert avg_loss == losses_by_type[('user', 'likes', 'item')]


def test_train_epoch_updates_weights(homog_graph, model, train_batches):
    """Test that training actually updates model weights."""
    device = torch.device('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get initial weights
    initial_weights = [p.clone() for p in model.parameters()]

    # Train for one epoch
    train_epoch(model, homog_graph, train_batches, optimizer, device)

    # Check that at least some weights changed
    weights_changed = False
    for initial, current in zip(initial_weights, model.parameters()):
        if not torch.allclose(initial, current):
            weights_changed = True
            break

    assert weights_changed, "Model weights should update during training"


def test_evaluate_cpu(homog_graph, model, eval_batches):
    """Test evaluation on CPU."""
    device = torch.device('cpu')

    metrics, mrr, auc = evaluate(model, homog_graph, eval_batches, device)

    # Check output structure
    assert isinstance(metrics, dict)
    assert isinstance(mrr, dict)
    assert isinstance(auc, dict)

    # Check all edge types have metrics
    for edge_type in eval_batches.keys():
        assert edge_type in metrics
        assert edge_type in mrr
        assert edge_type in auc

        # Check metric ranges
        assert 0 <= metrics[edge_type] <= 1
        assert 0 <= mrr[edge_type] <= 1
        assert 0 <= auc[edge_type] <= 1


def test_evaluate_multiple_edge_types(homog_graph, hetero_graph):
    """Test evaluation with multiple edge types."""
    device = torch.device('cpu')
    model = HGTLinkPredictor(hetero_graph, n_hidden=8, n_layers=1, n_heads=2)

    batches = {
        ('user', 'likes', 'item'): (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
            torch.tensor([1, 2])
        ),
        ('item', 'belongs_to', 'category'): (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
            torch.tensor([1, 1])
        ),
    }

    metrics, mrr, auc = evaluate(model, homog_graph, batches, device)

    assert len(metrics) == 2
    assert len(mrr) == 2
    assert len(auc) == 2


def test_evaluate_no_grad(homog_graph, model, eval_batches):
    """Test that evaluation doesn't compute gradients."""
    device = torch.device('cpu')

    # Enable gradient tracking
    for p in model.parameters():
        p.requires_grad = True

    metrics, mrr, auc = evaluate(model, homog_graph, eval_batches, device)

    # Check that no gradients were computed
    for p in model.parameters():
        assert p.grad is None, "Evaluation should not compute gradients"


def test_evaluate_accuracy_metric(homog_graph, hetero_graph):
    """Test accuracy metric calculation."""
    device = torch.device('cpu')

    # Create model with deterministic initialization
    torch.manual_seed(42)
    model = HGTLinkPredictor(hetero_graph, n_hidden=16, n_layers=1, n_heads=2)

    batches = {
        ('user', 'likes', 'item'): (
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([2, 0, 1, 2])
        ),
    }

    metrics, mrr, auc = evaluate(model, homog_graph, batches, device)

    # Accuracy should be between 0 and 1
    assert 0 <= metrics[('user', 'likes', 'item')] <= 1


def test_print_metrics_basic(capsys):
    """Test print_metrics with basic inputs."""
    metrics = {
        ('user', 'likes', 'item'): 0.75,
        ('item', 'belongs_to', 'category'): 0.85,
    }
    mrr = {
        ('user', 'likes', 'item'): 0.70,
        ('item', 'belongs_to', 'category'): 0.80,
    }
    auc = {
        ('user', 'likes', 'item'): 0.72,
        ('item', 'belongs_to', 'category'): 0.82,
    }

    print_metrics(1, 'train', metrics, mrr, auc, loss=0.5)

    captured = capsys.readouterr()
    output = captured.out

    # Check that key information is printed
    assert 'Epoch 1' in output
    assert 'TRAIN' in output
    assert 'Loss: 0.5000' in output
    assert 'Accuracy' in output
    assert 'MRR' in output
    assert 'AUC' in output


def test_print_metrics_with_focus_predicates(capsys):
    """Test print_metrics with focus predicates."""
    metrics = {
        ('user', 'likes', 'item'): 0.75,
        ('item', 'belongs_to', 'category'): 0.85,
    }
    mrr = {
        ('user', 'likes', 'item'): 0.70,
        ('item', 'belongs_to', 'category'): 0.80,
    }
    auc = {
        ('user', 'likes', 'item'): 0.72,
        ('item', 'belongs_to', 'category'): 0.82,
    }

    print_metrics(2, 'val', metrics, mrr, auc, focus_predicates=['likes'])

    captured = capsys.readouterr()
    output = captured.out

    assert 'Focus Predicates' in output
    assert 'likes' in output


def test_print_metrics_no_loss(capsys):
    """Test print_metrics without loss (for validation/test)."""
    metrics = {('user', 'likes', 'item'): 0.75}
    mrr = {('user', 'likes', 'item'): 0.70}
    auc = {('user', 'likes', 'item'): 0.72}

    print_metrics(1, 'test', metrics, mrr, auc)

    captured = capsys.readouterr()
    output = captured.out

    assert 'TEST' in output
    assert 'Loss' not in output  # Should not print loss if None


def test_print_metrics_averages(capsys):
    """Test that overall averages are computed correctly."""
    metrics = {
        ('a', 'r1', 'b'): 0.6,
        ('c', 'r2', 'd'): 0.8,
        ('e', 'r3', 'f'): 1.0,
    }
    mrr = {
        ('a', 'r1', 'b'): 0.5,
        ('c', 'r2', 'd'): 0.7,
        ('e', 'r3', 'f'): 0.9,
    }
    auc = {
        ('a', 'r1', 'b'): 0.55,
        ('c', 'r2', 'd'): 0.75,
        ('e', 'r3', 'f'): 0.95,
    }

    print_metrics(1, 'train', metrics, mrr, auc)

    captured = capsys.readouterr()
    output = captured.out

    # Average accuracy should be (0.6 + 0.8 + 1.0) / 3 = 0.8
    assert '0.8000' in output or '0.80' in output


# GPU tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_train_epoch_gpu(homog_graph, model, train_batches):
    """Test training epoch on GPU."""
    device = torch.device('cuda')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg_loss, losses_by_type = train_epoch(model, homog_graph, train_batches, optimizer, device)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
    assert len(losses_by_type) == len(train_batches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_evaluate_gpu(homog_graph, model, eval_batches):
    """Test evaluation on GPU."""
    device = torch.device('cuda')
    model = model.to(device)

    metrics, mrr, auc = evaluate(model, homog_graph, eval_batches, device)

    assert len(metrics) == len(eval_batches)
    assert len(mrr) == len(eval_batches)
    assert len(auc) == len(eval_batches)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
