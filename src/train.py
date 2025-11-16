"""
Training loop for HGT link prediction.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
import dgl


def train_epoch(
    model,
    g: dgl.DGLGraph,
    train_batches: Dict,
    optimizer,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: HGT link prediction model
        g: Training graph
        train_batches: Dictionary of {edge_type: (pos_src, pos_dst, neg_src, neg_dst)}
        optimizer: Optimizer
        device: Device to train on

    Returns:
        average_loss: Average loss across all edge types
        losses_by_type: Dictionary of losses per edge type
    """
    model.train()
    total_loss = 0
    losses_by_type = {}

    g = g.to(device)

    for edge_type, (pos_src, pos_dst, neg_src, neg_dst) in train_batches.items():
        # Move data to device
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = model.compute_loss(g, edge_type, pos_src, pos_dst, neg_src, neg_dst)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record loss
        loss_val = loss.item()
        total_loss += loss_val
        losses_by_type[edge_type] = loss_val

    avg_loss = total_loss / len(train_batches)

    return avg_loss, losses_by_type


@torch.no_grad()
def evaluate(
    model,
    g: dgl.DGLGraph,
    eval_batches: Dict,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Evaluate model on validation or test set.

    Args:
        model: HGT link prediction model
        g: Graph (can be train graph for validation/test)
        eval_batches: Dictionary of {edge_type: (pos_src, pos_dst, neg_src, neg_dst)}
        device: Device

    Returns:
        metrics_by_type: Dictionary of {edge_type: accuracy}
        mrr_by_type: Dictionary of {edge_type: MRR}
        auc_by_type: Dictionary of {edge_type: AUC}
    """
    model.eval()

    g = g.to(device)

    metrics_by_type = {}
    mrr_by_type = {}
    auc_by_type = {}

    for edge_type, (pos_src, pos_dst, neg_src, neg_dst) in eval_batches.items():
        pos_src = pos_src.to(device)
        pos_dst = pos_dst.to(device)
        neg_src = neg_src.to(device)
        neg_dst = neg_dst.to(device)

        # Get predictions
        pos_scores = model.predict(g, edge_type, pos_src, pos_dst)
        neg_scores = model.predict(g, edge_type, neg_src, neg_dst)

        # Compute accuracy (proportion of pos scores > neg scores)
        # For each positive, compare against all negatives
        num_pos = len(pos_scores)
        num_neg = len(neg_scores)

        # Simple accuracy: what fraction of positives score higher than negatives
        pos_mean = pos_scores.mean().item()
        neg_mean = neg_scores.mean().item()
        accuracy = float(pos_mean > neg_mean)

        # Better metric: for each positive, count how many negatives it beats
        correct = 0
        for p_score in pos_scores:
            correct += (p_score > neg_scores).sum().item()
        accuracy = correct / (num_pos * num_neg)

        metrics_by_type[edge_type] = accuracy

        # Compute MRR (Mean Reciprocal Rank)
        # For each positive edge, rank it among negative edges
        ranks = []
        for p_score in pos_scores:
            # Count how many negatives score higher
            rank = (neg_scores >= p_score).sum().item() + 1
            ranks.append(1.0 / rank)

        mrr = sum(ranks) / len(ranks) if ranks else 0.0
        mrr_by_type[edge_type] = mrr

        # Compute AUC-like metric
        # Probability that a positive scores higher than a negative
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([
            torch.ones(len(pos_scores), device=device),
            torch.zeros(len(neg_scores), device=device)
        ])

        # Simple AUC approximation: fraction of (pos, neg) pairs correctly ranked
        auc = 0
        for p_score in pos_scores:
            auc += (p_score > neg_scores).float().mean().item()
        auc /= len(pos_scores)

        auc_by_type[edge_type] = auc

    return metrics_by_type, mrr_by_type, auc_by_type


def print_metrics(
    epoch: int,
    split: str,
    metrics: Dict,
    mrr: Dict,
    auc: Dict,
    loss: float = None,
    focus_predicates: list = None
):
    """
    Print metrics in a readable format.

    Args:
        epoch: Current epoch
        split: 'train', 'val', or 'test'
        metrics: Accuracy metrics by edge type
        mrr: MRR by edge type
        auc: AUC by edge type
        loss: Optional loss value
        focus_predicates: List of predicates to highlight
    """
    print(f"\n{'='*80}")
    print(f"Epoch {epoch} - {split.upper()}")
    print(f"{'='*80}")

    if loss is not None:
        print(f"Loss: {loss:.4f}")

    # Compute overall averages
    avg_acc = sum(metrics.values()) / len(metrics) if metrics else 0
    avg_mrr = sum(mrr.values()) / len(mrr) if mrr else 0
    avg_auc = sum(auc.values()) / len(auc) if auc else 0

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  MRR:      {avg_mrr:.4f}")
    print(f"  AUC:      {avg_auc:.4f}")

    # Print focus predicates first
    if focus_predicates:
        print(f"\nFocus Predicates:")
        for etype in sorted(metrics.keys()):
            src_type, rel, dst_type = etype
            if rel in focus_predicates:
                acc = metrics[etype]
                mrr_val = mrr[etype]
                auc_val = auc[etype]
                print(f"  {src_type} --[{rel}]--> {dst_type}")
                print(f"    Acc: {acc:.4f} | MRR: {mrr_val:.4f} | AUC: {auc_val:.4f}")

    # Print all metrics sorted by predicate frequency (most common first)
    print(f"\nAll Edge Types:")
    for etype in sorted(metrics.keys(), key=lambda e: metrics[e], reverse=True):
        src_type, rel, dst_type = etype
        acc = metrics[etype]
        mrr_val = mrr[etype]
        auc_val = auc[etype]

        # Mark focus predicates
        marker = " *" if focus_predicates and rel in focus_predicates else ""

        print(f"  {src_type} --[{rel}]--> {dst_type}{marker}")
        print(f"    Acc: {acc:.4f} | MRR: {mrr_val:.4f} | AUC: {auc_val:.4f}")


if __name__ == "__main__":
    print("Training utilities loaded. Use train.py through main training script.")
