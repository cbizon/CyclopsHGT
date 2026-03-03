"""
Heterogeneous Graph Transformer (HGT) model for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HGTConv
from typing import Dict, List


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer for link prediction.
    """

    def __init__(
        self,
        g: dgl.DGLGraph,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.2
    ):
        """
        Args:
            g: DGL heterogeneous graph (used to get node/edge types)
            n_hidden: Hidden dimension size
            n_layers: Number of HGT layers
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.node_types = g.ntypes
        self.edge_types = g.canonical_etypes

        # Random initial embeddings for each node type
        self.embed = nn.ModuleDict({
            ntype: nn.Embedding(g.num_nodes(ntype), n_hidden)
            for ntype in self.node_types
        })

        # Initialize embeddings
        for ntype in self.node_types:
            nn.init.xavier_uniform_(self.embed[ntype].weight)

        # HGT layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                HGTConv(
                    in_size=n_hidden,
                    head_size=n_hidden // n_heads,
                    num_heads=n_heads,
                    num_ntypes=len(self.node_types),
                    num_etypes=len(self.edge_types),
                    dropout=dropout
                )
            )

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(n_hidden) for _ in range(n_layers)
        ])

    def forward(self, g: dgl.DGLGraph, return_all_layers: bool = False):
        """
        Forward pass through HGT.

        Args:
            g: Input homogeneous graph (converted from heterogeneous)
            return_all_layers: If True, return embeddings from all layers

        Returns:
            Node embeddings dictionary {node_type: embeddings}
            If return_all_layers=True, returns list of embedding dicts
        """
        device = next(self.parameters()).device

        # Get initial embeddings from node types
        # g should have 'ntype' attribute from to_homogeneous
        node_types = g.ndata[dgl.NTYPE]

        # Create embedding lookup for all nodes based on their type
        h = torch.zeros(g.num_nodes(), self.n_hidden, device=device)
        for ntype_id, ntype in enumerate(self.node_types):
            mask = (node_types == ntype_id)
            if mask.any():
                # For homogeneous graph, node IDs are global
                # Use sequential IDs within each type for embedding lookup
                num_nodes_of_type = mask.sum().item()
                h[mask] = self.embed[ntype](torch.arange(num_nodes_of_type, device=device))

        if return_all_layers:
            all_h = [h]

        # Apply HGT layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            # Store previous embeddings for residual
            h_prev = h

            # Apply HGT layer with homogeneous format
            h = layer(g, h, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=False)

            # Apply layer norm and residual connection
            h = norm(h) + h_prev

            if return_all_layers:
                all_h.append(h)

        # Convert back to heterogeneous format (dict)
        h_dict = {}
        for ntype_id, ntype in enumerate(self.node_types):
            mask = (node_types == ntype_id)
            if mask.any():
                h_dict[ntype] = h[mask]

        if return_all_layers:
            # Convert all layers to dict format
            all_h_dict = []
            for h_layer in all_h:
                h_layer_dict = {}
                for ntype_id, ntype in enumerate(self.node_types):
                    mask = (node_types == ntype_id)
                    if mask.any():
                        h_layer_dict[ntype] = h_layer[mask]
                all_h_dict.append(h_layer_dict)
            return all_h_dict

        return h_dict


class LinkPredictor(nn.Module):
    """
    Link prediction head for heterogeneous graphs.

    Uses a simple dot product between source and destination node embeddings,
    with optional learned relation-specific transformations.
    """

    def __init__(
        self,
        n_hidden: int,
        edge_types: List[tuple],
        use_relation_transform: bool = True
    ):
        """
        Args:
            n_hidden: Hidden dimension
            edge_types: List of canonical edge types
            use_relation_transform: If True, learn relation-specific transformations
        """
        super().__init__()

        self.n_hidden = n_hidden
        self.edge_types = edge_types
        self.use_relation_transform = use_relation_transform

        if use_relation_transform:
            # Learn a transformation matrix for each relation type
            self.relation_transforms = nn.ModuleDict({
                f"{src_type}_{rel}_{dst_type}": nn.Linear(n_hidden, n_hidden, bias=False)
                for src_type, rel, dst_type in edge_types
            })

    def forward(
        self,
        h: Dict[str, torch.Tensor],
        edge_type: tuple,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link scores for given edge type and node pairs.

        Args:
            h: Node embeddings dictionary
            edge_type: Canonical edge type (src_type, rel, dst_type)
            src_idx: Source node indices
            dst_idx: Destination node indices

        Returns:
            Link scores (logits)
        """
        src_type, rel, dst_type = edge_type

        # Get embeddings
        src_emb = h[src_type][src_idx]
        dst_emb = h[dst_type][dst_idx]

        if self.use_relation_transform:
            # Apply relation-specific transformation to source embeddings
            key = f"{src_type}_{rel}_{dst_type}"
            src_emb = self.relation_transforms[key](src_emb)

        # Dot product score
        scores = (src_emb * dst_emb).sum(dim=-1)

        return scores


class HGTLinkPredictor(nn.Module):
    """
    Complete HGT-based link prediction model.
    """

    def __init__(
        self,
        g: dgl.DGLGraph,
        n_hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_relation_transform: bool = True
    ):
        """
        Args:
            g: DGL heterogeneous graph
            n_hidden: Hidden dimension
            n_layers: Number of HGT layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_relation_transform: Use relation-specific transformations
        """
        super().__init__()

        self.encoder = HGT(g, n_hidden, n_layers, n_heads, dropout)
        self.predictor = LinkPredictor(n_hidden, g.canonical_etypes, use_relation_transform)

    def forward(self, g: dgl.DGLGraph):
        """
        Encode graph and return embeddings.
        """
        return self.encoder(g)

    def predict(
        self,
        g: dgl.DGLGraph,
        edge_type: tuple,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link scores.
        """
        h = self.encoder(g)
        return self.predictor(h, edge_type, src_idx, dst_idx)

    def compute_loss(
        self,
        g: dgl.DGLGraph,
        edge_type: tuple,
        pos_src: torch.Tensor,
        pos_dst: torch.Tensor,
        neg_src: torch.Tensor,
        neg_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for link prediction.

        Args:
            g: Input graph
            edge_type: Edge type to predict
            pos_src, pos_dst: Positive edge endpoints
            neg_src, neg_dst: Negative edge endpoints

        Returns:
            Loss value
        """
        # Get embeddings once
        h = self.encoder(g)

        # Predict positive and negative scores
        pos_scores = self.predictor(h, edge_type, pos_src, pos_dst)
        neg_scores = self.predictor(h, edge_type, neg_src, neg_dst)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        return (pos_loss + neg_loss) / 2


if __name__ == "__main__":
    # Test model with synthetic graph
    print("Testing HGT model with synthetic graph...")

    graph_data = {
        ('drug', 'treats', 'disease'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0])),
        ('gene', 'associated_with', 'disease'): (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])),
    }

    g = dgl.heterograph(graph_data)

    # Add required type information
    g.ndata['_TYPE'] = {ntype: torch.zeros(g.num_nodes(ntype), dtype=torch.long) for ntype in g.ntypes}
    g.edata['_TYPE'] = {etype: torch.zeros(g.num_edges(etype), dtype=torch.long) for etype in g.canonical_etypes}

    model = HGTLinkPredictor(g, n_hidden=64, n_layers=2, n_heads=4)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    h = model(g)
    print(f"\nNode embeddings:")
    for ntype, emb in h.items():
        print(f"  {ntype}: {emb.shape}")

    # Test prediction
    pos_scores = model.predict(g, ('drug', 'treats', 'disease'), torch.tensor([0, 1]), torch.tensor([0, 1]))
    print(f"\nPrediction scores: {pos_scores}")

    # Test loss computation
    loss = model.compute_loss(
        g, ('drug', 'treats', 'disease'),
        torch.tensor([0]), torch.tensor([0]),
        torch.tensor([0]), torch.tensor([1])
    )
    print(f"Loss: {loss.item():.4f}")
