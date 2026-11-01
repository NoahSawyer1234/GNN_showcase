"""Graph Convolutional Network (GCN) building blocks for node-, edge- and
graph-level tasks, built on PyTorch Geometric.

All three task models share one `GCNEncoder`; each task adds its own head.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GAE, GATConv, GCNConv, SAGEConv, global_add_pool, global_max_pool,
                                global_mean_pool)


class SkipGCNConv(nn.Module):
    """GCNConv plus a separate linear term on the node's own features: h_i = GCN(x)_i + W_self x_i.

    Plain GCNConv mixes a node's own features into the neighbour average with weight ~1/(degree+1),
    so on high-degree nodes they are almost lost. The extra term keeps them, like GraphSAGE's root weight."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.root = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight) + self.root(x)


class _SAGE(SAGEConv):
    """SAGEConv (mean aggregation + root weight) that accepts, and ignores, edge weights."""

    def forward(self, x, edge_index, edge_weight=None):
        return super().forward(x, edge_index)


class _GAT(nn.Module):
    """Graph attention layer (GATConv): each node weights its neighbours by learned attention scores.

    `heads` attention heads are concatenated so the output width matches the other layers
    (64 = 4 heads x 16); a layer whose width is not a multiple of `heads` (e.g. the 2-class
    output) uses one head. Edge weights are accepted and ignored."""

    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        h = heads if out_channels % heads == 0 and out_channels >= 4 * heads else 1
        self.conv = GATConv(in_channels, out_channels // h, heads=h)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)


CONVS = {"gcn": GCNConv, "gcn_skip": SkipGCNConv, "sage": _SAGE, "gat": _GAT}


class GCNEncoder(nn.Module):
    """Stack of GCNConv layers producing node embeddings.

    Args:
        in_channels: input feature dimension.
        hidden_channels: width of intermediate layers.
        out_channels: dimension of the returned node embeddings.
        num_layers: number of GCNConv layers (>= 1).
        dropout: dropout probability applied between layers.
        batch_norm: add BatchNorm after each hidden layer.
        final_activation: apply ReLU (+norm/dropout) after the last layer too.
            Use False when the output feeds a linear head or is a logit.
        conv: "gcn" (plain GCNConv, the default), "gcn_skip" (GCNConv + own-feature term),
            "sage" (GraphSAGE, mean aggregation) or "gat" (graph attention). The variants are reported
            as separately named models.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 batch_norm: bool = False, final_activation: bool = False, conv: str = "gcn"):
        super().__init__()
        assert num_layers >= 1
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(CONVS[conv](dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(dims[i + 1]) if batch_norm else nn.Identity()
            for i in range(num_layers)
        )
        self.dropout = dropout
        self.final_activation = final_activation

    def forward(self, x, edge_index, edge_weight: Optional[torch.Tensor] = None):
        last = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_weight)
            if i < last or self.final_activation:
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# --------------------------------------------------------------------------
# 1. Node-level: classification (out_channels = #classes) or regression
#    (out_channels = #targets). Output is one row per node.
# --------------------------------------------------------------------------
class NodeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, batch_norm=False, conv="gcn"):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels,
                                  num_layers, dropout, batch_norm,
                                  final_activation=False, conv=conv)

    def forward(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)


# --------------------------------------------------------------------------
# 2. Edge-level: link prediction and edge classification.
#    Encode nodes with the GCN, then score/classify node pairs.
# --------------------------------------------------------------------------
class EdgeGCN(nn.Module):
    """
    encode(...)        -> node embeddings z
    decode_link(...)   -> dot-product logit per node pair (link prediction)
    classify_edges(...) -> class logits per edge (edge classification);
                           only available when `num_edge_classes` is given.
    """

    def __init__(self, in_channels, hidden_channels, embed_channels,
                 num_layers=2, dropout=0.5, num_edge_classes: Optional[int] = None, conv="gcn"):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, embed_channels,
                                  num_layers, dropout, final_activation=False, conv=conv)
        self.edge_head = None
        if num_edge_classes is not None:
            self.edge_head = nn.Sequential(
                nn.Linear(3 * embed_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_edge_classes),
            )

    def encode(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)

    @staticmethod
    def decode_link(z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def classify_edges(self, z, edge_label_index):
        assert self.edge_head is not None, "construct with num_edge_classes=..."
        src, dst = edge_label_index
        # Symmetric in (src, dst) so undirected edges get one prediction.
        feats = torch.cat([z[src] + z[dst], (z[src] - z[dst]).abs(), z[src] * z[dst]], dim=-1)
        return self.edge_head(feats)


# --------------------------------------------------------------------------
# 3. Graph-level: classification or regression. One output row per graph.
# --------------------------------------------------------------------------
_POOL = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


class GraphGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.5, pooling="mean", batch_norm=True, conv="gcn"):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, hidden_channels,
                                  num_layers, dropout, batch_norm,
                                  final_activation=True, conv=conv)
        self.pool = _POOL[pooling]
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, batch, edge_weight=None):
        h = self.encoder(x, edge_index, edge_weight)
        return self.head(self.pool(h, batch))


# --------------------------------------------------------------------------
# 4. Graph autoencoder (unsupervised): a GCN encoder trained to reconstruct
#    the graph's edges with an inner-product decoder (Kipf & Welling, 2016).
#    No labels are used; the embeddings feed a downstream classifier.
# --------------------------------------------------------------------------
def GraphAutoEncoder(in_channels, hidden_channels, embed_channels, num_layers=2, dropout=0.5, conv="gcn"):
    return GAE(GCNEncoder(in_channels, hidden_channels, embed_channels, num_layers, dropout,
                          final_activation=False, conv=conv))
