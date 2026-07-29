"""Stage 1: Graph Transformer netlist encoder.

Maps a component/net graph (see `src/utils/graph_io.py` for the conversion
from the raw JSON netlist) to a dense 128-dim embedding per component,
learned via self-supervised link prediction: given two component
embeddings, predict whether they share a net. That pretraining signal
teaches the encoder a notion of "these parts are electrically close" which
Stage 2's placement policy then uses as its per-component state.

Built on PyG's `TransformerConv` (Shi et al., "Masked Label Prediction: Unified
Message Passing Model for Semi-Supervised Classification" — the standard
"graph transformer" conv in PyTorch Geometric), which natively supports
edge features, matching the "net type" edge attributes from `graph_io`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    head_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_feat_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = norm(h + residual)
        return self.output_proj(h)


class LinkPredictor(nn.Module):
    """Pretraining head only: scores whether a pair of component embeddings
    shares a net. Discarded after Stage 1 — Stage 2 only ever uses the
    encoder's node embeddings, not this head.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(pair).squeeze(-1)


def positive_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    """Deduplicated undirected (i<j) pairs from a bidirectional edge_index,
    for use as link-prediction positives (graph_io emits both directions
    per net membership).
    """
    src, dst = edge_index
    mask = src < dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def sample_negative_pairs(num_nodes: int, positive: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Uniform random (i, j) pairs with i != j that aren't in `positive`.

    Small, densely-connected graphs are common here: a shared VCC/GND rail
    alone can make every component pair "positive", leaving zero valid
    negatives. Rejection sampling for more negatives than actually exist
    would loop forever, so the request is capped at the true number
    available; `_batched_link_prediction`/`link_prediction_loss` both
    already tolerate fewer (or zero) negatives for a given graph.
    """
    positive_set = {(int(i), int(j)) for i, j in positive.t().tolist()}
    positive_set |= {(j, i) for i, j in positive_set}

    max_available = max(0, num_nodes * (num_nodes - 1) - len(positive_set))
    num_samples = min(num_samples, max_available)
    if num_samples == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    out = []
    seen = set()
    max_attempts = max(1000, num_samples * 50)
    attempts = 0
    while len(out) < num_samples and attempts < max_attempts:
        batch = min(num_samples * 2, 256)
        i = torch.randint(0, num_nodes, (batch,))
        j = torch.randint(0, num_nodes, (batch,))
        for a, b in zip(i.tolist(), j.tolist()):
            attempts += 1
            if a == b or (a, b) in positive_set or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
            if len(out) >= num_samples:
                break
    return torch.tensor(out, dtype=torch.long).t() if out else torch.zeros((2, 0), dtype=torch.long)


def link_prediction_loss(
    encoder: GraphTransformerEncoder,
    predictor: LinkPredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    negative_sampling_ratio: float = 1.0,
) -> torch.Tensor:
    """One BCE-with-logits loss over positive net-connection pairs and an
    equal (by ratio) count of random negative pairs, as the Stage 1
    pretraining objective.
    """
    embeddings = encoder(x, edge_index, edge_attr)

    pos_pairs = positive_pairs(edge_index)
    num_neg = max(1, int(pos_pairs.shape[1] * negative_sampling_ratio))
    neg_pairs = sample_negative_pairs(x.shape[0], pos_pairs, num_neg)

    pos_logits = predictor(embeddings[pos_pairs[0]], embeddings[pos_pairs[1]])
    neg_logits = predictor(embeddings[neg_pairs[0]], embeddings[neg_pairs[1]])

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, labels)
