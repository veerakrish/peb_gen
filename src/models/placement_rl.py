"""Stage 2: PPO placement policy on top of the Stage 1 Graph Transformer.

Three pieces:
  - `PCBFeatureExtractor`: turns `PCBPlacementEnv`'s Dict observation
    (padded node embeddings + mask, active-component one-hot, occupancy
    grid) into one feature vector for SB3's actor/critic heads.
  - `load_frozen_encoder` / `compute_node_embeddings`: bridges a trained
    `checkpoints/encoder_best.pth` to the environment — run the netlist
    through the frozen Stage 1 encoder once per episode (not once per
    step: the embeddings only depend on the fixed netlist graph, not on
    where components have been placed so far) and hand the resulting
    per-component vectors to the env via `set_embeddings`.
  - `build_ppo_agent`: a thin PPO factory wiring the two together.

The actor (mean/log-std of a Gaussian over the continuous (x, y, theta)
action) and critic (V(s)) heads themselves are *not* custom — SB3's
standard `MultiInputActorCriticPolicy` already builds exactly that MLP
pair on top of whatever `features_extractor_class` produces, so there's
nothing this file needs to reimplement there.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..utils.graph_io import netlist_to_data
from .graph_transformer import GraphTransformerEncoder


class PCBFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_channels: int = 16,
        pooled_proj_dim: int = 128,
        features_dim: int = 256,
    ):
        embedding_dim = observation_space["node_embeddings"].shape[1]
        grid_h, grid_w = observation_space["occupancy"].shape
        super().__init__(observation_space, features_dim=features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out_dim = cnn_channels * 2 * 4 * 4

        self.context_proj = nn.Linear(embedding_dim, pooled_proj_dim)

        combined_dim = embedding_dim + pooled_proj_dim + cnn_out_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        node_embeddings = observations["node_embeddings"]  # (B, N, D)
        node_mask = observations["node_mask"]  # (B, N)
        active_mask = observations["active_mask"]  # (B, N)
        occupancy = observations["occupancy"]  # (B, H, W)

        active_embedding = torch.einsum("bn,bnd->bd", active_mask, node_embeddings)

        mask_sum = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = torch.einsum("bn,bnd->bd", node_mask, node_embeddings) / mask_sum
        pooled = self.context_proj(pooled)

        cnn_features = self.cnn(occupancy.unsqueeze(1))

        combined = torch.cat([active_embedding, pooled, cnn_features], dim=-1)
        return self.head(combined)


def load_frozen_encoder(checkpoint_path: str, device: str = "cpu") -> GraphTransformerEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["encoder_config"]
    encoder = GraphTransformerEncoder(
        node_feat_dim=checkpoint["node_feat_dim"],
        edge_feat_dim=checkpoint["edge_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.to(device)


@torch.no_grad()
def compute_node_embeddings(
    encoder: GraphTransformerEncoder,
    netlist: dict,
    categories: list[str],
    pin_types: list[str],
) -> dict[str, np.ndarray]:
    """Run one netlist through the frozen Stage 1 encoder. Returns a
    component_id -> embedding dict, exactly the shape `PCBPlacementEnv`
    (and `PCBPlacementEnv.set_embeddings`) expect.
    """
    data = netlist_to_data(netlist, categories, pin_types)
    embeddings = encoder(data.x, data.edge_index, data.edge_attr)
    return {cid: embeddings[i].numpy() for i, cid in enumerate(data.component_ids)}


def build_ppo_agent(
    env,
    features_extractor_kwargs: dict | None = None,
    encoder_checkpoint_path: str | None = None,
    **ppo_kwargs,
) -> tuple[PPO, GraphTransformerEncoder | None]:
    """Build a PPO agent wired to `PCBFeatureExtractor`. If
    `encoder_checkpoint_path` is given, also loads the frozen Stage 1
    encoder and returns it (the caller is responsible for running it over
    each episode's netlist and calling `env.set_embeddings(...)` before
    `reset()` — that's environment/episode setup, not something the agent
    itself should own).
    """
    policy_kwargs = ppo_kwargs.pop("policy_kwargs", {})
    policy_kwargs.setdefault("features_extractor_class", PCBFeatureExtractor)
    policy_kwargs.setdefault("features_extractor_kwargs", features_extractor_kwargs or {})

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs)

    encoder = load_frozen_encoder(encoder_checkpoint_path) if encoder_checkpoint_path else None
    return model, encoder
