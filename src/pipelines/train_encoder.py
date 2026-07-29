"""Stage 1 training pipeline: pretrain the Graph Transformer encoder via
link prediction over synthetic drone-module netlists.

Usage::

    python -m src.pipelines.train_encoder
    python -m src.pipelines.train_encoder --num-graphs 200 --epochs 5

Design note on batching: PyG's DataLoader collates a list of graphs into
one disjoint-union `Batch` for an efficient single forward pass. But naive
whole-batch negative sampling (uniform random node pairs across the whole
batch) would let a negative pair land on two different netlists, which is
a *trivially* easy negative (those nodes were never candidates for sharing
a net at all) and would inflate the link-prediction ROC-AUC without
teaching anything real. `_batched_link_prediction` avoids this by slicing
the batch back into per-graph node ranges (via `batch.ptr`) and sampling
positives/negatives only within each graph, then pooling all pairs across
the batch for the loss / ROC-AUC.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.graph_transformer import (
    GraphTransformerEncoder,
    LinkPredictor,
    positive_pairs,
    sample_negative_pairs,
)
from src.utils.graph_io import netlist_to_data


def _load_configs(default_config_path: str, dataset_config_path: str) -> tuple[dict, dict]:
    with open(default_config_path, "r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    return default_config, dataset_config


def load_dataset_as_pyg(
    default_config: dict,
    dataset_config: dict,
    num_graphs: int | None = None,
    val_split: float | None = None,
    seed: int | None = None,
    include_real_data: bool = True,
) -> tuple[list[Data], list[Data]]:
    """Load synthetic netlist JSONs (auto-generating them if the raw
    directory doesn't have enough) plus any real netlists harvested by
    `data/real_netlist_harvester.py` (see `paths.real_data_dir`), convert
    each to a PyG `Data` graph, cache the converted graphs under
    `data/processed/`, and return an (train, val) split.

    Real graphs are added on top of `num_graphs`, not counted against it —
    there's normally far fewer of them than synthetic graphs, and capping
    would risk losing real data to make room for more synthetic data,
    which is the opposite of the point of gathering it.
    """
    raw_dir = dataset_config["paths"]["output_dir"]
    target_n = num_graphs or dataset_config["synthetic"]["num_graphs"]
    seed = seed if seed is not None else dataset_config["synthetic"].get("seed", 42)

    existing = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if len(existing) < target_n:
        from data.synthetic_generator import generate_dataset

        os.makedirs(raw_dir, exist_ok=True)
        graphs = generate_dataset(dataset_config, default_config["board"], target_n, seed)
        for i, graph in enumerate(graphs):
            with open(os.path.join(raw_dir, f"pcb_{i:05d}.json"), "w", encoding="utf-8") as f:
                json.dump(graph, f)
        existing = sorted(glob.glob(os.path.join(raw_dir, "*.json")))

    json_paths = existing[:target_n]

    real_dir = dataset_config["paths"].get("real_data_dir")
    if include_real_data and real_dir and os.path.isdir(real_dir):
        real_paths = sorted(glob.glob(os.path.join(real_dir, "*.json")))
        if real_paths:
            print(f"including {len(real_paths)} real netlist graph(s) from {real_dir}")
        json_paths = json_paths + real_paths

    categories = default_config["encoder"]["node_categories"]
    pin_types = default_config["encoder"]["pin_types"]

    processed_dir = default_config["paths"]["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    data_list = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            netlist = json.load(f)
        data = netlist_to_data(netlist, categories, pin_types)
        data_list.append(data)
        cache_path = os.path.join(processed_dir, os.path.basename(path).replace(".json", ".pt"))
        torch.save(data, cache_path)

    val_split = val_split if val_split is not None else default_config["encoder"]["training"]["val_split"]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(data_list), generator=generator).tolist()

    split_idx = int(len(perm) * (1 - val_split))
    train_list = [data_list[i] for i in perm[:split_idx]]
    val_list = [data_list[i] for i in perm[split_idx:]]
    return train_list, val_list


def _batched_link_prediction(
    encoder: GraphTransformerEncoder,
    predictor: LinkPredictor,
    batch,
    negative_sampling_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One encoder forward pass over the whole batch, then per-graph
    positive/negative sampling (see module docstring). Returns
    (loss, logits, labels) pooled across every graph in the batch.
    """
    embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr)

    all_logits = []
    all_labels = []
    ptr = batch.ptr.tolist()
    for lo, hi in zip(ptr[:-1], ptr[1:]):
        if hi - lo < 2:
            continue
        node_mask = (batch.edge_index[0] >= lo) & (batch.edge_index[0] < hi)
        local_edge_index = batch.edge_index[:, node_mask] - lo
        if local_edge_index.shape[1] == 0:
            continue

        pos_pairs = positive_pairs(local_edge_index)
        if pos_pairs.shape[1] == 0:
            continue
        num_neg = max(1, int(pos_pairs.shape[1] * negative_sampling_ratio))
        neg_pairs = sample_negative_pairs(hi - lo, pos_pairs, num_neg)

        local_emb = embeddings[lo:hi]
        pos_logits = predictor(local_emb[pos_pairs[0]], local_emb[pos_pairs[1]])
        neg_logits = predictor(local_emb[neg_pairs[0]], local_emb[neg_pairs[1]])

        all_logits.append(pos_logits)
        all_logits.append(neg_logits)
        all_labels.append(torch.ones_like(pos_logits))
        all_labels.append(torch.zeros_like(neg_logits))

    if not all_logits:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, torch.zeros(0), torch.zeros(0)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss, logits, labels


def train(
    default_config_path: str = "config/default_config.yaml",
    dataset_config_path: str = "config/dataset_config.yaml",
    num_graphs: int | None = None,
    epochs: int | None = None,
    checkpoint_dir: str | None = None,
) -> dict:
    default_config, dataset_config = _load_configs(default_config_path, dataset_config_path)
    train_cfg = default_config["encoder"]["training"]

    torch.manual_seed(default_config.get("seed", 42))

    train_list, val_list = load_dataset_as_pyg(default_config, dataset_config, num_graphs=num_graphs)
    print(f"dataset: {len(train_list)} train graphs, {len(val_list)} val graphs")

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)

    enc_cfg = default_config["encoder"]
    node_feat_dim = train_list[0].x.shape[1]
    edge_feat_dim = train_list[0].edge_attr.shape[1]

    encoder = GraphTransformerEncoder(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=enc_cfg["hidden_dim"],
        embedding_dim=enc_cfg["embedding_dim"],
        num_layers=enc_cfg["num_layers"],
        num_heads=enc_cfg["num_heads"],
        dropout=enc_cfg["dropout"],
    )
    predictor = LinkPredictor(embedding_dim=enc_cfg["embedding_dim"])

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    num_epochs = epochs or train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint_dir = checkpoint_dir or train_cfg["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "encoder_best.pth")
    summary_path = os.path.join(checkpoint_dir, "encoder_training_summary.json")

    neg_ratio = train_cfg["negative_sampling_ratio"]
    summary = {"epochs": []}
    best_val_auc = -1.0

    for epoch in range(num_epochs):
        encoder.train()
        predictor.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _ = _batched_link_prediction(encoder, predictor, batch, neg_ratio)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        encoder.eval()
        predictor.eval()
        val_losses = []
        all_labels: list[float] = []
        all_scores: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                loss, logits, labels = _batched_link_prediction(encoder, predictor, batch, neg_ratio)
                if labels.numel() == 0:
                    continue
                val_losses.append(loss.item())
                all_scores.extend(torch.sigmoid(logits).tolist())
                all_labels.extend(labels.tolist())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        if len(set(all_labels)) > 1:
            val_roc_auc = roc_auc_score(all_labels, all_scores)
        else:
            val_roc_auc = float("nan")

        train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_roc_auc,
            "lr": scheduler.get_last_lr()[0],
        }
        summary["epochs"].append(epoch_record)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_roc_auc={val_roc_auc:.4f}"
        )

        if val_roc_auc == val_roc_auc and val_roc_auc > best_val_auc:  # NaN-safe check
            best_val_auc = val_roc_auc
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "predictor_state_dict": predictor.state_dict(),
                    "node_feat_dim": node_feat_dim,
                    "edge_feat_dim": edge_feat_dim,
                    "encoder_config": {
                        "hidden_dim": enc_cfg["hidden_dim"],
                        "embedding_dim": enc_cfg["embedding_dim"],
                        "num_layers": enc_cfg["num_layers"],
                        "num_heads": enc_cfg["num_heads"],
                        "dropout": enc_cfg["dropout"],
                    },
                    "epoch": epoch,
                    "val_roc_auc": val_roc_auc,
                },
                checkpoint_path,
            )

    summary["best_val_roc_auc"] = best_val_auc
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {checkpoint_path} and {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-config", default="config/default_config.yaml")
    parser.add_argument("--dataset-config", default="config/dataset_config.yaml")
    parser.add_argument("--num-graphs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    train(
        default_config_path=args.default_config,
        dataset_config_path=args.dataset_config,
        num_graphs=args.num_graphs,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
