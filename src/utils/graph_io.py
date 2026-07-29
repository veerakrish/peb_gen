"""Convert the framework-agnostic netlist JSON (see `data/synthetic_generator.py`)
into a PyTorch Geometric `Data` graph for the Stage 1 encoder.

Design note: every component is the same PyG node "type" and every net
membership is the same edge "type" — categorical distinctions (module
category, net signal class) are encoded as node/edge *features* rather than
separate relation types. That gets the same modeling benefit as a
`HeteroData` graph here (the categories still shape the learned embedding
through the transformer's attention over edge features) without the added
complexity of per-relation weight matrices, which would only pay off if
different net types needed genuinely different message-passing behavior.

Nets aren't pairwise: a net can list any number of component pins. GNN
convs operate on ordinary edges, so each net is *clique-expanded* — every
distinct pair of components on that net becomes a bidirectional edge,
tagged with that net's type. A pair sharing more than one net gets one
edge per shared net (not deduplicated), since each net membership is a
distinct real-world signal.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def _component_features(
    component: dict, categories: list[str], size_norm: float, max_pin_count: float
) -> list[float]:
    width_n = component["width_mm"] / size_norm
    height_n = component["height_mm"] / size_norm
    pin_count_n = len(component["pins"]) / max_pin_count

    onehot = [0.0] * len(categories)
    if component["category"] in categories:
        onehot[categories.index(component["category"])] = 1.0

    return [width_n, height_n, pin_count_n, *onehot]


def _net_component_pairs(net: dict) -> list[tuple[str, str]]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for pin in net["pins"]:
        cid = pin["component_id"]
        if cid not in seen_set:
            seen_set.add(cid)
            seen.append(cid)

    pairs = []
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            pairs.append((seen[i], seen[j]))
    return pairs


def netlist_to_data(
    netlist: dict,
    categories: list[str],
    pin_types: list[str],
    size_norm: float = 50.0,
    max_pin_count: float = 32.0,
) -> Data:
    components = netlist["components"]
    component_ids = [c["id"] for c in components]
    id_to_idx = {cid: i for i, cid in enumerate(component_ids)}

    x = torch.tensor(
        [_component_features(c, categories, size_norm, max_pin_count) for c in components],
        dtype=torch.float32,
    )

    src, dst, edge_feats = [], [], []
    for net in netlist["nets"]:
        net_type_onehot = [0.0] * len(pin_types)
        if net["type"] in pin_types:
            net_type_onehot[pin_types.index(net["type"])] = 1.0
        fanout_n = len(net["pins"]) / 8.0  # rough normalization, most nets are small

        for a, b in _net_component_pairs(net):
            ia, ib = id_to_idx[a], id_to_idx[b]
            feat = [*net_type_onehot, fanout_n]
            # add both directions so message passing treats the graph as undirected
            src += [ia, ib]
            dst += [ib, ia]
            edge_feats += [feat, feat]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(pin_types) + 1), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(components))
    data.component_ids = component_ids
    return data
