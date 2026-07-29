"""Gymnasium environment for sequential 2D PCB component placement.

One episode places every component of a single netlist graph (the JSON
schema produced by `data/synthetic_generator.py`, or an equivalent dict)
exactly once, in a fixed order, one component per step:

  - Observation: embeddings for *every* component in the netlist (from a
    pretrained Stage 1 Graph Transformer if supplied, else a cheap
    hand-built per-node feature vector), padded to a fixed `max_nodes` so
    the Dict/Box observation space has a static shape despite netlists
    having a variable component count, plus a mask marking which rows are
    real vs padding, plus a one-hot mask marking which node is the one
    being placed this step, plus the board's current occupancy grid.
    Exposing the *whole* graph's embeddings (not just the active
    component's) matters here: the entire reason Stage 1 learns a
    net-topology-aware embedding is so the placement policy can reason
    about a component relative to the rest of the board, which a
    single-node observation would throw away.
  - Action: continuous, normalized to [-1, 1]^3; `step()` rescales to real
    (x, y, theta) board coordinates / radians. Normalized rather than
    raw board-coordinate bounds because SB3's default Gaussian policy
    starts at mean~0, std=1 — against a [0, board_width] range nearly
    every initial sample would be negative and clip straight to the
    (0, 0) corner with no gradient to ever move away from it.
  - Reward: dense, paid out incrementally so summing it over the episode
    equals the full-board objective exactly once each:
      - overlap: only between the just-placed component and previously
        placed ones (each pair is scored exactly once, when the later of
        the two gets placed)
      - HPWL: only for nets that become fully placed on this step (all
        member components now have a position); computed at component-
        center granularity since this stage doesn't carry per-pin
        geometry (see `_net_component_ids`)
      - out-of-bounds: for the just-placed component's own box

See `src/utils/geometry.py` for the underlying vectorized math.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from ..utils.geometry import out_of_bounds_area, rotated_extent, to_aabb

DEFAULT_CATEGORIES = [
    "MCU", "ESC", "GPS", "PDB", "IMU", "RX", "CONNECTOR", "PASSIVE",
]


def _default_embedding(component: dict, embedding_dim: int, categories: list[str]) -> np.ndarray:
    """Cheap stand-in for a Stage 1 encoder embedding, used when none is
    supplied (e.g. testing the environment in isolation).
    """
    vec = np.zeros(embedding_dim, dtype=np.float32)
    vec[0] = component["width_mm"] / 50.0
    vec[1] = component["height_mm"] / 50.0
    vec[2] = len(component["pins"]) / 32.0
    cat_idx = categories.index(component["category"]) if component["category"] in categories else len(categories)
    onehot_start = 3
    if onehot_start + cat_idx < embedding_dim:
        vec[onehot_start + cat_idx] = 1.0
    return vec


def _net_component_ids(net: dict) -> list[str]:
    """Distinct component ids on a net, in the order first seen."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for pin in net["pins"]:
        cid = pin["component_id"]
        if cid not in seen_set:
            seen_set.add(cid)
            seen.append(cid)
    return seen


class PCBPlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str = "config/default_config.yaml",
        netlist: dict | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
        max_nodes: int | None = None,
    ):
        super().__init__()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.board_width = float(config["board"]["width_mm"])
        self.board_height = float(config["board"]["height_mm"])
        self.grid_res = float(config["board"]["grid_resolution_mm"])
        self.clearance = float(config["board"]["min_clearance_mm"])
        self.embedding_dim = int(config["encoder"]["embedding_dim"])
        self.categories = list(config["encoder"]["node_categories"])
        self.max_nodes = max_nodes or int(config["encoder"].get("max_nodes", 40))

        weights = config["reward_weights"]
        self.w_overlap = float(weights["overlap_area"])
        self.w_hpwl = float(weights["hpwl"])
        self.w_oob = float(weights["out_of_bounds"])

        self.grid_w = max(1, int(math.ceil(self.board_width / self.grid_res)))
        self.grid_h = max(1, int(math.ceil(self.board_height / self.grid_res)))

        # Normalized to [-1, 1]^3, not real board coordinates directly.
        # SB3's default continuous policy is a Gaussian starting at mean~0,
        # std=1 — against an UN-normalized [0, board_width] range, nearly
        # every raw sample is negative and gets clipped straight to the
        # (0, 0) corner regardless of what the policy actually outputs,
        # which gives zero gradient signal to ever move away from it (the
        # clip has no gradient). Against this symmetric range, a std=1
        # Gaussian's mass actually spans a useful chunk of the board from
        # step one. `step()` rescales to real coordinates.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "node_embeddings": spaces.Box(
                    -np.inf, np.inf, shape=(self.max_nodes, self.embedding_dim), dtype=np.float32
                ),
                "node_mask": spaces.Box(0.0, 1.0, shape=(self.max_nodes,), dtype=np.float32),
                "active_mask": spaces.Box(0.0, 1.0, shape=(self.max_nodes,), dtype=np.float32),
                "occupancy": spaces.Box(0.0, 1.0, shape=(self.grid_h, self.grid_w), dtype=np.float32),
            }
        )

        self._external_embeddings = embeddings
        self._netlist: dict | None = None
        self._rng = np.random.default_rng()

        if netlist is not None:
            self.load_netlist(netlist)

    def set_embeddings(self, embeddings: dict[str, np.ndarray] | None) -> None:
        """Inject externally computed (e.g. frozen Stage 1 encoder) embeddings,
        keyed by component id. Call before `reset()`; see
        `src/models/placement_rl.py`'s encoder integration helper.
        """
        self._external_embeddings = embeddings
        if self._netlist is not None:
            self._build_node_embeddings()

    def load_netlist(self, netlist: dict) -> None:
        self._netlist = netlist
        self._components = {c["id"]: c for c in netlist["components"]}
        self._nets = netlist["nets"]
        self._net_members = [_net_component_ids(n) for n in self._nets]

        # fixed row ordering for the padded node arrays — independent of
        # `self.order`, which is reshuffled per episode in reset()
        self._node_order = list(self._components.keys())
        if len(self._node_order) > self.max_nodes:
            raise ValueError(
                f"netlist has {len(self._node_order)} components, exceeding max_nodes={self.max_nodes}"
            )
        self._node_row = {cid: i for i, cid in enumerate(self._node_order)}
        self._build_node_embeddings()

    def _build_node_embeddings(self) -> None:
        embeddings = np.zeros((self.max_nodes, self.embedding_dim), dtype=np.float32)
        mask = np.zeros((self.max_nodes,), dtype=np.float32)
        for i, comp_id in enumerate(self._node_order):
            if self._external_embeddings is not None and comp_id in self._external_embeddings:
                emb = np.asarray(self._external_embeddings[comp_id], dtype=np.float32)
                if emb.shape[0] != self.embedding_dim:
                    raise ValueError(
                        f"embedding for {comp_id} has dim {emb.shape[0]}, expected {self.embedding_dim}"
                    )
            else:
                emb = _default_embedding(self._components[comp_id], self.embedding_dim, self.categories)
            embeddings[i] = emb
            mask[i] = 1.0
        self._node_embeddings = embeddings
        self._node_mask = mask

    def _mark_occupancy(self, center: tuple[float, float], size: tuple[float, float], theta: float) -> None:
        half_w, half_h = rotated_extent(np.array([size[0]]), np.array([size[1]]), np.array([theta]))
        xmin, xmax = center[0] - half_w[0], center[0] + half_w[0]
        ymin, ymax = center[1] - half_h[0], center[1] + half_h[0]

        col_lo = int(np.clip(np.floor(xmin / self.grid_res), 0, self.grid_w - 1))
        col_hi = int(np.clip(np.ceil(xmax / self.grid_res), 0, self.grid_w))
        row_lo = int(np.clip(np.floor(ymin / self.grid_res), 0, self.grid_h - 1))
        row_hi = int(np.clip(np.ceil(ymax / self.grid_res), 0, self.grid_h))
        if col_hi > col_lo and row_hi > row_lo:
            self.occupancy[row_lo:row_hi, col_lo:col_hi] = 1.0

    def _current_obs(self) -> dict:
        active_mask = np.zeros((self.max_nodes,), dtype=np.float32)
        if self.step_index < len(self.order):
            comp_id = self.order[self.step_index]
            active_mask[self._node_row[comp_id]] = 1.0
        return {
            "node_embeddings": self._node_embeddings,
            "node_mask": self._node_mask,
            "active_mask": active_mask,
            "occupancy": self.occupancy.copy(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "netlist" in options:
            self.load_netlist(options["netlist"])
        if self._netlist is None:
            raise RuntimeError("PCBPlacementEnv.reset() called with no netlist loaded; "
                               "pass one to __init__ or reset(options={'netlist': ...})")

        self.order = list(self._components.keys())
        self._rng.shuffle(self.order)

        self.step_index = 0
        self.positions: dict[str, tuple[float, float, float]] = {}
        self.occupancy = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._completed_nets: set[int] = set()

        return self._current_obs(), {}

    def step(self, action: np.ndarray):
        if self.step_index >= len(self.order):
            raise RuntimeError("step() called after episode already terminated; call reset() first")

        # rescale from the normalized [-1, 1]^3 action_space to real board
        # coordinates; still clip defensively since a Gaussian's tails can
        # land slightly outside [-1, 1]
        clipped = np.clip(action, -1.0, 1.0)
        x = float((clipped[0] + 1.0) / 2.0 * self.board_width)
        y = float((clipped[1] + 1.0) / 2.0 * self.board_height)
        theta = float(clipped[2] * math.pi)

        comp_id = self.order[self.step_index]
        comp = self._components[comp_id]
        size = (comp["width_mm"], comp["height_mm"])
        self.positions[comp_id] = (x, y, theta)
        self._mark_occupancy((x, y), size, theta)

        placed_ids = self.order[: self.step_index + 1]
        overlap_penalty = self._incremental_overlap(comp_id, placed_ids)
        hpwl_penalty = self._newly_completed_hpwl()
        oob_penalty = self._out_of_bounds_penalty(comp_id)

        reward = -(
            self.w_overlap * overlap_penalty
            + self.w_hpwl * hpwl_penalty
            + self.w_oob * oob_penalty
        )

        self.step_index += 1
        terminated = self.step_index >= len(self.order)
        truncated = False

        info = {
            "overlap_area": overlap_penalty,
            "hpwl": hpwl_penalty,
            "out_of_bounds_area": oob_penalty,
        }
        return self._current_obs(), reward, terminated, truncated, info

    def _incremental_overlap(self, new_comp_id: str, placed_ids: list[str]) -> float:
        if len(placed_ids) < 2:
            return 0.0
        centers, sizes, angles = self._boxes_for(placed_ids)
        boxes = to_aabb(centers, sizes, angles)
        new_idx = placed_ids.index(new_comp_id)
        new_box = boxes[new_idx]

        total = 0.0
        for i, box in enumerate(boxes):
            if i == new_idx:
                continue
            ix_min = max(new_box[0] - self.clearance, box[0] - self.clearance)
            ix_max = min(new_box[2] + self.clearance, box[2] + self.clearance)
            iy_min = max(new_box[1] - self.clearance, box[1] - self.clearance)
            iy_max = min(new_box[3] + self.clearance, box[3] + self.clearance)
            w = max(0.0, ix_max - ix_min)
            h = max(0.0, iy_max - iy_min)
            total += w * h
        return total

    def _boxes_for(self, comp_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = np.array([self.positions[c][:2] for c in comp_ids], dtype=np.float64)
        sizes = np.array(
            [(self._components[c]["width_mm"], self._components[c]["height_mm"]) for c in comp_ids],
            dtype=np.float64,
        )
        angles = np.array([self.positions[c][2] for c in comp_ids], dtype=np.float64)
        return centers, sizes, angles

    def _newly_completed_hpwl(self) -> float:
        total = 0.0
        for net_idx, members in enumerate(self._net_members):
            if net_idx in self._completed_nets:
                continue
            if not all(m in self.positions for m in members):
                continue
            self._completed_nets.add(net_idx)
            if len(members) < 2:
                continue
            pts = np.array([self.positions[m][:2] for m in members], dtype=np.float64)
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            total += float(w + h)
        return total

    def _out_of_bounds_penalty(self, comp_id: str) -> float:
        centers, sizes, angles = self._boxes_for([comp_id])
        box = to_aabb(centers, sizes, angles)
        return out_of_bounds_area(box, self.board_width, self.board_height)

    def render(self):
        return None
