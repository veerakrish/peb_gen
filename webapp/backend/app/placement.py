"""Component placement: given a component list and the nets derived from
the user's connection table, find non-overlapping (x, y, theta) positions
that keep the board compact (low HPWL) and in-bounds.

`PlacementBackend` is deliberately a narrow interface so the RL policy
from the Kaggle pipeline (`src/environment/pcb_gym_env.py`,
`src/models/placement_rl.py`) can be dropped in later as a second
implementation without touching the API layer above it. For now,
`SimulatedAnnealingPlacement` is the only (and default) backend: it needs
no training and reuses the exact same overlap/HPWL/out-of-bounds cost terms
from `src/utils/geometry.py` that the RL reward function uses, so swapping
backends later doesn't change what "a good layout" means.

Note on `clearance_mm`'s default (2.0mm, well above a typical DRC minimum):
a "zero overlap" placement can still pack components edge-to-edge with no
gap at all, which is fine for the placement cost itself but leaves
`routing.py` nowhere to actually run a trace between neighbors. This
clearance is doing double duty as a routing-channel reservation, not just
a collision margin — dropping it back down to a bare DRC minimum will
visibly increase how many nets `GridRouter` fails to complete.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.utils.geometry import hpwl as hpwl_of_nets
from src.utils.geometry import out_of_bounds_area, to_aabb, total_overlap_area

from .netlist import Net
from .schema import Component

Position = tuple[float, float, float]  # x_mm, y_mm, theta_rad


@dataclass
class PlacementResult:
    positions: dict[str, Position]
    cost: float
    overlap_area: float
    hpwl: float
    out_of_bounds_area: float


class PlacementBackend(ABC):
    @abstractmethod
    def solve(
        self,
        components: list[Component],
        nets: list[Net],
        board_width: float,
        board_height: float,
    ) -> PlacementResult: ...


class SimulatedAnnealingPlacement(PlacementBackend):
    def __init__(
        self,
        overlap_weight: float = 100.0,
        hpwl_weight: float = 1.0,
        out_of_bounds_weight: float = 50.0,
        clearance_mm: float = 2.0,
        iterations: int = 8000,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.997,
        allowed_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
        seed: int | None = None,
    ):
        self.overlap_weight = overlap_weight
        self.hpwl_weight = hpwl_weight
        self.out_of_bounds_weight = out_of_bounds_weight
        self.clearance_mm = clearance_mm
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.allowed_angles = [math.radians(a) for a in allowed_angles_deg]
        self._rng = random.Random(seed)

    def solve(
        self,
        components: list[Component],
        nets: list[Net],
        board_width: float,
        board_height: float,
    ) -> PlacementResult:
        if not components:
            return PlacementResult({}, 0.0, 0.0, 0.0, 0.0)

        ids = [c.id for c in components]
        sizes = {c.id: (c.width_mm, c.height_mm) for c in components}
        net_members = [n.component_ids() for n in nets if len(n.component_ids()) >= 2]

        def random_position(comp_id: str) -> Position:
            w, h = sizes[comp_id]
            x = self._rng.uniform(w / 2, max(w / 2, board_width - w / 2))
            y = self._rng.uniform(h / 2, max(h / 2, board_height - h / 2))
            theta = self._rng.choice(self.allowed_angles)
            return (x, y, theta)

        def evaluate(positions: dict[str, Position]) -> tuple[float, float, float, float]:
            centers = np.array([positions[cid][:2] for cid in ids], dtype=np.float64)
            box_sizes = np.array([sizes[cid] for cid in ids], dtype=np.float64)
            angles = np.array([positions[cid][2] for cid in ids], dtype=np.float64)
            boxes = to_aabb(centers, box_sizes, angles)

            overlap = total_overlap_area(boxes, clearance=self.clearance_mm)
            oob = out_of_bounds_area(boxes, board_width, board_height)

            net_point_arrays = []
            for members in net_members:
                pts = np.array([positions[cid][:2] for cid in members], dtype=np.float64)
                net_point_arrays.append(pts)
            hpwl_val = hpwl_of_nets(net_point_arrays)

            cost = (
                self.overlap_weight * overlap
                + self.hpwl_weight * hpwl_val
                + self.out_of_bounds_weight * oob
            )
            return cost, overlap, hpwl_val, oob

        current = {cid: random_position(cid) for cid in ids}
        current_cost, current_overlap, current_hpwl, current_oob = evaluate(current)
        best = dict(current)
        best_cost, best_overlap, best_hpwl, best_oob = (
            current_cost,
            current_overlap,
            current_hpwl,
            current_oob,
        )

        temperature = self.initial_temp
        for _ in range(self.iterations):
            comp_id = self._rng.choice(ids)
            old_position = current[comp_id]

            if self._rng.random() < 0.3:
                proposal = random_position(comp_id)
            else:
                w, h = sizes[comp_id]
                step = max(0.5, temperature / self.initial_temp * max(board_width, board_height) * 0.2)
                x = min(max(old_position[0] + self._rng.uniform(-step, step), w / 2), max(w / 2, board_width - w / 2))
                y = min(max(old_position[1] + self._rng.uniform(-step, step), h / 2), max(h / 2, board_height - h / 2))
                theta = self._rng.choice(self.allowed_angles) if self._rng.random() < 0.2 else old_position[2]
                proposal = (x, y, theta)

            current[comp_id] = proposal
            new_cost, new_overlap, new_hpwl, new_oob = evaluate(current)
            delta = new_cost - current_cost

            accept = delta <= 0 or self._rng.random() < math.exp(-delta / max(temperature, 1e-9))
            if accept:
                current_cost, current_overlap, current_hpwl, current_oob = (
                    new_cost,
                    new_overlap,
                    new_hpwl,
                    new_oob,
                )
                if new_cost < best_cost:
                    best = dict(current)
                    best_cost, best_overlap, best_hpwl, best_oob = new_cost, new_overlap, new_hpwl, new_oob
            else:
                current[comp_id] = old_position

            temperature *= self.cooling_rate

        return PlacementResult(
            positions=best,
            cost=best_cost,
            overlap_area=best_overlap,
            hpwl=best_hpwl,
            out_of_bounds_area=best_oob,
        )
