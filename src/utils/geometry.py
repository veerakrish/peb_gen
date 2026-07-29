"""Vectorized geometry helpers for the placement environment and reward:
bounding-box overlap/collision, board-clearance, and HPWL.

Rotation is handled via the *rotated axis-aligned bounding box* (the AABB
that circumscribes a rectangle rotated by theta) rather than exact rotated-
rectangle (SAT) intersection. That keeps overlap/out-of-bounds checks O(1)
per component pair with plain numpy broadcasting, at the cost of being
conservative (it can report a small overlap between two rotated rectangles
that don't quite touch). That tradeoff is fine for a placement reward
signal; a routing/DRC-final pass should re-check with exact geometry
(e.g. Shapely) before treating a layout as clean.
"""

from __future__ import annotations

import numpy as np


def rotated_extent(width: np.ndarray, height: np.ndarray, theta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-width/half-height of the AABB of a WxH rectangle rotated by theta."""
    cos_t, sin_t = np.abs(np.cos(theta_rad)), np.abs(np.sin(theta_rad))
    eff_w = width * cos_t + height * sin_t
    eff_h = width * sin_t + height * cos_t
    return eff_w / 2.0, eff_h / 2.0


def to_aabb(centers: np.ndarray, sizes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """centers: (N,2) x,y. sizes: (N,2) w,h. angles: (N,) radians.

    Returns (N,4) array of [xmin, ymin, xmax, ymax].
    """
    half_w, half_h = rotated_extent(sizes[:, 0], sizes[:, 1], angles)
    xmin = centers[:, 0] - half_w
    xmax = centers[:, 0] + half_w
    ymin = centers[:, 1] - half_h
    ymax = centers[:, 1] + half_h
    return np.stack([xmin, ymin, xmax, ymax], axis=1)


def pairwise_overlap_area(boxes: np.ndarray, clearance: float = 0.0) -> np.ndarray:
    """boxes: (N,4) [xmin,ymin,xmax,ymax]. Returns (N,N) overlap area matrix
    (diagonal is 0), after inflating every box by `clearance` on each side
    so near-touching components are also penalized.
    """
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 0))

    xmin = boxes[:, 0] - clearance
    ymin = boxes[:, 1] - clearance
    xmax = boxes[:, 2] + clearance
    ymax = boxes[:, 3] + clearance

    ix_min = np.maximum(xmin[:, None], xmin[None, :])
    ix_max = np.minimum(xmax[:, None], xmax[None, :])
    iy_min = np.maximum(ymin[:, None], ymin[None, :])
    iy_max = np.minimum(ymax[:, None], ymax[None, :])

    overlap_w = np.clip(ix_max - ix_min, a_min=0, a_max=None)
    overlap_h = np.clip(iy_max - iy_min, a_min=0, a_max=None)
    area = overlap_w * overlap_h
    np.fill_diagonal(area, 0.0)
    return area


def total_overlap_area(boxes: np.ndarray, clearance: float = 0.0) -> float:
    """Sum of overlap area over all unordered component pairs."""
    area = pairwise_overlap_area(boxes, clearance)
    return float(area.sum() / 2.0)


def out_of_bounds_area(boxes: np.ndarray, board_width: float, board_height: float) -> float:
    """Sum, over all boxes, of the box area that falls outside [0,W]x[0,H]."""
    if boxes.shape[0] == 0:
        return 0.0
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    over_left = np.clip(-xmin, a_min=0, a_max=None)
    over_right = np.clip(xmax - board_width, a_min=0, a_max=None)
    over_bottom = np.clip(-ymin, a_min=0, a_max=None)
    over_top = np.clip(ymax - board_height, a_min=0, a_max=None)

    box_h = ymax - ymin
    box_w = xmax - xmin
    out_w = np.clip(over_left + over_right, a_min=0, a_max=np.maximum(box_w, 1e-9))
    out_h = np.clip(over_bottom + over_top, a_min=0, a_max=np.maximum(box_h, 1e-9))

    # approximate out-of-bounds area as the union of the two overhanging strips
    area = out_w * box_h + out_h * box_w - out_w * out_h
    return float(np.clip(area, a_min=0, a_max=None).sum())


def hpwl(nets: list[np.ndarray]) -> float:
    """Half-Perimeter Wire Length: sum over nets of (bbox width + bbox height)
    of that net's pin positions. `nets` is a list of (K,2) arrays of pin
    (x, y) coordinates; nets with fewer than 2 pins contribute 0.
    """
    total = 0.0
    for pins in nets:
        if pins.shape[0] < 2:
            continue
        w = pins[:, 0].max() - pins[:, 0].min()
        h = pins[:, 1].max() - pins[:, 1].min()
        total += float(w + h)
    return total
