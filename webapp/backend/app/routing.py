"""Stage 3: grid-based multi-layer maze router.

Given the components' solved placement and the nets derived from the
connection table, route actual copper: a 2-layer (F.Cu / B.Cu, matching
KiCad's own layer names so export is a direct translation) grid, A*
pathfinding per connection, vias for layer changes.

Each >2-pin net is turned into a spanning tree grown one pin at a time
(`_route_net`): at every step the closest not-yet-connected pin tries to
reach the tree via its nearest already-connected anchor first, falling
back to farther anchors if that path is blocked — more resilient than a
fixed MST edge list, where one blocked edge would fail the whole net even
when a different anchor had a clear path. This is still a sequential
router with no rip-up/reroute: once a net's copper is on the grid it
becomes a permanent obstacle to every other net, so under heavy congestion
some nets may still fail to route — those are reported in
`RoutingResult.unrouted_nets` rather than silently producing a shorted or
broken board.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .footprint import pin_global_positions
from .netlist import Net
from .placement import Position
from .schema import Component

Layer = str  # "F.Cu" or "B.Cu"
GridCell = tuple[int, int]  # (row, col)

OBSTACLE = -1
FREE = 0


@dataclass
class RoutedSegment:
    net_name: str
    layer: Layer
    points: list[tuple[float, float]]


@dataclass
class Via:
    net_name: str
    position: tuple[float, float]


@dataclass
class RoutingResult:
    segments: list[RoutedSegment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    unrouted_nets: list[str] = field(default_factory=list)

    @property
    def total_trace_length_mm(self) -> float:
        total = 0.0
        for seg in self.segments:
            for a, b in zip(seg.points, seg.points[1:]):
                total += math.hypot(b[0] - a[0], b[1] - a[1])
        return total


class GridRouter:
    def __init__(
        self,
        resolution_mm: float = 0.5,
        clearance_mm: float = 0.2,
        via_cost_mm: float = 5.0,
        layers: tuple[Layer, Layer] = ("F.Cu", "B.Cu"),
    ):
        self.res = resolution_mm
        self.clearance_mm = clearance_mm
        self.via_cost_mm = via_cost_mm
        self.layers = layers

    # -- grid <-> board mm conversion -------------------------------------------------
    def _to_cell(self, x: float, y: float) -> GridCell:
        return (int(round(y / self.res)), int(round(x / self.res)))

    def _to_mm(self, cell: GridCell) -> tuple[float, float]:
        row, col = cell
        return (col * self.res, row * self.res)

    # -- obstacle construction ----------------------------------------------------------
    def _build_base_obstacles(
        self, components: list[Component], positions: dict[str, Position], rows: int, cols: int
    ) -> np.ndarray:
        from src.utils.geometry import to_aabb  # local import: keeps webapp decoupled from src at module load time

        obstacles = np.zeros((rows, cols), dtype=np.int8)
        if components:
            centers = np.array([positions[c.id][:2] for c in components])
            sizes = np.array([(c.width_mm, c.height_mm) for c in components])
            angles = np.array([positions[c.id][2] for c in components])
            boxes = to_aabb(centers, sizes, angles)

            for xmin, ymin, xmax, ymax in boxes:
                r0, c0 = self._to_cell(xmin - self.clearance_mm, ymin - self.clearance_mm)
                r1, c1 = self._to_cell(xmax + self.clearance_mm, ymax + self.clearance_mm)
                r0, r1 = max(0, r0), min(rows - 1, r1)
                c0, c1 = max(0, c0), min(cols - 1, c1)
                if r1 >= r0 and c1 >= c0:
                    obstacles[r0 : r1 + 1, c0 : c1 + 1] = OBSTACLE
        return obstacles

    def _carve_pin_clearance(self, obstacles: np.ndarray, cell: GridCell, rows: int, cols: int) -> None:
        radius_cells = max(1, math.ceil(self.clearance_mm / self.res))
        r0, c0 = cell
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < rows and 0 <= c < cols and dr * dr + dc * dc <= radius_cells * radius_cells:
                    obstacles[r, c] = FREE

    # -- A* -------------------------------------------------------------------------------
    def _astar(
        self,
        occupancy: dict[Layer, np.ndarray],
        start_cell: GridCell,
        goal_cell: GridCell,
        net_idx: int,
        rows: int,
        cols: int,
    ) -> list[tuple[int, int, int]] | None:
        """State: (row, col, layer_index). Goal test ignores layer (a pad is
        reachable from either side). Returns the path or None if unreachable.
        """
        num_layers = len(self.layers)
        neighbors_2d = [
            (-1, 0, self.res), (1, 0, self.res), (0, -1, self.res), (0, 1, self.res),
            (-1, -1, self.res * math.sqrt(2)), (-1, 1, self.res * math.sqrt(2)),
            (1, -1, self.res * math.sqrt(2)), (1, 1, self.res * math.sqrt(2)),
        ]

        def passable(r: int, c: int, layer_idx: int) -> bool:
            if not (0 <= r < rows and 0 <= c < cols):
                return False
            cell_val = occupancy[self.layers[layer_idx]][r, c]
            return cell_val == FREE or cell_val == net_idx

        def heuristic(r: int, c: int) -> float:
            dr, dc = abs(goal_cell[0] - r), abs(goal_cell[1] - c)
            # octile distance: admissible for 8-directional grid movement
            return self.res * (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))

        start_states = [
            (start_cell[0], start_cell[1], layer_idx)
            for layer_idx in range(num_layers)
            if passable(start_cell[0], start_cell[1], layer_idx)
        ]
        if not start_states or not any(
            passable(goal_cell[0], goal_cell[1], layer_idx) for layer_idx in range(num_layers)
        ):
            return None

        counter = 0
        open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        g_score = {}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}

        for state in start_states:
            g_score[state] = 0.0
            heapq.heappush(open_heap, (heuristic(state[0], state[1]), counter, state))
            counter += 1

        visited = set()
        while open_heap:
            _, _, state = heapq.heappop(open_heap)
            if state in visited:
                continue
            visited.add(state)
            r, c, layer_idx = state

            if (r, c) == goal_cell:
                path = [state]
                while state in came_from:
                    state = came_from[state]
                    path.append(state)
                path.reverse()
                return path

            for dr, dc, move_cost in neighbors_2d:
                nr, nc = r + dr, c + dc
                if not passable(nr, nc, layer_idx):
                    continue
                neighbor = (nr, nc, layer_idx)
                tentative = g_score[state] + move_cost
                if tentative < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = state
                    heapq.heappush(open_heap, (tentative + heuristic(nr, nc), counter, neighbor))
                    counter += 1

            for other_layer in range(num_layers):
                if other_layer == layer_idx or not passable(r, c, other_layer):
                    continue
                neighbor = (r, c, other_layer)
                tentative = g_score[state] + self.via_cost_mm
                if tentative < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = state
                    heapq.heappush(open_heap, (tentative + heuristic(r, c), counter, neighbor))
                    counter += 1

        return None

    def _mark_path(
        self, occupancy: dict[Layer, np.ndarray], path: list[tuple[int, int, int]], net_idx: int, rows: int, cols: int
    ) -> None:
        radius_cells = max(1, math.ceil(self.clearance_mm / self.res))
        for r, c, layer_idx in path:
            layer = self.layers[layer_idx]
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and dr * dr + dc * dc <= radius_cells * radius_cells:
                        if occupancy[layer][rr, cc] == FREE:
                            occupancy[layer][rr, cc] = net_idx

    @staticmethod
    def _simplify_collinear(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Drop interior points where the path doesn't change direction —
        A* walks the grid one cell at a time, so a straight run of 40+
        cells would otherwise become 40+ one-cell-long segments instead of
        one. Keeps every point where the incoming and outgoing direction
        differ (i.e. real corners) plus both endpoints.
        """
        if len(points) <= 2:
            return points
        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            (ax, ay), (bx, by), (cx, cy) = points[i - 1], points[i], points[i + 1]
            d1 = (round(bx - ax, 6), round(by - ay, 6))
            d2 = (round(cx - bx, 6), round(cy - by, 6))
            if d1 != d2:
                simplified.append(points[i])
        simplified.append(points[-1])
        return simplified

    def _path_to_segments_and_vias(
        self, path: list[tuple[int, int, int]], net_name: str
    ) -> tuple[list[RoutedSegment], list[Via]]:
        segments: list[RoutedSegment] = []
        vias: list[Via] = []

        current_layer_idx = path[0][2]
        current_points = [self._to_mm((path[0][0], path[0][1]))]

        for prev, curr in zip(path, path[1:]):
            if curr[2] != prev[2]:
                via_point = self._to_mm((prev[0], prev[1]))
                current_points.append(via_point)
                segments.append(
                    RoutedSegment(
                        net_name=net_name,
                        layer=self.layers[current_layer_idx],
                        points=self._simplify_collinear(current_points),
                    )
                )
                vias.append(Via(net_name=net_name, position=via_point))
                current_layer_idx = curr[2]
                current_points = [via_point]
            current_points.append(self._to_mm((curr[0], curr[1])))

        segments.append(
            RoutedSegment(
                net_name=net_name,
                layer=self.layers[current_layer_idx],
                points=self._simplify_collinear(current_points),
            )
        )
        return segments, vias

    # -- top-level entry point -----------------------------------------------------------
    def route(
        self,
        components: list[Component],
        nets: list[Net],
        positions: dict[str, Position],
        board_width: float,
        board_height: float,
    ) -> RoutingResult:
        rows = max(1, math.ceil(board_height / self.res)) + 1
        cols = max(1, math.ceil(board_width / self.res)) + 1

        base_obstacles = self._build_base_obstacles(components, positions, rows, cols)
        pin_positions = pin_global_positions(components, positions)
        for (x, y) in pin_positions.values():
            self._carve_pin_clearance(base_obstacles, self._to_cell(x, y), rows, cols)

        occupancy = {layer: base_obstacles.copy() for layer in self.layers}

        routable_nets = [n for n in nets if len(set(n.pins)) >= 2]
        routable_nets.sort(key=lambda n: len(n.pins))

        result = RoutingResult()
        for net_idx, net in enumerate(routable_nets, start=1):
            unique_pins = sorted(set(net.pins))
            segments, vias, failed = self._route_net(
                net.name, unique_pins, pin_positions, occupancy, net_idx, rows, cols
            )
            result.segments.extend(segments)
            result.vias.extend(vias)
            if failed:
                result.unrouted_nets.append(net.name)

        return result

    def _route_net(
        self,
        net_name: str,
        unique_pins: list[tuple[str, str]],
        pin_positions: dict,
        occupancy: dict[Layer, np.ndarray],
        net_idx: int,
        rows: int,
        cols: int,
    ) -> tuple[list[RoutedSegment], list[Via], bool]:
        """Grow a spanning tree one pin at a time: at each step, connect
        whichever unconnected pin is closest to the tree, trying every
        already-connected anchor (nearest first) rather than a single fixed
        MST partner. If a pin can't reach any anchor yet, it's deferred —
        another pin joining the tree first may open up a path for it — and
        only reported as failed if nothing more can be connected in a full
        pass over the remaining pins.
        """
        connected = [unique_pins[0]]
        remaining = list(unique_pins[1:])
        segments: list[RoutedSegment] = []
        vias: list[Via] = []

        while remaining:
            remaining.sort(key=lambda p: min(
                math.hypot(pin_positions[p][0] - pin_positions[c][0], pin_positions[p][1] - pin_positions[c][1])
                for c in connected
            ))

            made_progress = False
            for pin in list(remaining):
                anchors = sorted(
                    connected,
                    key=lambda c: math.hypot(
                        pin_positions[pin][0] - pin_positions[c][0], pin_positions[pin][1] - pin_positions[c][1]
                    ),
                )
                for anchor in anchors:
                    start_cell = self._to_cell(*pin_positions[anchor])
                    goal_cell = self._to_cell(*pin_positions[pin])
                    path = self._astar(occupancy, start_cell, goal_cell, net_idx, rows, cols)
                    if path is not None:
                        self._mark_path(occupancy, path, net_idx, rows, cols)
                        seg, via = self._path_to_segments_and_vias(path, net_name)
                        segments.extend(seg)
                        vias.extend(via)
                        connected.append(pin)
                        remaining.remove(pin)
                        made_progress = True
                        break
                if made_progress:
                    break

            if not made_progress:
                return segments, vias, True

        return segments, vias, False
