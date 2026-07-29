"""Ties schema -> netlist -> placement -> routing together into the single
call behind the "Prepare PCB" button.

Placement is a stochastic search (`SimulatedAnnealingPlacement`), and
"zero overlap" doesn't by itself guarantee every net has room to route —
some placements pack components into a layout that's geometrically valid
but leaves the router no channel to reach a particular pin. Rather than
accept whatever the first placement attempt produces, `prepare_pcb` runs a
handful of independent placement attempts (different random seeds) and
keeps whichever one the router actually completes best — the same
"multiple restarts, keep the best" idea `SimulatedAnnealingPlacement` uses
internally, just one level up, treating routability itself as part of
what makes a placement good.
"""

from __future__ import annotations

from dataclasses import dataclass

from .netlist import Net, build_nets, unconnected_pins
from .placement import PlacementBackend, PlacementResult, SimulatedAnnealingPlacement
from .routing import GridRouter, RoutingResult
from .schema import PcbProject


@dataclass
class PcbResult:
    nets: list[Net]
    placement: PlacementResult
    routing: RoutingResult
    unconnected_pins: list[tuple[str, str]]
    attempts_tried: int


def prepare_pcb(
    project: PcbProject,
    placement_backend: PlacementBackend | None = None,
    router: GridRouter | None = None,
    max_attempts: int = 5,
    base_seed: int = 0,
) -> PcbResult:
    nets = build_nets(project)
    router = router or GridRouter()

    # retries only help when we can vary the placement's randomness; a
    # caller-supplied backend runs once since we have no seed to vary
    attempts = max_attempts if placement_backend is None else 1

    best: tuple[PlacementResult, RoutingResult] | None = None
    attempts_used = 0
    for attempt in range(attempts):
        attempts_used += 1
        backend = placement_backend or SimulatedAnnealingPlacement(seed=base_seed + attempt)
        placement = backend.solve(project.components, nets, project.board.width_mm, project.board.height_mm)
        routing = router.route(project.components, nets, placement.positions, project.board.width_mm, project.board.height_mm)

        if best is None or _is_better(routing, best[1]):
            best = (placement, routing)
        if not routing.unrouted_nets:
            break

    placement, routing = best
    return PcbResult(
        nets=nets,
        placement=placement,
        routing=routing,
        unconnected_pins=unconnected_pins(project),
        attempts_tried=attempts_used,
    )


def _is_better(routing: RoutingResult, best_routing: RoutingResult) -> bool:
    if len(routing.unrouted_nets) != len(best_routing.unrouted_nets):
        return len(routing.unrouted_nets) < len(best_routing.unrouted_nets)
    return routing.total_trace_length_mm < best_routing.total_trace_length_mm
