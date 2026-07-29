"""Pin geometry for user-authored components.

The web UI only collects a component's outline (width/height) and its pin
list (number, label, function) — there's no real footprint library behind
it, so no per-pin pad coordinates exist. To route anything we still need a
physical location for every pin, so this module auto-assigns one:
pins are spread evenly around the rectangle's perimeter, clockwise from the
top-left corner, the same way a generic connector/QFP-style footprint would
lay out an arbitrary pin count without needing package-specific rules.

This is a placement-quality approximation, not a real footprint — if/when
real footprints are wired in (e.g. matched from a KiCad library by
component category), this module is the one place that needs to change.
"""

from __future__ import annotations

import math

from .placement import Position
from .schema import Component

LocalPosition = tuple[float, float]


def assign_pin_local_positions(component: Component) -> dict[str, LocalPosition]:
    """Pin number -> (x, y) in the component's own centered local frame
    (origin at the component's center, same frame `placement.py` and
    `geometry.py` use before rotation/translation).
    """
    w, h = component.width_mm, component.height_mm
    n = len(component.pins)
    if n == 0:
        return {}

    perimeter = 2 * (w + h)
    positions: dict[str, LocalPosition] = {}
    for i, pin in enumerate(component.pins):
        # offset by half a step so no pin lands exactly on a corner
        t = (i + 0.5) / n * perimeter
        positions[pin.number] = _perimeter_point(t, w, h)
    return positions


def _perimeter_point(t: float, w: float, h: float) -> LocalPosition:
    """Walk clockwise from the top-left corner: across the top edge, down
    the right edge, back along the bottom edge, up the left edge.
    """
    if t < w:
        return (-w / 2 + t, -h / 2)
    t -= w
    if t < h:
        return (w / 2, -h / 2 + t)
    t -= h
    if t < w:
        return (w / 2 - t, h / 2)
    t -= w
    return (-w / 2, h / 2 - t)


def to_global(local: LocalPosition, position: Position) -> tuple[float, float]:
    """Rotate+translate a local pin position by a solved component
    placement (x, y, theta_rad) — same CCW rotation convention used
    throughout `placement.py`/`geometry.py`.
    """
    x, y, theta = position
    lx, ly = local
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    gx = x + lx * cos_t - ly * sin_t
    gy = y + lx * sin_t + ly * cos_t
    return (gx, gy)


def pin_global_positions(
    components: list[Component], positions: dict[str, Position]
) -> dict[tuple[str, str], tuple[float, float]]:
    """(component_id, pin_number) -> global (x, y) for every pin of every
    placed component.
    """
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for comp in components:
        local_positions = assign_pin_local_positions(comp)
        comp_position = positions[comp.id]
        for pin in comp.pins:
            out[(comp.id, pin.number)] = to_global(local_positions[pin.number], comp_position)
    return out
