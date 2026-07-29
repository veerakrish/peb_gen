from __future__ import annotations

from webapp.backend.app.pipeline import prepare_pcb
from webapp.backend.tests.test_drone_example import build_drone_project


def test_prepare_pcb_routes_cleanly_within_a_few_attempts():
    project = build_drone_project()
    result = prepare_pcb(project, max_attempts=5, base_seed=0)

    routable = [n for n in result.nets if len(set(n.pins)) >= 2]
    print(
        f"\nattempts used: {result.attempts_tried}, {len(routable)} routable nets, "
        f"{len(result.routing.unrouted_nets)} unrouted: {result.routing.unrouted_nets}"
    )
    print(
        f"placement overlap={result.placement.overlap_area:.3f} oob={result.placement.out_of_bounds_area:.3f}, "
        f"trace length={result.routing.total_trace_length_mm:.2f} mm, vias={len(result.routing.vias)}"
    )

    assert result.placement.overlap_area < 1.0
    assert result.placement.out_of_bounds_area < 1.0
    assert not result.routing.unrouted_nets, (
        f"expected 'prepare PCB' to fully route within {result.attempts_tried} attempts: "
        f"{result.routing.unrouted_nets}"
    )
    for seg in result.routing.segments:
        assert seg.layer in ("F.Cu", "B.Cu")
        assert len(seg.points) >= 2


def test_no_cross_net_shorts():
    """No cell on any layer should ever be claimed by two different nets
    without a via between them — that would be an electrical short.
    """
    project = build_drone_project()
    result = prepare_pcb(project, max_attempts=5, base_seed=0)

    res = 0.5
    occupied_by: dict[tuple[str, int, int], str] = {}
    conflicts = []

    for seg in result.routing.segments:
        for x, y in seg.points:
            cell = (seg.layer, round(x / res), round(y / res))
            owner = occupied_by.get(cell)
            if owner is not None and owner != seg.net_name:
                conflicts.append((cell, owner, seg.net_name))
            occupied_by[cell] = seg.net_name

    assert conflicts == [], f"found {len(conflicts)} cross-net cell collisions, e.g. {conflicts[:5]}"
