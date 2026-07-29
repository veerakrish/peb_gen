"""End-to-end check of the backend core against the exact workflow the
user described: hand-enter a drone's components (flight controller, ESCs,
GPS, power distribution board) with custom pin labels, wire them up as a
point-to-point connection table, then solve for a placement.
"""

from __future__ import annotations

import math

import numpy as np

from src.utils.geometry import out_of_bounds_area, to_aabb, total_overlap_area
from webapp.backend.app.netlist import build_nets, unconnected_pins
from webapp.backend.app.placement import SimulatedAnnealingPlacement
from webapp.backend.app.schema import BoardSpec, Component, Connection, ConnectionEndpoint, PcbProject, Pin


def _pin(number: str, label: str, function: str = "signal") -> Pin:
    return Pin(number=number, label=label, function=function)


def _endpoint(component_id: str, pin_number: str) -> ConnectionEndpoint:
    return ConnectionEndpoint(component_id=component_id, pin_number=pin_number)


def _connect(a: tuple[str, str], b: tuple[str, str]) -> Connection:
    return Connection(source=_endpoint(*a), target=_endpoint(*b))


def build_drone_project() -> PcbProject:
    fc = Component(
        id="FC",
        label="Flight Controller",
        width_mm=30,
        height_mm=30,
        pins=[
            _pin("1", "VCC", "power"),
            _pin("2", "GND", "ground"),
            _pin("3", "TX1"),
            _pin("4", "RX1"),
            _pin("5", "PWM1"),
            _pin("6", "PWM2"),
            _pin("7", "PWM3"),
            _pin("8", "PWM4"),
        ],
    )
    escs = [
        Component(
            id=f"ESC{i}",
            label=f"ESC {i}",
            width_mm=12,
            height_mm=12,
            pins=[_pin("1", "VCC", "power"), _pin("2", "GND", "ground"), _pin("3", "SIGNAL")],
        )
        for i in range(1, 5)
    ]
    gps = Component(
        id="GPS",
        label="GPS Module",
        width_mm=18,
        height_mm=18,
        pins=[_pin("1", "VCC", "power"), _pin("2", "GND", "ground"), _pin("3", "TX"), _pin("4", "RX")],
    )
    pdb = Component(
        id="PDB",
        label="Power Distribution Board",
        width_mm=25,
        height_mm=25,
        pins=[
            _pin("1", "VBAT+", "power"),
            _pin("2", "VBAT-", "ground"),
            _pin("3", "5V_OUT", "power"),
            _pin("4", "GND_OUT", "ground"),
        ],
    )
    battery = Component(
        id="BATT",
        label="Battery Connector",
        width_mm=8,
        height_mm=5,
        pins=[_pin("1", "VBAT+", "power"), _pin("2", "VBAT-", "ground")],
    )

    components = [fc, pdb, gps, battery, *escs]

    connections = [
        # battery input into the PDB
        _connect(("BATT", "1"), ("PDB", "1")),
        _connect(("BATT", "2"), ("PDB", "2")),
        # power rail fanout from the PDB to every module
        _connect(("PDB", "3"), ("FC", "1")),
        _connect(("PDB", "4"), ("FC", "2")),
        _connect(("PDB", "3"), ("GPS", "1")),
        _connect(("PDB", "4"), ("GPS", "2")),
    ]
    for esc in escs:
        connections.append(_connect(("PDB", "3"), (esc.id, "1")))
        connections.append(_connect(("PDB", "4"), (esc.id, "2")))

    # flight controller PWM outputs to each ESC's signal pin
    for i, esc in enumerate(escs, start=5):
        connections.append(_connect(("FC", str(i)), (esc.id, "3")))

    # GPS serial link
    connections.append(_connect(("FC", "3"), ("GPS", "4")))
    connections.append(_connect(("FC", "4"), ("GPS", "3")))

    return PcbProject(board=BoardSpec(width_mm=80, height_mm=80), components=components, connections=connections)


def test_project_validates():
    project = build_drone_project()
    assert len(project.components) == 8
    assert len(project.connections) == 2 + 4 + 8 + 4 + 2


def test_nets_merge_shared_power_rails():
    project = build_drone_project()
    nets = build_nets(project)

    by_name = {n.name: n for n in nets}
    # PDB's output pins are labeled "5V_OUT"/"GND_OUT" but every consumer
    # module calls its own input pin "VCC"/"GND" — the majority label wins.
    assert "VCC" in by_name
    assert set(by_name["VCC"].component_ids()) == {"PDB", "FC", "GPS", "ESC1", "ESC2", "ESC3", "ESC4"}
    assert "GND" in by_name
    assert set(by_name["GND"].component_ids()) == {"PDB", "FC", "GPS", "ESC1", "ESC2", "ESC3", "ESC4"}

    # each FC PWM output pairs 1:1 with exactly one ESC's SIGNAL pin; the
    # net's display name is a tie between the two pins' labels ("PWM1" vs
    # "SIGNAL") so we check membership rather than the name itself.
    esc_ids = {"ESC1", "ESC2", "ESC3", "ESC4"}
    motor_signal_nets = [
        n for n in nets
        if len(n.component_ids()) == 2 and "FC" in n.component_ids() and set(n.component_ids()) & esc_ids
    ]
    assert len(motor_signal_nets) == 4


def test_no_unconnected_pins_in_this_example():
    project = build_drone_project()
    assert unconnected_pins(project) == []


def test_placement_converges_to_a_clean_layout():
    project = build_drone_project()
    nets = build_nets(project)

    solver = SimulatedAnnealingPlacement(seed=7, iterations=6000)
    result = solver.solve(project.components, nets, project.board.width_mm, project.board.height_mm)

    assert set(result.positions.keys()) == {c.id for c in project.components}

    centers = np.array([result.positions[c.id][:2] for c in project.components])
    sizes = np.array([(c.width_mm, c.height_mm) for c in project.components])
    angles = np.array([result.positions[c.id][2] for c in project.components])
    boxes = to_aabb(centers, sizes, angles)

    overlap = total_overlap_area(boxes)
    oob = out_of_bounds_area(boxes, project.board.width_mm, project.board.height_mm)

    print(f"\nfinal cost={result.cost:.2f} overlap={overlap:.3f} hpwl={result.hpwl:.2f} oob={oob:.3f}")
    for c in project.components:
        x, y, theta = result.positions[c.id]
        print(f"  {c.label:28s} center=({x:6.2f},{y:6.2f}) rot={math.degrees(theta):5.1f} deg")

    assert overlap < 1.0, "components should not meaningfully overlap after annealing"
    assert oob < 1.0, "components should stay on the board"
