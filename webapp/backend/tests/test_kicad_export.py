"""Validates the exported .kicad_pcb without a real KiCad install: this
sandbox has neither the KiCad GUI nor `kicad-cli`, so "opens cleanly in
KiCad with no DRC errors" can't literally be checked here. What *can* be
checked, and is checked below: the file is well-formed S-expression syntax
(round-tripped through the same sexpdata-based parser used for the real
KiCad schematics earlier in this project), and a set of structural
invariants a broken exporter would very likely violate (net table
completeness, pad/net consistency, coordinates on-board, no duplicate net
names). Passing this is necessary but not sufficient for "KiCad accepts
it" — treat it as a strong static check, not a substitute for opening the
file in KiCad at least once before relying on it.
"""

from __future__ import annotations

import os

import sexpdata
from sexpdata import Symbol

from webapp.backend.app.kicad_export import export_kicad_pcb
from webapp.backend.app.pipeline import prepare_pcb
from webapp.backend.tests.test_drone_example import build_drone_project

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_drone.kicad_pcb")


def _prepare_and_export():
    project = build_drone_project()
    result = prepare_pcb(project, max_attempts=5, base_seed=0)
    assert not result.routing.unrouted_nets, "fixture assumes a fully-routed board"
    export_kicad_pcb(project, result, OUTPUT_PATH)
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tree = sexpdata.loads(text)
    return project, result, text, tree


def _tag(node) -> str | None:
    return str(node[0]) if isinstance(node, list) and node and isinstance(node[0], Symbol) else None


def _children(node: list, name: str) -> list:
    return [n for n in node[1:] if _tag(n) == name]


def test_file_is_well_formed_sexpr():
    project, result, text, tree = _prepare_and_export()
    assert text.count("(") == text.count(")")
    assert _tag(tree) == "kicad_pcb"


def test_net_table_matches_result_nets_exactly():
    _, result, _, tree = _prepare_and_export()
    net_nodes = _children(tree, "net")
    declared = {(int(n[1]), str(n[2])) for n in net_nodes}

    expected = {(0, "")}
    for i, net in enumerate(result.nets, start=1):
        expected.add((i, net.name))
    assert declared == expected

    names = [name for _, name in declared if name]
    assert len(names) == len(set(names)), "net table must not contain duplicate names"


def test_footprint_and_pad_counts_match_components():
    project, result, _, tree = _prepare_and_export()
    footprints = _children(tree, "footprint")
    assert len(footprints) == len(project.components)

    by_label = {c.label: c for c in project.components}
    for fp in footprints:
        ref_prop = next(n for n in _children(fp, "property") if str(n[1]) == "Reference")
        label = str(ref_prop[2])
        component = by_label[label]
        pads = _children(fp, "pad")
        assert len(pads) == len(component.pins)


def test_every_pad_and_trace_net_is_declared():
    _, result, _, tree = _prepare_and_export()
    declared_ids = {int(n[1]) for n in _children(tree, "net")}

    for fp in _children(tree, "footprint"):
        for pad in _children(fp, "pad"):
            net_nodes = _children(pad, "net")
            for net_node in net_nodes:
                assert int(net_node[1]) in declared_ids

    for seg in _children(tree, "segment"):
        net_node = _children(seg, "net")[0]
        assert int(net_node[1]) in declared_ids
    for via in _children(tree, "via"):
        net_node = _children(via, "net")[0]
        assert int(net_node[1]) in declared_ids


def test_footprints_and_traces_stay_on_board():
    project, result, _, tree = _prepare_and_export()
    w, h = project.board.width_mm, project.board.height_mm
    margin = 20.0  # footprints can extend past their center by up to ~half their own size

    for fp in _children(tree, "footprint"):
        at = _children(fp, "at")[0]
        x, y = float(at[1]), float(at[2])
        assert -margin <= x <= w + margin
        assert -margin <= y <= h + margin

    for seg in _children(tree, "segment"):
        start = _children(seg, "start")[0]
        end = _children(seg, "end")[0]
        for node in (start, end):
            x, y = float(node[1]), float(node[2])
            assert -1.0 <= x <= w + 1.0
            assert -1.0 <= y <= h + 1.0
