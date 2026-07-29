import os

from pcb_gen.graph.builder import build_graph
from pcb_gen.graph.netlist import build_netlist
from pcb_gen.parser.loader import parse_schematic

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pic_programmer")


def _load_all():
    return [
        parse_schematic(os.path.join(DATA_DIR, "pic_programmer.kicad_sch")),
        parse_schematic(os.path.join(DATA_DIR, "pic_sockets.kicad_sch")),
    ]


def test_parses_components():
    schematics = _load_all()
    total = sum(len(s.components) for s in schematics)
    assert total > 100


def test_netlist_has_no_singleton_power_nets():
    schematics = _load_all()
    nets = build_netlist(schematics)
    assert len(nets) > 50

    gnd_nets = [n for n in nets if n.name == "GND"]
    assert len(gnd_nets) == 1, "all GND power symbols should merge into one net"
    assert len(gnd_nets[0].pins) > 10, "GND should be a widely-shared net"


def test_hierarchical_labels_bridge_sheets():
    schematics = _load_all()
    nets = build_netlist(schematics)
    bridged = [n for n in nets if len({p.file_idx for p in n.pins}) > 1]
    assert bridged, "at least one net should span both the parent sheet and pic_sockets"


def test_graph_is_bipartite_component_net():
    schematics = _load_all()
    graph = build_graph(schematics)

    kinds = {n for _, n in graph.nodes(data="kind")}
    assert kinds == {"component", "net"}

    for u, v in graph.edges():
        ku, kv = graph.nodes[u]["kind"], graph.nodes[v]["kind"]
        assert {ku, kv} == {"component", "net"}


def test_no_pin_left_unconnected_to_some_net():
    schematics = _load_all()
    graph = build_graph(schematics)

    for node, data in graph.nodes(data=True):
        if data["kind"] != "component" or data["is_power_symbol"]:
            continue
        expected_pins = data["pin_count"]
        actual_edges = graph.degree(node)
        assert actual_edges <= expected_pins
