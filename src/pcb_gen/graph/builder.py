"""Bipartite component/net graph, built from one or more parsed schematics.

Nodes come in two kinds, distinguished by the "kind" node attribute:
  - "component": one physical part (merged across all its unit placements)
  - "net": one electrical net

Edges connect a component to every net it has a pin on, tagged with the
pin number. Nets aren't necessarily 2-pin, so this bipartite form (rather
than a component-component graph) avoids losing multi-pin nets to an
arbitrary clique expansion.
"""

from __future__ import annotations

import networkx as nx

from ..parser.model import Schematic
from .netlist import Net, build_netlist


def build_graph(schematics: list[Schematic]) -> nx.Graph:
    graph = nx.Graph()

    for file_idx, sch in enumerate(schematics):
        for comp in sch.components:
            node_id = ("component", file_idx, comp.reference)
            graph.add_node(
                node_id,
                kind="component",
                reference=comp.reference,
                value=comp.value,
                lib_id=comp.lib_id,
                footprint=comp.footprint,
                pos=comp.pos,
                angle=comp.angle,
                pin_count=len(comp.pin_positions),
                is_power_symbol=comp.reference.startswith("#PWR"),
            )

    nets: list[Net] = build_netlist(schematics)
    for i, net in enumerate(nets):
        net_id = ("net", net.name, i)
        graph.add_node(net_id, kind="net", name=net.name, pin_count=len(net.pins))
        for pin in net.pins:
            comp_id = ("component", pin.file_idx, pin.reference)
            if comp_id not in graph:
                continue
            graph.add_edge(comp_id, net_id, pin_number=pin.pin_number, pos=pin.pos)

    return graph
