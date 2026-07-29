"""Electrical connectivity extraction across one or more parsed schematic
sheets.

KiCad schematics encode connectivity geometrically: two things are on the
same net if they share a coordinate (wire endpoints, pins, junctions,
labels), plus a few name-based rules (global labels, power-symbol values,
and hierarchical-label / sheet-pin pairs connect points that aren't
geometrically coincident at all, possibly across files). We build one
union-find over "coordinate keys" scoped per source file, then apply the
name-based merges on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..parser.model import Schematic


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


@dataclass
class PinRef:
    file_idx: int
    reference: str
    pin_number: str
    pos: tuple[float, float]


@dataclass
class Net:
    name: str
    pins: list[PinRef] = field(default_factory=list)


def _coord_key(file_idx: int, pos: tuple[float, float]):
    return (file_idx, round(pos[0], 2), round(pos[1], 2))


def build_netlist(schematics: list[Schematic]) -> list[Net]:
    uf = UnionFind()

    # wire segments: union consecutive points within the same file
    for file_idx, sch in enumerate(schematics):
        for wire in sch.wires:
            for a, b in zip(wire.points, wire.points[1:]):
                uf.union(_coord_key(file_idx, a), _coord_key(file_idx, b))

    # gather all pin entries (these are also connection points for free,
    # via sharing a coordinate key with wires/junctions/labels)
    pin_entries: list[PinRef] = []
    for file_idx, sch in enumerate(schematics):
        for comp in sch.components:
            for num, pos in comp.pin_positions.items():
                pin_entries.append(PinRef(file_idx, comp.reference, num, pos))

    # local labels: sheet-scoped merge by text
    local_groups: dict[tuple[int, str], list] = {}
    for file_idx, sch in enumerate(schematics):
        for label in sch.labels:
            if label.kind != "local":
                continue
            local_groups.setdefault((file_idx, label.text), []).append(
                _coord_key(file_idx, label.pos)
            )
    for keys in local_groups.values():
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    # global labels: merge by text across all files
    global_groups: dict[str, list] = {}
    for file_idx, sch in enumerate(schematics):
        for label in sch.labels:
            if label.kind != "global":
                continue
            global_groups.setdefault(label.text, []).append(_coord_key(file_idx, label.pos))
    for keys in global_groups.values():
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    # hierarchical labels (child sheets) <-> sheet pins (parent sheets), by name
    hier_groups: dict[str, list] = {}
    for file_idx, sch in enumerate(schematics):
        for label in sch.labels:
            if label.kind != "hierarchical":
                continue
            hier_groups.setdefault(label.text, []).append(_coord_key(file_idx, label.pos))
        for sheet in sch.sheets:
            for pin in sheet.pins:
                hier_groups.setdefault(pin.text, []).append(_coord_key(file_idx, pin.pos))
    for keys in hier_groups.values():
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    # power symbols (#PWR... reference): globally connected by their Value (net name)
    power_groups: dict[str, list] = {}
    for file_idx, sch in enumerate(schematics):
        for comp in sch.components:
            if not comp.reference.startswith("#PWR"):
                continue
            for pos in comp.pin_positions.values():
                power_groups.setdefault(comp.value, []).append(_coord_key(file_idx, pos))
    for keys in power_groups.values():
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)

    # group pins by their union-find root
    root_to_pins: dict = {}
    for pin in pin_entries:
        root = uf.find(_coord_key(pin.file_idx, pin.pos))
        root_to_pins.setdefault(root, []).append(pin)

    # name each net: prefer a label/power name found at that root, else auto-name
    def name_lookup() -> dict:
        lookup = {}
        for (file_idx, text), keys in local_groups.items():
            for k in keys:
                lookup[uf.find(k)] = text
        for text, keys in global_groups.items():
            for k in keys:
                lookup[uf.find(k)] = text
        for text, keys in hier_groups.items():
            for k in keys:
                lookup[uf.find(k)] = text
        for text, keys in power_groups.items():
            for k in keys:
                lookup[uf.find(k)] = text
        return lookup

    root_to_name = name_lookup()

    nets = []
    for root, pins in root_to_pins.items():
        pins_sorted = sorted(pins, key=lambda p: (p.file_idx, p.reference, p.pin_number))
        name = root_to_name.get(root)
        if name is None:
            first = pins_sorted[0]
            name = f"Net-({first.reference}-Pad{first.pin_number})"
        nets.append(Net(name=name, pins=pins_sorted))

    return nets
