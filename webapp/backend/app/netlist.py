"""Build multi-pin nets from the user's point-to-point connection table.

The web UI collects connections as pairwise rows (source component/pin ->
target component/pin), same as wiring up a breadboard one jumper at a
time. Several rows can chain into one electrical net (A-1 -> B-2, B-2 ->
C-1 means A-1, B-2, and C-1 are all one net) via union-find, the same
approach used for real KiCad wire connectivity in
`src/pcb_gen/graph/netlist.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import PcbProject

PinRef = tuple[str, str]  # (component_id, pin_number)


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
class Net:
    name: str
    pins: list[PinRef] = field(default_factory=list)

    def component_ids(self) -> list[str]:
        seen: list[str] = []
        seen_set: set[str] = set()
        for cid, _ in self.pins:
            if cid not in seen_set:
                seen_set.add(cid)
                seen.append(cid)
        return seen


def _net_name(pins: list[PinRef], pin_labels: dict[PinRef, str]) -> str:
    """The net's display name is whichever pin label is most common among
    its members (e.g. a rail wired as "5V_OUT" on the PDB but "VCC" on
    every consumer module still gets named "VCC", the majority label) —
    real designs rarely have every pin on a net spelled identically, so
    requiring unanimous agreement would leave most real nets with an
    unhelpful auto-generated name instead.
    """
    labels = [pin_labels[p] for p in pins]
    if not labels:
        return "NET"
    most_common_label, _ = Counter(labels).most_common(1)[0]
    return most_common_label


def build_nets(project: PcbProject) -> list[Net]:
    uf = UnionFind()
    pin_labels: dict[PinRef, str] = {}
    for comp in project.components:
        for pin in comp.pins:
            pin_labels[(comp.id, pin.number)] = pin.label

    for conn in project.connections:
        source = (conn.source.component_id, conn.source.pin_number)
        target = (conn.target.component_id, conn.target.pin_number)
        uf.union(source, target)

    referenced_pins: list[PinRef] = []
    seen: set[PinRef] = set()
    for conn in project.connections:
        for endpoint in (conn.source, conn.target):
            ref = (endpoint.component_id, endpoint.pin_number)
            if ref not in seen:
                seen.add(ref)
                referenced_pins.append(ref)

    groups: dict = {}
    for pin_ref in referenced_pins:
        root = uf.find(pin_ref)
        groups.setdefault(root, []).append(pin_ref)

    nets = []
    for pins in groups.values():
        pins_sorted = sorted(pins)
        nets.append(Net(name=_net_name(pins_sorted, pin_labels), pins=pins_sorted))

    nets.sort(key=lambda n: n.name)
    _dedupe_net_names(nets)
    return nets


def _dedupe_net_names(nets: list[Net]) -> None:
    """Two electrically distinct nets can legitimately get the same
    majority-vote display name (e.g. two different motor-signal nets each
    named "SIGNAL"). Every downstream consumer — the router's segment
    labels, the KiCad net table — treats `Net.name` as a unique key, so a
    collision here would silently merge unrelated nets. Mutates in place,
    appending `_2`, `_3`, ... to repeats; the first occurrence keeps its
    plain name.
    """
    seen: dict[str, int] = {}
    for net in nets:
        if net.name in seen:
            seen[net.name] += 1
            net.name = f"{net.name}_{seen[net.name]}"
        else:
            seen[net.name] = 1


def unconnected_pins(project: PcbProject) -> list[PinRef]:
    """Pins that never appear in any connection row — useful for the UI to
    flag as "did you forget to wire this?" without blocking generation.
    """
    referenced: set[PinRef] = set()
    for conn in project.connections:
        referenced.add((conn.source.component_id, conn.source.pin_number))
        referenced.add((conn.target.component_id, conn.target.pin_number))

    unconnected = []
    for comp in project.components:
        for pin in comp.pins:
            ref = (comp.id, pin.number)
            if ref not in referenced:
                unconnected.append(ref)
    return unconnected
