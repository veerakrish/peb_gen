from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pin:
    number: str
    name: str
    electrical_type: str
    local_pos: tuple[float, float]
    local_angle: float


@dataclass
class SymbolDef:
    lib_id: str
    pins: dict[str, Pin] = field(default_factory=dict)


@dataclass
class Component:
    reference: str
    value: str
    lib_id: str
    footprint: str
    uuid: str
    pos: tuple[float, float]
    angle: float
    mirror: str | None  # None, "x", or "y"
    pin_uuids: dict[str, str] = field(default_factory=dict)  # pin number -> uuid
    pin_positions: dict[str, tuple[float, float]] = field(default_factory=dict)  # pin number -> global (x, y)


@dataclass
class Wire:
    points: list[tuple[float, float]]


@dataclass
class Junction:
    pos: tuple[float, float]


@dataclass
class Label:
    text: str
    pos: tuple[float, float]
    kind: str  # "local", "global", "hierarchical", "sheet_pin"


@dataclass
class NoConnect:
    pos: tuple[float, float]


@dataclass
class SheetRef:
    name: str
    file: str
    pins: list[Label]  # kind="sheet_pin"


@dataclass
class Schematic:
    path: str
    symbol_defs: dict[str, SymbolDef] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    no_connects: list[NoConnect] = field(default_factory=list)
    sheets: list[SheetRef] = field(default_factory=list)
