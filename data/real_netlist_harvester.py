"""Converts real open-source KiCad schematics into the same JSON netlist
schema `data/synthetic_generator.py` produces, so real and synthetic boards
can train Stage 1/Stage 2 together as one corpus.

Uses `src/pcb_gen`'s schematic parser and netlist builder (originally built
to parse real .kicad_sch files like demos/pic_programmer, see that
project's tests) rather than re-deriving connectivity here — this module's
only job is reshaping that already-correct output into the training
schema.

Two real gaps this fills in with heuristics, since a bare schematic parse
doesn't have them:
  - Physical footprint dimensions: `src/pcb_gen` only parses schematics,
    not the .kicad_mod footprint files a component's Footprint property
    points at (those live in KiCad's system libraries, not the project
    repo, so they're not reliably available at all). `estimate_size`
    approximates width/height from component category + pin count instead.
  - A "category" a Stage 1 categorical feature can use: inferred from the
    reference designator prefix (U/Q/D/R/C/J/... — a genuinely standard
    KiCad convention, not a guess specific to this codebase).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

from src.pcb_gen.graph.netlist import Net, build_netlist
from src.pcb_gen.parser.loader import parse_schematic
from src.pcb_gen.parser.model import Schematic

CATEGORY_BY_PREFIX = {
    "U": "IC",
    "IC": "IC",
    "Q": "TRANSISTOR",
    "D": "DIODE",
    "LED": "DIODE",
    "R": "PASSIVE",
    "C": "PASSIVE",
    "L": "PASSIVE",
    "FB": "PASSIVE",
    "J": "CONNECTOR",
    "P": "CONNECTOR",
    "CN": "CONNECTOR",
    "SW": "SWITCH",
    "S": "SWITCH",
    "Y": "CRYSTAL",
    "X": "CRYSTAL",
}
DEFAULT_CATEGORY = "PASSIVE"

# (base_width_mm, base_height_mm, growth_per_extra_pin) — rough real-world
# packages by category; deliberately coarse, not a footprint database.
SIZE_ESTIMATE_MM = {
    "IC": (5.0, 5.0, 0.6),
    "TRANSISTOR": (2.5, 2.5, 0.2),
    "DIODE": (2.5, 1.5, 0.1),
    "PASSIVE": (1.6, 0.8, 0.05),
    "CONNECTOR": (3.0, 3.0, 1.2),
    "SWITCH": (4.0, 4.0, 0.5),
    "CRYSTAL": (3.2, 2.5, 0.2),
}

GROUND_NAME_HINTS = ("GND", "VSS", "AGND", "DGND", "0V")
POWER_NAME_HINTS = ("VCC", "VDD", "V+", "VBAT", "3V3", "5V0", "5V", "12V", "PWR")
HIGH_SPEED_HINTS = ("CLK", "USB", "SCL", "SDA", "SPI", "MISO", "MOSI", "HDMI")


def infer_category(reference: str) -> str:
    prefix = "".join(ch for ch in reference if ch.isalpha())
    return CATEGORY_BY_PREFIX.get(prefix.upper(), DEFAULT_CATEGORY)


def infer_pin_function(pin_name: str, electrical_type: str) -> str:
    upper = (pin_name or "").upper()
    if any(h in upper for h in GROUND_NAME_HINTS):
        return "ground"
    if any(h in upper for h in POWER_NAME_HINTS) or electrical_type in ("power_in", "power_out"):
        return "power"
    if any(h in upper for h in HIGH_SPEED_HINTS):
        return "high_speed"
    return "signal"


def infer_net_type(net_name: str, pin_functions: list[str]) -> str:
    """Prefer the net's own name — for real schematics this comes from
    power-flag symbols or labels (see src/pcb_gen/graph/netlist.py), which
    is reliable ground truth. Falls back to a majority vote over member
    pins' functions, but that alone is weak here: passive components
    (resistors, capacitors) carry bare numeric pin labels with no semantic
    name, so on a real GND net dominated by decoupling-cap pins the vote
    can wrongly favor "signal" over the one or two named GND/VCC pins.
    """
    upper = net_name.upper()
    if any(h in upper for h in GROUND_NAME_HINTS):
        return "ground"
    if any(h in upper for h in POWER_NAME_HINTS):
        return "power"
    if any(h in upper for h in HIGH_SPEED_HINTS):
        return "high_speed"
    if pin_functions:
        return max(set(pin_functions), key=pin_functions.count)
    return "signal"


def estimate_size(category: str, pin_count: int) -> tuple[float, float]:
    base_w, base_h, growth = SIZE_ESTIMATE_MM.get(category, SIZE_ESTIMATE_MM[DEFAULT_CATEGORY])
    scale = 1.0 + growth * max(0, pin_count - 2)
    return round(base_w * scale, 2), round(base_h * scale, 2)


def estimate_board_size(components: list[dict], packing_slack: float = 2.5) -> tuple[float, float]:
    total_area = sum(c["width_mm"] * c["height_mm"] for c in components)
    side = math.sqrt(max(total_area, 1.0) * packing_slack)
    side = max(30.0, round(side, 1))
    return side, side


def harvest(schematics: list[Schematic]) -> dict:
    components: list[dict] = []
    comp_id_by_ref: dict[tuple[int, str], str] = {}
    pin_function_by_key: dict[tuple[str, str], str] = {}

    for file_idx, sch in enumerate(schematics):
        for comp in sch.components:
            if comp.reference.startswith("#"):
                continue  # power-flag / ERC annotation symbols, not real parts

            symbol_def = sch.symbol_defs.get(comp.lib_id)
            pins = []
            for number in comp.pin_positions:
                pin_meta = symbol_def.pins.get(number) if symbol_def else None
                pin_name = (pin_meta.name if pin_meta else "") or number
                electrical_type = pin_meta.electrical_type if pin_meta else "passive"
                function = infer_pin_function(pin_name, electrical_type)
                pins.append({"number": number, "label": pin_name, "function": function})

            if not pins:
                continue  # e.g. mounting holes: real, but nothing to route

            comp_id = comp.reference if file_idx == 0 else f"{comp.reference}_f{file_idx}"
            category = infer_category(comp.reference)
            width_mm, height_mm = estimate_size(category, len(pins))

            for pin in pins:
                pin_function_by_key[(comp_id, pin["number"])] = pin["function"]

            components.append(
                {
                    "id": comp_id,
                    "category": category,
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "pins": pins,
                }
            )
            comp_id_by_ref[(file_idx, comp.reference)] = comp_id

    nets_out = []
    for net in build_netlist(schematics):
        seen: set[tuple[str, str]] = set()
        pins_out = []
        for pin_ref in net.pins:
            comp_id = comp_id_by_ref.get((pin_ref.file_idx, pin_ref.reference))
            if comp_id is None:
                continue  # filtered-out power-flag symbol or a pinless component
            key = (comp_id, pin_ref.pin_number)
            if key in seen:
                continue
            seen.add(key)
            pins_out.append({"component_id": comp_id, "pin_number": pin_ref.pin_number})

        if len(pins_out) < 2:
            continue

        functions = [pin_function_by_key.get((p["component_id"], p["pin_number"]), "signal") for p in pins_out]
        net_type = infer_net_type(net.name, functions)
        nets_out.append({"name": net.name, "type": net_type, "pins": pins_out})

    board_width, board_height = estimate_board_size(components)
    return {
        "board": {"width_mm": board_width, "height_mm": board_height},
        "components": components,
        "nets": nets_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "schematics", nargs="+", help="one or more .kicad_sch files forming a single project (root sheet first)"
    )
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    sch_paths = []
    for pattern in args.schematics:
        sch_paths.extend(sorted(glob.glob(pattern)) or [pattern])

    schematics = [parse_schematic(p) for p in sch_paths]
    graph = harvest(schematics)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(
        f"wrote {args.output}: {len(graph['components'])} components, "
        f"{len(graph['nets'])} nets, board {graph['board']['width_mm']}x{graph['board']['height_mm']}mm"
    )


if __name__ == "__main__":
    main()
