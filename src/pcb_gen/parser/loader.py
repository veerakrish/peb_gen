from __future__ import annotations

from . import sexpr
from .model import Component, Junction, Label, NoConnect, Pin, Schematic, SheetRef, SymbolDef, Wire
from .transform import transform_point


_KICAD_TEXT_ESCAPES = {
    "{slash}": "/",
    "{backslash}": "\\",
    "{dollar}": "$",
    "{colon}": ":",
    "{space}": " ",
    "{lt}": "<",
    "{gt}": ">",
    "{quote}": '"',
    "{brace}": "{",
}


def _unescape_kicad_text(text: str) -> str:
    """KiCad escapes a few characters (notably '/', its hierarchical-path
    separator) as e.g. "{slash}" inside label/net text. Different instances
    of what is logically the same net name can end up with different
    escaping depending on where they were typed, so normalize on read.
    """
    for escaped, literal in _KICAD_TEXT_ESCAPES.items():
        text = text.replace(escaped, literal)
    return text


def _get_at(node: list) -> tuple[float, float, float]:
    at = sexpr.child(node, "at")
    vals = sexpr.values(at)
    x, y = float(vals[0]), float(vals[1])
    angle = float(vals[2]) if len(vals) > 2 else 0.0
    return x, y, angle


def _get_property(node: list, name: str) -> str | None:
    for p in sexpr.children(node, "property"):
        vals = sexpr.values(p)
        if str(vals[0]) == name:
            return str(vals[1])
    return None


def _parse_lib_symbol_defs(root: list) -> dict[str, dict[str, dict[str, Pin]]]:
    """lib_id -> unit (str, "0" for common) -> pin number -> Pin (local coords)."""
    result: dict[str, dict[str, dict[str, Pin]]] = {}
    lib_symbols_node = sexpr.child(root, "lib_symbols")
    if lib_symbols_node is None:
        return result

    for top in sexpr.children(lib_symbols_node, "symbol"):
        lib_id = sexpr.value_str(top)
        units: dict[str, dict[str, Pin]] = {}
        for sub in sexpr.children(top, "symbol"):
            sub_name = sexpr.value_str(sub)
            unit = sub_name.split("_")[-2]
            for pin_node in sexpr.children(sub, "pin"):
                pvals = sexpr.values(pin_node)
                electrical_type = str(pvals[0])
                px, py, pangle = _get_at(pin_node)
                number_node = sexpr.child(pin_node, "number")
                number = sexpr.value_str(number_node)
                name_node = sexpr.child(pin_node, "name")
                name = sexpr.value_str(name_node) if name_node else ""
                units.setdefault(unit, {})[number] = Pin(
                    number=number,
                    name=name,
                    electrical_type=electrical_type,
                    local_pos=(px, py),
                    local_angle=pangle,
                )
        result[lib_id] = units
    return result


def _parse_components(
    root: list, symbol_defs: dict[str, dict[str, dict[str, Pin]]]
) -> list[Component]:
    by_reference: dict[str, Component] = {}

    for inst in sexpr.children(root, "symbol"):
        lib_id_node = sexpr.child(inst, "lib_id")
        if lib_id_node is None:
            continue
        lib_id = sexpr.value_str(lib_id_node)

        x, y, angle = _get_at(inst)
        mirror_node = sexpr.child(inst, "mirror")
        mirror = sexpr.value_str(mirror_node) if mirror_node else None
        unit_node = sexpr.child(inst, "unit")
        unit = sexpr.value_str(unit_node) if unit_node else "1"
        uuid_node = sexpr.child(inst, "uuid")
        uuid = sexpr.value_str(uuid_node) if uuid_node else ""

        reference = _get_property(inst, "Reference") or f"?{uuid}"
        value = _get_property(inst, "Value") or ""
        footprint = _get_property(inst, "Footprint") or ""

        pin_uuids = {}
        for pin_node in sexpr.children(inst, "pin"):
            num = sexpr.value_str(pin_node)
            puuid_node = sexpr.child(pin_node, "uuid")
            pin_uuids[num] = sexpr.value_str(puuid_node) if puuid_node else ""

        component = by_reference.get(reference)
        if component is None:
            component = Component(
                reference=reference,
                value=value,
                lib_id=lib_id,
                footprint=footprint,
                uuid=uuid,
                pos=(x, y),
                angle=angle,
                mirror=mirror,
                pin_uuids={},
            )
            by_reference[reference] = component
        elif not component.footprint and footprint:
            component.footprint = footprint

        units = symbol_defs.get(lib_id, {})
        visible_pins = dict(units.get("0", {}))
        visible_pins.update(units.get(unit, {}))
        for number, pin in visible_pins.items():
            global_pos = transform_point(pin.local_pos, angle, mirror, (x, y))
            component.pin_uuids[number] = pin_uuids.get(number, "")
            component.pin_positions[number] = global_pos

    return list(by_reference.values())


def _parse_wires(root: list) -> list[Wire]:
    wires = []
    for w in sexpr.children(root, "wire"):
        pts_node = sexpr.child(w, "pts")
        points = []
        for xy in sexpr.children(pts_node, "xy"):
            vals = sexpr.values(xy)
            points.append((float(vals[0]), float(vals[1])))
        wires.append(Wire(points=points))
    return wires


def _parse_junctions(root: list) -> list[tuple[float, float]]:
    out = []
    for j in sexpr.children(root, "junction"):
        x, y, _ = _get_at(j)
        out.append((x, y))
    return out


def _parse_no_connects(root: list) -> list[tuple[float, float]]:
    out = []
    for nc in sexpr.children(root, "no_connect"):
        x, y, _ = _get_at(nc)
        out.append((x, y))
    return out


def _parse_labels(root: list) -> list[Label]:
    labels = []
    tag_kind = [
        ("label", "local"),
        ("global_label", "global"),
        ("hierarchical_label", "hierarchical"),
    ]
    for tag_name, kind in tag_kind:
        for node in sexpr.children(root, tag_name):
            text = _unescape_kicad_text(sexpr.value_str(node))
            x, y, _ = _get_at(node)
            labels.append(Label(text=text, pos=(x, y), kind=kind))
    return labels


def _parse_sheets(root: list) -> list[SheetRef]:
    sheets = []
    for s in sexpr.children(root, "sheet"):
        name = _get_property(s, "Sheetname") or ""
        file = _get_property(s, "Sheetfile") or ""
        pins = []
        for pin_node in sexpr.children(s, "pin"):
            text = _unescape_kicad_text(sexpr.value_str(pin_node))
            x, y, _ = _get_at(pin_node)
            pins.append(Label(text=text, pos=(x, y), kind="sheet_pin"))
        sheets.append(SheetRef(name=name, file=file, pins=pins))
    return sheets


def parse_schematic(path: str) -> Schematic:
    root = sexpr.load(path)

    symbol_def_pins = _parse_lib_symbol_defs(root)
    symbol_defs = {}
    for lib_id, units in symbol_def_pins.items():
        merged: dict[str, Pin] = {}
        for unit_pins in units.values():
            merged.update(unit_pins)
        symbol_defs[lib_id] = SymbolDef(lib_id=lib_id, pins=merged)

    components = _parse_components(root, symbol_def_pins)

    sch = Schematic(path=path, symbol_defs=symbol_defs, components=components)
    sch.wires = _parse_wires(root)
    sch.junctions = [Junction(pos=p) for p in _parse_junctions(root)]
    sch.no_connects = [NoConnect(pos=p) for p in _parse_no_connects(root)]
    sch.labels = _parse_labels(root)
    sch.sheets = _parse_sheets(root)
    return sch
