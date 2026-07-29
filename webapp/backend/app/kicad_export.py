"""Write a `prepare_pcb()` result out as a real `.kicad_pcb` file.

Targets the KiCad 8 board file format (`(version 20221018)` — the format
version KiCad 7/8 both write and any KiCad 8+ install opens without a
"newer file format" warning; picking an old-but-stable version number here
is deliberate, not an oversight).

Since none of our components come from a real footprint library, every
component gets a small auto-generated footprint: one through-hole pad per
pin at the same perimeter position `footprint.py` used for routing (so the
copper this module draws lines up exactly with the pads), plus a
silkscreen outline and reference/value text for readability in KiCad.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sexpdata import Symbol

from .footprint import assign_pin_local_positions
from .netlist import Net
from .pipeline import PcbResult
from .schema import Component, PcbProject

TRACE_WIDTH_MM = 0.25
PAD_DIAMETER_MM = 1.2
PAD_DRILL_MM = 0.7
VIA_DIAMETER_MM = 0.6
VIA_DRILL_MM = 0.3


def _uuid() -> str:
    return str(uuid.uuid4())


def _bare(name: str) -> Symbol:
    """Mark a token as an unquoted keyword (e.g. `thru_hole`, `solid`,
    `none`) rather than a quoted string value. KiCad's grammar isn't
    "first token bare, rest quoted" — plenty of non-leading fields are
    bareword enums too (`(stroke (type solid))`, `(fill none)`,
    `(pad "1" thru_hole circle ...)`), verified against real files in
    `data/pic_programmer/`. Every other Python str passed to `line`/`open`
    is treated as a genuine quoted string value.
    """
    return Symbol(name)


class SExprWriter:
    """Minimal S-expression pretty-printer matching KiCad's own tab-indented
    style (verified against real files in `data/pic_programmer/`). The
    leading token of every `line`/`open` call — the tag — is always
    emitted bare automatically; use `_bare(...)` for any other token that
    must stay unquoted.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    @staticmethod
    def _fmt(value, *, is_tag: bool = False) -> str:
        if is_tag or isinstance(value, Symbol):
            return str(value)
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, float):
            text = f"{value:.6f}".rstrip("0").rstrip(".")
            return text if text else "0"
        return str(value)

    def _format_parts(self, parts: tuple) -> str:
        formatted = [self._fmt(parts[0], is_tag=True)]
        formatted += [self._fmt(p) for p in parts[1:]]
        return " ".join(formatted)

    def line(self, *parts) -> None:
        self.lines.append("\t" * self.depth + f"({self._format_parts(parts)})")

    def open(self, *parts) -> None:
        self.lines.append("\t" * self.depth + f"({self._format_parts(parts)}")
        self.depth += 1

    def close(self) -> None:
        self.depth -= 1
        self.lines.append("\t" * self.depth + ")")

    def block(self, *parts):
        return _Block(self, parts)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class _Block:
    def __init__(self, writer: SExprWriter, parts: tuple):
        self.writer = writer
        self.parts = parts

    def __enter__(self) -> SExprWriter:
        self.writer.open(*self.parts)
        return self.writer

    def __exit__(self, *exc) -> None:
        self.writer.close()


@dataclass
class _NetTable:
    id_by_name: dict[str, int]
    id_by_pin: dict[tuple[str, str], int]

    def id_for_pin(self, component_id: str, pin_number: str) -> int:
        return self.id_by_pin.get((component_id, pin_number), 0)

    def name_for_id(self, net_id: int) -> str:
        for name, nid in self.id_by_name.items():
            if nid == net_id:
                return name
        return ""


def _build_net_table(nets: list[Net]) -> _NetTable:
    id_by_name: dict[str, int] = {"": 0}
    id_by_pin: dict[tuple[str, str], int] = {}
    for i, net in enumerate(nets, start=1):
        id_by_name[net.name] = i
        for pin_ref in net.pins:
            id_by_pin[pin_ref] = i
    return _NetTable(id_by_name=id_by_name, id_by_pin=id_by_pin)


def _write_header(w: SExprWriter) -> None:
    w.line("version", 20221018)
    w.line("generator", "pcb_gen")
    with w.block("general"):
        w.line("thickness", 1.6)
    w.line("paper", "A4")
    with w.block("layers"):
        w.line(0, "F.Cu", _bare("signal"))
        w.line(2, "B.Cu", _bare("signal"))
        w.line(9, "F.Adhes", _bare("user"), "F.Adhesive")
        w.line(11, "B.Adhes", _bare("user"), "B.Adhesive")
        w.line(13, "F.Paste", _bare("user"))
        w.line(15, "B.Paste", _bare("user"))
        w.line(5, "F.SilkS", _bare("user"), "F.Silkscreen")
        w.line(7, "B.SilkS", _bare("user"), "B.Silkscreen")
        w.line(1, "F.Mask", _bare("user"))
        w.line(3, "B.Mask", _bare("user"))
        w.line(25, "Edge.Cuts", _bare("user"))
        w.line(35, "F.Fab", _bare("user"))
        w.line(33, "B.Fab", _bare("user"))
    with w.block("setup"):
        w.line("pad_to_mask_clearance", 0)


def _write_board_outline(w: SExprWriter, width_mm: float, height_mm: float) -> None:
    with w.block("gr_rect"):
        w.line("start", 0.0, 0.0)
        w.line("end", width_mm, height_mm)
        with w.block("stroke"):
            w.line("width", 0.1)
            w.line("type", _bare("default"))
        w.line("fill", _bare("none"))
        w.line("layer", "Edge.Cuts")
        w.line("uuid", _uuid())


def _write_nets(w: SExprWriter, net_table: _NetTable) -> None:
    for name, net_id in sorted(net_table.id_by_name.items(), key=lambda kv: kv[1]):
        w.line("net", net_id, name)


def _write_footprint(w: SExprWriter, component: Component, position, net_table: _NetTable) -> None:
    x, y, theta_rad = position
    angle_deg = (theta_rad * 180.0 / 3.141592653589793) % 360.0
    fp_id = f"pcb_gen:{component.label.replace(' ', '_')}_{component.width_mm:g}x{component.height_mm:g}mm"
    local_positions = assign_pin_local_positions(component)

    with w.block("footprint", fp_id):
        w.line("layer", "F.Cu")
        w.line("uuid", _uuid())
        w.line("at", x, y, angle_deg)
        with w.block("property", "Reference", component.label):
            w.line("at", 0.0, -(component.height_mm / 2 + 1.0), 0.0)
            w.line("layer", "F.SilkS")
            w.line("uuid", _uuid())
            with w.block("effects"):
                with w.block("font"):
                    w.line("size", 1.0, 1.0)
        with w.block("fp_rect"):
            w.line("start", -component.width_mm / 2, -component.height_mm / 2)
            w.line("end", component.width_mm / 2, component.height_mm / 2)
            with w.block("stroke"):
                w.line("width", 0.15)
                w.line("type", _bare("default"))
            w.line("fill", _bare("none"))
            w.line("layer", "F.SilkS")
            w.line("uuid", _uuid())

        for pin in component.pins:
            lx, ly = local_positions[pin.number]
            net_id = net_table.id_for_pin(component.id, pin.number)
            net_name = net_table.name_for_id(net_id)
            shape = _bare("rect") if pin.number == component.pins[0].number else _bare("circle")
            with w.block("pad", pin.number, _bare("thru_hole"), shape):
                w.line("at", lx, ly)
                w.line("size", PAD_DIAMETER_MM, PAD_DIAMETER_MM)
                w.line("drill", PAD_DRILL_MM)
                w.line("layers", "*.Cu", "*.Mask")
                if net_id:
                    w.line("net", net_id, net_name)
                w.line("uuid", _uuid())


def _write_segments_and_vias(w: SExprWriter, result: PcbResult, net_table: _NetTable) -> None:
    for seg in result.routing.segments:
        net_id = net_table.id_by_name.get(seg.net_name, 0)
        for (x1, y1), (x2, y2) in zip(seg.points, seg.points[1:]):
            with w.block("segment"):
                w.line("start", x1, y1)
                w.line("end", x2, y2)
                w.line("width", TRACE_WIDTH_MM)
                w.line("layer", seg.layer)
                w.line("net", net_id)
                w.line("uuid", _uuid())

    for via in result.routing.vias:
        net_id = net_table.id_by_name.get(via.net_name, 0)
        x, y = via.position
        with w.block("via"):
            w.line("at", x, y)
            w.line("size", VIA_DIAMETER_MM)
            w.line("drill", VIA_DRILL_MM)
            w.line("layers", "F.Cu", "B.Cu")
            w.line("net", net_id)
            w.line("uuid", _uuid())


def export_kicad_pcb(project: PcbProject, result: PcbResult, output_path: str) -> str:
    net_table = _build_net_table(result.nets)

    w = SExprWriter()
    with w.block("kicad_pcb"):
        _write_header(w)
        _write_nets(w, net_table)
        _write_board_outline(w, project.board.width_mm, project.board.height_mm)
        for component in project.components:
            _write_footprint(w, component, result.placement.positions[component.id], net_table)
        _write_segments_and_vias(w, result, net_table)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(w.text())
    return output_path
