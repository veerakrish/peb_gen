import os

from data.real_netlist_harvester import harvest, infer_category, infer_net_type, infer_pin_function
from src.pcb_gen.parser.loader import parse_schematic

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pic_programmer")


def test_infer_category_from_reference_prefix():
    assert infer_category("U2") == "IC"
    assert infer_category("R14") == "PASSIVE"
    assert infer_category("C3") == "PASSIVE"
    assert infer_category("Q1") == "TRANSISTOR"
    assert infer_category("D5") == "DIODE"
    assert infer_category("J1") == "CONNECTOR"
    assert infer_category("Y1") == "CRYSTAL"
    assert infer_category("ZZZ99") == "PASSIVE"  # unknown prefix falls back to a safe default


def test_infer_pin_function_prefers_name_over_generic_electrical_type():
    assert infer_pin_function("GND", "passive") == "ground"
    assert infer_pin_function("VCC", "passive") == "power"
    assert infer_pin_function("power_in_but_unnamed", "power_in") == "power"
    assert infer_pin_function("SCL", "bidirectional") == "high_speed"
    assert infer_pin_function("3", "passive") == "signal"  # bare pin number, no semantic name


def test_infer_net_type_uses_net_name_not_just_pin_majority():
    # a GND net dominated by anonymous capacitor pins (function "signal",
    # since passive pins carry no semantic name) must still resolve to
    # "ground" from the net's own name — this was a real bug caught while
    # harvesting pic_programmer (see git history / conversation).
    mostly_signal_pins = ["signal"] * 10 + ["ground"] * 2
    assert infer_net_type("GND", mostly_signal_pins) == "ground"
    assert infer_net_type("VCC", mostly_signal_pins) == "power"
    assert infer_net_type("SPI_CLK", []) == "high_speed"
    assert infer_net_type("Net-(U1-Pad3)", ["signal", "signal"]) == "signal"


def test_harvest_pic_programmer_end_to_end():
    schematics = [
        parse_schematic(os.path.join(DATA_DIR, "pic_programmer.kicad_sch")),
        parse_schematic(os.path.join(DATA_DIR, "pic_sockets.kicad_sch")),
    ]
    graph = harvest(schematics)

    assert len(graph["components"]) > 40
    assert len(graph["nets"]) > 10
    assert graph["board"]["width_mm"] > 0 and graph["board"]["height_mm"] > 0

    # no power-flag/ERC-annotation symbols leaked through as fake components
    assert all(not c["id"].startswith("#") for c in graph["components"])
    # every component has at least one real pin
    assert all(len(c["pins"]) > 0 for c in graph["components"])

    by_name = {n["name"]: n for n in graph["nets"]}
    assert by_name["GND"]["type"] == "ground"
    assert by_name["VCC"]["type"] == "power"
    assert len(by_name["GND"]["pins"]) > 10  # GND should be a widely-shared net

    # every net pin must reference a component that's actually in the output
    component_ids = {c["id"] for c in graph["components"]}
    for net in graph["nets"]:
        for pin in net["pins"]:
            assert pin["component_id"] in component_ids
