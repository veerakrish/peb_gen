"""Synthetic drone-module netlist generator.

Produces JSON graphs simulating a drone's electronics stack (flight
controller MCU, ESCs, GPS, power distribution board, IMU, RX) with
randomized footprint dimensions, pin assignments, and inter-module nets.

The JSON schema is intentionally framework-agnostic (plain dicts/lists) so
it has no dependency on PyTorch/PyG; `src/utils/graph_io.py` converts it
into a PyG `Data` graph for the Stage 1 encoder.

Schema::

    {
      "board": {"width_mm": .., "height_mm": ..},
      "components": [
        {"id": "MCU_0", "category": "MCU", "width_mm": .., "height_mm": ..,
         "pins": [{"number": "1", "type": "power"}, ...]},
        ...
      ],
      "nets": [
        {"name": "VCC", "type": "power",
         "pins": [{"component_id": "MCU_0", "pin_number": "1"}, ...]},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import yaml


@dataclass
class ModuleTemplate:
    category: str
    width_range: tuple[float, float]
    height_range: tuple[float, float]
    pin_count_range: tuple[int, int]
    count_range: tuple[int, int]


def _load_catalog(dataset_config: dict) -> list[ModuleTemplate]:
    templates = []
    for entry in dataset_config["synthetic"]["module_catalog"]:
        templates.append(
            ModuleTemplate(
                category=entry["category"],
                width_range=tuple(entry["width_mm"]),
                height_range=tuple(entry["height_mm"]),
                pin_count_range=tuple(entry["pin_count"]),
                count_range=tuple(entry["count"]),
            )
        )
    return templates


def _sample_pin_type(rng: random.Random, weights: dict[str, float]) -> str:
    types = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(types, weights=probs, k=1)[0]


def _instantiate_component(
    rng: random.Random, template: ModuleTemplate, instance_idx: int, pin_type_weights: dict[str, float]
) -> dict:
    width = round(rng.uniform(*template.width_range), 2)
    height = round(rng.uniform(*template.height_range), 2)
    pin_count = rng.randint(*template.pin_count_range)

    pins = []
    # every non-passive, non-connector module gets at least one dedicated
    # power and ground pin so the shared-rail net wiring below is meaningful
    guaranteed_types = []
    if template.category not in ("PASSIVE",):
        guaranteed_types = ["power", "ground"]

    for i in range(pin_count):
        if i < len(guaranteed_types):
            pin_type = guaranteed_types[i]
        else:
            pin_type = _sample_pin_type(rng, pin_type_weights)
        pins.append({"number": str(i + 1), "type": pin_type})

    return {
        "id": f"{template.category}_{instance_idx}",
        "category": template.category,
        "width_mm": width,
        "height_mm": height,
        "pins": pins,
    }


def _generate_components(rng: random.Random, templates: list[ModuleTemplate], pin_type_weights: dict) -> list[dict]:
    components = []
    for template in templates:
        n = rng.randint(*template.count_range)
        for i in range(n):
            components.append(_instantiate_component(rng, template, i, pin_type_weights))
    return components


def _generate_nets(rng: random.Random, components: list[dict], net_cfg: dict) -> list[dict]:
    nets: list[dict] = []

    def pins_of_type(pin_type: str) -> list[tuple[str, str]]:
        out = []
        for comp in components:
            for pin in comp["pins"]:
                if pin["type"] == pin_type:
                    out.append((comp["id"], pin["number"]))
        return out

    # shared power/ground rails, like a real PDB distributing VCC/GND to every module
    for net_name, pin_type in (("VCC", "power"), ("GND", "ground")):
        members = pins_of_type(pin_type)
        if members:
            nets.append(
                {
                    "name": net_name,
                    "type": pin_type,
                    "pins": [{"component_id": c, "pin_number": p} for c, p in members],
                }
            )

    # remaining signal/high_speed pins: randomly paired into small point-to-point
    # or small-fanout nets, a handful per module
    high_speed_pins = set(pins_of_type("high_speed"))
    free_pins = pins_of_type("signal") + list(high_speed_pins)
    rng.shuffle(free_pins)

    max_fanout = net_cfg.get("max_net_fanout", 4)
    lo, hi = net_cfg.get("extra_random_nets_per_module", [1, 3])
    target_net_count = max(1, int(len(components) * rng.randint(lo, hi) / 2))

    idx = 0
    net_counter = 0
    while idx < len(free_pins) and net_counter < target_net_count:
        fanout = rng.randint(2, max_fanout)
        chunk = free_pins[idx : idx + fanout]
        idx += fanout
        if len(chunk) < 2:
            break
        pin_type = "high_speed" if any(p in high_speed_pins for p in chunk) else "signal"
        nets.append(
            {
                "name": f"NET_{net_counter}",
                "type": pin_type,
                "pins": [{"component_id": c, "pin_number": p} for c, p in chunk],
            }
        )
        net_counter += 1

    return nets


def generate_graph(rng: random.Random, dataset_config: dict, board_config: dict) -> dict:
    templates = _load_catalog(dataset_config)
    pin_type_weights = dataset_config["synthetic"]["pin_type_weights"]
    net_cfg = dataset_config["synthetic"]["net_generation"]

    components = _generate_components(rng, templates, pin_type_weights)
    nets = _generate_nets(rng, components, net_cfg)

    return {
        "board": {
            "width_mm": board_config["width_mm"],
            "height_mm": board_config["height_mm"],
        },
        "components": components,
        "nets": nets,
    }


def generate_dataset(dataset_config: dict, board_config: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [generate_graph(rng, dataset_config, board_config) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="config/dataset_config.yaml")
    parser.add_argument("--default-config", default="config/default_config.yaml")
    parser.add_argument("--num-graphs", type=int, default=None, help="override synthetic.num_graphs")
    parser.add_argument("--output-dir", default=None, help="override paths.output_dir")
    args = parser.parse_args()

    with open(args.dataset_config, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    with open(args.default_config, "r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f)

    n = args.num_graphs or dataset_config["synthetic"]["num_graphs"]
    seed = dataset_config["synthetic"].get("seed", 42)
    output_dir = args.output_dir or dataset_config["paths"]["output_dir"]
    board_config = default_config["board"]

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(seed)
    for i in range(n):
        graph = generate_graph(rng, dataset_config, board_config)
        out_path = os.path.join(output_dir, f"pcb_{i:05d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    print(f"Wrote {n} synthetic netlist graphs to {output_dir}")


if __name__ == "__main__":
    main()
