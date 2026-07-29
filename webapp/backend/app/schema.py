"""Request/response data model for a user-authored PCB project: components
with custom-labeled pins, and a point-to-point connection table between
them (exactly the two-step input the web UI collects before "Prepare PCB").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PinFunction = Literal["power", "ground", "signal", "high_speed"]


class Pin(BaseModel):
    number: str = Field(..., description="Physical pin number/name on the component, e.g. '1' or 'A3'")
    label: str = Field(..., description="User-facing functional label, e.g. 'VCC', 'SCL', 'TX1'")
    function: PinFunction = "signal"


class Component(BaseModel):
    id: str = Field(..., description="Stable unique identifier for this component instance")
    label: str = Field(..., description="User-visible name, e.g. 'Flight Controller', 'ESC 1'")
    width_mm: float = Field(..., gt=0)
    height_mm: float = Field(..., gt=0)
    pins: list[Pin]

    @model_validator(mode="after")
    def _unique_pin_numbers(self) -> "Component":
        numbers = [p.number for p in self.pins]
        if len(numbers) != len(set(numbers)):
            dupes = {n for n in numbers if numbers.count(n) > 1}
            raise ValueError(f"component '{self.id}' has duplicate pin numbers: {sorted(dupes)}")
        return self


class ConnectionEndpoint(BaseModel):
    component_id: str
    pin_number: str


class Connection(BaseModel):
    source: ConnectionEndpoint
    target: ConnectionEndpoint


class BoardSpec(BaseModel):
    width_mm: float = 50.0
    height_mm: float = 50.0


class PcbProject(BaseModel):
    board: BoardSpec = BoardSpec()
    components: list[Component]
    connections: list[Connection]

    @model_validator(mode="after")
    def _connections_reference_real_pins(self) -> "PcbProject":
        valid_pins: dict[str, set[str]] = {c.id: {p.number for p in c.pins} for c in self.components}
        component_ids = set(valid_pins.keys())
        if len(component_ids) != len(self.components):
            raise ValueError("component ids must be unique")

        for conn in self.connections:
            for endpoint, side in ((conn.source, "source"), (conn.target, "target")):
                if endpoint.component_id not in component_ids:
                    raise ValueError(
                        f"connection {side} references unknown component '{endpoint.component_id}'"
                    )
                if endpoint.pin_number not in valid_pins[endpoint.component_id]:
                    raise ValueError(
                        f"connection {side} references unknown pin '{endpoint.pin_number}' "
                        f"on component '{endpoint.component_id}'"
                    )
        return self
