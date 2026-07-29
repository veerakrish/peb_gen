"""FastAPI service layer wrapping the place/route/export pipeline for the
React frontend: submit a project, get back a renderable layout plus a
session id for downloading the generated files.

Session state is an in-memory dict — good enough for a single-process dev
backend, not for multi-worker/production deployment (a restart loses every
session, and `download_gerber`'s lazy generation would race under
concurrent workers). Swapping in a real store (Redis, a DB row, even just
a per-session JSON file) is the natural next step before this goes beyond
local use.
"""

from __future__ import annotations

import math
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .gerber_export import KICAD_CLI_MISSING_MESSAGE, export_gerbers
from .kicad_export import export_kicad_pcb
from .pipeline import PcbResult, prepare_pcb
from .schema import PcbProject

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")

app = FastAPI(title="pcb_gen API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict] = {}


class ComponentPlacement(BaseModel):
    component_id: str
    x_mm: float
    y_mm: float
    rotation_deg: float


class RoutedSegmentOut(BaseModel):
    net_name: str
    layer: str
    points: list[tuple[float, float]]


class ViaOut(BaseModel):
    net_name: str
    position: tuple[float, float]


class Metrics(BaseModel):
    attempts_tried: int
    unrouted_nets_count: int
    unrouted_nets: list[str]
    total_trace_length_mm: float
    via_count: int
    overlap_area_mm2: float
    out_of_bounds_area_mm2: float
    unconnected_pin_count: int


class PreparePcbResponse(BaseModel):
    session_id: str
    board_width_mm: float
    board_height_mm: float
    placement: list[ComponentPlacement]
    segments: list[RoutedSegmentOut]
    vias: list[ViaOut]
    metrics: Metrics


def _to_response(project: PcbProject, result: PcbResult, session_id: str) -> PreparePcbResponse:
    placement = [
        ComponentPlacement(
            component_id=cid,
            x_mm=x,
            y_mm=y,
            rotation_deg=math.degrees(theta) % 360.0,
        )
        for cid, (x, y, theta) in result.placement.positions.items()
    ]
    segments = [
        RoutedSegmentOut(net_name=s.net_name, layer=s.layer, points=s.points) for s in result.routing.segments
    ]
    vias = [ViaOut(net_name=v.net_name, position=v.position) for v in result.routing.vias]
    metrics = Metrics(
        attempts_tried=result.attempts_tried,
        unrouted_nets_count=len(result.routing.unrouted_nets),
        unrouted_nets=result.routing.unrouted_nets,
        total_trace_length_mm=result.routing.total_trace_length_mm,
        via_count=len(result.routing.vias),
        overlap_area_mm2=result.placement.overlap_area,
        out_of_bounds_area_mm2=result.placement.out_of_bounds_area,
        unconnected_pin_count=len(result.unconnected_pins),
    )
    return PreparePcbResponse(
        session_id=session_id,
        board_width_mm=project.board.width_mm,
        board_height_mm=project.board.height_mm,
        placement=placement,
        segments=segments,
        vias=vias,
        metrics=metrics,
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/prepare-pcb", response_model=PreparePcbResponse)
def prepare_pcb_endpoint(project: PcbProject) -> PreparePcbResponse:
    result = prepare_pcb(project)

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(OUTPUT_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)

    kicad_path = os.path.join(session_dir, "board.kicad_pcb")
    export_kicad_pcb(project, result, kicad_path)

    _sessions[session_id] = {
        "project": project,
        "result": result,
        "kicad_path": kicad_path,
        "gerber_zip_path": None,
    }

    return _to_response(project, result, session_id)


def _get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session_id '{session_id}'")
    return session


@app.get("/api/download/kicad/{session_id}")
def download_kicad(session_id: str) -> FileResponse:
    session = _get_session(session_id)
    return FileResponse(
        session["kicad_path"],
        media_type="application/octet-stream",
        filename="board.kicad_pcb",
    )


@app.get("/api/download/gerber/{session_id}")
def download_gerber(session_id: str) -> FileResponse:
    session = _get_session(session_id)

    if session["gerber_zip_path"] is None:
        gerber_dir = os.path.join(OUTPUT_ROOT, session_id, "gerbers")
        try:
            zip_path = export_gerbers(session["kicad_path"], gerber_dir)
        except RuntimeError as exc:
            status = 503 if str(exc).startswith(KICAD_CLI_MISSING_MESSAGE) else 500
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        session["gerber_zip_path"] = zip_path

    return FileResponse(
        session["gerber_zip_path"],
        media_type="application/zip",
        filename="gerbers.zip",
    )
