from __future__ import annotations

from fastapi.testclient import TestClient

from webapp.backend.app.main import app
from webapp.backend.tests.test_drone_example import build_drone_project

client = TestClient(app)


def _drone_payload() -> dict:
    project = build_drone_project()
    return project.model_dump()


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_prepare_pcb_returns_full_layout():
    resp = client.post("/api/prepare-pcb", json=_drone_payload())
    assert resp.status_code == 200
    body = resp.json()

    assert body["metrics"]["unrouted_nets_count"] == 0
    assert len(body["placement"]) == 8
    assert len(body["segments"]) > 0
    assert body["board_width_mm"] == 80
    assert "session_id" in body


def test_prepare_pcb_rejects_invalid_connection():
    payload = _drone_payload()
    payload["connections"][0]["source"]["pin_number"] = "does-not-exist"
    resp = client.post("/api/prepare-pcb", json=payload)
    assert resp.status_code == 422


def test_download_kicad_file_after_prepare():
    resp = client.post("/api/prepare-pcb", json=_drone_payload())
    session_id = resp.json()["session_id"]

    download = client.get(f"/api/download/kicad/{session_id}")
    assert download.status_code == 200
    assert download.content.startswith(b"(kicad_pcb")


def test_download_kicad_unknown_session_404():
    resp = client.get("/api/download/kicad/not-a-real-session")
    assert resp.status_code == 404


def test_download_gerber_reports_kicad_cli_missing():
    resp = client.post("/api/prepare-pcb", json=_drone_payload())
    session_id = resp.json()["session_id"]

    gerber = client.get(f"/api/download/gerber/{session_id}")
    assert gerber.status_code == 503
    assert "kicad-cli" in gerber.json()["detail"]
