"""kicad-cli isn't installed in this sandbox, so only the
"cleanly reports unavailable" path can actually be exercised here — the
real subprocess/zip path needs a machine with KiCad installed.
"""

import pytest

from webapp.backend.app.gerber_export import export_gerbers, is_kicad_cli_available


def test_kicad_cli_reported_unavailable_in_this_environment():
    assert is_kicad_cli_available() is False


def test_export_gerbers_fails_clearly_without_kicad_cli(tmp_path):
    with pytest.raises(RuntimeError, match="kicad-cli not found"):
        export_gerbers(str(tmp_path / "board.kicad_pcb"), str(tmp_path / "gerbers"))
