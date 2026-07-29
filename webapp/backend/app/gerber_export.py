"""Gerber/drill export via KiCad's own CLI.

Deliberately shells out to `kicad-cli` rather than generating Gerber (RS-274X)
or Excellon drill files by hand — those are their own file formats with
their own footguns (aperture macros, format specifiers, coordinate modes),
and KiCad's own plotting code already produces manufacturer-correct output
from the exact same `.kicad_pcb` `kicad_export.py` writes. This module's
job is just invoking that correctly and packaging the result.

Note: this sandbox has no KiCad installation, so `kicad-cli` is not on
PATH here and the actual subprocess calls below have not been run against
real KiCad in this environment — only the "kicad-cli is missing" failure
path has been exercised. `is_kicad_cli_available()` lets a caller check
before promising a user a Gerber download.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile

KICAD_CLI_MISSING_MESSAGE = (
    "kicad-cli not found on PATH. Gerber export requires a local KiCad 7+ "
    "installation (kicad-cli ships with KiCad since v7) on the machine "
    "running this backend."
)


def is_kicad_cli_available() -> bool:
    return shutil.which("kicad-cli") is not None


def _run(cmd: list[str], timeout_s: int = 120) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kicad-cli command failed ({' '.join(cmd)}):\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def export_gerbers(kicad_pcb_path: str, output_dir: str) -> str:
    """Runs `kicad-cli pcb export gerbers` and `... export drill` against
    `kicad_pcb_path`, zips everything written to `output_dir`, and returns
    the zip's path. Raises RuntimeError (with `KICAD_CLI_MISSING_MESSAGE`)
    if kicad-cli isn't installed, rather than letting a bare
    FileNotFoundError leak out of subprocess.run.
    """
    kicad_cli = shutil.which("kicad-cli")
    if kicad_cli is None:
        raise RuntimeError(KICAD_CLI_MISSING_MESSAGE)

    os.makedirs(output_dir, exist_ok=True)
    _run([kicad_cli, "pcb", "export", "gerbers", "-o", output_dir, kicad_pcb_path])
    _run([kicad_cli, "pcb", "export", "drill", "-o", output_dir, kicad_pcb_path])

    zip_path = output_dir.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                zf.write(full_path, arcname=os.path.relpath(full_path, output_dir))

    return zip_path
