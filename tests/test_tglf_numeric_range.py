# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — TGLF numeric token range and execution custody.

"""Refuse malformed out-of-range decimal evidence without rejecting true zeros or subnormals."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from scpn_control.core.tglf_flux import TGLFFluxError, TGLFFluxSolver, read_tglf_fluxes

_FIXTURE = Path(__file__).parent / "data/tglf/default"


def _electron_token_case(directory: Path, token: str, column: int = 0) -> None:
    """Modify a real captured spectrum with one controlled token and a compatible rounded summary."""
    shutil.copytree(_FIXTURE, directory, dirs_exist_ok=True)
    path = directory / "out.tglf.sum_flux_spectrum"
    lines = path.read_text().splitlines()
    for index in range(2, 23):
        row = lines[index].split()
        row[column] = token if index == 2 else "0"
        lines[index] = " ".join(row)
    path.write_text("\n".join(lines) + "\n")
    if column != 3:
        summary = directory / "out.tglf.gbflux"
        coarse = summary.read_text().split()
        coarse[{0: 0, 1: 2, 2: 4, 4: 6}[column]] = "0.0000E+00"
        summary.write_text(" ".join(coarse) + "\n")


@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "1D-400", "-1D-400", "1e-99999999999999999999"])
@pytest.mark.parametrize("column", range(5))
def test_nonzero_out_of_range_spectral_token_is_rejected(tmp_path: Path, token: str, column: int) -> None:
    """Even a coarse zero summary cannot legitimize loss of a nonzero token in any spectral column."""
    _electron_token_case(tmp_path, token, column)
    with pytest.raises(TGLFFluxError, match="spectrum"):
        read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize("token", ["0", "-0.0", "0D-400", "-0D-400", "5e-324", "-5e-324", "5D-324", "-5D-324"])
def test_true_zero_and_signed_subnormal_tokens_are_admissible(tmp_path: Path, token: str) -> None:
    """Keep exact zeros and the smallest representable signed particle moment without a magnitude floor."""
    _electron_token_case(tmp_path, token)
    result = read_tglf_fluxes(tmp_path)
    assert result.particle_flux_gb[0] == float(token.replace("D", "E"))
    if token.startswith("-5"):
        assert result.particle_flux_gb[0] < 0
    elif token.startswith("5"):
        assert result.particle_flux_gb[0] > 0


@pytest.mark.parametrize("token", ["1e-400", "-1D-400"])
def test_nonzero_underflow_in_eigenvalue_spectrum_is_rejected(tmp_path: Path, token: str) -> None:
    """Growth/frequency evidence uses the same nonzero-preserving numeric decoding as flux moments."""
    shutil.copytree(_FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "out.tglf.eigenvalue_spectrum"
    lines = path.read_text().splitlines()
    row = lines[2].split()
    row[0] = token
    lines[2] = " ".join(row)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TGLFFluxError, match="parse"):
        read_tglf_fluxes(tmp_path)


@pytest.mark.parametrize("token", ["1e-400", "-1D-400"])
def test_corrupted_real_execution_never_gets_a_valid_receipt(tmp_path: Path, token: str) -> None:
    """A real provider followed by deliberate output corruption must retain a failed validation receipt."""
    binary = os.environ.get("SCPN_TGLF_BINARY")
    if not binary:
        pytest.skip("SCPN_TGLF_BINARY must name an actual configured GACODE launcher")
    environment = json.loads(os.environ.get("SCPN_TGLF_ENV_JSON", "{}"))
    launcher = tmp_path / "tglf_corruption_probe"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\nfrom pathlib import Path\n"
        f"code = subprocess.call([{binary!r}, '-e', '.'])\n"
        "if code: sys.exit(code)\n"
        "path = Path('out.tglf.sum_flux_spectrum')\n"
        "Path(str(path)+'.provider_original').write_bytes(path.read_bytes())\n"
        "lines = path.read_text().splitlines()\n"
        "nky = int(Path('out.tglf.ky_spectrum').read_text().splitlines()[1])\n"
        "for index in range(2, nky+2):\n"
        "    row = lines[index].split()\n"
        f"    row[0] = {token!r} if index == 2 else '0'\n"
        "    lines[index] = ' '.join(row)\n"
        "path.write_text('\\n'.join(lines)+'\\n')\n"
        "path = Path('out.tglf.gbflux')\n"
        "Path(str(path)+'.provider_original').write_bytes(path.read_bytes())\n"
        "tokens = path.read_text().split()\n"
        "tokens[0] = '0.0000E+00'\n"
        "path.write_text(' '.join(tokens)+'\\n')\n"
    )
    launcher.chmod(0o700)
    solver = TGLFFluxSolver(tmp_path / "runs", binary=str(launcher), environment=environment)
    with pytest.raises(TGLFFluxError, match="spectrum"):
        solver.run(_FIXTURE / "input.tglf")
    (run,) = solver.work_dir.iterdir()
    receipt = json.loads((run / "execution.json").read_text())
    assert receipt["returncode"] == 0 and not receipt["output_validated"]
    spectrum = run / "out.tglf.sum_flux_spectrum"
    assert receipt["output_sha256"][spectrum.name] == hashlib.sha256(spectrum.read_bytes()).hexdigest()
    assert spectrum.read_bytes() != (run / (spectrum.name + ".provider_original")).read_bytes()
    with pytest.raises(TGLFFluxError, match="retained artifacts"):
        read_tglf_fluxes(run)
