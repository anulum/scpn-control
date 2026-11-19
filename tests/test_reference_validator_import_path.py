# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Import-path effects of the reference validators.
"""The reference validators change the import path only when run as scripts.

Imported as ``validation.<module>`` by tests, tools or other validators, a
validator must leave the caller's import path as it found it. Run as a script
from a path without the checkout, it must put the checkout (and, for two of
them, the sources) first so its ``validation.*`` imports resolve. Both checks
strip the checkout and its sources first; with them already present an
unguarded "insert if absent" would also leave the path unchanged.
"""

from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Module name -> import-path entries a script run puts first, in order.
SCRIPT_PREFIX: dict[str, tuple[Path, ...]] = {
    **{
        name: (ROOT,)
        for name in (
            "validate_blob_transport_reference",
            "validate_burn_reference",
            "validate_current_drive_reference",
            "validate_density_reference",
            "validate_digital_twin_reference",
            "validate_disruption_reference",
            "validate_elm_reference",
            "validate_eped_reference",
            "validate_free_boundary_reference",
            "validate_gk_crosscode",
            "validate_marfe_reference",
            "validate_mu_synthesis_reference",
            "validate_neural_equilibrium_reference",
            "validate_neural_transport_reference",
            "validate_neural_turbulence_reference",
            "validate_ntm_reference",
            "validate_orbit_reference",
            "validate_rzip_reference",
            "validate_soc_reference",
            "validate_static_mu_analysis_reference",
            "validate_uncertainty_reference",
            "validate_vmec_reference",
            "validate_volt_second_reference",
        )
    },
    "validate_gk_geometry_reference": (ROOT / "src", ROOT),
    "validate_gk_species_reference": (ROOT, ROOT / "src"),
}


def _stripped() -> list[str]:
    """Return the current import path without the checkout and its sources."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() not in {ROOT, ROOT / "src"}]


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_package_import_leaves_the_import_path_unchanged(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading the validator as a package module inserts nothing into a path without the checkout."""
    module = importlib.import_module(f"validation.{name}")
    stripped = _stripped()
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(module)
    assert sys.path == stripped


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_puts_the_checkout_first(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script from a path without the checkout, the validator inserts its entries and shows its help."""
    script = ROOT / "validation" / f"{name}.py"
    monkeypatch.setattr(sys, "path", _stripped())
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
    expected = [str(entry) for entry in SCRIPT_PREFIX[name]]
    assert sys.path[: len(expected)] == expected


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_with_the_checkout_present_adds_no_duplicate(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with the checkout and sources already importable, the validator adds no second entry."""
    script = ROOT / "validation" / f"{name}.py"
    present = [str(ROOT), str(ROOT / "src"), *_stripped()]
    monkeypatch.setattr(sys, "path", list(present))
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert sys.path == present


def test_every_reference_validator_is_listed() -> None:
    """The table covers every Git-tracked reference validator, so a new one cannot skip these checks."""
    listing = subprocess.run(
        ["git", "ls-files", "validation/validate_*_reference.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    tracked = {Path(path).stem for path in listing}
    assert tracked, "the tracked-file listing must not be empty"
    assert tracked <= set(SCRIPT_PREFIX)
