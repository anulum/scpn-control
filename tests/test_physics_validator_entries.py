# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Script entries and import-path effects of the physics validators.
"""Every physics validator answers ``--help`` as a script and edits the import path only then.

The help run reaches each module's ``__main__`` line without running a
validation or writing a report. Seven validators put the checkout (and some
their sources) on the import path for a script run; imported as
``validation.<module>`` they must leave it alone. The import-path checks strip
the checkout and its sources first, because with them present an unguarded
"insert if absent" would also leave the path unchanged.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PHYSICS_VALIDATORS = (
    "validate_burn_control",
    "validate_current_drive",
    "validate_density_control",
    "validate_disruption_sequence",
    "validate_elm_peeling_ballooning",
    "validate_eped_pedestal",
    "validate_grad_shafranov_solovev",
    "validate_guiding_centre_conservation",
    "validate_h_infinity_control",
    "validate_halo_current",
    "validate_kuramoto_synchronisation",
    "validate_marfe_onset",
    "validate_mhd_stability",
    "validate_momentum_transport",
    "validate_mu_structured_singular_value",
    "validate_ntm_island_dynamics",
    "validate_runaway_electron",
    "validate_rwm_feedback",
    "validate_rzip_vertical_stability",
    "validate_sawtooth_kadomtsev",
    "validate_sol_two_point",
    "validate_stability_mhd_ballooning",
    "validate_transport_diffusion",
    "validate_vmec_lite_geometry",
    "validate_volt_second",
)

# Module name -> import-path entries a script run puts first, in order.
SCRIPT_PREFIX: dict[str, tuple[Path, ...]] = {
    "validate_density_control": (ROOT,),
    "validate_disruption_sequence": (SRC, ROOT),
    "validate_marfe_onset": (SRC, ROOT),
    "validate_sol_two_point": (SRC, ROOT),
    "validate_stability_mhd_ballooning": (ROOT, SRC),
    "validate_vmec_lite_geometry": (SRC, ROOT),
    "validate_volt_second": (ROOT,),
}


def _stripped() -> list[str]:
    """Return the current import path without the checkout and its sources."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() not in {ROOT, SRC}]


def _run_help(name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run the validator as a script with ``--help`` and require argparse's help and exit zero."""
    script = ROOT / "validation" / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("name", PHYSICS_VALIDATORS)
def test_script_entry_answers_help_without_validating(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The script entry hands the command line to ``main``, whose parser prints help and exits zero."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    _run_help(name, monkeypatch, capsys)


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_package_import_leaves_the_import_path_unchanged(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading the validator as a package module inserts nothing into a path without the checkout."""
    module = importlib.import_module(f"validation.{name}")
    stripped = _stripped()
    monkeypatch.setattr(sys, "path", list(stripped))
    importlib.reload(module)
    assert sys.path == stripped


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_puts_its_entries_first(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script from a path without the checkout, the validator inserts its entries in order."""
    monkeypatch.setattr(sys, "path", _stripped())
    _run_help(name, monkeypatch, capsys)
    expected = [str(entry) for entry in SCRIPT_PREFIX[name]]
    assert sys.path[: len(expected)] == expected


@pytest.mark.parametrize("name", sorted(SCRIPT_PREFIX))
def test_script_run_with_the_checkout_present_adds_no_duplicate(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with the checkout and sources already importable, the validator adds no second entry."""
    present = [str(ROOT), str(SRC), *_stripped()]
    monkeypatch.setattr(sys, "path", list(present))
    _run_help(name, monkeypatch, capsys)
    assert sys.path == present
