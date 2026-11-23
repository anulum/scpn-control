# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Import-path contract of validation and tool commands.
"""Validation and tool commands edit the import path only when they run as scripts.

Imported as ``validation.<module>`` or ``tools.<module>`` they must leave the
caller's import path alone. Loaded from their file outside a package, as a
script run is, they must put their documented entries first, in order, and
add no duplicate when the entries are already present. Every check strips the
checkout and its ``src`` first: with them present, an unguarded "insert if
absent" would also leave the path unchanged. Each module is imported normally
beforehand so that its dependencies resolve from the module cache. Run as
``__main__``, each MAST disruption command hands its arguments to ``main``.
"""

from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

ENTRIES: dict[str, tuple[Path, ...]] = {
    "validation/acquire_mast_disruption_shots.py": (ROOT,),
    "validation/audit_mast_disruption_feature_sources.py": (ROOT,),
    "validation/audit_mast_efm_feature_provenance.py": (ROOT,),
    "validation/audit_mast_label_proxy_sensitivity.py": (ROOT,),
    "validation/build_disruption_replay_channels.py": (ROOT,),
    "validation/build_fairmast_disruption_manifest.py": (ROOT,),
    "validation/build_mast_disruption_dataset.py": (ROOT,),
    "validation/build_mast_efm_neural_equilibrium_dataset.py": (ROOT,),
    "validation/code_to_code_benchmark.py": (SRC, ROOT),
    "validation/control_resilience_campaign.py": (ROOT,),
    "validation/controller_comparison.py": (SRC, ROOT),
    "validation/design_mast_matched_cohort.py": (ROOT,),
    "validation/extract_mast_disruption_channels.py": (ROOT,),
    "validation/free_boundary_tracking_acceptance.py": (SRC, ROOT),
    "validation/generate_physics_traceability_report.py": (ROOT,),
    "validation/plan_neural_equilibrium_training_campaign.py": (ROOT,),
    "validation/stress_test_campaign.py": (SRC,),
    "validation/train_mast_efm_neural_equilibrium.py": (ROOT,),
    "tools/check_generated_traceability.py": (ROOT,),
    "tools/publish.py": (ROOT,),
    "tools/run_benchmark_suite.py": (SRC, ROOT),
}

DISRUPTION_COMMANDS = (
    "validation/acquire_mast_disruption_shots.py",
    "validation/audit_mast_disruption_feature_sources.py",
    "validation/audit_mast_label_proxy_sensitivity.py",
    "validation/build_disruption_replay_channels.py",
    "validation/build_fairmast_disruption_manifest.py",
    "validation/build_mast_disruption_dataset.py",
    "validation/design_mast_matched_cohort.py",
    "validation/evaluate_mast_disruption.py",
    "validation/extract_mast_disruption_channels.py",
)


def _module_name(relative: str) -> str:
    """Return the dotted package name of a repository-relative module file."""
    return relative.removesuffix(".py").replace("/", ".")


def _without_checkout() -> list[str]:
    """Return the current import path without the checkout and its src."""
    return [entry for entry in sys.path if Path(entry or ".").resolve() not in {ROOT, SRC}]


@pytest.mark.parametrize("relative", sorted(ENTRIES))
def test_package_import_leaves_the_import_path_unchanged(relative: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executed as a module of its package, the command inserts nothing into a path without the checkout."""
    name = _module_name(relative)
    importlib.import_module(name)
    stripped = _without_checkout()
    monkeypatch.setattr(sys, "path", list(stripped))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__package__ == name.rpartition(".")[0]
    assert sys.path == stripped


@pytest.mark.parametrize("relative", sorted(ENTRIES))
def test_script_load_puts_the_documented_entries_first(relative: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaded from its file outside a package, the command puts its entries first, in order."""
    importlib.import_module(_module_name(relative))
    stripped = _without_checkout()
    monkeypatch.setattr(sys, "path", list(stripped))
    runpy.run_path(str(ROOT / relative), run_name="script_import_probe")
    expected = [str(entry) for entry in ENTRIES[relative]]
    assert sys.path == [*expected, *stripped]


@pytest.mark.parametrize("relative", sorted(ENTRIES))
def test_script_load_with_the_entries_present_adds_no_duplicate(relative: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaded from its file with its entries already importable, the command leaves the path as it was."""
    importlib.import_module(_module_name(relative))
    present = [*(str(entry) for entry in ENTRIES[relative]), *_without_checkout()]
    monkeypatch.setattr(sys, "path", list(present))
    runpy.run_path(str(ROOT / relative), run_name="script_import_probe")
    assert sys.path == present


@pytest.mark.parametrize("relative", DISRUPTION_COMMANDS)
def test_disruption_command_run_as_main_reaches_its_parser(
    relative: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as ``__main__`` from a path without the checkout, the command imports and answers ``--help``."""
    importlib.import_module(_module_name(relative))
    monkeypatch.setattr(sys, "path", _without_checkout())
    monkeypatch.setattr(sys, "argv", [str(ROOT / relative), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / relative), run_name="__main__")
    assert stopped.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ")
