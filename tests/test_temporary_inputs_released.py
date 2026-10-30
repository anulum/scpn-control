# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — temporary solver inputs are released

"""Temporary inputs the package creates do not outlive their use.

Each test points :mod:`tempfile` at an empty directory of its own and checks
that it is empty again afterwards: gyrokinetic input decks after every run,
successful or not, and the scenario's transport configuration once its
solver is gone or failed to build. Configured working directories and paths
the solver did not create are never removed.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import pytest

from scpn_control.core.gk_cgyro import CGYROSolver
from scpn_control.core.gk_gene import GENESolver
from scpn_control.core.gk_gs2 import GS2Solver
from scpn_control.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase
from scpn_control.core.gk_qualikiz import QuaLiKizSolver
from scpn_control.core.gk_tglf import TGLFSolver
from scpn_control.core.integrated_scenario import IntegratedScenarioSimulator
from scpn_control.core.integrated_scenario_presets import ScenarioConfig

MISSING = "nonexistent_gk_binary_for_release_tests"
PARAMS = GKLocalParams(R_L_Ti=6.9, R_L_Te=6.9, R_L_ne=2.2, q=1.4, s_hat=0.78, rho=0.5, R0=2.78, a=1.0, B0=2.0)


@pytest.fixture
def temporary_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send every :mod:`tempfile` creation into a fresh empty directory."""
    root = tmp_path / "tempfile-root"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _strict_solvers(work_dir: Path | None) -> list[GKSolverBase]:
    """Return one of each external solver, failing closed because its code is absent."""
    return [
        TGLFSolver(binary=MISSING, work_dir=work_dir),
        GS2Solver(binary=MISSING, work_dir=work_dir),
        CGYROSolver(binary=MISSING, work_dir=work_dir),
        GENESolver(binary=MISSING, work_dir=work_dir),
        QuaLiKizSolver(binary=MISSING, work_dir=work_dir),
    ]


def test_failed_runs_remove_their_temporary_decks(temporary_root: Path) -> None:
    """A run that fails closed still removes the deck directory it created."""
    for solver in _strict_solvers(None):
        with pytest.raises(RuntimeError):
            solver.run_from_params(PARAMS)
        assert list(temporary_root.iterdir()) == [], type(solver).__name__


def test_successful_runs_remove_their_temporary_decks(temporary_root: Path) -> None:
    """A run that returns a result removes the deck directory too."""
    for solver_type in (GS2Solver, CGYROSolver, GENESolver, QuaLiKizSolver):
        solver = solver_type(binary=MISSING, allow_fallback=True, allow_legacy_fallback=True)
        assert isinstance(solver.run_from_params(PARAMS), GKOutput)
        assert list(temporary_root.iterdir()) == [], solver_type.__name__


def test_configured_work_directory_is_kept(tmp_path: Path, temporary_root: Path) -> None:
    """A configured working directory survives a failed run, deck included.

    QuaLiKiz runs through its Python API and writes no deck, so only its
    directory is checked.
    """
    for index in range(len(_strict_solvers(None))):
        work_dir = tmp_path / f"work-{index}"
        solver = _strict_solvers(work_dir)[index]
        with pytest.raises(RuntimeError):
            solver.run_from_params(PARAMS)
        assert work_dir.is_dir(), type(solver).__name__
        assert isinstance(solver, QuaLiKizSolver) or any(work_dir.iterdir()), type(solver).__name__
    assert list(temporary_root.iterdir()) == []


def test_release_input_removes_only_what_the_solver_created(tmp_path: Path, temporary_root: Path) -> None:
    """Release a prepared temporary deck once; leave foreign paths alone."""
    solver = TGLFSolver(binary=MISSING)
    deck = solver.prepare_input(PARAMS)
    assert deck.parent == temporary_root
    assert (deck / "input.tglf").is_file()

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    solver.release_input(foreign)
    assert foreign.is_dir()

    solver.release_input(deck)
    assert not deck.exists()
    solver.release_input(deck)


class _ForeignPathSolver(GKSolverBase):
    """Subclass without a base constructor that returns a directory it does not own."""

    def __init__(self, path: Path) -> None:
        """Remember the directory to hand out."""
        self.path = path

    def prepare_input(self, params: GKLocalParams) -> Path:
        """Return the remembered directory without creating anything."""
        return self.path

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Return a fixed result."""
        return GKOutput(chi_i=1.0, chi_e=0.8, D_e=0.1)

    def is_available(self) -> bool:
        """Report the solver as available."""
        return True


def test_run_keeps_a_directory_the_solver_did_not_create(tmp_path: Path) -> None:
    """A subclass's own directory survives ``run_from_params``."""
    assert _ForeignPathSolver(tmp_path).run_from_params(PARAMS).chi_i == 1.0
    assert tmp_path.is_dir()


def _scenario() -> IntegratedScenarioSimulator:
    """Return a short scenario with only the transport core enabled."""
    return IntegratedScenarioSimulator(
        ScenarioConfig(
            R0=0.93,
            a=0.58,
            B0=1.0,
            kappa=2.0,
            delta=0.4,
            Ip_MA=1.0,
            P_aux_MW=0.5,
            P_eccd_MW=0.0,
            P_nbi_MW=0.0,
            t_start=0.0,
            t_end=0.05,
            dt=0.01,
            include_sawteeth=False,
            include_ntm=False,
            include_sol=False,
            include_elm=False,
            include_stability=False,
        )
    )


def test_scenario_configuration_lives_as_long_as_its_solver(temporary_root: Path) -> None:
    """The configuration stays readable while the solver exists and goes with it."""
    solver = _scenario()._setup_transport_solver()
    (config,) = temporary_root.iterdir()
    assert config.suffix == ".json"
    assert config.read_text(encoding="utf-8").startswith("{")

    del solver
    gc.collect()
    assert list(temporary_root.iterdir()) == []


def test_failed_scenario_solver_leaves_no_configuration(temporary_root: Path) -> None:
    """A solver that refuses its grid leaves no configuration file behind."""
    simulator = _scenario()
    simulator.nr = 1
    with pytest.raises(ValueError, match="nr must be an integer >= 2"):
        simulator._setup_transport_solver()
    assert list(temporary_root.iterdir()) == []
