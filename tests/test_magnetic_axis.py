# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Checked magnetic-axis accessor and its readers.
"""The axis of a solved flux map must be an interior O-point; every reader refuses an edge maximum."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.director_interface import DirectorInterface
from scpn_control.control.fusion_neural_mpc import NeuralSurrogate
from scpn_control.control.fusion_optimal_control import OptimalController
from scpn_control.control.tokamak_flight_sim import run_flight_sim
from scpn_control.core.fusion_kernel import FusionKernel
from scpn_control.core.magnetic_axis import (
    MagneticAxisError,
    locate_magnetic_axis,
    require_magnetic_axis,
)

FloatArray = NDArray[np.float64]

_R = np.linspace(4.0, 8.0, 21)
_Z = np.linspace(-2.0, 2.0, 17)
_RR, _ZZ = np.meshgrid(_R, _Z)


def _peaked(r0: float, z0: float) -> FloatArray:
    """Return a flux map with a single smooth maximum at ``(r0, z0)``."""
    return np.asarray(1.0 - ((_RR - r0) ** 2 + (_ZZ - z0) ** 2), dtype=np.float64)


def _edge_flux() -> FloatArray:
    """Return a flux map that rises monotonically towards the outer box edge."""
    return np.asarray(_RR + 0.0 * _ZZ, dtype=np.float64)


class _EdgeKernel:
    """Kernel stand-in whose solved flux peaks on the outer box edge."""

    def __init__(self, _config_file: str = "edge.json") -> None:
        """Build a 21 x 17 grid with five named coils and the edge-peaked flux."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"name": f"C{i}", "current": 0.0} for i in range(5)],
        }
        self.R = _R.copy()
        self.Z = _Z.copy()
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = _edge_flux()

    def solve_equilibrium(self) -> None:
        """Keep the degenerate edge-peaked flux."""
        self.Psi = _edge_flux()

    def find_x_point(self, _psi: FloatArray) -> tuple[tuple[float, float], float]:
        """Return a fixed X-point."""
        return (float(self.R[3]), float(self.Z[2])), 0.0


def test_interior_peak_is_an_admissible_axis() -> None:
    """A smooth interior maximum is located at its grid point and admitted."""
    axis = require_magnetic_axis(_peaked(6.0, 0.5), _R, _Z)
    assert axis.interior
    assert axis.r == pytest.approx(6.0)
    assert axis.z == pytest.approx(0.5)
    assert axis.psi == pytest.approx(1.0)
    assert (axis.iz, axis.ir) == (int(np.argmin(np.abs(_Z - 0.5))), int(np.argmin(np.abs(_R - 6.0))))


@pytest.mark.parametrize(
    "flux",
    [
        pytest.param(_edge_flux(), id="outer-edge"),
        pytest.param(_peaked(4.0, 0.0), id="inner-edge"),
        pytest.param(_peaked(8.0, 2.0), id="corner"),
        pytest.param(_peaked(6.0, 2.0), id="top-edge"),
    ],
)
def test_edge_maximum_is_located_but_refused(flux: FloatArray) -> None:
    """A maximum on the box edge is reported as not interior and refused by the checked reader."""
    axis = locate_magnetic_axis(flux, _R, _Z)
    assert not axis.interior
    assert axis.iz in (0, _Z.size - 1) or axis.ir in (0, _R.size - 1)
    with pytest.raises(MagneticAxisError, match="not an interior O-point"):
        axis.require_interior()


def test_interior_peak_tied_with_the_boundary_is_refused() -> None:
    """An interior maximum no higher than some boundary value is not an O-point."""
    flux = np.zeros((_Z.size, _R.size))
    flux[8, 10] = 1.0
    flux[0, 4] = 1.0
    flux_interior_first = flux.copy()
    flux_interior_first[0, 4] = 0.0
    flux_interior_first[8, 10] = 1.0
    flux_interior_first[16, 20] = 1.0
    for case in (flux, flux_interior_first):
        assert not locate_magnetic_axis(case, _R, _Z).interior


def test_non_finite_flux_is_never_admissible() -> None:
    """A NaN anywhere, even on the boundary away from the peak, makes the map inadmissible."""
    flux = _peaked(6.0, 0.0)
    flux[0, -1] = np.nan
    assert not locate_magnetic_axis(flux, _R, _Z).interior
    flux = _peaked(6.0, 0.0)
    flux[8, 10] = np.inf
    with pytest.raises(MagneticAxisError):
        require_magnetic_axis(flux, _R, _Z)


@pytest.mark.parametrize(
    ("flux", "r", "z", "message"),
    [
        pytest.param(np.zeros((17, 20)), _R, _Z, "must be", id="shape"),
        pytest.param(np.zeros(21), _R, _Z, "must be", id="rank"),
        pytest.param(np.zeros((2, 21)), _R, _Z[:2], "at least 3 x 3", id="too-small"),
    ],
)
def test_malformed_grids_are_refused(flux: FloatArray, r: FloatArray, z: FloatArray, message: str) -> None:
    """Shape errors are ``ValueError``, not axis findings."""
    with pytest.raises(ValueError, match=message):
        locate_magnetic_axis(flux, r, z)


def test_kernel_iteration_reads_an_edge_maximum_without_refusing() -> None:
    """The solver's own axis lookup stays non-refusing for intermediate iterates."""
    kernel = cast(FusionKernel, SimpleNamespace(Psi=_edge_flux(), R=_R, Z=_Z))
    iz, ir, psi_axis = FusionKernel._find_magnetic_axis(kernel)
    assert ir == _R.size - 1
    assert psi_axis == pytest.approx(8.0)


def test_kernel_iteration_keeps_the_small_flux_floor() -> None:
    """A near-zero maximum is replaced by 1e-6 for normalisation, as before."""
    kernel = cast(FusionKernel, SimpleNamespace(Psi=np.zeros((_Z.size, _R.size)), R=_R, Z=_Z))
    assert FusionKernel._find_magnetic_axis(kernel)[2] == 1e-6


def test_neural_mpc_state_refuses_an_edge_axis() -> None:
    """The MPC shape state is not read from an edge maximum."""
    surrogate = NeuralSurrogate(n_coils=5, n_state=4, verbose=False)
    with pytest.raises(MagneticAxisError):
        surrogate.get_state(_EdgeKernel())


def test_neural_mpc_state_reads_an_interior_axis() -> None:
    """With an interior O-point the state carries its coordinates."""
    kernel = _EdgeKernel()
    kernel.Psi = _peaked(6.0, -0.5)
    state = NeuralSurrogate(n_coils=5, n_state=4, verbose=False).get_state(kernel)
    np.testing.assert_allclose(state, [6.0, -0.5, kernel.R[3], kernel.Z[2]])


def test_optimal_control_position_refuses_an_edge_axis() -> None:
    """The optimal controller does not report an edge cell as the plasma position."""
    controller = OptimalController("edge.json", kernel_factory=_EdgeKernel, verbose=False)
    with pytest.raises(MagneticAxisError):
        controller.get_plasma_pos()


def test_flight_sim_refuses_an_edge_axis() -> None:
    """The iso-flux flight simulation stops at the first degenerate equilibrium."""
    with pytest.raises(MagneticAxisError):
        run_flight_sim(
            config_file="edge.json",
            shot_duration=3,
            seed=1,
            save_plot=False,
            verbose=False,
            kernel_factory=_EdgeKernel,
        )


class _EdgeAxisBrain:
    """Proportional stand-in for a spiking controller pool."""

    def step(self, error: float) -> float:
        """Return a small proportional command."""
        return -0.1 * error


class _EdgeAxisController:
    """Neuro-cybernetic controller stand-in whose kernel has no interior O-point."""

    def __init__(self, _config_path: str) -> None:
        """Bind the edge-peaked kernel and two proportional brains."""
        self.kernel = _EdgeKernel()
        self.brain_R = _EdgeAxisBrain()
        self.brain_Z = _EdgeAxisBrain()

    def initialize_brains(self, use_quantum: bool = True) -> None:
        """Keep the proportional stand-ins."""


def test_directed_mission_refuses_an_edge_axis() -> None:
    """The director does not read an edge cell as the plasma position."""
    director = DirectorInterface(
        "edge.json",
        controller_factory=_EdgeAxisController,
        allow_fallback=True,
        allow_legacy_fallback=True,
    )
    with pytest.raises(MagneticAxisError, match="not an interior O-point"):
        director.run_directed_mission(duration=3, save_plot=False, verbose=False)
