# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: COV-1 tests for native solver bridge delegation fallbacks.
"""COV-1 regression tests for defensive HPC bridge delegation paths."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.hpc_bridge import HPCBridge


class _PreparedBridge(HPCBridge):
    """Initialized bridge shell whose input preparation succeeds."""

    def __init__(self) -> None:
        self.loaded = True
        self.solver_ptr = cast(Any, object())
        self.nr = 2
        self.nz = 3


class _SolveIntoNoneBridge(_PreparedBridge):
    """Bridge whose `solve_into` delegate fails closed."""

    def solve_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        iterations: int = 100,
    ) -> NDArray[np.float64] | None:
        del j_phi, psi_out, iterations
        return None


class _ConvergedIntoNoneBridge(_PreparedBridge):
    """Bridge whose convergence delegate fails closed."""

    def solve_until_converged_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        omega: float = 1.8,
    ) -> tuple[int, float] | None:
        del j_phi, psi_out, max_iterations, tolerance, omega
        return None


def test_solve_returns_none_when_delegated_into_solver_fails_closed() -> None:
    """`solve` preserves `None` when `solve_into` fails after preparation."""
    bridge = _SolveIntoNoneBridge()

    result = bridge.solve(np.zeros((3, 2), dtype=np.float64))

    assert result is None


def test_solve_until_converged_returns_none_when_delegated_into_solver_fails_closed() -> None:
    """`solve_until_converged` preserves `None` when the into-variant fails."""
    bridge = _ConvergedIntoNoneBridge()

    result = bridge.solve_until_converged(np.zeros((3, 2), dtype=np.float64))

    assert result is None
