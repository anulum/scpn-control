# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Adaptive transport time controller

"""Richardson time step control for the bounded integrated transport facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver


class AdaptiveTimeController:
    """Select a Crank–Nicolson transport step using Richardson error.

    One full step and two half steps begin from the same solver snapshot. The
    two-half-step result remains in the solver; a proportional-integral update
    chooses the next step size in seconds.

    Parameters
    ----------
    dt_init : float
        Initial time step in seconds.
    dt_min : float
        Minimum time step in seconds.
    dt_max : float
        Maximum time step in seconds.
    tol : float
        Target norm of the local ion-temperature error in keV.
    safety : float
        Multiplicative factor in the step-size update.
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        dt_min: float = 1e-5,
        dt_max: float = 1.0,
        tol: float = 1e-3,
        safety: float = 0.9,
    ) -> None:
        """Set step bounds, tolerance and PI history for subsequent adaptive error estimates."""
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol
        self.safety = safety
        self.p = 2  # CN is second-order

        self.dt_history: list[float] = []
        self.error_history: list[float] = []
        self._err_prev: float = tol

    def estimate_error(self, solver: "TransportSolver", P_aux: float) -> float:
        """Estimate local error via Richardson extrapolation.

        The full-step trial is restored before the two half steps. The solver
        retains their accepted state, including species and rotation.

        Parameters
        ----------
        solver : TransportSolver
            Integrated transport solver whose state is advanced.
        P_aux : float
            Auxiliary heating power in megawatts.

        Returns
        -------
        float
            Richardson norm of the ion-temperature difference in keV.
        """
        entry_state = solver.capture_evolution_state()

        solver.evolve_profiles(self.dt, P_aux)
        T_full = solver.Ti.copy()

        # The full trial must not seed the two half-step trials.
        solver.restore_evolution_state(entry_state)
        solver.evolve_profiles(self.dt / 2.0, P_aux)
        solver.evolve_profiles(self.dt / 2.0, P_aux)
        T_half = solver.Ti.copy()

        # Crank–Nicolson has order p=2.
        error: float = float(np.linalg.norm(T_full - T_half)) / (2**self.p - 1)
        error = max(error, 1e-15)

        return error

    def adapt_dt(self, error: float) -> None:
        """Bound the next time step using the current and preceding errors.

        Parameters
        ----------
        error : float
            Positive Richardson error norm in keV from :meth:`estimate_error`.
        """
        self.error_history.append(error)
        self.dt_history.append(self.dt)

        ratio_i = (self.tol / error) ** (0.7 / self.p)
        ratio_p = (self._err_prev / error) ** (0.4 / self.p)
        factor = self.safety * ratio_i * ratio_p
        factor = min(factor, 2.0)
        factor = max(factor, 0.1)

        self.dt *= factor
        self.dt = max(self.dt, self.dt_min)
        self.dt = min(self.dt, self.dt_max)

        self._err_prev = error
