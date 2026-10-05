# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Adaptive transport controller tests
"""Exercise accepted Richardson steps and legacy import compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_control.core.adaptive_time_controller import AdaptiveTimeController as ExtractedAdaptiveTimeController
from scpn_control.core.integrated_transport_solver import AdaptiveTimeController, TransportSolver


def test_adaptive_controller_keeps_the_integrated_solver_import() -> None:
    """The historical solver-module import names the extracted controller."""
    assert AdaptiveTimeController is ExtractedAdaptiveTimeController


class TestAdaptiveRichardsonState:
    """The accepted adaptive result equals two direct half steps."""

    @staticmethod
    def _seeded(config_file: Path) -> TransportSolver:
        """Build a multi-ion solver whose ion and electron temperatures differ."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 3.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.n_D = 0.5 * ts.ne.copy()
        ts.n_T = 0.5 * ts.ne.copy()
        ts.n_He = np.zeros(ts.nr)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        return ts

    def test_adaptive_step_equals_the_two_half_steps_it_accepts(self, config_file: Path) -> None:
        """Every evolved profile matches a direct two-half-step run.

        A full step is taken first to estimate the error. If its mutations are
        not rolled back, species and density are advanced three times instead of
        twice and diverge from this reference.
        """
        reference = self._seeded(config_file)
        reference.update_transport_model(50.0)
        reference.evolve_profiles(0.005, 50.0)
        reference.evolve_profiles(0.005, 50.0)

        adaptive = self._seeded(config_file)
        adaptive.run_to_steady_state(P_aux=50.0, n_steps=1, dt=0.01, adaptive=True)

        for name in ("Ti", "Te", "ne", "n_D", "n_T", "n_He", "n_impurity", "omega_phi", "J_phi"):
            np.testing.assert_allclose(
                getattr(adaptive, name),
                getattr(reference, name),
                rtol=0.0,
                atol=0.0,
                err_msg=f"{name} diverged from the accepted two-half-step path",
            )

    def test_adaptive_step_keeps_electron_temperature_distinct(self, config_file: Path) -> None:
        """The accepted electron temperature is its own profile, not the ion one."""
        adaptive = self._seeded(config_file)
        adaptive.run_to_steady_state(P_aux=50.0, n_steps=1, dt=0.01, adaptive=True)
        assert np.max(np.abs(adaptive.Te - adaptive.Ti)) > 1e-6
