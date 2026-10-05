# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EFIT Picard status regression tests.

"""Exercise reconstruction termination status through the public solver."""

from __future__ import annotations

import numpy as np

from scpn_control.control.realtime_efit import MagneticDiagnostics, RealtimeEFIT


def test_geometric_and_picard_termination_are_distinct() -> None:
    """A finite max-iteration result must not be reported as converged."""
    diagnostics = MagneticDiagnostics(
        flux_loops=[(5.0, 1.0), (6.2, 1.5), (7.4, 1.0)],
        b_probes=[(5.0, 1.0, "R"), (5.0, 1.0, "Z"), (7.4, 1.0, "R")],
        rogowski_radius=6.2,
    )
    solver = RealtimeEFIT(diagnostics, np.linspace(4.2, 8.2, 17), np.linspace(-3.0, 3.0, 17))
    psi = solver._solve_gs_with_sources(np.array([2.0, -1.5, 0.4]), np.array([1.0, -0.6, 0.1]))
    measurements = solver.response.simulate_measurements(psi, np.zeros(1))

    geometric = solver.reconstruct(measurements, mode="geometric")
    limited = solver.reconstruct(measurements, mode="psi_n", max_iter=1, tol=1.0e-12)
    converged = solver.reconstruct(measurements, mode="psi_n", max_iter=2, tol=1.0e6)

    assert geometric.iteration_status == "geometric_solve"
    assert geometric.final_relative_change is None
    assert limited.iteration_status == "picard_limit"
    assert limited.n_iterations == 1
    assert limited.final_relative_change is not None
    assert limited.final_relative_change >= 1.0e-12
    assert converged.iteration_status == "picard_converged"
    assert converged.final_relative_change is not None
    assert converged.final_relative_change < 1.0e6
