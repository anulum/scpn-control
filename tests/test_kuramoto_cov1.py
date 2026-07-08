# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto COV-1 fallback coverage tests.
"""Focused COV-1 tests for reachable Kuramoto Python fallback behavior."""

from __future__ import annotations

import numpy as np

from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step


def test_kuramoto_step_uses_python_fallback_when_wrap_disabled() -> None:
    """Exercise the NumPy fallback return even when the optional Rust backend exists."""

    theta = np.array([3.2, -3.0, 0.25], dtype=np.float64)
    omega = np.array([0.5, -0.25, 0.125], dtype=np.float64)

    out = kuramoto_sakaguchi_step(
        theta,
        omega,
        dt=0.1,
        K=0.0,
        zeta=0.0,
        psi_driver=0.0,
        psi_mode="external",
        wrap=False,
    )

    expected = theta + 0.1 * omega
    np.testing.assert_allclose(np.asarray(out["theta1"], dtype=np.float64), expected)
    np.testing.assert_allclose(np.asarray(out["dtheta"], dtype=np.float64), omega)
    assert float(out["Psi"]) == 0.0
