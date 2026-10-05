# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EFIT inverse admission regression tests.

"""Exercise public inverse configuration and measurement refusal paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit import MagneticDiagnostics, RealtimeEFIT


def _solver() -> RealtimeEFIT:
    """Build a small valid reconstruction grid and sensor set."""
    diagnostics = MagneticDiagnostics([(2.0, 0.0)], [(2.0, 0.0, "Z")], 2.0)
    return RealtimeEFIT(diagnostics, np.linspace(1.0, 3.0, 9), np.linspace(-1.0, 1.0, 9))


def _measurements() -> dict[str, float | AnyFloatArray]:
    """Provide finite observations with a nonzero current channel."""
    return {"flux_loops": np.array([0.0]), "b_probes": np.array([0.0]), "Ip": 1.0e6}


def test_reconstruct_rejects_invalid_solver_configuration() -> None:
    """An inverse cannot use boolean steps or invalid tolerances and weights."""
    solver = _solver()
    observations = _measurements()
    with pytest.raises(ValueError, match="max_iter"):
        solver.reconstruct(observations, max_iter=True)
    with pytest.raises(ValueError, match="tol"):
        solver.reconstruct(observations, tol=0.0)
    with pytest.raises(ValueError, match="tol"):
        solver.reconstruct(observations, tol=float("nan"))
    with pytest.raises(ValueError, match="regularization"):
        solver.reconstruct(observations, regularization=-1.0)
    with pytest.raises(ValueError, match="regularization"):
        solver.reconstruct(observations, regularization=float("inf"))
    with pytest.raises(ValueError, match="rel_sigma"):
        solver.reconstruct(observations, rel_sigma=0.0)
    with pytest.raises(ValueError, match="rel_sigma"):
        solver.reconstruct(observations, rel_sigma=float("nan"))


def test_reconstruct_rejects_nonfinite_observation_and_weight_overflow() -> None:
    """A nonfinite measured channel or derived weight cannot reach least squares."""
    solver = _solver()
    invalid = _measurements()
    invalid["Ip"] = float("inf")
    with pytest.raises(ValueError, match="measurements must be finite"):
        solver.reconstruct(invalid)
    with np.errstate(over="ignore", divide="ignore"):
        with pytest.raises(ValueError, match="weights must be finite"):
            solver.reconstruct(_measurements(), rel_sigma=1.0e-300)
        with pytest.raises(ValueError, match="weights must be finite"):
            solver.reconstruct(_measurements(), rel_sigma=1.0e-160)
