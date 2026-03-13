# ──────────────────────────────────────────────────────────────────────
# SCPN Control — GS Solver Mesh Convergence Regression
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Regression tests for spatial convergence order of the GS solver.
Ensures that 2nd-order accuracy is maintained.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.mesh_convergence_study import run_solovev_benchmark


def test_mesh_convergence_rate_p2():
    """Verify that the GS solver achieves ~2nd order convergence.
    
    Checks 17x17 vs 33x33 grid resolutions.
    Error should reduce by a factor of ~4 (2nd order).
    We use smaller grids for CI speed.
    """
    res17 = run_solovev_benchmark(17, 17, max_iter=2000, tol=1e-10)
    res33 = run_solovev_benchmark(33, 33, max_iter=4000, tol=1e-10)
    
    h_ratio = res17["h"] / res33["h"]
    err_ratio = res17["nrmse"] / res33["nrmse"]
    rate = np.log(err_ratio) / np.log(h_ratio)
    
    # Expected rate is 2.0 for central differences.
    assert 1.8 <= rate <= 2.2, f"Convergence rate {rate:.2f} is not 2nd order."
