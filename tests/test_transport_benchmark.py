# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Transport Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Transport Validation Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Regression tests for the transport validation benchmark.
"""

from __future__ import annotations


from validation.benchmark_transport import (
    PURE_DIFFUSION_MAX_RELATIVE_ERROR,
    run_pure_diffusion_benchmark,
    run_threshold_benchmark,
)


def test_transport_pure_diffusion():
    """Verify that pure diffusion matches analytic solution trend."""
    # We use a loose threshold due to 1.5D vs 1D cylindrical discrepancies
    # found during validation.
    result = run_pure_diffusion_benchmark(nr=50)
    assert result["max_relative_error"] < PURE_DIFFUSION_MAX_RELATIVE_ERROR


def test_transport_threshold():
    """Verify that the transport model respects critical gradients."""
    result = run_threshold_benchmark()
    assert result["pass"]
    assert result["chi_sub"] == 0.0
    assert result["chi_super"] > 0.0
