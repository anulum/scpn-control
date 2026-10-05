# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark Free Boundary.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Benchmark
# © 1996–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Vacuum software diagnostics with solver-expression and off-axis field limits.

These local samples do not provide independent physical flux normalization,
Helmholtz-axis agreement or a validated magnetic-null location.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ellipe, ellipk

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.core.fusion_kernel import FusionKernel


def jackson_psi(Rc: float, Zc: float, R: float, Z: float, I: float = 1.0) -> float:
    """Evaluate the local clipped loop expression used for solver self-consistency.

    Parameters
    ----------
    Rc, Zc : float
        Source coil radial/vertical coordinates in the code's metre convention.
    R, Z : float
        Observation coordinates in that same convention. Physical-domain
        validation is not performed by this function.
    I : float, default 1.0
        Coil current in the code's ampere convention, with mu0=4e-7*pi.

    Returns
    -------
    float
        Scalar raw Psi from the displayed expression. k2 is clipped to
        [1e-9, 0.999999], so singular/axis limits are not exact evaluations.
        The formula grouping matches the current vacuum-field implementation;
        agreement is self-consistency, not an independent normalization check.

    Raises
    ------
    TypeError, ValueError
        Underlying arithmetic/special functions cannot consume supplied values.
    FloatingPointError
        Invalid floating operations under a caller's NumPy error policy.

    Notes
    -----
    NumPy's global error policy governs warnings/errors. Invalid coordinates
    can produce nonfinite values without a custom refusal. The historical
    Jackson label is not byte-bound evidence that Wb/Wb-per-radian conventions
    or a published analytic derivation have been independently verified.
    """
    mu0 = 4e-7 * np.pi
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-9, 0.999999)
    K = ellipk(k2)
    E = ellipe(k2)
    # Keep the solver's grouping for a self-consistency comparison.
    term = ((2.0 - k2) * K - 2.0 * E) / k2
    pre = mu0 * I / (2 * np.pi) * np.sqrt((R + Rc) ** 2 + (Z - Zc) ** 2)
    return float(pre * term)


def run_free_boundary_benchmark() -> dict[str, Any]:
    """Compute three fixed vacuum diagnostics with a real local FusionKernel.

    Returns
    -------
    dict
        single_coil calculated/reference/raw relative error and threshold flag;
        helmholtz off-axis B_z/axis reference and legacy unconditional True
        qualitative marker; x_point grid-gradient minimum and a Z-only flag.
        This API retains its legacy diagnostic marker. main() reports that
        unassessed Helmholtz field as pass=None with an explicit assessment.

    Raises
    ------
    OSError
        Temporary config allocation/write or cleanup fails. Initial config
        writing precedes the cleanup try/finally and may leave a file on error.
    RuntimeError, TypeError, ValueError
        The defining kernel/config/numerical operations refuse execution.

    Notes
    -----
    Each call creates three fresh kernels on a 65x65 grid, R=[0.5,2.5] m and
    Z=[-1.5,1.5] m, with fixed 1e6 A coils and mu0=4e-7*pi. It writes/reuses a
    uniquely allocated temporary config and removes it after the computation
    try/finally. No persistent report is written and no plasma solve is run.
    The single-coil reference uses the same formula grouping as the solver.
    The Helmholtz sample is R=0.5, Z=0; its reference is on the unreachable
    R=0 axis, so no same-point error or tolerance is evaluated. The reported
    X-point is a whole-grid gradient-norm minimum, with only abs(Z)<0.1 checked;
    it is not a saddle/null test and expected R=0 lies outside the grid.
    These diagnostics establish no independent physical unit normalization,
    analytic/external-code validation, topology or facility-control admission.
    NumPy/runtime resources are shared; no timing or concurrency guarantee.
    """
    results = {}

    # Setup a minimal kernel
    cfg = {
        "reactor_name": "Benchmark-Free",
        "grid_resolution": [65, 65],
        "dimensions": {"R_min": 0.5, "R_max": 2.5, "Z_min": -1.5, "Z_max": 1.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 4e-7 * np.pi},
        "coils": [{"name": "Coil1", "r": 1.0, "z": 0.0, "current": 1e6, "turns": 1}],
        "solver": {"max_iterations": 1, "convergence_threshold": 1.0},
    }
    descriptor, raw_path = tempfile.mkstemp(prefix="scpn-control-free-boundary-", suffix=".json")
    os.close(descriptor)
    cfg_path = Path(raw_path)
    cfg_path.write_text(json.dumps(cfg))

    try:
        kernel = FusionKernel(cfg_path)

        # 1. Single Coil Flux
        psi_calc = kernel.calculate_vacuum_field()
        # Sample at R=1.5, Z=0.5
        ir = np.searchsorted(kernel.R, 1.5)
        iz = np.searchsorted(kernel.Z, 0.5)
        val_calc = psi_calc[iz, ir]
        val_ref = jackson_psi(1.0, 0.0, kernel.R[ir], kernel.Z[iz], 1e6)

        results["single_coil"] = {
            "calculated": float(val_calc),
            "reference": float(val_ref),
            "error_rel": float(abs(val_calc - val_ref) / val_ref),
            "pass": bool(abs(val_calc - val_ref) / val_ref < 1e-6),
        }

        # 2. Helmholtz Pair Field
        # R=1.0, Z= +/- 0.5. Field at axis (R=0) should be uniform.
        # Note: B_z(axis) = mu0 * I / R * (8 / 5*sqrt(5))
        mu0 = 4e-7 * np.pi
        I_helm = 1e6
        R_helm = 1.0
        cfg["coils"] = [
            {"name": "H1", "r": R_helm, "z": 0.5, "current": I_helm},
            {"name": "H2", "r": R_helm, "z": -0.5, "current": I_helm},
        ]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        kernel_h = FusionKernel(cfg_path)
        psi_h = kernel_h.calculate_vacuum_field()
        kernel_h.Psi = psi_h
        kernel_h.compute_b_field()

        # The R=0.5 sample and the R=0 axis reference are different points.
        iz_mid = kernel_h.NZ // 2
        ir_min = 0  # R = 0.5
        bz_calc = kernel_h.B_Z[iz_mid, ir_min]

        # Retain the axis reference as an unassessed diagnostic value.
        bz_ref_axis = mu0 * I_helm / R_helm * (8.0 / (5.0 * np.sqrt(5.0)))

        results["helmholtz"] = {
            "bz_axis_ref": float(bz_ref_axis),
            "bz_at_min_r": float(bz_calc),
            "pass": True,  # Qualitative check since grid doesn't reach R=0
        }

        # 3. Antisymmetric coil preset and grid gradient-minimum diagnostic.
        cfg["coils"] = [
            {"name": "X1", "r": 1.0, "z": 1.0, "current": 1e6},
            {"name": "X2", "r": 1.0, "z": -1.0, "current": -1e6},
        ]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        kernel_x = FusionKernel(cfg_path)
        psi_x = kernel_x.calculate_vacuum_field()
        # Search for X-point
        # find_x_point in fusion_kernel usually searches divertor region.
        # We manually find min gradient.
        dPsi_dR, dPsi_dZ = np.gradient(psi_x, kernel_x.dZ, kernel_x.dR)
        grad_norm = np.hypot(dPsi_dR, dPsi_dZ)
        iz_x, ir_x = np.unravel_index(np.argmin(grad_norm), grad_norm.shape)
        rx, zx = kernel_x.R[ir_x], kernel_x.Z[iz_x]

        results["x_point"] = {
            "detected_r": float(rx),
            "detected_z": float(zx),
            "expected_r": 0.0,  # Not reachable on grid
            "expected_z": 0.0,
            "pass": bool(abs(zx) < 0.1),
        }

        return results
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def main() -> None:
    """Write caller-relative software reports with an unassessed off-axis field.

    Returns
    -------
    None
        Write validation/reports/free_boundary_benchmark.json and .md, then
        print their directory. JSON changes only Helmholtz pass to None and
        adds assessment=diagnostic_only_off_axis_sample; Markdown keeps N/A.

    Raises
    ------
    RuntimeError, ValueError
        Persistent output guard or defining diagnostics refuse execution.
    OSError
        Directory/report/config IO fails. Writes are sequential with the
        platform text codec; partial output can remain without a transaction.

    Notes
    -----
    There are no CLI parameters. Destinations resolve from the current working
    directory, while campaign persistence scope uses this module's root.
    Guarded persistent paths require a campaign ID; outside-root scratch paths
    do not. ID presence/syntax is not authenticity or physical-reference proof;
    actual recorded wrapper custody is a separate API. Legacy json.dump(indent=2)
    retains its nonfinite convention. Shared output paths have no producer
    locking or atomic replacement. Numeric diagnostics and flags other than
    the unassessed Helmholtz report marker retain their original arithmetic.
    No physical normalization, axis agreement or topology admission is granted.
    """
    report_dir = Path("validation/reports")
    json_path = report_dir / "free_boundary_benchmark.json"
    markdown_path = report_dir / "free_boundary_benchmark.md"
    require_recorded_campaign(json_path, markdown_path, repository_root=Path(__file__).resolve().parents[1])
    res = run_free_boundary_benchmark()
    res["helmholtz"]["pass"] = None
    res["helmholtz"]["assessment"] = "diagnostic_only_off_axis_sample"

    report_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("w") as f:
        json.dump(res, f, indent=2)

    with markdown_path.open("w") as f:
        f.write("# Free-Boundary Software Diagnostics\n\n")
        f.write("Single-coil equality is solver-expression self-consistency. The off-axis Helmholtz sample is\n")
        f.write("unassessed against its axis reference; the gradient minimum is not a validated magnetic null.\n\n")
        f.write("| Test | Metric | Result | Pass |\n")
        f.write("|------|--------|--------|------|\n")
        sc = res["single_coil"]
        f.write(f"| Single Coil | Rel Error | {sc['error_rel']:.2e} | {sc['pass']} |\n")
        hm = res["helmholtz"]
        f.write(f"| Helmholtz | B_z Axis Ref | {hm['bz_axis_ref']:.4f} T | N/A |\n")
        xp = res["x_point"]
        f.write(f"| X-point | Detected Z | {xp['detected_z']:.4f} | {xp['pass']} |\n")

    print(f"Results saved to {report_dir}")


if __name__ == "__main__":
    main()
