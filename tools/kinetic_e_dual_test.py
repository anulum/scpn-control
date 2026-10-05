#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kinetic Electron Dual Test.

"""Compare fixed explicit-JAX and implicit-NumPy kinetic-electron cases.

JAX selects its available device; this script does not require or certify a
GPU. The two cases use different electron masses, step counts and backends,
so their elapsed times are not a paired backend performance comparison.
Importing this module creates no solver and runs no campaign.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np


class KineticDualResult(TypedDict):
    """Carry one case's scalar summary and saved code-unit histories.

    chi_i_gB is the producer's mean ion flux divided by R_L_Ti, with nonfinite
    scalar values replaced by None. History arrays are copied without finite
    filtering. converged is the solver's flag, not a saturation or facility
    validation verdict; elapsed_s times run(), excluding construction.
    """

    label: str
    chi_i_gB: float | None
    elapsed_s: float
    converged: bool
    phi_rms: list[float]
    Q_i: list[float]
    time: list[float]


def run_case(label: str, ke: bool, implicit: bool, mass_ratio: float, n_steps: int) -> KineticDualResult:
    """Run one fresh fixed-grid CBC case and print its saved diagnostics.

    Parameters
    ----------
    label
        Display and returned case name; it does not change the configuration.
    ke, implicit
        Kinetic-electron switch and backend selection. implicit=True selects
        NonlinearGKSolver with its implicit-electron configuration; otherwise
        JaxNonlinearGKSolver is requested without a NumPy fallback.
    mass_ratio
        Dimensionless electron/ion mass ratio passed to NonlinearGKConfig.
    n_steps
        Requested solver iterations; histories save every 100 iterations and
        may end earlier on nonfinite state. No input validation is added here.

    Returns
    -------
    KineticDualResult
        Fresh-run summary, phi RMS, ion-flux and normalized solver-time lists.
        Grid is128 x 16 x 32 x 16 x 8 with two species; nominal dt 0.05 is CFL-adaptive,
        so neither sample times nor elapsed_s are a fixed physical duration.

    Raises
    ------
    RuntimeError
        The JAX backend is unavailable under its default no-fallback contract.
    Exception
        Backend import, allocation or solver errors propagate. The case may
        allocate large arrays and does not supply a process timeout.

    Notes
    -----
    Timing includes work in run(), including JAX work first triggered there.
    chi_i_gB is a code-normalized quantity, not an independently calibrated
    m^2/s measurement. Late growth is a last-quarter endpoint fractional
    change per saved solver-time span, not a fitted exponential growth rate.
    The returned flag and finite saved traces do not certify the final state.
    """
    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

    print(f"\n=== {label} ===", flush=True)
    cfg = NonlinearGKConfig(
        n_kx=128,
        n_ky=16,
        n_theta=32,
        n_vpar=16,
        n_mu=8,
        dt=0.05,
        n_steps=n_steps,
        save_interval=100,
        R_L_Ti=6.9,
        R_L_Te=6.9,
        R_L_ne=2.2,
        q=1.4,
        s_hat=0.78,
        R0=2.78,
        a=1.0,
        B0=2.0,
        cfl_adapt=True,
        cfl_factor=0.5,
        nonlinear=True,
        collisions=True,
        nu_collision=0.01,
        hyper_coeff=0.2,
        kinetic_electrons=ke,
        implicit_electrons=implicit,
        mass_ratio_me_mi=mass_ratio,
    )
    # Implicit electrons use NumPy solver (JAX doesn't have the tridiag solve)
    solver: NonlinearGKSolver | JaxNonlinearGKSolver
    if implicit:
        from scpn_control.core.gk_nonlinear import NonlinearGKSolver

        solver = NonlinearGKSolver(cfg)
    else:
        solver = JaxNonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    r = solver.run()
    elapsed = time.perf_counter() - t0
    chi_gB = r.chi_i / max(cfg.R_L_Ti, 0.01)
    print(f"  elapsed={elapsed:.0f}s, chi_gB={chi_gB:.4f}", flush=True)
    if len(r.phi_rms_t) > 0:
        print(f"  phi: [{r.phi_rms_t[0]:.4e}...{r.phi_rms_t[-1]:.4e}]", flush=True)
        print(f"  time: [{r.time[0]:.2f}, {r.time[-1]:.2f}]", flush=True)
        n4 = 3 * len(r.phi_rms_t) // 4
        phi_lq = r.phi_rms_t[n4:]
        if len(phi_lq) > 2 and phi_lq[0] > 0:
            g = (phi_lq[-1] - phi_lq[0]) / (phi_lq[0] * max(r.time[-1] - r.time[n4], 0.01))
            print(f"  late_growth={g:.4f}", flush=True)
        for i in range(0, len(r.phi_rms_t), max(1, len(r.phi_rms_t) // 8)):
            pr = r.phi_rms_t[i]
            print(f"    t={r.time[i]:.2f} phi={pr:.3e} Q={r.Q_i_t[i]:.3e}", flush=True)
    return {
        "label": label,
        "chi_i_gB": float(chi_gB) if np.isfinite(chi_gB) else None,
        "elapsed_s": elapsed,
        "converged": bool(r.converged),
        "phi_rms": r.phi_rms_t.tolist(),
        "Q_i": r.Q_i_t.tolist(),
        "time": r.time.tolist(),
    }


def main() -> None:
    """Run the two original unequal-work cases and overwrite their cwd report.

    Explicit JAX uses mass 1/25 and 5,000 steps; implicit NumPy uses mass 1/400 and
    200 steps. JAX device printing is best effort, while the explicit case
    still requires JAX. No CLI options are parsed. Results are written to
    gpu_results/kinetic_e_dual.json relative to the caller's cwd only after
    both cases return. JSON histories may contain nonstandard NaN/Infinity;
    native and filesystem errors propagate. This is a long campaign entrypoint.
    """
    try:
        import jax

        print(f"JAX {jax.__version__}, {jax.devices()}", flush=True)
    except ImportError:
        pass

    results = {}

    # B: Reduced mass ratio (JAX, fast)
    results["explicit_1_25"] = run_case(
        "B: explicit, m_e/m_i=1/25",
        ke=True,
        implicit=False,
        mass_ratio=1.0 / 25.0,
        n_steps=5000,
    )

    # A: Semi-implicit (NumPy only, slower per step but no electron CFL)
    # Use fewer steps since NumPy is ~60× slower per step
    results["implicit_1_400"] = run_case(
        "A: implicit, m_e/m_i=1/400",
        ke=True,
        implicit=True,
        mass_ratio=1.0 / 400.0,
        n_steps=200,
    )

    out = Path("gpu_results")
    out.mkdir(exist_ok=True)
    outpath = out / "kinetic_e_dual.json"
    outpath.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nSaved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
