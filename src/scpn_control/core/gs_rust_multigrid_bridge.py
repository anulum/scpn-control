# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust multigrid equilibrium bridge

"""Thin Python orchestration for the Rust multigrid equilibrium backend.

This leaf owns the CONTROL ``solver_method=rust_multigrid`` dispatch path:
availability probing, boundary-constrained fallback to Python SOR, state sync
from :class:`~scpn_control.core._rust_compat.RustAcceleratedKernel`, and
structured result packaging. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C with a thin wrapper. Rust algorithm semantics are
not changed here (polyglot boundary).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, cast, runtime_checkable

from scpn_control._typing import FloatArray

logger = logging.getLogger(__name__)


@runtime_checkable
class RustMultigridOwner(Protocol):
    """Minimal FusionKernel surface required by the rust_multigrid bridge."""

    Psi: FloatArray
    J_phi: FloatArray
    RR: FloatArray
    cfg: dict[str, Any]
    _config_path: str

    def solve_equilibrium(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Run a fixed-boundary equilibrium solve (used for Python SOR fallback)."""
        ...

    def _compute_gs_residual_rms(self, Source: FloatArray) -> float:
        """Return RMS GS residual over interior points."""
        ...


def solve_via_rust_multigrid(
    kernel: RustMultigridOwner,
    preserve_initial_state: bool = False,
    boundary_flux: FloatArray | None = None,
) -> dict[str, Any]:
    """Delegate the full equilibrium solve to the Rust multigrid backend.

    Falls back to Python SOR if:

    - a boundary-constrained solve is requested (preserve_initial_state or
      explicit boundary_flux), or
    - the Rust extension is not installed / available.

    Parameters
    ----------
    kernel :
        Live FusionKernel-like owner providing mesh state and Python SOR path.
    preserve_initial_state :
        When True, fall back to Python SOR (Rust path does not honour boundary
        constraints in this bridge).
    boundary_flux :
        Explicit boundary map; when set, fall back to Python SOR.

    Returns
    -------
    dict[str, Any]
        Structured equilibrium result including ``solver_method`` of either
        ``rust_multigrid`` or the Python fallback method (``sor``/``anderson``).
    """
    from scpn_control.core._rust_compat import RustAcceleratedKernel, _rust_available

    if preserve_initial_state or boundary_flux is not None:
        logger.warning("Boundary-constrained solve requested with rust_multigrid; falling back to Python SOR.")
        prior_method = kernel.cfg["solver"].get("solver_method", "rust_multigrid")
        kernel.cfg["solver"]["solver_method"] = "sor"
        try:
            return kernel.solve_equilibrium(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )
        finally:
            kernel.cfg["solver"]["solver_method"] = prior_method

    if not _rust_available():
        logger.warning("Rust unavailable; falling back to Python SOR.")
        prior_method = kernel.cfg["solver"].get("solver_method", "rust_multigrid")
        kernel.cfg["solver"]["solver_method"] = "sor"
        try:
            return kernel.solve_equilibrium()
        finally:
            kernel.cfg["solver"]["solver_method"] = prior_method

    t0 = time.time()
    rk = RustAcceleratedKernel(kernel._config_path)
    rk.set_solver_method("multigrid")
    rust_result = rk.solve_equilibrium()

    # Sync state back onto the owner surface (B_R/B_Z may be first assigned here).
    owner = cast(Any, kernel)
    owner.Psi = rk.Psi
    owner.J_phi = rk.J_phi
    owner.B_R = rk.B_R
    owner.B_Z = rk.B_Z

    mu0: float = kernel.cfg["physics"]["vacuum_permeability"]
    source = -mu0 * kernel.RR * kernel.J_phi
    gs_residual = kernel._compute_gs_residual_rms(source)
    elapsed = time.time() - t0
    solver_tol = float(kernel.cfg.get("solver", {}).get("convergence_threshold", 1e-4))
    practical_tol = max(solver_tol, 2e-3)
    converged = bool(rust_result.converged or rust_result.residual <= practical_tol)
    return {
        "psi": kernel.Psi,
        "converged": converged,
        "iterations": rust_result.iterations,
        "residual": rust_result.residual,
        "residual_history": [],
        "gs_residual": gs_residual,
        "gs_residual_best": gs_residual,
        "gs_residual_history": [],
        "wall_time_s": elapsed,
        "solver_method": "rust_multigrid",
    }


# Historical private name for FusionKernel wrappers.
_solve_via_rust_multigrid = solve_via_rust_multigrid
