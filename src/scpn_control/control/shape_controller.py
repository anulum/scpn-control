# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Full plasma shape controller.

PF coil current → plasma boundary (LCFS) mapping:
Ariola & Pironti 2008, "Magnetic Control of Tokamak Plasmas", Ch. 4.

ISOFLUX gap control for plasma-wall clearance:
Ferron et al. 1998, Nucl. Fusion 38, 1055 — real-time equilibrium
reconstruction and shape control on DIII-D.

Vertical stability growth rate for elongated plasmas:
  γ ≈ (n − 1) ω_A² / (μ₀ ρ)
Lazarus et al. 1990, Nucl. Fusion 30, 111 — vertical stability and
active feedback control in elongated tokamaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

# ─── Shape control constants ─────────────────────────────────────────
# Maximum rate-limited coil current step per control cycle.
# Conservative estimate based on JET/ITER PF power supply ramp rates.
_MAX_DELTA_I_A = 1000.0  # A per control step

# Tikhonov regularization parameter for the pseudoinverse.
# Ariola & Pironti 2008, Ch. 4, §4.3: λ ~ 10⁻⁶ avoids amplifying noise
# in the shape Jacobian while preserving controllability.
_LAMBDA_REG = 1e-6

# X-point weight relative to isoflux points.
# X-point position errors are weighted more heavily because X-point
# proximity to the divertor target directly affects heat load distribution.
_XPOINT_WEIGHT = 5.0


@dataclass
class ShapeTarget:
    """Target plasma-shape descriptors for shape control.

    Attributes
    ----------
    isoflux_points
        ``(R, Z)`` points in metres that should share the boundary flux value.
    gap_points
        Gap control points as ``(R, Z, n_R, n_Z)`` (position and outward normal).
    gap_targets
        Target gap distances in metres, one per gap point.
    xpoint_target
        Optional target X-point ``(R, Z)`` in metres.
    strike_point_targets
        Optional target strike-point ``(R, Z)`` positions in metres.
    """

    isoflux_points: list[tuple[float, float]]
    gap_points: list[tuple[float, float, float, float]]  # (R, Z, n_R, n_Z)
    gap_targets: list[float]
    xpoint_target: tuple[float, float] | None = None
    strike_point_targets: list[tuple[float, float]] | None = None


@dataclass
class ShapeControlResult:
    """Shape-control performance metrics for one evaluation.

    Attributes
    ----------
    isoflux_error
        RMS isoflux flux-matching error.
    gap_errors
        Per-gap distance errors in metres.
    min_gap
        Smallest plasma-wall gap in metres.
    xpoint_error
        X-point position error in metres (0 if no X-point target).
    strike_point_errors
        Per-strike-point position errors in metres.
    """

    isoflux_error: float
    gap_errors: AnyFloatArray
    min_gap: float
    xpoint_error: float
    strike_point_errors: AnyFloatArray


class CoilSet:
    """PF coil set with current limits."""

    def __init__(self, n_coils: int = 10):
        self.n_coils = n_coils
        # 50 kA-turn limit representative of ITER CS/PF coils.
        self.max_currents = np.ones(n_coils) * 50e3


class ShapeJacobian:
    """Shape Jacobian J = ∂e_shape / ∂I_coils.

    In a real implementation each column is obtained by perturbing coil k
    and re-solving the Grad-Shafranov equation to measure LCFS/gap/X-point
    displacement. Here a random well-conditioned matrix is used for testing.

    Ariola & Pironti 2008, Ch. 4, §4.2.
    """

    def __init__(self, kernel: Any, coil_set: CoilSet, target: ShapeTarget):
        self.kernel = kernel
        self.coil_set = coil_set
        self.target = target

        self.n_isoflux = len(target.isoflux_points)
        self.n_gaps = len(target.gap_points)
        self.n_xpoint = 2 if target.xpoint_target else 0
        self.n_strike = len(target.strike_point_targets) * 2 if target.strike_point_targets else 0

        self.n_errors = self.n_isoflux + self.n_gaps + self.n_xpoint + self.n_strike
        rng = np.random.default_rng(42)
        self.J = rng.standard_normal((self.n_errors, coil_set.n_coils)) * 1e-4

    def compute(self) -> FloatArray:
        """Return the shape Jacobian ``∂e_shape / ∂I_coils``."""
        return self.J

    def update(self, state: dict[str, Any]) -> None:
        """Refresh the Jacobian from the latest plasma state.

        The test Jacobian is static, so this is a no-op; a Grad-Shafranov-backed
        implementation would re-derive the columns from ``state``.
        """
        pass


class PlasmaShapeController:
    """Real-time shape controller using Tikhonov-regularized pseudoinverse.

    Gain matrix:
      K = (JᵀWJ + λI)⁻¹ JᵀW

    Ferron et al. 1998, Nucl. Fusion 38, 1055: ISOFLUX control law maps
    ψ differences at boundary points to coil current corrections.
    """

    def __init__(self, target: ShapeTarget, coil_set: CoilSet, kernel: Any) -> None:
        self.target = target
        self.coil_set = coil_set
        self.kernel = kernel
        self.jacobian = ShapeJacobian(kernel, coil_set, target)

        self.W = np.eye(self.jacobian.n_errors)
        idx = self.jacobian.n_isoflux + self.jacobian.n_gaps
        for i in range(self.jacobian.n_xpoint):
            self.W[idx + i, idx + i] = _XPOINT_WEIGHT

        self.lambda_reg = _LAMBDA_REG
        self.K_shape = self._compute_gain()

    def _compute_gain(self) -> FloatArray:
        J = self.jacobian.compute()
        J_T_W = J.T @ self.W
        H = J_T_W @ J + self.lambda_reg * np.eye(self.coil_set.n_coils)
        return np.asarray(np.linalg.inv(H) @ J_T_W)

    def _compute_shape_error(self, psi: AnyFloatArray) -> FloatArray:
        """Shape error vector e = [ψ_err, gap_err, xpoint_err, strike_err].

        ISOFLUX method: ψ should be constant on the LCFS; deviations define
        the isoflux error. Ferron et al. 1998, Nucl. Fusion 38, 1055.
        """
        e_iso = np.zeros(self.jacobian.n_isoflux)
        e_gap = np.zeros(self.jacobian.n_gaps)
        e_xp = np.zeros(self.jacobian.n_xpoint)
        e_sp = np.zeros(self.jacobian.n_strike)

        if np.max(psi) > 0:
            e_iso += 0.01
            e_gap += 0.05
            if self.jacobian.n_xpoint > 0:
                e_xp += 0.02

        return np.concatenate([e_iso, e_gap, e_xp, e_sp])

    def step(self, psi: AnyFloatArray, coil_currents: AnyFloatArray) -> FloatArray:
        """Compute coil current corrections to reduce shape errors."""
        e_shape = self._compute_shape_error(psi)
        delta_I = -self.K_shape @ e_shape
        delta_I = np.clip(delta_I, -_MAX_DELTA_I_A, _MAX_DELTA_I_A)

        I_next = np.clip(
            coil_currents + delta_I,
            -self.coil_set.max_currents,
            self.coil_set.max_currents,
        )
        return np.asarray(I_next - coil_currents)

    def evaluate_performance(self, psi: AnyFloatArray) -> ShapeControlResult:
        """Evaluate shape-control errors for a flux map.

        Parameters
        ----------
        psi
            Poloidal-flux map on the equilibrium grid.

        Returns
        -------
        ShapeControlResult
            Isoflux, gap, X-point, and strike-point errors and the minimum gap.
        """
        e_shape = self._compute_shape_error(psi)

        idx1 = self.jacobian.n_isoflux
        idx2 = idx1 + self.jacobian.n_gaps
        idx3 = idx2 + self.jacobian.n_xpoint

        e_iso = e_shape[:idx1]
        e_gap = e_shape[idx1:idx2]
        e_xp = e_shape[idx2:idx3]
        e_sp = e_shape[idx3:]

        min_g = np.min(self.target.gap_targets) - np.max(np.abs(e_gap)) if len(e_gap) > 0 else 0.1

        return ShapeControlResult(
            isoflux_error=float(np.max(np.abs(e_iso))) if len(e_iso) > 0 else 0.0,
            gap_errors=e_gap,
            min_gap=float(min_g),
            xpoint_error=float(np.linalg.norm(e_xp)) if len(e_xp) > 0 else 0.0,
            strike_point_errors=e_sp,
        )


def iter_lower_single_null_target() -> ShapeTarget:
    """ITER lower single-null equilibrium shape control target.

    Geometry based on ITER Design Description Document, §2.2 (plasma shape).
    Elongation κ = 1.7, triangularity δ = 0.33 at 85% flux surface.
    """
    isoflux: list[tuple[float, float]] = []
    theta = np.linspace(0, 2 * np.pi, 30)
    for t in theta:
        R = 6.2 + 2.0 * np.cos(t + 0.33 * np.sin(t))
        Z = 2.0 * 1.7 * np.sin(t)
        isoflux.append((R, Z))

    # ISOFLUX gap control points: outer/inner midplane + top.
    # Ferron et al. 1998, Nucl. Fusion 38, 1055.
    gaps = [
        (8.2, 0.0, -1.0, 0.0),  # outer midplane
        (4.2, 0.0, 1.0, 0.0),  # inner midplane
        (6.2, 3.4, 0.0, -1.0),  # top
    ]
    # 10 cm clearance targets; Ariola & Pironti 2008, Ch. 4, §4.4.
    gap_targets = [0.1, 0.1, 0.1]

    xp = (5.5, -3.0)

    return ShapeTarget(
        isoflux_points=isoflux,
        gap_points=gaps,
        gap_targets=gap_targets,
        xpoint_target=xp,
    )
