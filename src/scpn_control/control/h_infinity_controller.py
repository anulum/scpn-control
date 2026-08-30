# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — normalized continuous-time H-infinity output feedback.

"""Normalized DGKF H-infinity output-feedback synthesis and runtime.

The supported generalized plant is the continuous-time standard form

``dx/dt = A x + B1 w + B2 u``
``z     = C1 x + D12 u``
``y     = C2 x + D21 w``

with ``D11 = 0`` and ``D22 = 0``. The implementation deliberately admits only
the normalized Doyle-Glover-Khargonekar-Francis (DGKF) problem:

``D12.T @ [C1, D12] = [0, I]`` and
``[B1; D21] @ D21.T = [0; I]``.

For a strictly feasible attenuation ``gamma``, the two stabilizing Riccati
solutions define the central dynamic controller from Doyle et al., IEEE TAC
34(8), 1989, Theorem 3, DOI 10.1109/9.29425. The runtime uses an exact
zero-order-hold discretization of that admitted realization. The H-infinity
theorem applies to the unsaturated linear continuous-time interconnection;
output clipping is a separate safety boundary without that guarantee.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, solve_continuous_are

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core._validators import require_finite_array, require_positive_float

_NORMALIZATION_ATOL = 1.0e-9
_PSD_ATOL = 1.0e-8
_STABILITY_ATOL = 1.0e-9
_SPECTRAL_RELATIVE_GAP = 1.0e-9


def _zoh_discretize(
    state_matrix: AnyFloatArray,
    input_matrix: AnyFloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the exact zero-order-hold realization for one sample period."""
    n_states = state_matrix.shape[0]
    n_inputs = input_matrix.shape[1]
    augmented = np.zeros((n_states + n_inputs, n_states + n_inputs))
    augmented[:n_states, :n_states] = state_matrix * dt
    augmented[:n_states, n_states:] = input_matrix * dt
    exponential = expm(augmented)
    return exponential[:n_states, :n_states], exponential[:n_states, n_states:]


def _matrix_rank(matrix: AnyFloatArray, tolerance: float) -> int:
    """Return a deterministic singular-value rank at ``tolerance``."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _is_stabilizable(
    state_matrix: AnyFloatArray,
    input_matrix: AnyFloatArray,
    tolerance: float,
) -> bool:
    """Evaluate the continuous-time PBH stabilizability condition."""
    n_states = state_matrix.shape[0]
    for eigenvalue in np.linalg.eigvals(state_matrix):
        if eigenvalue.real >= -tolerance:
            pbh = np.hstack((eigenvalue * np.eye(n_states) - state_matrix, input_matrix))
            if _matrix_rank(pbh, tolerance) < n_states:
                return False
    return True


def _is_detectable(
    output_matrix: AnyFloatArray,
    state_matrix: AnyFloatArray,
    tolerance: float,
) -> bool:
    """Evaluate detectability by duality with PBH stabilizability."""
    return _is_stabilizable(state_matrix.T, output_matrix.T, tolerance)


@dataclass(frozen=True)
class _Synthesis:
    """Internal immutable result of one feasible DGKF synthesis."""

    X: FloatArray
    Y: FloatArray
    F: FloatArray
    L: FloatArray
    Z: FloatArray
    Ak: FloatArray
    Bk: FloatArray
    Ck: FloatArray
    spectral_radius_xy: float
    closed_loop_eigenvalues: npt.NDArray[np.complex128]


class HInfinityController:
    """Central DGKF controller for a normalized continuous-time standard plant.

    ``D12`` and ``D21`` are required because silently inventing performance or
    sensor-noise channels would change the declared robust-control problem. If
    ``gamma`` is omitted, logarithmic bisection finds a feasible near-infimum
    and adds 0.5 percent admission headroom.

    ``enforce_robust_feasibility`` is a deprecated compatibility flag.
    Feasibility is always enforced; passing ``False`` warns but cannot bypass
    the DGKF existence conditions.
    """

    _GAMMA_SEARCH_MIN = 1.0e-6
    _GAMMA_SEARCH_MAX = 1.0e8
    _GAMMA_FEASIBILITY_PAD = 1.005

    def __init__(
        self,
        A: npt.ArrayLike,
        B1: npt.ArrayLike,
        B2: npt.ArrayLike,
        C1: npt.ArrayLike,
        C2: npt.ArrayLike,
        gamma: float | None = None,
        D12: npt.ArrayLike | None = None,
        D21: npt.ArrayLike | None = None,
        enforce_robust_feasibility: bool = True,
    ) -> None:
        self.A = self._as_matrix("A", A)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a finite square matrix.")

        self.n = self.A.shape[0]
        self.B1 = self._as_oriented_matrix("B1", B1, rows=self.n)
        self.B2 = self._as_oriented_matrix("B2", B2, rows=self.n)
        self.C1 = self._as_oriented_matrix("C1", C1, columns=self.n)
        self.C2 = self._as_oriented_matrix("C2", C2, columns=self.n)
        self.p = self.B1.shape[1]
        self.m = self.B2.shape[1]
        self.q = self.C1.shape[0]
        self.l = self.C2.shape[0]

        if D12 is None or D21 is None:
            raise ValueError(
                "D12 and D21 are required for the normalized DGKF contract; "
                "use a documented factory or supply explicit feedthrough matrices."
            )
        self.D12 = self._as_exact_matrix("D12", D12, (self.q, self.m))
        self.D21 = self._as_exact_matrix("D21", D21, (self.l, self.p))
        self._validate_normalized_standard_plant()

        if not enforce_robust_feasibility:
            warnings.warn(
                "enforce_robust_feasibility=False no longer bypasses the DGKF "
                "existence conditions; infeasible synthesis always fails closed.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.gamma = self._find_optimal_gamma() if gamma is None else require_positive_float("gamma", gamma)
        synthesis = self._synthesize(self.gamma)
        self.X = synthesis.X
        self.Y = synthesis.Y
        self.F = synthesis.F
        self.L = synthesis.L
        self.L_gain = -synthesis.L
        self.Z = synthesis.Z
        self.Ak = synthesis.Ak
        self.Bk = synthesis.Bk
        self.Ck = synthesis.Ck
        self.Dk = np.zeros((self.m, self.l))
        self.spectral_radius_xy = synthesis.spectral_radius_xy
        self.closed_loop_eigenvalues = synthesis.closed_loop_eigenvalues
        self.robust_feasible = True

        self.u_max = 1.0e8
        self.state = np.zeros(self.n)
        self._cached_dt = 0.0
        self._Ak_d: FloatArray = np.eye(self.n)
        self._Bk_d: FloatArray = np.zeros((self.n, self.l))

    @staticmethod
    def _as_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
        """Convert one finite one- or two-dimensional value to a matrix."""
        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            array = np.atleast_2d(array)
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name} must be a non-empty one- or two-dimensional matrix.")
        require_finite_array(name, array)
        return np.asarray(array)

    @classmethod
    def _as_oriented_matrix(
        cls,
        name: str,
        value: npt.ArrayLike,
        *,
        rows: int | None = None,
        columns: int | None = None,
    ) -> FloatArray:
        """Orient a one-dimensional B as a column or C as a row."""
        raw: FloatArray = np.asarray(value, dtype=float)
        array: FloatArray
        if raw.ndim == 1:
            array = raw.reshape((-1, 1)) if rows is not None else raw.reshape((1, -1))
        else:
            array = cls._as_matrix(name, raw)
        require_finite_array(name, array)
        if rows is not None and array.shape[0] != rows:
            raise ValueError(f"{name} row count must match A ({rows}).")
        if columns is not None and array.shape[1] != columns:
            raise ValueError(f"{name} column count must match A ({columns}).")
        return np.asarray(array)

    @classmethod
    def _as_exact_matrix(
        cls,
        name: str,
        value: npt.ArrayLike,
        shape: tuple[int, int],
    ) -> FloatArray:
        """Return a finite matrix with one exact required shape."""
        matrix = cls._as_matrix(name, value)
        if matrix.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        return matrix

    def _validate_normalized_standard_plant(self) -> None:
        """Fail closed unless the normalized standard-plant assumptions hold."""
        tolerance = _NORMALIZATION_ATOL
        if not _is_stabilizable(self.A, self.B1, tolerance):
            raise ValueError("(A, B1) must be stabilizable for normalized DGKF synthesis.")
        if not _is_detectable(self.C1, self.A, tolerance):
            raise ValueError("(C1, A) must be detectable for normalized DGKF synthesis.")
        if not _is_stabilizable(self.A, self.B2, tolerance):
            raise ValueError("(A, B2) must be stabilizable for normalized DGKF synthesis.")
        if not _is_detectable(self.C2, self.A, tolerance):
            raise ValueError("(C2, A) must be detectable for normalized DGKF synthesis.")

        if not np.allclose(self.D12.T @ self.C1, 0.0, atol=tolerance, rtol=0.0):
            raise ValueError("D12.T @ C1 must be zero for the normalized DGKF contract.")
        if not np.allclose(self.D12.T @ self.D12, np.eye(self.m), atol=tolerance, rtol=0.0):
            raise ValueError("D12.T @ D12 must equal I for the normalized DGKF contract.")
        if not np.allclose(self.B1 @ self.D21.T, 0.0, atol=tolerance, rtol=0.0):
            raise ValueError("B1 @ D21.T must be zero for the normalized DGKF contract.")
        if not np.allclose(self.D21 @ self.D21.T, np.eye(self.l), atol=tolerance, rtol=0.0):
            raise ValueError("D21 @ D21.T must equal I for the normalized DGKF contract.")

    def normalization_residual_norms(self) -> tuple[float, float, float, float]:
        """Return Frobenius residuals for the four normalization identities."""
        return (
            float(np.linalg.norm(self.D12.T @ self.C1, ord="fro")),
            float(np.linalg.norm(self.D12.T @ self.D12 - np.eye(self.m), ord="fro")),
            float(np.linalg.norm(self.B1 @ self.D21.T, ord="fro")),
            float(np.linalg.norm(self.D21 @ self.D21.T - np.eye(self.l), ord="fro")),
        )

    def _synthesize(self, gamma: float) -> _Synthesis:
        """Synthesize and admit the central DGKF controller at ``gamma``."""
        gamma = require_positive_float("gamma", gamma)
        gamma_squared = gamma * gamma

        b_x = np.hstack((self.B2, self.B1 / gamma))
        r_x = np.diag(np.concatenate((np.ones(self.m), -np.ones(self.p))))
        X = solve_continuous_are(self.A, b_x, self.C1.T @ self.C1, r_x)
        X = np.asarray(0.5 * (X + X.T))

        b_y = np.hstack((self.C2.T, self.C1.T / gamma))
        r_y = np.diag(np.concatenate((np.ones(self.l), -np.ones(self.q))))
        Y = solve_continuous_are(self.A.T, b_y, self.B1 @ self.B1.T, r_y)
        Y = np.asarray(0.5 * (Y + Y.T))

        require_finite_array("X Riccati solution", X)
        require_finite_array("Y Riccati solution", Y)
        if float(np.min(np.linalg.eigvalsh(X))) < -_PSD_ATOL:
            raise ValueError("X is not positive semidefinite; gamma is not DGKF-feasible.")
        if float(np.min(np.linalg.eigvalsh(Y))) < -_PSD_ATOL:
            raise ValueError("Y is not positive semidefinite; gamma is not DGKF-feasible.")

        x_closed = self.A + (self.B1 @ self.B1.T / gamma_squared - self.B2 @ self.B2.T) @ X
        y_closed = self.A.T + (self.C1.T @ self.C1 / gamma_squared - self.C2.T @ self.C2) @ Y
        if np.max(np.real(np.linalg.eigvals(x_closed))) >= -_STABILITY_ATOL:
            raise ValueError("X is not the stabilizing DGKF Riccati solution.")
        if np.max(np.real(np.linalg.eigvals(y_closed))) >= -_STABILITY_ATOL:
            raise ValueError("Y is not the stabilizing DGKF Riccati solution.")

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(X @ Y))))
        if spectral_radius >= gamma_squared * (1.0 - _SPECTRAL_RELATIVE_GAP):
            raise ValueError(
                "H-infinity spectral feasibility condition failed: "
                f"rho(XY)={spectral_radius:.9g} is not strictly below gamma^2={gamma_squared:.9g}."
            )

        F = np.asarray(-self.B2.T @ X)
        L = np.asarray(-Y @ self.C2.T)
        coupling = np.eye(self.n) - Y @ X / gamma_squared
        try:
            Z = np.asarray(np.linalg.solve(coupling, np.eye(self.n)))
        except np.linalg.LinAlgError as error:
            raise ValueError("DGKF coupling matrix is singular.") from error
        Ak = np.asarray(self.A + self.B1 @ self.B1.T @ X / gamma_squared + self.B2 @ F + Z @ L @ self.C2)
        Bk = np.asarray(-Z @ L)
        Ck = F.copy()
        for name, matrix in (("F", F), ("L", L), ("Z", Z), ("Ak", Ak), ("Bk", Bk), ("Ck", Ck)):
            require_finite_array(name, matrix)

        closed_loop_state, _, _, _ = self._closed_loop_realization(Ak, Bk, Ck)
        closed_loop_eigenvalues = np.asarray(np.linalg.eigvals(closed_loop_state))
        if np.max(np.real(closed_loop_eigenvalues)) >= -_STABILITY_ATOL:
            raise ValueError("DGKF central controller does not internally stabilize the plant.")

        return _Synthesis(
            X=X,
            Y=Y,
            F=F,
            L=L,
            Z=Z,
            Ak=Ak,
            Bk=Bk,
            Ck=Ck,
            spectral_radius_xy=spectral_radius,
            closed_loop_eigenvalues=closed_loop_eigenvalues,
        )

    def _find_optimal_gamma(
        self,
        gamma_min: float = _GAMMA_SEARCH_MIN,
        gamma_max: float = _GAMMA_SEARCH_MAX,
        rtol: float = 1.0e-6,
        max_iter: int = 100,
    ) -> float:
        """Return a padded near-infimum feasible attenuation by log bisection."""
        gamma_min = require_positive_float("gamma_min", gamma_min)
        gamma_max = require_positive_float("gamma_max", gamma_max)
        rtol = require_positive_float("rtol", rtol)
        if gamma_min >= gamma_max:
            raise ValueError("gamma_min must be strictly below gamma_max.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        try:
            self._synthesize(gamma_max)
        except (np.linalg.LinAlgError, ValueError) as error:
            raise ValueError(f"No feasible gamma found at search upper bound {gamma_max:g}.") from error

        lower = gamma_min
        upper = gamma_max
        for _ in range(max_iter):
            candidate = float(np.sqrt(lower * upper))
            try:
                self._synthesize(candidate)
            except (np.linalg.LinAlgError, ValueError):
                lower = candidate
            else:
                upper = candidate
            if upper / lower - 1.0 <= rtol:
                break

        admitted = upper * self._GAMMA_FEASIBILITY_PAD
        self._synthesize(admitted)
        return admitted

    def _closed_loop_realization(
        self,
        controller_state: AnyFloatArray,
        controller_input: AnyFloatArray,
        controller_output: AnyFloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Assemble the augmented plant-controller map from ``w`` to ``z``."""
        state = np.block(
            [
                [self.A, self.B2 @ controller_output],
                [controller_input @ self.C2, controller_state],
            ]
        )
        disturbance = np.vstack((self.B1, controller_input @ self.D21))
        performance = np.hstack((self.C1, self.D12 @ controller_output))
        feedthrough = np.zeros((self.q, self.p))
        return np.asarray(state), np.asarray(disturbance), np.asarray(performance), feedthrough

    def closed_loop_realization(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return the admitted continuous closed-loop realization from ``w`` to ``z``."""
        return self._closed_loop_realization(self.Ak, self.Bk, self.Ck)

    def _update_discretization(self, dt: float) -> None:
        """Cache the exact ZOH discretization of the controller realization."""
        self._Ak_d, self._Bk_d = _zoh_discretize(self.Ak, self.Bk, dt)
        self._cached_dt = dt

    def step(self, error: npt.ArrayLike, dt: float) -> float | FloatArray:
        """Advance the controller from one finite measurement sample.

        The output is evaluated before the state update, avoiding an artificial
        sample delay. ``u_max`` clipping is outside the linear H-infinity claim.
        """
        dt = require_positive_float("dt", dt)
        measurement = np.asarray(error, dtype=float)
        if measurement.ndim == 0:
            measurement = measurement.reshape(1)
        if measurement.shape != (self.l,):
            raise ValueError(f"error must have shape ({self.l},).")
        require_finite_array("error", measurement)
        u_limit = require_positive_float("u_max", self.u_max)

        if dt != self._cached_dt:
            self._update_discretization(dt)

        raw_control = np.asarray(self.Ck @ self.state)
        control = np.asarray(np.clip(raw_control, -u_limit, u_limit))
        self.state = np.asarray(self._Ak_d @ self.state + self._Bk_d @ measurement)
        if self.m == 1:
            return float(control.item())
        return control

    def riccati_residual_norms(self) -> tuple[float, float]:
        """Return Frobenius residuals of the two normalized DGKF equations."""
        gamma_squared = self.gamma * self.gamma
        residual_x = (
            self.A.T @ self.X
            + self.X @ self.A
            + self.C1.T @ self.C1
            + self.X @ (self.B1 @ self.B1.T / gamma_squared - self.B2 @ self.B2.T) @ self.X
        )
        residual_y = (
            self.A @ self.Y
            + self.Y @ self.A.T
            + self.B1 @ self.B1.T
            + self.Y @ (self.C1.T @ self.C1 / gamma_squared - self.C2.T @ self.C2) @ self.Y
        )
        return float(np.linalg.norm(residual_x, ord="fro")), float(np.linalg.norm(residual_y, ord="fro"))

    def robust_feasibility_margin(self) -> float:
        """Return ``gamma**2 - rho(XY)``; admitted controllers are positive."""
        return float(self.gamma * self.gamma - self.spectral_radius_xy)

    def reset(self) -> None:
        """Reset the dynamic controller state to zero."""
        self.state = np.zeros(self.n)

    @property
    def is_stable(self) -> bool:
        """Return whether the augmented unsaturated continuous closed loop is stable."""
        return bool(np.max(np.real(self.closed_loop_eigenvalues)) < -_STABILITY_ATOL)

    @property
    def stability_margin_db(self) -> float:
        """Return a legacy pole-displacement diagnostic, not a Bode gain margin."""
        dominant_closed = float(np.max(np.real(self.closed_loop_eigenvalues)))
        if dominant_closed >= 0.0:
            return 0.0
        dominant_open = float(np.max(np.real(np.linalg.eigvals(self.A))))
        if dominant_open <= 0.0:
            return float("inf")
        return float(20.0 * np.log10(1.0 + (-dominant_closed / dominant_open)))

    @property
    def gain_margin_db(self) -> float:
        """Return the deprecated pole-displacement diagnostic."""
        warnings.warn(
            "gain_margin_db is a pole-displacement diagnostic, not a classical gain margin; "
            "use explicit loop-transfer analysis for a gain margin.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stability_margin_db


def get_flight_sim_controller(
    response_gain: float = 0.05,
    actuator_tau: float = 0.06,
    enforce_robust_feasibility: bool = True,
) -> HInfinityController:
    """Return a normalized two-state flight-simulator H-infinity example.

    The first disturbance channel drives position and the second represents
    unit sensor noise. The third performance output is the unit control channel
    required by normalized synthesis. This is not a facility-identified model.
    """
    response_gain = require_positive_float("response_gain", response_gain)
    actuator_tau = require_positive_float("actuator_tau", actuator_tau)
    inverse_tau = 1.0 / actuator_tau
    A = np.array([[1.0, -response_gain], [0.0, -inverse_tau]])
    B1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    B2 = np.array([[0.0], [inverse_tau]])
    C1 = np.array([[1.0, 0.0], [0.0, 0.01], [0.0, 0.0]])
    C2 = np.array([[1.0, 0.0]])
    D12 = np.array([[0.0], [0.0], [1.0]])
    D21 = np.array([[0.0, 1.0]])
    return HInfinityController(
        A,
        B1,
        B2,
        C1,
        C2,
        D12=D12,
        D21=D21,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )


def get_radial_robust_controller(
    gamma_growth: float = 100.0,
    *,
    damping: float = 10.0,
    enforce_robust_feasibility: bool = True,
) -> HInfinityController:
    """Return a normalized reduced vertical-instability H-infinity example.

    A physical disturbance channel enters acceleration; an orthogonal unit
    channel represents measurement noise. Performance penalizes position and
    unit control effort. The model is pedagogical until SPO identifies a
    scenario and FUSION releases matched plant, diagnostic, and actuator
    contracts.
    """
    gamma_growth = require_positive_float("gamma_growth", gamma_growth)
    damping = require_positive_float("damping", damping)
    A = np.array([[0.0, 1.0], [gamma_growth * gamma_growth, -damping]])
    B1 = np.array([[0.0, 0.0], [0.5, 0.0]])
    B2 = np.array([[0.0], [1.0]])
    C1 = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    C2 = np.array([[1.0, 0.0]])
    D12 = np.array([[0.0], [0.0], [1.0]])
    D21 = np.array([[0.0, 1.0]])
    return HInfinityController(
        A,
        B1,
        B2,
        C1,
        C2,
        D12=D12,
        D21=D21,
        enforce_robust_feasibility=enforce_robust_feasibility,
    )
