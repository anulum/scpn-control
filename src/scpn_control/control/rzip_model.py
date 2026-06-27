# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP rigid plasma response model
"""Rigid-plasma vertical response model, stability analysis, and controller utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.vessel_model import VesselElement, VesselModel

MU_0 = 4.0 * np.pi * 1e-7
_RZIP_CALIBRATION_SCHEMA_VERSION = 1
_FACILITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "external_code_benchmark", "measured_discharge"}
)
_BOUNDED_REFERENCE_SOURCES = frozenset({"local_regression_reference", *_FACILITY_REFERENCE_SOURCES})


@dataclass(frozen=True)
class RZIPCalibrationEvidence:
    """Serialisable calibration/admission evidence for a bounded RZIP plant."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    vertical_inertia_kg: float
    wall_time_constant_s: float
    growth_rate_s_inv: float
    growth_time_ms: float
    reference_growth_rate_s_inv: float | None
    growth_rate_relative_error: float | None
    growth_rate_relative_tolerance: float
    facility_claim_allowed: bool
    claim_status: str
    evidence_payload_sha256: str


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _evidence_payload_digest(payload: dict[str, object]) -> str:
    digest_payload = dict(payload)
    digest_payload.pop("evidence_payload_sha256", None)
    return hashlib.sha256(_canonical_json(digest_payload).encode("utf-8")).hexdigest()


def _assert_evidence_digest_matches(evidence: RZIPCalibrationEvidence) -> None:
    payload = asdict(evidence)
    expected = _evidence_payload_digest(payload)
    if evidence.evidence_payload_sha256 != expected:
        raise ValueError("RZIP calibration evidence payload digest mismatch")


def _finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def rzip_calibration_evidence(
    rzip: "RZIPModel",
    *,
    source: str,
    source_id: str,
    wall_time_constant_s: float,
    model_id: str = "bounded_rzip",
    reference_growth_rate_s_inv: float | None = None,
    growth_rate_relative_tolerance: float = 0.2,
) -> RZIPCalibrationEvidence:
    """Build fail-closed calibration evidence for RZIP claim admission.

    ``local_regression_reference`` evidence is useful for deterministic
    regression reports but never permits facility claims. Facility claims require
    documented public, external-code, or measured-discharge reference sources
    and must satisfy the declared growth-rate tolerance.
    """

    if source not in _BOUNDED_REFERENCE_SOURCES:
        raise ValueError("source must be a declared RZIP reference source")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    tau_wall = _finite_positive("wall_time_constant_s", wall_time_constant_s)
    tolerance = _finite_positive("growth_rate_relative_tolerance", growth_rate_relative_tolerance)
    gamma = float(rzip.vertical_growth_rate())
    tau_ms = float(rzip.vertical_growth_time())
    if not np.isfinite(gamma):
        raise ValueError("RZIP growth rate must be finite")

    reference_gamma: float | None = None
    relative_error: float | None = None
    if reference_growth_rate_s_inv is not None:
        reference_gamma = _finite_positive("reference_growth_rate_s_inv", reference_growth_rate_s_inv)
        relative_error = abs(gamma - reference_gamma) / max(abs(reference_gamma), 1.0e-30)

    source_can_support_facility = source in _FACILITY_REFERENCE_SOURCES
    facility_allowed = bool(source_can_support_facility and relative_error is not None and relative_error <= tolerance)
    if source == "local_regression_reference":
        claim_status = "bounded local RZIP regression evidence only; external reference required for facility claims"
    elif relative_error is None:
        claim_status = "external RZIP reference source declared but quantitative growth-rate comparison is missing"
    elif facility_allowed:
        claim_status = "external RZIP reference admission passed for declared tolerance"
    else:
        claim_status = "external RZIP reference admission failed declared growth-rate tolerance"

    payload: dict[str, object] = {
        "schema_version": _RZIP_CALIBRATION_SCHEMA_VERSION,
        "source": source,
        "source_id": source_id.strip(),
        "model_id": model_id.strip(),
        "vertical_inertia_kg": float(rzip.M_eff),
        "wall_time_constant_s": tau_wall,
        "growth_rate_s_inv": gamma,
        "growth_time_ms": tau_ms,
        "reference_growth_rate_s_inv": reference_gamma,
        "growth_rate_relative_error": None if relative_error is None else float(relative_error),
        "growth_rate_relative_tolerance": tolerance,
        "facility_claim_allowed": facility_allowed,
        "claim_status": claim_status,
    }
    evidence_payload_sha256 = _evidence_payload_digest(payload)
    return RZIPCalibrationEvidence(
        schema_version=_RZIP_CALIBRATION_SCHEMA_VERSION,
        source=source,
        source_id=source_id.strip(),
        model_id=model_id.strip(),
        vertical_inertia_kg=float(rzip.M_eff),
        wall_time_constant_s=tau_wall,
        growth_rate_s_inv=gamma,
        growth_time_ms=tau_ms,
        reference_growth_rate_s_inv=reference_gamma,
        growth_rate_relative_error=None if relative_error is None else float(relative_error),
        growth_rate_relative_tolerance=tolerance,
        facility_claim_allowed=facility_allowed,
        claim_status=claim_status,
        evidence_payload_sha256=evidence_payload_sha256,
    )


def assert_rzip_facility_claim_admissible(evidence: RZIPCalibrationEvidence) -> RZIPCalibrationEvidence:
    """Return evidence or fail closed before a RZIP facility-control claim."""

    if not isinstance(evidence, RZIPCalibrationEvidence):
        raise ValueError("evidence must be RZIPCalibrationEvidence")
    if evidence.schema_version != _RZIP_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("RZIP calibration evidence schema_version is unsupported")
    if evidence.source not in _FACILITY_REFERENCE_SOURCES:
        raise ValueError("RZIP facility claim requires a facility reference source")
    if not isinstance(evidence.source_id, str) or not evidence.source_id.strip():
        raise ValueError("RZIP facility claim requires a non-empty source_id")
    if not isinstance(evidence.model_id, str) or not evidence.model_id.strip():
        raise ValueError("RZIP facility claim requires a non-empty model_id")
    _finite_positive("vertical_inertia_kg", evidence.vertical_inertia_kg)
    _finite_positive("wall_time_constant_s", evidence.wall_time_constant_s)
    _finite_positive("growth_time_ms", evidence.growth_time_ms)
    _finite_positive("growth_rate_relative_tolerance", evidence.growth_rate_relative_tolerance)
    if evidence.reference_growth_rate_s_inv is None:
        raise ValueError("RZIP facility claim requires a reference growth rate")
    _finite_positive("reference_growth_rate_s_inv", evidence.reference_growth_rate_s_inv)
    if evidence.growth_rate_relative_error is None or not np.isfinite(evidence.growth_rate_relative_error):
        raise ValueError("RZIP facility claim requires finite growth-rate comparison error")
    if evidence.growth_rate_relative_error > evidence.growth_rate_relative_tolerance:
        raise ValueError("RZIP facility claim failed declared growth-rate tolerance")
    if not np.isfinite(evidence.growth_rate_s_inv):
        raise ValueError("RZIP facility claim requires a finite model growth rate")
    _assert_evidence_digest_matches(evidence)
    if not evidence.facility_claim_allowed:
        raise ValueError(f"RZIP facility claim is not admissible: {evidence.claim_status}")
    return evidence


def save_rzip_calibration_evidence(evidence: RZIPCalibrationEvidence, path: str | Path) -> None:
    """Persist RZIP calibration evidence as deterministic JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RZIPModel:
    """Rigid-plasma (RZIP) vertical-displacement response and circuit model.

    Assembles the linear state space ``x = [Z, dZ/dt, I_1, …, I_n]`` coupling
    the vertical plasma motion to the passive vessel and active-coil currents,
    where the destabilising force constant is ``K = n_index μ₀ I_p² / (4π R0)``.

    Parameters
    ----------
    R0
        Major radius in metres; must be finite and positive.
    a
        Minor radius in metres; must be finite, positive, and below ``R0``.
    kappa
        Plasma elongation (dimensionless); must be positive.
    Ip_MA
        Plasma current in MA; must be positive.
    B0
        Toroidal field on axis in tesla; must be positive.
    n_index
        Field decay index (dimensionless); ``n < 0`` is vertically unstable.
    vessel
        Conducting-vessel model supplying element inductances and resistances.
    active_coils
        Optional active control coils appended to the circuit.
    vertical_inertia_kg
        Effective vertical inertia of the bounded RZIP plant in kg; must be
        positive.
    """

    def __init__(
        self,
        R0: float,
        a: float,
        kappa: float,
        Ip_MA: float,
        B0: float,
        n_index: float,
        vessel: VesselModel,
        active_coils: list[VesselElement] | None = None,
        vertical_inertia_kg: float = 1.0,
    ):
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and positive for a physical major radius.")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and positive for a physical minor radius.")
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering.")
        if not np.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be finite and positive.")
        if not np.isfinite(Ip_MA) or Ip_MA <= 0.0:
            raise ValueError("Ip_MA must be finite and positive.")
        if not np.isfinite(B0) or B0 <= 0.0:
            raise ValueError("B0 must be finite and positive.")
        if not np.isfinite(n_index):
            raise ValueError("n_index must be finite.")
        if not np.isfinite(vertical_inertia_kg) or vertical_inertia_kg <= 0.0:
            raise ValueError("vertical_inertia_kg must be finite and positive.")
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.Ip = Ip_MA * 1e6
        self.B0 = B0
        self.n_index = n_index
        self.vessel = vessel
        self.active_coils = active_coils if active_coils is not None else []

        # Combine vessel elements and active coils for the full circuit
        self.all_elements = self.vessel.elements + self.active_coils
        self.n_wall = len(self.vessel.elements)
        self.n_coils = len(self.active_coils)
        self.n_circuits = len(self.all_elements)

        # User-declared vertical inertia for the bounded RZIP plant. Facility
        # claims require calibration through the RZIP reference-artifact gate.
        self.M_eff = float(vertical_inertia_kg)

    def _calc_dM_dz(self, R1: float, Z1: float, R2: float, Z2: float) -> float:
        # Finite difference for mutual inductance derivative w.r.t Z1
        # vessel._calculate_mutual_inductance takes (R1, Z1, R2, Z2)
        dZ = 1e-4
        M_plus = self.vessel._mutual_inductance(R1, Z1 + dZ, R2, Z2)
        M_minus = self.vessel._mutual_inductance(R1, Z1 - dZ, R2, Z2)
        return float((M_plus - M_minus) / (2.0 * dZ))

    def build_state_space(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Assemble the linear state-space matrices of the RZIP plant.

        States are ``[Z, dZ/dt, I_1, …, I_n_circuits]`` and inputs are the
        active-coil voltages.

        Returns
        -------
        tuple of NDArray[np.float64]
            The ``(A, B, C, D)`` matrices: ``A`` is ``(n_states, n_states)``,
            ``B`` is ``(n_states, n_coils)``, ``C`` selects ``Z`` as the output,
            and ``D`` is zero.

        Raises
        ------
        ValueError
            If the circuit inductance matrix is singular.
        """
        # x = [Z, dZ/dt, I_1, ..., I_n]
        n_states = 2 + self.n_circuits
        n_inputs = self.n_coils

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.zeros((1, n_states))  # Output is just Z
        D = np.zeros((1, n_inputs))

        C[0, 0] = 1.0  # Output y = Z

        # K = n_index * mu0 * Ip^2 / (4 * pi * R0)
        K = self.n_index * MU_0 * self.Ip**2 / (4.0 * np.pi * self.R0)

        # Compute M_mat and R_mat for all circuits
        M_mat = np.zeros((self.n_circuits, self.n_circuits))
        R_mat = np.zeros((self.n_circuits, self.n_circuits))

        C_vec = np.zeros(self.n_circuits)  # dM_pj/dz * Ip

        for i in range(self.n_circuits):
            el_i = self.all_elements[i]
            R_mat[i, i] = el_i.resistance
            C_vec[i] = self._calc_dM_dz(self.R0, 0.0, el_i.R, el_i.Z) * self.Ip

            for j in range(self.n_circuits):
                el_j = self.all_elements[j]
                if i == j:
                    M_mat[i, j] = el_i.inductance
                else:
                    M_mat[i, j] = self.vessel._mutual_inductance(el_i.R, el_i.Z, el_j.R, el_j.Z)

        try:
            M_inv = np.linalg.inv(M_mat)
        except np.linalg.LinAlgError:
            raise ValueError("RZIP circuit inductance matrix must be non-singular.") from None

        # Z dot
        A[0, 1] = 1.0

        # dZ/dt dot = (-K * Z + sum(C_k * I_k)) / M_eff
        A[1, 0] = -K / self.M_eff
        for i in range(self.n_circuits):
            A[1, 2 + i] = C_vec[i] / self.M_eff

        # I dot = M_inv * (V - R*I - C * dZ/dt)
        # dI/dt = -M_inv * C * dZ/dt - M_inv * R * I + M_inv * V
        M_inv_R = M_inv @ R_mat
        M_inv_C = M_inv @ C_vec

        for i in range(self.n_circuits):
            A[2 + i, 1] = -M_inv_C[i]
            for j in range(self.n_circuits):
                A[2 + i, 2 + j] = -M_inv_R[i, j]

        # B matrix (inputs to active coils)
        for i in range(self.n_circuits):
            for j in range(self.n_coils):
                coil_idx = self.n_wall + j
                B[2 + i, j] = M_inv[i, coil_idx]

        return A, B, C, D

    def vertical_growth_rate(self) -> float:
        """Largest real eigenvalue of the open-loop plant (growth rate).

        Returns
        -------
        float
            The vertical instability growth rate in s⁻¹ (negative if stable).
        """
        A, _, _, _ = self.build_state_space()
        eigvals = np.linalg.eigvals(A)
        return float(np.max(np.real(eigvals)))

    def vertical_growth_time(self) -> float:
        """Vertical-instability e-folding time.

        Returns
        -------
        float
            The growth time in milliseconds (``inf`` if the plant is stable).
        """
        gamma = self.vertical_growth_rate()
        if gamma <= 0.0:
            return float("inf")
        return 1000.0 / gamma  # in ms

    def stability_margin(self) -> float:
        """Passive vertical-stability margin.

        Returns
        -------
        float
            The field decay index ``n_index``; positive values are passively
            stable, the marginal point being ``n_index = 0``.
        """
        # Distance from marginality (n_index = 0 typically)
        return float(self.n_index)


class VerticalStabilityAnalysis:
    """Field-based decay-index and feedback-gain stability diagnostics.

    Static helpers that compute the vertical-field decay index from a poloidal
    field map and the controller bounds for delayed RZIP stabilisation.
    """

    @staticmethod
    def _differentiate_radial(field: NDArray[np.float64], r_axis: NDArray[np.float64]) -> NDArray[np.float64]:
        derivative = np.empty_like(field, dtype=float)
        derivative[:, 0] = (field[:, 1] - field[:, 0]) / (r_axis[1] - r_axis[0])
        derivative[:, -1] = (field[:, -1] - field[:, -2]) / (r_axis[-1] - r_axis[-2])

        for idx in range(1, r_axis.size - 1):
            h_left = r_axis[idx] - r_axis[idx - 1]
            h_right = r_axis[idx + 1] - r_axis[idx]
            derivative[:, idx] = (
                -(h_right / (h_left * (h_left + h_right))) * field[:, idx - 1]
                + ((h_right - h_left) / (h_left * h_right)) * field[:, idx]
                + (h_left / (h_right * (h_left + h_right))) * field[:, idx + 1]
            )
        return derivative

    @staticmethod
    def _grid_axes(
        R: NDArray[np.float64], Z: NDArray[np.float64], shape: tuple[int, int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if R.ndim == 1 and Z.ndim == 1:
            if R.size != shape[1] or Z.size != shape[0]:
                raise ValueError("R/Z axes must match psi shape")
            r_axis = R.astype(float)
            z_axis = Z.astype(float)
        elif R.shape == shape and Z.shape == shape:
            r_axis = R[0, :].astype(float)
            z_axis = Z[:, 0].astype(float)
            if not np.allclose(R, r_axis[None, :]) or not np.allclose(Z, z_axis[:, None]):
                raise ValueError("R/Z grids must be rectilinear")
        else:
            raise ValueError("R and Z must be 1-D axes or rectilinear 2-D grids")

        if r_axis.size < 3 or z_axis.size < 3:
            raise ValueError("psi grid requires at least three R and Z points")
        if not (np.all(np.isfinite(r_axis)) and np.all(np.isfinite(z_axis))):
            raise ValueError("R/Z axes must be finite")
        if np.any(np.diff(r_axis) <= 0.0) or np.any(np.diff(z_axis) <= 0.0):
            raise ValueError("R/Z axes must be strictly increasing")
        if np.any(r_axis <= 0.0):
            raise ValueError("R axis must be positive")
        return r_axis, z_axis

    @staticmethod
    def compute_n_index(psi: NDArray[np.float64], R: NDArray[np.float64], Z: NDArray[np.float64], R0: float) -> float:
        """
        Compute the vertical field index at the magnetic-axis radius.

        The axisymmetric poloidal flux convention gives
        ``B_Z = (1 / R) d psi / dR``.  The RZIP vertical stability index is
        then evaluated on the midplane as ``n = -(R0 / B_Z) dB_Z / dR``.
        """
        psi_arr = np.asarray(psi, dtype=float)
        r_arr = np.asarray(R, dtype=float)
        z_arr = np.asarray(Z, dtype=float)
        r0 = float(R0)

        if psi_arr.ndim != 2:
            raise ValueError("psi must be a 2-D rectilinear flux grid")
        if not np.all(np.isfinite(psi_arr)):
            raise ValueError("psi grid must be finite")
        if not np.isfinite(r0) or r0 <= 0.0:
            raise ValueError("R0 must be finite and positive")

        psi_shape = (psi_arr.shape[0], psi_arr.shape[1])
        r_axis, z_axis = VerticalStabilityAnalysis._grid_axes(r_arr, z_arr, psi_shape)
        if r0 < r_axis[0] or r0 > r_axis[-1]:
            raise ValueError("R0 must lie inside the R grid")

        dpsi_dR = VerticalStabilityAnalysis._differentiate_radial(psi_arr, r_axis)
        bz = dpsi_dR / r_axis[None, :]
        dBz_dR = VerticalStabilityAnalysis._differentiate_radial(bz, r_axis)

        midplane_idx = int(np.argmin(np.abs(z_axis)))
        bz_midplane = bz[midplane_idx, :]
        dBz_dR_midplane = dBz_dR[midplane_idx, :]

        bz_at_r0 = float(np.interp(r0, r_axis, bz_midplane))
        dBz_dR_at_r0 = float(np.interp(r0, r_axis, dBz_dR_midplane))
        if not np.isfinite(bz_at_r0) or abs(bz_at_r0) <= 1e-12:
            raise ValueError("vertical field at R0 is degenerate")
        # psi is guarded finite and r_axis is finite, positive and strictly
        # increasing, so the finite-difference field and its radial gradient are
        # finite everywhere; this guard is defence in depth and is unreachable.
        if not np.isfinite(dBz_dR_at_r0):
            raise ValueError(
                "vertical field radial gradient is not finite"
            )  # pragma: no cover - validator rejects non-finite gradient earlier

        return float(-(r0 / bz_at_r0) * dBz_dR_at_r0)

    @staticmethod
    def passive_stability_margin(n_index: float, tau_wall: float) -> float:
        """Passive vertical-stability margin set by the field decay index.

        Parameters
        ----------
        n_index
            Field decay index (dimensionless); must be finite.
        tau_wall
            Wall vertical-stabilisation time constant in seconds; must be
            positive.

        Returns
        -------
        float
            The decay index ``n_index`` (the passive margin proxy).
        """
        if not np.isfinite(n_index):
            raise ValueError("n_index must be finite.")
        if not np.isfinite(tau_wall) or tau_wall <= 0.0:
            raise ValueError("tau_wall must be finite and positive.")
        return float(n_index)

    @staticmethod
    def required_feedback_gain(gamma: float, tau_wall: float, tau_controller: float) -> float:
        """Minimum wall-normalised feedback gain for delayed RZIP stabilisation.

        For the bounded linear RZIP channel, a proportional vertical controller
        with first-order latency reduces the open-loop growth rate as

            gamma_cl = gamma - K / (tau_wall * (1 + gamma * tau_controller)).

        Requiring ``gamma_cl < 0`` gives the dimensionless threshold

            K > gamma * tau_wall * (1 + gamma * tau_controller).

        The expression is a controller-design bound for the local RZIP plant,
        not a facility gain-calibration claim.
        """
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative.")
        if not np.isfinite(tau_wall) or tau_wall <= 0.0:
            raise ValueError("tau_wall must be finite and positive.")
        if not np.isfinite(tau_controller) or tau_controller < 0.0:
            raise ValueError("tau_controller must be finite and non-negative.")
        if gamma == 0.0:
            return 0.0
        return float(gamma * tau_wall * (1.0 + gamma * tau_controller))


class RZIPController:
    """LQR vertical-position controller for the RZIP plant.

    Solves the continuous-time algebraic Riccati equation for the optimal gain
    weighting position (``Kp``) and velocity (``Kd``). If SciPy's Riccati input
    validation fails because of a local numerical-stack mismatch, the controller
    falls back to a finite-horizon discrete Riccati iteration for the same
    plant. Zero gain is the final fail-closed fallback when no feedback design
    can be computed.

    Parameters
    ----------
    rzip
        The RZIP plant model providing the state space.
    Kp
        Position state-cost weight (dimensionless); floored at 1.0, must be
        non-negative.
    Kd
        Velocity state-cost weight (dimensionless); floored at 1.0, must be
        non-negative.
    """

    def __init__(self, rzip: RZIPModel, Kp: float, Kd: float):
        self.rzip = rzip
        if not np.isfinite(Kp) or Kp < 0.0:
            raise ValueError("Kp must be finite and non-negative.")
        if not np.isfinite(Kd) or Kd < 0.0:
            raise ValueError("Kd must be finite and non-negative.")
        self.Kp = max(float(Kp), 1.0)
        self.Kd = max(float(Kd), 1.0)
        self.prev_Z = 0.0

        # Precompute LQR optimal gain matrix
        A, B, C, D = self.rzip.build_state_space()

        try:
            import scipy.linalg

            Q = np.zeros_like(A)
            Q[0, 0] = self.Kp
            Q[1, 1] = self.Kd
            R = np.eye(B.shape[1]) * 1.0

            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.K_gain = np.linalg.inv(R) @ B.T @ P
        except TypeError:
            try:
                self.K_gain = self._discrete_riccati_fallback_gain(A, B, Q, R)
            except (np.linalg.LinAlgError, TypeError, ValueError):
                self.K_gain = np.zeros((B.shape[1], A.shape[1]))
        except (ImportError, np.linalg.LinAlgError):
            # Fallback if scipy not available or LQR fails
            self.K_gain = np.zeros((B.shape[1], A.shape[1]))

    @staticmethod
    def _discrete_riccati_fallback_gain(
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute a bounded NumPy-only state-feedback fallback gain."""

        spectral_radius = max(float(np.max(np.abs(np.linalg.eigvals(A)))), 1.0)
        dt = 0.1 / spectral_radius
        A_d = np.eye(A.shape[0]) + dt * A
        B_d = dt * B
        P = Q.copy()
        for _ in range(256):
            gain = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
            if not np.all(np.isfinite(gain)):
                raise np.linalg.LinAlgError("discrete Riccati fallback produced non-finite gain")
            next_P = Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ gain
            if not np.all(np.isfinite(next_P)):
                raise np.linalg.LinAlgError("discrete Riccati fallback produced non-finite covariance")
            P = next_P
        return np.asarray(np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d), dtype=float)

    def step(self, dZ_measured: float, dt: float) -> NDArray[np.float64]:
        """Compute the active-coil voltage command for one control cycle.

        Estimates the vertical velocity from successive position measurements and
        applies the LQR law ``u = -K x`` with coil and wall currents assumed zero
        (no observer).

        Parameters
        ----------
        dZ_measured
            Measured vertical displacement in metres; must be finite.
        dt
            Control time step in seconds; must be positive.

        Returns
        -------
        NDArray[np.float64]
            The commanded active-coil voltages in volts, shape ``(n_coils,)``.
        """
        if not np.isfinite(dZ_measured):
            raise ValueError("dZ_measured must be finite.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        dZ_dt = (dZ_measured - self.prev_Z) / dt

        self.prev_Z = dZ_measured

        # We only have state measurement for Z and dZ/dt in this simple step.
        # Assume coil and wall currents are 0 for the instantaneous voltage command
        # or require a full state observer (not implemented here).
        x = np.zeros(self.rzip.n_circuits + 2)
        x[0] = dZ_measured
        x[1] = dZ_dt

        # Voltage command u = -K x
        V_coils = -self.K_gain @ x
        return np.asarray(V_coils)

    def closed_loop_eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the LQR closed-loop system matrix ``A - B K``.

        Returns
        -------
        NDArray[np.float64]
            The complex closed-loop eigenvalues; negative real parts indicate a
            stabilised vertical mode.
        """
        A, B, C, D = self.rzip.build_state_space()
        A_cl = A - B @ self.K_gain
        return np.linalg.eigvals(A_cl)
