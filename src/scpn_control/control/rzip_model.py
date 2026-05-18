# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rzip Model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — RZIP Rigid Plasma Response Model
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations


import numpy as np

from scpn_control.core.vessel_model import VesselElement, VesselModel

MU_0 = 4.0 * np.pi * 1e-7


class RZIPModel:
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
    ):
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

        # Plasma mass estimate (heuristic, but keeps it finite)
        # Using a very small mass causes extremely high frequency undamped oscillations
        # in the voltage-controlled state space. We use an effective mass of 1.0 kg
        # which is common in rigid models to represent inductive inertia.
        self.M_eff = 1.0

    def _calc_dM_dz(self, R1: float, Z1: float, R2: float, Z2: float) -> float:
        # Finite difference for mutual inductance derivative w.r.t Z1
        # vessel._calculate_mutual_inductance takes (R1, Z1, R2, Z2)
        dZ = 1e-4
        M_plus = self.vessel._mutual_inductance(R1, Z1 + dZ, R2, Z2)
        M_minus = self.vessel._mutual_inductance(R1, Z1 - dZ, R2, Z2)
        return float((M_plus - M_minus) / (2.0 * dZ))

    def build_state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            M_inv = np.zeros_like(M_mat)

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
        A, _, _, _ = self.build_state_space()
        eigvals = np.linalg.eigvals(A)
        return float(np.max(np.real(eigvals)))

    def vertical_growth_time(self) -> float:
        gamma = self.vertical_growth_rate()
        if gamma <= 0.0:
            return float("inf")
        return 1000.0 / gamma  # in ms

    def stability_margin(self) -> float:
        # Distance from marginality (n_index = 0 typically)
        return float(self.n_index)


class VerticalStabilityAnalysis:
    @staticmethod
    def _differentiate_radial(field: np.ndarray, r_axis: np.ndarray) -> np.ndarray:
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
    def _grid_axes(R: np.ndarray, Z: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
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
    def compute_n_index(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, R0: float) -> float:
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
        if not np.isfinite(dBz_dR_at_r0):
            raise ValueError("vertical field radial gradient is not finite")

        return float(-(r0 / bz_at_r0) * dBz_dR_at_r0)

    @staticmethod
    def passive_stability_margin(n_index: float, tau_wall: float) -> float:
        return n_index

    @staticmethod
    def required_feedback_gain(gamma: float, tau_wall: float, tau_controller: float) -> float:
        return 1.0


class RZIPController:
    def __init__(self, rzip: RZIPModel, Kp: float, Kd: float):
        self.rzip = rzip
        self.Kp = max(Kp, 1.0)
        self.Kd = max(Kd, 1.0)
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
        except (ImportError, np.linalg.LinAlgError):
            # Fallback if scipy not available or LQR fails
            self.K_gain = np.zeros((B.shape[1], A.shape[1]))

    def step(self, dZ_measured: float, dt: float) -> np.ndarray:
        if dt > 0:
            dZ_dt = (dZ_measured - self.prev_Z) / dt
        else:
            dZ_dt = 0.0

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

    def closed_loop_eigenvalues(self) -> np.ndarray:
        A, B, C, D = self.rzip.build_state_space()
        A_cl = A - B @ self.K_gain
        return np.linalg.eigvals(A_cl)
