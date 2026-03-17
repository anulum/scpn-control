# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Extended Kalman Filter (EKF) for plasma state estimation.

EKF predict-update cycle:
  x_{k+1|k} = f(x_k, u_k)
  P_{k+1|k} = F P_k Fᵀ + Q
  y_k = z_k − H x_{k|k−1}
  S_k = H P_{k|k−1} Hᵀ + R
  K_k = P_{k|k−1} Hᵀ S_k⁻¹
  x_{k|k} = x_{k|k−1} + K_k y_k
  P_{k|k} = (I − K_k H) P_{k|k−1}

Simon 2006, "Optimal State Estimation", Ch. 13.

Magnetic diagnostic-based tokamak state estimation:
Lister et al. 1997, Nucl. Fusion 37, 1633 — real-time equilibrium
reconstruction from magnetic measurements on TCV.

Observer-based current profile estimation:
Moreau et al. 2008, Nucl. Fusion 48, 106001 — model-based estimation of
the safety-factor profile q(ρ) from partial magnetic and kinetic data.
"""

from __future__ import annotations

import numpy as np

# ─── EKF state-space dimensions ─────────────────────────────────────
# State x: [R (m), Z (m), vR (m/s), vZ (m/s), Ip (MA), Te_core (keV)]
# Measurement z: [R (m), Z (m), Ip (MA), Te_core (keV)]
_NX = 6
_NZ = 4

# Index mapping for clarity (Simon 2006, Ch. 13, §13.1)
_IDX_R = 0
_IDX_Z = 1
_IDX_VR = 2
_IDX_VZ = 3
_IDX_IP = 4
_IDX_TE = 5


class ExtendedKalmanFilter:
    """EKF for 6D plasma state estimation.

    State vector: [R, Z, vR, vZ, Ip, Te_core]
    Measurement vector: [R, Z, Ip, Te_core]

    Process model: constant-velocity for (R, Z); Ip and Te_core modelled
    as random walks (zero-input prediction). The linearization is exact
    because the process model is affine, so this degenerates to a standard
    Kalman filter for the current implementation.

    Lister et al. 1997, Nucl. Fusion 37, 1633: magnetic flux and field
    measurements provide [R, Z, Ip] observability in real time.
    Moreau et al. 2008, Nucl. Fusion 48, 106001: Te_core estimated via
    ECE + model observer.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state estimate (6D).
    P0 : np.ndarray
        Initial error covariance (6×6).
    Q : np.ndarray
        Process noise covariance (6×6).
    R_cov : np.ndarray
        Measurement noise covariance (4×4).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R_cov: np.ndarray,
    ) -> None:
        self.x = x0.astype(float)
        self.P = P0.astype(float)
        self.Q = Q.astype(float)
        self.R = R_cov.astype(float)

        # Linear measurement matrix H (4×6).
        # Simon 2006, Ch. 13, Eq. (13.3).
        self.H = np.zeros((_NZ, _NX))
        self.H[0, _IDX_R] = 1.0
        self.H[1, _IDX_Z] = 1.0
        self.H[2, _IDX_IP] = 1.0
        self.H[3, _IDX_TE] = 1.0

    def predict(self, dt: float, u: np.ndarray | None = None) -> np.ndarray:
        """Advance state estimate and covariance.

        x_{k+1|k} = F x_k       (constant-velocity kinematics + random walk)
        P_{k+1|k} = F P_k Fᵀ + Q dt

        Simon 2006, Ch. 13, Eqs. (13.7)–(13.8).

        Parameters
        ----------
        dt : float
            Time step [s].
        u : np.ndarray, optional
            Control input (unused in constant-velocity model).

        Returns
        -------
        np.ndarray — Predicted state.
        """
        F = np.eye(_NX)
        F[_IDX_R, _IDX_VR] = dt
        F[_IDX_Z, _IDX_VZ] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """Correct state estimate with a new measurement.

        y   = z − H x_{k|k−1}
        S   = H P_{k|k−1} Hᵀ + R
        K   = P_{k|k−1} Hᵀ S⁻¹
        x   = x_{k|k−1} + K y
        P   = (I − K H) P_{k|k−1}

        Simon 2006, Ch. 13, Eqs. (13.9)–(13.13).

        Parameters
        ----------
        z : np.ndarray
            Measured values [R, Z, Ip, Te_core] (4D).

        Returns
        -------
        np.ndarray — Updated state.
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_NX) - K @ self.H) @ self.P
        return self.x

    def estimate(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.x.copy()
