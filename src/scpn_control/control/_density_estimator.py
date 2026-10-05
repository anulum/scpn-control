# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Interferometric density Kalman estimator

"""Interferometric density Kalman estimator."""

from __future__ import annotations

import math
import warnings
from typing import cast

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

_KALMAN_GEOMETRY_ALIAS_REMOVAL_VERSION = "0.25.0"


class KalmanDensityEstimator:
    """Linear Kalman filter for bounded interferometric density reconstruction.

    The state contains piecewise-constant electron densities in uniform annular
    shells over normalised minor radius ``rho in [0, 1]``. The measurement model
    integrates each shell along a straight chord through a circular,
    axisymmetric plasma. Consequently, the projection matrix has units of
    metres and maps density in m⁻³ to line-integrated density in m⁻².

    This is a bounded linear-Gaussian model. It does not model shaped magnetic
    surfaces, ray refraction, diagnostic calibration, or a constrained
    non-negative posterior.

    Parameters
    ----------
    n_rho
        Number of radial state grid points.
    n_chords
        Number of interferometer chords (measurements).
    minor_radius_m
        Circular plasma minor radius in metres.

    Raises
    ------
    ValueError
        If a dimension is not a positive integer or the minor radius is not
        finite and positive.
    """

    def __init__(self, n_rho: int, n_chords: int = 8, minor_radius_m: float = 1.0):
        if isinstance(n_rho, bool) or not isinstance(n_rho, int) or n_rho < 1:
            raise ValueError("n_rho must be a positive integer.")
        if isinstance(n_chords, bool) or not isinstance(n_chords, int) or n_chords < 1:
            raise ValueError("n_chords must be a positive integer.")
        if not math.isfinite(minor_radius_m) or minor_radius_m <= 0.0:
            raise ValueError("minor_radius_m must be finite and positive.")
        self.n_rho = n_rho
        self.n_chords = n_chords
        self.minor_radius_m = minor_radius_m
        self.x: FloatArray = np.zeros(n_rho)
        self.P = np.eye(n_rho) * 1e38
        self.Q = np.eye(n_rho) * 1e36  # density-state process covariance per second
        self.R = np.eye(n_chords) * 1e34  # line-integrated measurement covariance

    def measurement_matrix(
        self,
        chord_impacts: AnyFloatArray | None = None,
        *,
        chord_angles: AnyFloatArray | None = None,
    ) -> FloatArray:
        """Return the exact annular straight-chord projection matrix.

        Parameters
        ----------
        chord_impacts
            Signed chord impact coordinates divided by ``minor_radius_m``,
            shape ``(n_chords,)`` and bounded to ``[-1, 1]``. Sign is retained
            as geometry metadata but the circular projection is even in the
            impact coordinate.
        chord_angles
            Deprecated keyword alias for ``chord_impacts``. Values use the
            same normalised impact-coordinate contract; they are not angles in
            radians.

        Returns
        -------
        FloatArray
            Matrix with shape ``(n_chords, n_rho)`` and units metres. For shell
            ``[rho_j, rho_{j+1}]`` and normalised impact ``b``, the entry is
            ``2 a (sqrt(rho_{j+1}^2-b^2)_+ - sqrt(rho_j^2-b^2)_+)``.

        Raises
        ------
        ValueError
            If the geometry is absent, ambiguous, incorrectly shaped, non-finite, or
            outside the circular plasma.
        """
        impacts = self._resolve_chord_impacts(chord_impacts, chord_angles)
        shell_edges = np.linspace(0.0, 1.0, self.n_rho + 1)
        impact_sq = np.square(impacts)[:, np.newaxis]
        outer_sq = np.maximum(np.square(shell_edges[1:])[np.newaxis, :] - impact_sq, 0.0)
        inner_sq = np.maximum(np.square(shell_edges[:-1])[np.newaxis, :] - impact_sq, 0.0)
        projection = 2.0 * self.minor_radius_m * (np.sqrt(outer_sq) - np.sqrt(inner_sq))
        return cast(FloatArray, projection)

    def _resolve_chord_impacts(
        self, chord_impacts: AnyFloatArray | None, chord_angles: AnyFloatArray | None
    ) -> FloatArray:
        if chord_impacts is not None and chord_angles is not None:
            raise ValueError("Pass chord_impacts or deprecated chord_angles, not both.")
        if chord_impacts is None:
            if chord_angles is None:
                raise ValueError("chord_impacts must be provided.")
            warnings.warn(
                "chord_angles is deprecated; pass signed normalised chord_impacts instead; "
                f"removal is scheduled for {_KALMAN_GEOMETRY_ALIAS_REMOVAL_VERSION}.",
                DeprecationWarning,
                stacklevel=3,
            )
            chord_impacts = chord_angles
        impacts = self._validate_vector(chord_impacts, "chord_impacts", self.n_chords)
        if np.any(np.abs(impacts) > 1.0):
            raise ValueError("chord_impacts must lie within [-1, 1].")
        return impacts

    @staticmethod
    def _validate_vector(values: AnyFloatArray, name: str, size: int) -> FloatArray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},).")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values.")
        return vector.copy()

    @staticmethod
    def _validate_covariance(values: AnyFloatArray, name: str, size: int, *, positive_definite: bool) -> FloatArray:
        covariance = np.asarray(values, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}).")
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
            raise ValueError(f"{name} must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        if positive_definite:
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{name} must be positive definite.") from exc
        else:
            eigenvalues = np.linalg.eigvalsh(covariance)
            tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(covariance)))) * size
            if eigenvalues[0] < -tolerance:
                raise ValueError(f"{name} must be positive semidefinite.")
        return cast(FloatArray, covariance)

    def predict(self, ne: AnyFloatArray, dt: float) -> AnyFloatArray:
        """Propagate the state and inflate its covariance by process noise.

        Parameters
        ----------
        ne
            Predicted density profile in m⁻³, shape ``(n_rho,)``, used as the
            propagated state mean.
        dt
            Time step in seconds scaling the process-noise increment.

        Returns
        -------
        AnyFloatArray
            The predicted state (density profile) in m⁻³, shape ``(n_rho,)``.

        Raises
        ------
        ValueError
            If the state, timestep, state covariance, or process covariance is
            outside its declared finite-dimensional domain.
        """
        ne_arr = self._validate_vector(ne, "ne", self.n_rho)
        if np.any(ne_arr < 0.0):
            raise ValueError("ne must be non-negative.")
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative.")
        covariance = self._validate_covariance(self.P, "P", self.n_rho, positive_definite=False)
        process_covariance = self._validate_covariance(self.Q, "Q", self.n_rho, positive_definite=False)
        self.x = ne_arr
        self.P = covariance + process_covariance * dt
        return self.x

    def update(
        self,
        ne_pred: AnyFloatArray,
        measurements: AnyFloatArray,
        chord_impacts: AnyFloatArray | None = None,
        *,
        chord_angles: AnyFloatArray | None = None,
    ) -> AnyFloatArray:
        """Correct the predicted state with chord measurements (Kalman update).

        Parameters
        ----------
        ne_pred
            Predicted density profile in m⁻³, shape ``(n_rho,)``.
        measurements
            Line-integrated chord measurements, shape ``(n_chords,)``.
        chord_impacts
            Signed normalised chord impacts passed to
            :meth:`measurement_matrix`, shape ``(n_chords,)``.
        chord_angles
            Deprecated keyword alias for ``chord_impacts``; values are still
            normalised impact coordinates, not angles in radians.

        Returns
        -------
        AnyFloatArray
            Posterior density estimate in m⁻³, shape ``(n_rho,)``.

        Raises
        ------
        ValueError
            If state, measurements, chord geometry, or covariance matrices are
            invalid, or the innovation covariance is not positive definite.
        """
        ne_arr = self._validate_vector(ne_pred, "ne_pred", self.n_rho)
        if np.any(ne_arr < 0.0):
            raise ValueError("ne_pred must be non-negative.")
        measurement_arr = self._validate_vector(measurements, "measurements", self.n_chords)
        covariance = self._validate_covariance(self.P, "P", self.n_rho, positive_definite=False)
        measurement_covariance = self._validate_covariance(self.R, "R", self.n_chords, positive_definite=True)
        C = self.measurement_matrix(chord_impacts, chord_angles=chord_angles)
        covariance_measurement = covariance @ C.T
        innovation_covariance = C @ covariance_measurement + measurement_covariance
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        try:
            factor = np.linalg.cholesky(innovation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be positive definite.") from exc
        gain = np.linalg.solve(factor.T, np.linalg.solve(factor, covariance_measurement.T)).T
        innovation = measurement_arr - C @ ne_arr
        self.x = ne_arr + gain @ innovation
        correction = np.eye(self.n_rho) - gain @ C
        joseph_covariance = correction @ covariance @ correction.T + gain @ measurement_covariance @ gain.T
        self.P = 0.5 * (joseph_covariance + joseph_covariance.T)
        return self.x
