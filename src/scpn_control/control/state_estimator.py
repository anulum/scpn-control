# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear state estimator.

r"""Extended Kalman filtering for nonlinear plasma-state estimation.

The estimator executes the nonlinear predict/update equations

.. math::

   x^-_k = f(x_{k-1}, u_k, \Delta t),
   \qquad P^-_k = F_k P_{k-1} F_k^T + Q_k,

   y_k = r(z_k, h(x^-_k)),
   \qquad S_k = H_k P^-_k H_k^T + R,

where callers supply analytic process and measurement Jacobians ``F`` and
``H`` for custom nonlinear models. The correction covariance uses the Joseph
form, avoiding the loss of symmetry and positive semidefiniteness associated
with the simplified ``(I - KH)P`` update in finite precision.

The default model preserves the repository's six-state plasma observer:
``[R, Z, vR, vZ, Ip, Te_core]`` with measurements
``[R, Z, Ip, Te_core]``. That default is affine and is therefore the exact
linear special case of this nonlinear API. Custom models may use any positive
state and measurement dimensions consistent with their covariance matrices.

References
----------
Simon 2006, *Optimal State Estimation*, Ch. 13.
Lister et al. 1997, *Nuclear Fusion* 37, 1633.
Moreau et al. 2008, *Nuclear Fusion* 48, 106001.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

ControlInput: TypeAlias = FloatArray | None
ProcessModel: TypeAlias = Callable[[FloatArray, ControlInput, float], AnyFloatArray]
ProcessJacobian: TypeAlias = Callable[[FloatArray, ControlInput, float], AnyFloatArray]
MeasurementModel: TypeAlias = Callable[[FloatArray], AnyFloatArray]
MeasurementJacobian: TypeAlias = Callable[[FloatArray], AnyFloatArray]
ProcessNoiseModel: TypeAlias = Callable[[FloatArray, FloatArray, ControlInput, float], AnyFloatArray]
MeasurementResidual: TypeAlias = Callable[[FloatArray, FloatArray], AnyFloatArray]
StateAddition: TypeAlias = Callable[[FloatArray, FloatArray], AnyFloatArray]

_DEFAULT_STATE_DIMENSION = 6
_DEFAULT_MEASUREMENT_DIMENSION = 4
_IDX_R = 0
_IDX_Z = 1
_IDX_VR = 2
_IDX_VZ = 3
_IDX_IP = 4
_IDX_TE = 5
_SYMMETRY_RTOL = 1e-12
_PSD_RTOL = 1e-12


def _as_vector(name: str, value: AnyFloatArray, expected_size: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_size is not None and array.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_matrix(name: str, value: AnyFloatArray, shape: tuple[int, int]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_covariance(
    name: str,
    value: AnyFloatArray,
    expected_size: int | None = None,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_size is not None and array.shape != (expected_size, expected_size):
        raise ValueError(f"{name} must have shape ({expected_size}, {expected_size}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    scale = max(1.0, float(np.linalg.norm(array, ord=2)))
    tolerance = _SYMMETRY_RTOL * scale
    if not np.allclose(array, array.T, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (array + array.T)
    if float(np.linalg.eigvalsh(symmetric).min()) < -_PSD_RTOL * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric.copy()


def _default_process_model(x: FloatArray, u: ControlInput, dt: float) -> FloatArray:
    del u
    predicted = x.copy()
    predicted[_IDX_R] += dt * x[_IDX_VR]
    predicted[_IDX_Z] += dt * x[_IDX_VZ]
    return predicted


def _default_process_jacobian(x: FloatArray, u: ControlInput, dt: float) -> FloatArray:
    del x, u
    jacobian = np.eye(_DEFAULT_STATE_DIMENSION)
    jacobian[_IDX_R, _IDX_VR] = dt
    jacobian[_IDX_Z, _IDX_VZ] = dt
    return jacobian


def _default_measurement_model(x: FloatArray) -> FloatArray:
    return x[[_IDX_R, _IDX_Z, _IDX_IP, _IDX_TE]].copy()


def _default_measurement_jacobian(x: FloatArray) -> FloatArray:
    del x
    jacobian = np.zeros((_DEFAULT_MEASUREMENT_DIMENSION, _DEFAULT_STATE_DIMENSION))
    jacobian[0, _IDX_R] = 1.0
    jacobian[1, _IDX_Z] = 1.0
    jacobian[2, _IDX_IP] = 1.0
    jacobian[3, _IDX_TE] = 1.0
    return jacobian


def _default_process_noise(
    covariance_density: FloatArray,
    x: FloatArray,
    u: ControlInput,
    dt: float,
) -> FloatArray:
    del x, u
    return covariance_density * dt


def _default_measurement_residual(observed: FloatArray, predicted: FloatArray) -> FloatArray:
    return observed - predicted


def _default_state_addition(state: FloatArray, correction: FloatArray) -> FloatArray:
    return state + correction


class ExtendedKalmanFilter:
    """Stateful n-dimensional extended Kalman filter.

    Custom process and measurement models must be supplied together with their
    analytic Jacobians. The estimator intentionally provides no implicit finite-
    difference fallback: derivative accuracy and conventions remain part of the
    caller-owned model contract.

    Parameters
    ----------
    x0:
        Initial one-dimensional state estimate.
    P0:
        Symmetric positive-semidefinite initial state covariance.
    Q:
        Symmetric positive-semidefinite continuous-time process covariance
        density. The default discretisation is ``Q * dt``.
    R_cov:
        Symmetric positive-semidefinite measurement covariance.
    process_model, process_jacobian:
        Optional nonlinear ``f(x, u, dt)`` and analytic ``F(x, u, dt)``.
        Supplying exactly one is an error. When both are absent, the compatible
        six-state constant-velocity plasma model is used.
    measurement_model, measurement_jacobian:
        Optional nonlinear ``h(x)`` and analytic ``H(x)``. Supplying exactly one
        is an error. When both are absent, the compatible four-diagnostic plasma
        measurement model is used.
    process_noise_model:
        Optional discretisation ``Q_d(Q, x, u, dt)``. Its result must be a
        symmetric positive-semidefinite state covariance.
    measurement_residual:
        Optional ``r(z, h(x))`` operation for wrapped or otherwise non-Euclidean
        measurements. Defaults to subtraction.
    state_addition:
        Optional ``a(x, dx)`` retraction/addition for manifold-aware states.
        Defaults to vector addition.

    Notes
    -----
    The caller owns model equations, units, state ordering, control dimensions,
    differentiability and Jacobian correctness. This class validates numerical
    contracts but does not promote synthetic/local results to facility evidence.
    """

    def __init__(
        self,
        x0: AnyFloatArray,
        P0: AnyFloatArray,
        Q: AnyFloatArray,
        R_cov: AnyFloatArray,
        *,
        process_model: ProcessModel | None = None,
        process_jacobian: ProcessJacobian | None = None,
        measurement_model: MeasurementModel | None = None,
        measurement_jacobian: MeasurementJacobian | None = None,
        process_noise_model: ProcessNoiseModel | None = None,
        measurement_residual: MeasurementResidual | None = None,
        state_addition: StateAddition | None = None,
    ) -> None:
        if (process_model is None) != (process_jacobian is None):
            raise ValueError("process_model and process_jacobian must be supplied together")
        if (measurement_model is None) != (measurement_jacobian is None):
            raise ValueError("measurement_model and measurement_jacobian must be supplied together")

        self.x: FloatArray = _as_vector("x0", x0)
        self.state_dimension = int(self.x.size)
        self.P: FloatArray = _as_covariance("P0", P0, self.state_dimension)
        self.Q: FloatArray = _as_covariance("Q", Q, self.state_dimension)
        self.R: FloatArray = _as_covariance("R_cov", R_cov)
        self.measurement_dimension = int(self.R.shape[0])
        self._process_model: ProcessModel
        self._process_jacobian: ProcessJacobian
        self._measurement_model: MeasurementModel
        self._measurement_jacobian: MeasurementJacobian

        if process_model is None:
            if self.state_dimension != _DEFAULT_STATE_DIMENSION:
                raise ValueError(
                    "the default process model requires a six-dimensional state; "
                    "supply process_model and process_jacobian for custom dimensions"
                )
            self._process_model = _default_process_model
            self._process_jacobian = _default_process_jacobian
        else:
            self._process_model = process_model
            assert process_jacobian is not None
            self._process_jacobian = process_jacobian

        if measurement_model is None:
            if (
                self.state_dimension != _DEFAULT_STATE_DIMENSION
                or self.measurement_dimension != _DEFAULT_MEASUREMENT_DIMENSION
            ):
                raise ValueError(
                    "the default measurement model requires a six-dimensional state and "
                    "four-dimensional measurement; supply measurement_model and "
                    "measurement_jacobian for custom dimensions"
                )
            self._measurement_model = _default_measurement_model
            self._measurement_jacobian = _default_measurement_jacobian
        else:
            self._measurement_model = measurement_model
            assert measurement_jacobian is not None
            self._measurement_jacobian = measurement_jacobian

        self._process_noise_model = process_noise_model or _default_process_noise
        self._measurement_residual = measurement_residual or _default_measurement_residual
        self._state_addition = state_addition or _default_state_addition

        initial_state = self.x.copy()
        _as_vector(
            "measurement_model output",
            self._measurement_model(initial_state.copy()),
            self.measurement_dimension,
        )
        self.H: FloatArray = _as_matrix(
            "measurement_jacobian output",
            self._measurement_jacobian(initial_state.copy()),
            (self.measurement_dimension, self.state_dimension),
        )

    def predict(self, dt: float, u: AnyFloatArray | None = None) -> FloatArray:
        """Advance the nonlinear state and covariance prediction.

        Model, numerical and covariance validation failures leave the previous
        estimator state unchanged. This is an exception-safety guarantee, not
        synchronisation for concurrent callers or callbacks that directly
        mutate the estimator through external references.

        Parameters
        ----------
        dt:
            Finite positive time step in seconds.
        u:
            Optional finite one-dimensional control vector. Its ordering and
            units belong to the supplied process model. The default affine
            plasma model ignores it for backward compatibility.

        Returns
        -------
        numpy.ndarray
            Defensive copy of the predicted state.

        Raises
        ------
        ValueError
            If the time step, control, model output, Jacobian, process noise, or
            resulting covariance violates the declared numerical contract.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        control = None if u is None else _as_vector("u", u)
        prior_state = self.x.copy()
        predicted = _as_vector(
            "process_model output",
            self._process_model(prior_state.copy(), None if control is None else control.copy(), dt),
            self.state_dimension,
        )
        jacobian = _as_matrix(
            "process_jacobian output",
            self._process_jacobian(prior_state.copy(), None if control is None else control.copy(), dt),
            (self.state_dimension, self.state_dimension),
        )
        process_noise = _as_covariance(
            "process_noise_model output",
            self._process_noise_model(
                self.Q.copy(),
                prior_state.copy(),
                None if control is None else control.copy(),
                dt,
            ),
            self.state_dimension,
        )
        predicted_covariance = jacobian @ self.P @ jacobian.T + process_noise
        validated_covariance = _as_covariance(
            "predicted covariance",
            0.5 * (predicted_covariance + predicted_covariance.T),
            self.state_dimension,
        )
        self.x, self.P = predicted, validated_covariance
        return self.x.copy()

    def update(self, z: AnyFloatArray) -> FloatArray:
        """Correct the state estimate using a nonlinear measurement.

        The gain is computed from a Cholesky-factored positive-definite
        innovation covariance, without forming a matrix inverse. Covariance is
        updated in Joseph form
        ``(I-KH) P (I-KH)^T + K R K^T`` and then symmetrised.

        State, covariance and measurement Jacobian are published only after
        the complete correction validates. A rejected correction preserves
        their previous values, under the same serial-caller and callback
        ownership conditions as :meth:`predict`.

        Parameters
        ----------
        z:
            Finite measurement vector with the dimension declared by ``R_cov``.

        Returns
        -------
        numpy.ndarray
            Defensive copy of the corrected state.

        Raises
        ------
        ValueError
            If a measurement/model callback violates its contract or the
            innovation covariance is not positive definite.
        """
        measurement = _as_vector("z", z, self.measurement_dimension)
        prior_state = self.x.copy()
        predicted_measurement = _as_vector(
            "measurement_model output",
            self._measurement_model(prior_state.copy()),
            self.measurement_dimension,
        )
        jacobian = _as_matrix(
            "measurement_jacobian output",
            self._measurement_jacobian(prior_state.copy()),
            (self.measurement_dimension, self.state_dimension),
        )
        residual = _as_vector(
            "measurement_residual output",
            self._measurement_residual(measurement.copy(), predicted_measurement.copy()),
            self.measurement_dimension,
        )

        innovation_covariance = jacobian @ self.P @ jacobian.T + self.R
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        try:
            lower = np.linalg.cholesky(innovation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance must be positive definite") from exc

        covariance_times_jacobian_t = self.P @ jacobian.T
        gain = np.linalg.solve(
            lower.T,
            np.linalg.solve(lower, covariance_times_jacobian_t.T),
        ).T
        correction = gain @ residual
        corrected_state = _as_vector(
            "state_addition output",
            self._state_addition(prior_state.copy(), correction.copy()),
            self.state_dimension,
        )

        identity_minus_gain_jacobian = np.eye(self.state_dimension) - gain @ jacobian
        posterior_covariance = (
            identity_minus_gain_jacobian @ self.P @ identity_minus_gain_jacobian.T + gain @ self.R @ gain.T
        )
        validated_covariance = _as_covariance(
            "updated covariance",
            0.5 * (posterior_covariance + posterior_covariance.T),
            self.state_dimension,
        )
        self.x, self.P, self.H = corrected_state, validated_covariance, jacobian
        return self.x.copy()

    def estimate(self) -> FloatArray:
        """Return a defensive copy of the current state estimate."""
        return self.x.copy()


__all__ = [
    "ExtendedKalmanFilter",
    "MeasurementJacobian",
    "MeasurementModel",
    "MeasurementResidual",
    "ProcessJacobian",
    "ProcessModel",
    "ProcessNoiseModel",
    "StateAddition",
]
