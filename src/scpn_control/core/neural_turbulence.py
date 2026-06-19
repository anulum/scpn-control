# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Turbulence Surrogate
"""QLKNN-class neural turbulence surrogate, training, and flux prediction utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

_NEURAL_TURBULENCE_CLAIM_SCHEMA_VERSION = 1
_NEURAL_TURBULENCE_REFERENCE_SOURCES = frozenset({"real_gk_campaign", "documented_public_reference"})
_NEURAL_TURBULENCE_FEATURE_SCHEMA = (
    "R_LTi",
    "R_LTe",
    "R_Ln",
    "q",
    "s_hat",
    "alpha_MHD",
    "Ti_Te",
    "nu_star",
    "Z_eff",
    "epsilon",
)


@dataclass(frozen=True)
class NeuralTurbulenceClaimEvidence:
    """Serialisable admission evidence for neural-turbulence surrogate claims."""

    schema_version: int
    model_id: str
    source: str
    source_id: str
    weights_path: str
    weights_sha256: str
    feature_schema: tuple[str, ...]
    local_sample_count: int
    local_q_i_rmse_gB: float
    local_q_e_rmse_gB: float
    local_gamma_e_rmse_gB: float
    local_flux_relative_mae: float
    local_critical_gradient_accuracy: float
    reference_source: str
    reference_dataset_id: str
    reference_artifact_sha256: str
    reference_sample_count: int
    q_i_rmse_gB: float | None
    q_e_rmse_gB: float | None
    gamma_e_rmse_gB: float | None
    flux_relative_mae: float | None
    critical_gradient_accuracy: float | None
    q_i_rmse_tolerance_gB: float | None
    q_e_rmse_tolerance_gB: float | None
    gamma_e_rmse_tolerance_gB: float | None
    flux_relative_mae_tolerance: float | None
    critical_gradient_accuracy_min: float | None
    quantitative_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_nonnegative_or_none(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive_or_none(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval_or_none(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def _finite_nonnegative(name: str, value: object) -> float:
    result = _finite_nonnegative_or_none(name, value)
    if result is None:
        raise ValueError(f"{name} must be present")
    return result


def _unit_interval(name: str, value: object) -> float:
    result = _unit_interval_or_none(name, value)
    if result is None:
        raise ValueError(f"{name} must be present")
    return result


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _profile_array(
    name: str,
    values: AnyFloatArray,
    shape: tuple[int, ...] | None = None,
    *,
    positive: bool = False,
    allow_last_zero: bool = False,
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional profile with at least two points")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must match the radius grid shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and allow_last_zero and (np.any(arr[:-1] <= 0.0) or arr[-1] < 0.0):
        raise ValueError(f"{name} must be positive in the interior and non-negative at the boundary")
    if positive and not allow_last_zero and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive everywhere")
    return arr


class QLKNNSurrogate:
    """
    Pure NumPy inference for a QLKNN-like neural network.
    Predicts turbulent fluxes [Q_i, Q_e, Gamma_e] from 10 parameters.
    van de Plassche et al., Phys. Plasmas 27, 022310 (2020).

    Default construction auto-trains on a Jenko et al. (2001) critical gradient
    model so predictions are physically meaningful out of the box.
    """

    def __init__(self, hidden_layers: list[int] | None = None, activation: str = "elu", pretrained: bool = True):
        if hidden_layers is None:
            hidden_layers = [128, 128, 64]

        self.hidden_layers = hidden_layers
        self.activation = activation

        self.weights: list[AnyFloatArray] = []
        self.biases: list[AnyFloatArray] = []

        rng = np.random.RandomState(42) if pretrained else np.random.RandomState()

        layers = [10] + hidden_layers + [3]
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            w = rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros(n_out)
            self.weights.append(w)
            self.biases.append(b)

        if pretrained:
            self._pretrain(rng)

    def _activate(self, x: AnyFloatArray) -> AnyFloatArray:
        if self.activation == "elu":
            out = np.asarray(x, dtype=float).copy()
            negative = out <= 0.0
            out[negative] = np.exp(out[negative]) - 1.0
            return out
        if self.activation == "relu":
            return np.asarray(np.maximum(0, x))
        if self.activation == "tanh":
            return np.asarray(np.tanh(x))
        return x

    def _activate_deriv(self, x: AnyFloatArray) -> FloatArray:
        if self.activation == "elu":
            return np.where(x > 0, 1.0, np.exp(x))
        if self.activation == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if self.activation == "tanh":
            return np.asarray(1.0 - np.tanh(x) ** 2)
        return np.ones_like(x)

    def _pretrain(self, rng: np.random.RandomState) -> None:
        """Train on Jenko et al. (2001) analytic critical gradient model."""
        X = TrainingDataGenerator.generate_parameter_scan(500, rng=rng)
        y = TrainingDataGenerator.generate_analytic_targets(X)

        n_val = 50
        X_train, y_train = X[:-n_val], y[:-n_val]

        for _ in range(100):
            activations: list[AnyFloatArray] = [X_train]
            pre_acts: list[AnyFloatArray] = []
            out: AnyFloatArray = X_train
            for i in range(len(self.weights) - 1):
                z = out @ self.weights[i] + self.biases[i]
                pre_acts.append(z)
                out = self._activate(z)
                activations.append(out)
            z_last = out @ self.weights[-1] + self.biases[-1]
            pred = z_last

            n_train = X_train.shape[0]
            delta = 2.0 * (pred - y_train) / (n_train * y_train.shape[1])

            for i in range(len(self.weights) - 1, -1, -1):
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)
                dW_norm = float(np.linalg.norm(dW))
                if dW_norm > 1.0:
                    dW = dW / dW_norm
                    db = db / max(float(np.linalg.norm(db)), 1e-8)
                self.weights[i] -= 1e-3 * dW
                self.biases[i] -= 1e-3 * db
                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._activate_deriv(pre_acts[i - 1])
                    np.clip(delta, -1e6, 1e6, out=delta)

    def forward(self, x: AnyFloatArray) -> FloatArray:
        """
        x shape: (batch_size, 10)
        returns shape: (batch_size, 3)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        out = x
        for i in range(len(self.weights) - 1):
            out = out @ self.weights[i] + self.biases[i]
            out = self._activate(out)

        out = out @ self.weights[-1] + self.biases[-1]
        return np.asarray(out)

    def load_weights(self, path: str) -> None:
        """Load weights and biases from a NumPy ``.npz`` archive.

        Parameters
        ----------
        path
            Path to an ``.npz`` file with arrays ``w{i}`` and ``b{i}`` for each
            layer ``i``.
        """
        data = np.load(path, allow_pickle=True)
        self.weights = [data[f"w{i}"] for i in range(len(self.weights))]
        self.biases = [data[f"b{i}"] for i in range(len(self.biases))]

    def save_weights(self, path: str) -> None:
        """Save weights and biases to a NumPy ``.npz`` archive.

        Parameters
        ----------
        path
            Destination ``.npz`` path; each layer is stored as ``w{i}`` and
            ``b{i}``.
        """
        arrays: dict[str, Any] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)


class TransportInputNormalizer:
    """Convert physical plasma profiles into the 10 dimensionless QLKNN inputs."""

    @staticmethod
    def from_profiles(
        Te: AnyFloatArray, Ti: AnyFloatArray, ne: AnyFloatArray, q: AnyFloatArray, R0: float, a: float, B0: float, r: AnyFloatArray
    ) -> FloatArray:
        """
        Convert physical profiles into the 10 dimensionless QLKNN inputs.
        """
        r = _profile_array("r", r, positive=True)
        if np.any(np.diff(r) <= 0.0):
            raise ValueError("r must be strictly increasing")
        shape = r.shape
        Te = _profile_array("Te", Te, shape, positive=True, allow_last_zero=True)
        Ti = _profile_array("Ti", Ti, shape, positive=True, allow_last_zero=True)
        ne = _profile_array("ne", ne, shape, positive=True, allow_last_zero=True)
        q = _profile_array("q", q, shape, positive=True)
        R0 = _finite_scalar("R0", R0, positive=True)
        a = _finite_scalar("a", a, positive=True)
        B0 = _finite_scalar("B0", B0, positive=True)
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        dr = r[1] - r[0] if len(r) > 1 else 0.1

        # Gradients (simple central difference)
        grad_Te = np.gradient(Te, dr)
        grad_Ti = np.gradient(Ti, dr)
        grad_ne = np.gradient(ne, dr)
        grad_q = np.gradient(q, dr)

        # 1. R/L_Ti
        R_L_Ti = -R0 / np.maximum(Ti, 1e-3) * grad_Ti
        # 2. R/L_Te
        R_L_Te = -R0 / np.maximum(Te, 1e-3) * grad_Te
        # 3. R/L_ne
        R_L_ne = -R0 / np.maximum(ne, 1e-3) * grad_ne
        # 4. q
        q_norm = q
        # 5. s_hat (shear)
        s_hat = r / np.maximum(q, 1e-3) * grad_q
        # 6. alpha_MHD (pressure gradient)
        # p = 2 * n_e * T_e (roughly)
        p = 2.0 * ne * 1e19 * Te * 1e3 * 1.602e-19
        grad_p = np.gradient(p, dr)
        mu_0 = 4.0 * np.pi * 1e-7
        alpha_MHD = -(q**2) * R0 * grad_p * 2.0 * mu_0 / (B0**2)
        # 7. Ti/Te
        Ti_Te = Ti / np.maximum(Te, 1e-3)
        # 8. nu_star (banana-regime electron collisionality)
        epsilon = r / R0
        Te_safe_nu = np.maximum(Te, 1e-3)
        Z_eff_value = 1.5
        ln_lambda = 17.0
        ne_m3 = ne * 1.0e19
        Te_eV = Te_safe_nu * 1.0e3
        nu_star = (
            6.921e-18 * q_norm * R0 * ne_m3 * Z_eff_value * ln_lambda / (np.maximum(epsilon, 1e-3) ** 1.5 * Te_eV**2)
        )
        # 9. Z_eff (assumed flat 1.5)
        Z_eff = np.ones_like(r) * Z_eff_value
        # 10. epsilon
        eps = epsilon

        inputs = np.vstack([R_L_Ti, R_L_Te, R_L_ne, q_norm, s_hat, alpha_MHD, Ti_Te, nu_star, Z_eff, eps]).T

        return inputs


class TrainingDataGenerator:
    """Synthetic training-data generator for the QLKNN flux surrogate."""

    @staticmethod
    def generate_parameter_scan(n_samples: int, rng: np.random.RandomState | None = None) -> FloatArray:
        """Uniform random sampling in 10D QLKNN parameter space."""
        if rng is None:
            rng = np.random.RandomState()
        bounds = np.array(
            [
                [0.0, 15.0],
                [0.0, 15.0],
                [-5.0, 10.0],
                [0.5, 5.0],
                [-1.0, 3.0],
                [0.0, 2.0],
                [0.1, 2.0],
                [1e-3, 1.0],
                [1.0, 3.0],
                [0.01, 0.3],
            ]
        )

        X = np.zeros((n_samples, 10))
        for i in range(10):
            X[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_samples)

        return X

    @staticmethod
    def generate_analytic_targets(inputs: AnyFloatArray) -> FloatArray:
        """
        Compute bounded analytic quasilinear flux targets.
        Returns [Q_i, Q_e, Gamma_e] in gyro-Bohm units.
        """
        n_samples = inputs.shape[0]
        y = np.zeros((n_samples, 3))

        for i in range(n_samples):
            R_L_Ti = inputs[i, 0]
            R_L_Te = inputs[i, 1]
            R_L_ne = inputs[i, 2]
            q = inputs[i, 3]
            s_hat = inputs[i, 4]
            eps = inputs[i, 9]
            nu_star = inputs[i, 7]

            # Jenko et al. critical gradient formula
            R_L_Ti_crit = (1.0 + inputs[i, 6]) * max(1.33 + 1.91 * s_hat / q, 0.0) * (1.0 - 1.5 * eps)
            R_L_Ti_crit = max(R_L_Ti_crit, 0.0)

            # ITG Flux
            Q_i = 0.0
            if R_L_Ti > R_L_Ti_crit:
                # Q_i ~ (R/L_Ti - R/L_Ti_crit)^1.5
                Q_i = 5.0 * (R_L_Ti - R_L_Ti_crit) ** 1.5

            # TEM Flux
            Q_e = 0.0
            Gamma_e = 0.0
            R_L_ne_crit = 2.0
            if R_L_ne > R_L_ne_crit:
                # TEM driven flux, collisionality dampens it
                drive = R_L_ne - R_L_ne_crit
                Q_e = 2.0 * drive * np.sqrt(max(nu_star, 1e-4))
                Gamma_e = 1.0 * drive * np.sqrt(max(nu_star, 1e-4))

            y[i, 0] = Q_i
            y[i, 1] = Q_e
            y[i, 2] = Gamma_e

        return y


class NeuralTransportTrainer:
    """Backpropagation trainer for the QLKNN gyro-Bohm flux surrogate."""

    def _activate_deriv(self, x: AnyFloatArray, activation: str) -> FloatArray:
        if activation == "elu":
            return np.where(x > 0, 1.0, np.exp(x))
        if activation == "relu":
            return np.where(x > 0, 1.0, 0.0)
        if activation == "tanh":
            return np.asarray(1.0 - np.tanh(x) ** 2)
        return np.ones_like(x)

    def train(self, X: AnyFloatArray, y: AnyFloatArray, epochs: int = 200, lr: float = 1e-3, val_frac: float = 0.2) -> dict[str, Any]:
        """Train a QLKNN surrogate by gradient descent with clipping.

        Parameters
        ----------
        X
            Input samples, shape ``(n_samples, 10)`` of QLKNN inputs.
        y
            Target gyro-Bohm fluxes, shape ``(n_samples, 3)``.
        epochs
            Number of training epochs.
        lr
            Learning rate.
        val_frac
            Fraction of samples held out for validation.

        Returns
        -------
        dict[str, Any]
            Training history with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        n_samples = X.shape[0]
        n_val = max(int(n_samples * val_frac), 1)

        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        model = QLKNNSurrogate(pretrained=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Forward pass — store pre-activations for backprop
            activations: list[AnyFloatArray] = [X_train]
            pre_acts = []
            out: AnyFloatArray = X_train
            for i in range(len(model.weights) - 1):
                z = out @ model.weights[i] + model.biases[i]
                pre_acts.append(z)
                out = model._activate(z)
                activations.append(out)
            z_last = out @ model.weights[-1] + model.biases[-1]
            pre_acts.append(z_last)
            pred = z_last
            activations.append(pred)

            loss_train = float(np.mean((pred - y_train) ** 2))

            # Backprop
            n_train = X_train.shape[0]
            delta = 2.0 * (pred - y_train) / (n_train * y_train.shape[1])

            for i in range(len(model.weights) - 1, -1, -1):
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)
                # Gradient clipping to prevent explosion
                dW_norm = np.linalg.norm(dW)
                if dW_norm > 1.0:
                    dW = dW / dW_norm
                    db = db / max(float(np.linalg.norm(db)), 1e-8)
                model.weights[i] -= lr * dW
                model.biases[i] -= lr * db
                if i > 0:
                    delta = (delta @ model.weights[i].T) * self._activate_deriv(pre_acts[i - 1], model.activation)
                    np.clip(delta, -1e6, 1e6, out=delta)

            pred_val = model.forward(X_val)
            loss_val = float(np.mean((pred_val - y_val) ** 2))

            history["train_loss"].append(loss_train)
            history["val_loss"].append(loss_val)

        self._model = model
        return history


@dataclass
class TransportFluxes:
    """Dimensional turbulent transport fluxes on a radial grid.

    Attributes
    ----------
    Q_i_W_m2
        Ion heat flux in W/m².
    Q_e_W_m2
        Electron heat flux in W/m².
    Gamma_e_inv_m2_s
        Electron particle flux in m⁻² s⁻¹.
    """

    Q_i_W_m2: AnyFloatArray
    Q_e_W_m2: AnyFloatArray
    Gamma_e_inv_m2_s: AnyFloatArray


class QLKNNTransportModel:
    """QLKNN surrogate transport model mapping profiles to dimensional fluxes.

    Parameters
    ----------
    surrogate
        The trained gyro-Bohm flux surrogate network.
    """

    def __init__(self, surrogate: QLKNNSurrogate):
        self.surrogate = surrogate
        self.normalizer = TransportInputNormalizer()

    def compute_fluxes(
        self,
        Te: AnyFloatArray,
        Ti: AnyFloatArray,
        ne: AnyFloatArray,
        q: AnyFloatArray,
        R0: float,
        a: float,
        B0: float,
        r: AnyFloatArray,
    ) -> TransportFluxes:
        """Predict dimensional turbulent fluxes from physical profiles.

        Normalises the profiles to QLKNN inputs, evaluates the surrogate for
        gyro-Bohm fluxes, then de-normalises with the local gyro-Bohm unit.

        Parameters
        ----------
        Te
            Electron-temperature profile in keV.
        Ti
            Ion-temperature profile in keV.
        ne
            Electron-density profile in 10¹⁹ m⁻³.
        q
            Safety-factor profile (dimensionless).
        R0
            Major radius in metres.
        a
            Minor radius in metres.
        B0
            Toroidal field on axis in tesla.
        r
            Strictly increasing minor-radius grid in metres.

        Returns
        -------
        TransportFluxes
            The ion and electron heat fluxes and the electron particle flux.
        """
        inputs = self.normalizer.from_profiles(Te, Ti, ne, q, R0, a, B0, r)

        # Predict gyro-Bohm fluxes
        gB_fluxes = self.surrogate.forward(inputs)

        Q_i_gB = gB_fluxes[:, 0]
        Q_e_gB = gB_fluxes[:, 1]
        Gamma_e_gB = gB_fluxes[:, 2]

        # De-normalize
        e_charge = 1.602e-19
        m_i = 2.0 * 1.67e-27

        Te_safe = np.maximum(Te, 1e-3)
        Te_J = Te_safe * 1e3 * e_charge
        c_s = np.sqrt(Te_J / m_i)
        rho_s = np.maximum(m_i * c_s / (e_charge * B0), 1e-6)

        ne_safe = np.maximum(ne, 1e-3)
        ne_m3 = ne_safe * 1e19
        Q_gB_phys = ne_m3 * Te_J * c_s * (rho_s / a) ** 2
        Gamma_gB_phys = ne_m3 * c_s * (rho_s / a) ** 2

        Q_i_phys = Q_i_gB * Q_gB_phys
        Q_e_phys = Q_e_gB * Q_gB_phys
        Gamma_e_phys = Gamma_e_gB * Gamma_gB_phys

        return TransportFluxes(Q_i_phys, Q_e_phys, Gamma_e_phys)


def cross_validate_neural_turbulence(
    surrogate: QLKNNSurrogate | None = None,
    *,
    n_samples: int = 256,
    seed: int = 20240531,
) -> dict[str, Any]:
    """Compare the surrogate against bounded analytic quasilinear targets."""
    samples = int(n_samples)
    if samples < 8:
        raise ValueError("n_samples must be at least 8")
    rng = np.random.RandomState(seed)
    active = surrogate if surrogate is not None else QLKNNSurrogate(pretrained=True)
    inputs = TrainingDataGenerator.generate_parameter_scan(samples, rng=rng)
    reference = TrainingDataGenerator.generate_analytic_targets(inputs)
    predicted = active.forward(inputs)
    if predicted.shape != reference.shape or not np.all(np.isfinite(predicted)):
        raise RuntimeError("neural turbulence surrogate produced invalid local benchmark outputs")
    error = predicted - reference
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    ref_mae = np.maximum(np.mean(np.abs(reference), axis=0), 1.0e-8)
    rel_mae = mae / ref_mae
    reference_active = np.any(reference > 1.0e-12, axis=1)
    predicted_active = np.any(predicted > 1.0e-12, axis=1)
    critical_gradient_accuracy = float(np.mean(reference_active == predicted_active))
    return {
        "n_samples": samples,
        "seed": int(seed),
        "Q_i_rmse_gB": float(rmse[0]),
        "Q_e_rmse_gB": float(rmse[1]),
        "Gamma_e_rmse_gB": float(rmse[2]),
        "per_channel_relative_mae": [float(v) for v in rel_mae],
        "flux_relative_mae": float(np.mean(rel_mae)),
        "critical_gradient_accuracy": critical_gradient_accuracy,
    }


def neural_turbulence_claim_evidence(
    validation_result: dict[str, Any],
    *,
    source: str,
    source_id: str,
    model_id: str = "neural_turbulence_qlknn_facade",
    weights_path: str | Path | None = None,
    reference_artifact_path: str | Path | None = None,
) -> NeuralTurbulenceClaimEvidence:
    """Build fail-closed evidence for neural-turbulence quantitative claims."""
    local_samples = int(validation_result.get("n_samples", 0))
    if local_samples < 1:
        raise ValueError("validation_result must contain a positive n_samples")
    weights = Path(weights_path) if weights_path is not None else None
    weights_sha256 = ""
    if weights is not None:
        if not weights.is_file():
            raise FileNotFoundError(f"neural-turbulence weights not found: {weights}")
        weights_sha256 = _sha256_file(weights)

    metrics: dict[str, object] = {}
    tolerances: dict[str, object] = {}
    reference_source = "none"
    reference_dataset_id = ""
    reference_artifact_sha256 = ""
    reference_sample_count = 0
    claim_allowed = False
    if reference_artifact_path is not None:
        if weights is None:
            raise ValueError("reference admission requires the exact neural-turbulence weights_path")
        from validation.validate_neural_turbulence_reference import validate_neural_turbulence_reference

        artifact_path = Path(reference_artifact_path)
        report = validate_neural_turbulence_reference(artifact_path, require_reference_artifacts=True)
        if report["status"] != "pass":
            raise ValueError("neural-turbulence reference artifact failed strict validation")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        reference_source = _non_empty_text("source", str(payload["source"]))
        if reference_source not in _NEURAL_TURBULENCE_REFERENCE_SOURCES:
            raise ValueError("neural-turbulence reference source is not admissible")
        if payload["trained_weights_sha256"].lower() != weights_sha256.lower():
            raise ValueError("neural-turbulence reference artifact does not match supplied weights")
        reference_dataset_id = _non_empty_text("reference_dataset_id", str(payload["reference_dataset_id"]))
        reference_artifact_sha256 = _non_empty_text(
            "reference_artifact_sha256", str(payload["reference_artifact_sha256"])
        )
        reference_sample_count = int(payload["reference_sample_count"])
        if reference_sample_count < 1:
            raise ValueError("reference_sample_count must be positive")
        metrics = dict(payload["metrics"])
        tolerances = dict(payload["tolerances"])
        claim_allowed = True

    claim_status = (
        "matched neural-turbulence reference admission passed"
        if claim_allowed
        else "local analytic-target regression evidence only; quantitative turbulence claims blocked"
    )
    return NeuralTurbulenceClaimEvidence(
        schema_version=_NEURAL_TURBULENCE_CLAIM_SCHEMA_VERSION,
        model_id=_non_empty_text("model_id", model_id),
        source=_non_empty_text("source", source),
        source_id=_non_empty_text("source_id", source_id),
        weights_path=str(weights) if weights is not None else "",
        weights_sha256=weights_sha256,
        feature_schema=_NEURAL_TURBULENCE_FEATURE_SCHEMA,
        local_sample_count=local_samples,
        local_q_i_rmse_gB=_finite_nonnegative("Q_i_rmse_gB", validation_result.get("Q_i_rmse_gB")),
        local_q_e_rmse_gB=_finite_nonnegative("Q_e_rmse_gB", validation_result.get("Q_e_rmse_gB")),
        local_gamma_e_rmse_gB=_finite_nonnegative("Gamma_e_rmse_gB", validation_result.get("Gamma_e_rmse_gB")),
        local_flux_relative_mae=_finite_nonnegative("flux_relative_mae", validation_result.get("flux_relative_mae")),
        local_critical_gradient_accuracy=_unit_interval(
            "critical_gradient_accuracy", validation_result.get("critical_gradient_accuracy")
        ),
        reference_source=reference_source,
        reference_dataset_id=reference_dataset_id,
        reference_artifact_sha256=reference_artifact_sha256,
        reference_sample_count=reference_sample_count,
        q_i_rmse_gB=_finite_nonnegative_or_none("Q_i_rmse_gB", metrics.get("Q_i_rmse_gB")),
        q_e_rmse_gB=_finite_nonnegative_or_none("Q_e_rmse_gB", metrics.get("Q_e_rmse_gB")),
        gamma_e_rmse_gB=_finite_nonnegative_or_none("Gamma_e_rmse_gB", metrics.get("Gamma_e_rmse_gB")),
        flux_relative_mae=_finite_nonnegative_or_none("flux_relative_mae", metrics.get("flux_relative_mae")),
        critical_gradient_accuracy=_unit_interval_or_none(
            "critical_gradient_accuracy", metrics.get("critical_gradient_accuracy")
        ),
        q_i_rmse_tolerance_gB=_finite_positive_or_none("Q_i_rmse_tolerance_gB", tolerances.get("Q_i_rmse_gB")),
        q_e_rmse_tolerance_gB=_finite_positive_or_none("Q_e_rmse_tolerance_gB", tolerances.get("Q_e_rmse_gB")),
        gamma_e_rmse_tolerance_gB=_finite_positive_or_none(
            "Gamma_e_rmse_tolerance_gB", tolerances.get("Gamma_e_rmse_gB")
        ),
        flux_relative_mae_tolerance=_finite_positive_or_none(
            "flux_relative_mae_tolerance", tolerances.get("flux_relative_mae")
        ),
        critical_gradient_accuracy_min=_unit_interval_or_none(
            "critical_gradient_accuracy_min", tolerances.get("critical_gradient_accuracy_min")
        ),
        quantitative_claim_allowed=claim_allowed,
        claim_status=claim_status,
    )


def assert_neural_turbulence_quantitative_claim_admissible(
    evidence: NeuralTurbulenceClaimEvidence,
) -> NeuralTurbulenceClaimEvidence:
    """Return evidence only when strict matched-reference admission passed."""
    if not isinstance(evidence, NeuralTurbulenceClaimEvidence):
        raise ValueError("evidence must be NeuralTurbulenceClaimEvidence")
    if evidence.schema_version != _NEURAL_TURBULENCE_CLAIM_SCHEMA_VERSION:
        raise ValueError("neural-turbulence claim evidence schema_version is unsupported")
    if not evidence.quantitative_claim_allowed:
        raise ValueError("neural-turbulence quantitative claim is blocked without matched reference evidence")
    return evidence


def save_neural_turbulence_claim_evidence(evidence: NeuralTurbulenceClaimEvidence, path: str | Path) -> None:
    """Persist neural-turbulence claim evidence as deterministic JSON."""
    if not isinstance(evidence, NeuralTurbulenceClaimEvidence):
        raise ValueError("evidence must be NeuralTurbulenceClaimEvidence")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
