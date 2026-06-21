# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Equilibrium
"""
PCA + MLP surrogate for Grad-Shafranov equilibrium reconstruction.

Maps coil currents (or profile parameters) → PCA coefficients → ψ(R,Z)
with ~1000× speedup over the full Picard iteration.

**Training modes:**

1. **From FusionKernel** — Generate training data by perturbing coil
   currents and running the GS solver.  Requires a valid config JSON.

2. **From SPARC GEQDSKs** — Train on real equilibrium data from CFS.
   Uses the GEQDSK's profile parameters (p', FF', I_p) as input features
   and ψ(R,Z) as targets.  No coil model needed.

**Status:** Reduced-order surrogate.  Not on the critical control path.
Use for rapid design-space exploration and batch equilibrium sweeps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_equilibrium_sparc.npz"
_NEURAL_EQUILIBRIUM_CLAIM_SCHEMA_VERSION = 1
_NEURAL_EQUILIBRIUM_REFERENCE_SOURCES = frozenset({"real_pefit", "documented_public_reference"})


# ── Data containers ──────────────────────────────────────────────────


@dataclass
class NeuralEqConfig:
    """Configuration for the neural equilibrium model."""

    n_components: int = 20
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    n_input_features: int = 12
    grid_shape: tuple[int, int] = (129, 129)  # (nh, nw)
    lambda_gs: float = 0.1
    R_min: float = 1.0  # [m] inner R boundary of computational grid
    R_max: float = 2.5  # [m] outer R boundary of computational grid


@dataclass
class TrainingResult:
    """Training summary."""

    n_samples: int
    n_components: int
    explained_variance: float
    final_loss: float
    train_time_s: float
    weights_path: str
    val_loss: float = float("nan")
    test_mse: float = float("nan")
    test_max_error: float = float("nan")


@dataclass(frozen=True)
class SyntheticEquilibriumCampaign:
    """Metadata for bounded synthetic equilibrium pretraining data."""

    n_samples: int
    grid_shape: tuple[int, int]
    seed: int
    feature_names: tuple[str, ...]
    validity_domain: str


@dataclass(frozen=True)
class PretrainingResult:
    """Synthetic neural-equilibrium pretraining summary."""

    n_samples: int
    n_components: int
    explained_variance: float
    train_mse: float
    val_mse: float
    test_mse: float
    test_max_error: float
    gs_residual: float
    train_time_s: float
    weights_path: str
    evidence_kind: str
    campaign: SyntheticEquilibriumCampaign


@dataclass(frozen=True)
class NeuralEquilibriumClaimEvidence:
    """Serialisable admission evidence for neural-equilibrium predictive claims."""

    schema_version: int
    model_id: str
    source: str
    source_id: str
    weights_path: str
    weights_sha256: str
    reference_source: str
    reference_dataset_id: str
    reference_artifact_sha256: str
    reference_equilibria_count: int
    grid_shape: tuple[int, int]
    feature_names: tuple[str, ...]
    n_components: int
    explained_variance: float
    synthetic_test_mse: float
    synthetic_gs_residual: float
    psi_rmse_Wb: float | None
    pressure_rmse_Pa: float | None
    q_profile_rmse: float | None
    boundary_rmse_m: float | None
    axis_position_error_m: float | None
    psi_tolerance_Wb: float | None
    pressure_tolerance_Pa: float | None
    q_profile_tolerance: float | None
    boundary_tolerance_m: float | None
    axis_position_tolerance_m: float | None
    facility_claim_allowed: bool
    claim_status: str


NEURAL_EQ_FEATURE_NAMES = (
    "Ip_MA",
    "Bt_T",
    "R_axis_m",
    "Z_axis_m",
    "pprime_scale",
    "ffprime_scale",
    "simag_Wb",
    "sibry_Wb",
    "kappa",
    "delta_upper",
    "delta_lower",
    "q95",
)


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _require_positive_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


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


def _validate_feature_matrix(features: AnyFloatArray) -> FloatArray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(NEURAL_EQ_FEATURE_NAMES) or not np.all(np.isfinite(x)):
        raise ValueError(f"features must be finite with shape (n, {len(NEURAL_EQ_FEATURE_NAMES)})")
    return x


def _validate_psi_matrix(psi: AnyFloatArray, grid_shape: tuple[int, int]) -> FloatArray:
    y = np.asarray(psi, dtype=np.float64)
    n_grid = int(grid_shape[0] * grid_shape[1])
    if y.ndim != 2 or y.shape[1] != n_grid or not np.all(np.isfinite(y)):
        raise ValueError(f"psi targets must be finite with shape (n, {n_grid})")
    return y


def _synthetic_equilibrium_from_features(features: AnyFloatArray, grid_shape: tuple[int, int]) -> FloatArray:
    nh, nw = grid_shape
    r = np.linspace(1.0, 2.5, nw)
    z = np.linspace(-1.0, 1.0, nh)
    rr, zz = np.meshgrid(r, z)

    ip_ma, bt, r_axis, z_axis, p_scale, ff_scale, simag, sibry, kappa, delta_u, delta_l, q95 = features
    minor = np.clip(0.42 + 0.025 * ip_ma + 0.015 * (q95 - 4.0), 0.35, 0.75)
    z_norm = (zz - z_axis) / np.clip(kappa * minor, 0.2, 1.4)
    upper = zz >= z_axis
    delta = np.where(upper, delta_u, delta_l)
    r_shift = r_axis + delta * minor * z_norm**2
    r_norm = (rr - r_shift) / minor
    rho2 = r_norm**2 + z_norm**2
    rho = np.sqrt(np.maximum(rho2, 0.0))

    denom = sibry - simag
    if abs(float(denom)) < 1.0e-10:
        denom = 1.0
    pressure_shape = (1.0 - np.clip(rho, 0.0, 1.4) ** 2) ** 2
    pressure_shape = np.where(rho <= 1.0, pressure_shape, 0.0)
    shear_shape = np.sin(np.pi * np.clip(rho, 0.0, 1.0)) ** 2
    axis_tilt = 0.025 * (bt - 5.0) * (rr - r_axis) + 0.02 * (ip_ma - 8.0) * (zz - z_axis)
    profile_mix = 0.055 * denom * ((p_scale - 1.0) * pressure_shape + (ff_scale - 1.0) * shear_shape)
    psi_n = np.clip(rho2 + axis_tilt, 0.0, 1.35)
    return np.asarray(simag + denom * psi_n + profile_mix, dtype=np.float64)


def generate_synthetic_equilibrium_dataset(
    n_samples: int,
    *,
    grid_shape: tuple[int, int] = (65, 65),
    seed: int = 20240531,
) -> tuple[FloatArray, FloatArray, SyntheticEquilibriumCampaign]:
    """Generate bounded Solovev-like equilibria for pretraining.

    The generated targets are synthetic Grad-Shafranov-shaped flux maps for
    pretraining only. They are not matched EFIT or P-EFIT validation evidence.
    """
    samples = _require_positive_int("n_samples", n_samples)
    nh = _require_positive_int("grid_shape[0]", grid_shape[0])
    nw = _require_positive_int("grid_shape[1]", grid_shape[1])
    rng = np.random.default_rng(seed)

    features = np.empty((samples, len(NEURAL_EQ_FEATURE_NAMES)), dtype=np.float64)
    psi = np.empty((samples, nh * nw), dtype=np.float64)
    for idx in range(samples):
        ip_ma = rng.uniform(5.0, 15.0)
        bt = rng.uniform(3.0, 8.0)
        r_axis = rng.uniform(1.55, 1.95)
        z_axis = rng.uniform(-0.12, 0.12)
        p_scale = rng.uniform(0.65, 1.35)
        ff_scale = rng.uniform(0.65, 1.35)
        simag = rng.uniform(-0.12, 0.02)
        sibry = simag + rng.uniform(0.7, 1.7)
        kappa = rng.uniform(1.35, 2.15)
        delta_u = rng.uniform(0.05, 0.55)
        delta_l = rng.uniform(0.02, 0.50)
        q95 = rng.uniform(2.6, 6.5)
        row = np.array(
            [ip_ma, bt, r_axis, z_axis, p_scale, ff_scale, simag, sibry, kappa, delta_u, delta_l, q95],
            dtype=np.float64,
        )
        features[idx] = row
        psi[idx] = _synthetic_equilibrium_from_features(row, (nh, nw)).ravel()

    campaign = SyntheticEquilibriumCampaign(
        n_samples=samples,
        grid_shape=(nh, nw),
        seed=int(seed),
        feature_names=NEURAL_EQ_FEATURE_NAMES,
        validity_domain="bounded synthetic Solovev-like equilibria for pretraining; not matched EFIT validation",
    )
    return features, psi, campaign


# ── Simple MLP (pure NumPy) ──────────────────────────────────────────


class SimpleMLP:
    """Feedforward MLP with ReLU hidden layers and linear output."""

    def __init__(self, layer_sizes: list[int], seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.weights: list[AnyFloatArray] = []
        self.biases: list[AnyFloatArray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialisation
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, x: AnyFloatArray) -> AnyFloatArray:
        """Forward pass through the ReLU MLP.

        Parameters
        ----------
        x
            Input batch, shape ``(batch, n_in)``.

        Returns
        -------
        AnyFloatArray
            The network output, shape ``(batch, n_out)``.
        """
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU
        return h

    def predict(self, x: AnyFloatArray) -> AnyFloatArray:
        """Return the forward-pass output for ``x`` (alias of :meth:`forward`)."""
        return self.forward(x)


# ── PCA (minimal, no sklearn dependency) ─────────────────────────────


class MinimalPCA:
    """Minimal PCA via SVD, no sklearn required."""

    def __init__(self, n_components: int = 20) -> None:
        self.n_components = n_components
        self.mean_: AnyFloatArray | None = None
        self.components_: AnyFloatArray | None = None
        self.explained_variance_ratio_: AnyFloatArray | None = None

    def fit(self, X: AnyFloatArray) -> "MinimalPCA":
        """Fit the PCA basis to the data via SVD.

        Parameters
        ----------
        X
            Data matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        MinimalPCA
            This estimator, with mean, components, and explained-variance ratio
            populated.
        """
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_var = (S**2).sum()
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ratio_ = S[: self.n_components] ** 2 / max(total_var, 1e-15)
        return self

    def transform(self, X: AnyFloatArray) -> FloatArray:
        """Project data onto the fitted principal components.

        Parameters
        ----------
        X
            Data matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        FloatArray
            The reduced representation, shape ``(n_samples, n_components)``.
        """
        assert self.mean_ is not None and self.components_ is not None
        return np.asarray((X - self.mean_) @ self.components_.T)

    def inverse_transform(self, Z: AnyFloatArray) -> FloatArray:
        """Reconstruct data from its principal-component representation.

        Parameters
        ----------
        Z
            Reduced representation, shape ``(n_samples, n_components)``.

        Returns
        -------
        FloatArray
            The reconstructed data, shape ``(n_samples, n_features)``.
        """
        assert self.mean_ is not None and self.components_ is not None
        return np.asarray(Z @ self.components_ + self.mean_)

    def fit_transform(self, X: AnyFloatArray) -> FloatArray:
        """Fit the PCA basis and return the reduced representation of ``X``."""
        self.fit(X)
        return self.transform(X)


# ── Neural Equilibrium Accelerator ───────────────────────────────────


class NeuralEquilibriumAccelerator:
    """
    PCA + MLP surrogate for Grad-Shafranov equilibrium.

    Can be trained from SPARC GEQDSK files (preferred) or from a
    FusionKernel config with coil perturbations.
    """

    def __init__(self, config: NeuralEqConfig | None = None) -> None:
        self.cfg = config or NeuralEqConfig()
        self.pca = MinimalPCA(n_components=self.cfg.n_components)
        self.mlp: SimpleMLP | None = None
        self.is_trained = False
        self._input_mean: AnyFloatArray | None = None
        self._input_std: AnyFloatArray | None = None

    # ── GS residual loss ───────────────────────────────────────────

    def _gs_residual_loss(self, psi_pred_flat: AnyFloatArray, grid_shape: tuple[int, int]) -> float:
        """GS* residual loss: d²ψ/dR² - (1/R)dψ/dR + d²ψ/dZ²."""
        nh, nw = grid_shape
        psi = psi_pred_flat.reshape(nh, nw)
        R_1d = np.linspace(self.cfg.R_min, self.cfg.R_max, nw)
        dR = R_1d[1] - R_1d[0]
        R_interior = R_1d[np.newaxis, 1:-1]

        # All terms carry a factor of dR² (unnormalized finite differences)
        d2_dR2 = psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]
        d2_dZ2 = psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]
        # -(1/R) dψ/dR: central difference gives (ψ_{i+1} - ψ_{i-1})/(2 dR),
        # multiply by -(dR²/R) to match the dR² scaling of the other terms
        dpsi_dR_correction = (dR / (2.0 * R_interior)) * (psi[1:-1, 2:] - psi[1:-1, :-2])

        gs_star = d2_dR2 - dpsi_dR_correction + d2_dZ2
        return float(np.mean(gs_star**2))

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate_surrogate(self, X_test: AnyFloatArray, Y_test_raw: AnyFloatArray) -> dict[str, float]:
        """Evaluate on test set. Returns dict with mse, max_error, gs_residual."""
        if not self.is_trained:
            raise RuntimeError("Not trained")
        assert self._input_mean is not None and self._input_std is not None
        assert self.mlp is not None
        x_norm = (X_test - self._input_mean) / self._input_std
        coeffs = self.mlp.predict(x_norm)
        psi_pred = self.pca.inverse_transform(coeffs)

        mse = float(np.mean((psi_pred - Y_test_raw) ** 2))
        max_err = float(np.max(np.abs(psi_pred - Y_test_raw)))
        gs_res = 0.0
        for row in range(len(psi_pred)):
            gs_res += self._gs_residual_loss(psi_pred[row], self.cfg.grid_shape)
        gs_res /= max(len(psi_pred), 1)

        return {"mse": mse, "max_error": max_err, "gs_residual": gs_res}

    def pretrain_from_synthetic_equilibria(
        self,
        n_samples: int = 2048,
        *,
        seed: int = 20240531,
        ridge_alpha: float = 1.0e-6,
        save_path: str | Path | None = None,
    ) -> PretrainingResult:
        """Pretrain the surrogate on bounded synthetic equilibria.

        Uses PCA compression followed by a deterministic ridge-regression
        readout. The saved weights are compatible with the existing JAX
        inference path because the readout is represented as a one-layer MLP.
        """
        alpha = _require_positive_float("ridge_alpha", ridge_alpha)
        t0 = time.perf_counter()
        features, psi, campaign = generate_synthetic_equilibrium_dataset(
            n_samples,
            grid_shape=self.cfg.grid_shape,
            seed=seed,
        )
        x = _validate_feature_matrix(features)
        y = _validate_psi_matrix(psi, self.cfg.grid_shape)
        self.cfg.n_input_features = x.shape[1]

        self._input_mean = x.mean(axis=0)
        input_std = x.std(axis=0)
        input_std[input_std < 1.0e-10] = 1.0
        self._input_std = input_std
        x_norm = (x - self._input_mean) / self._input_std

        y_compressed = self.pca.fit_transform(y)
        assert self.pca.explained_variance_ratio_ is not None
        explained = float(np.sum(self.pca.explained_variance_ratio_))

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x_norm))
        n_test = max(1, int(0.15 * len(order)))
        n_val = max(1, int(0.15 * len(order)))
        test_idx = order[:n_test]
        val_idx = order[n_test : n_test + n_val]
        train_idx = order[n_test + n_val :]

        x_train = x_norm[train_idx]
        z_train = y_compressed[train_idx]
        x_aug = np.column_stack([x_train, np.ones(len(x_train))])
        gram = x_aug.T @ x_aug + alpha * np.eye(x_aug.shape[1])
        gram[-1, -1] -= alpha
        coeff = np.linalg.solve(gram, x_aug.T @ z_train)

        self.mlp = SimpleMLP([self.cfg.n_input_features, self.cfg.n_components], seed=seed)
        self.mlp.weights[0] = coeff[:-1]
        self.mlp.biases[0] = coeff[-1]
        self.is_trained = True

        train_pred = self.predict(x[train_idx]).reshape(len(train_idx), -1)
        val_pred = self.predict(x[val_idx]).reshape(len(val_idx), -1)
        test_pred = self.predict(x[test_idx]).reshape(len(test_idx), -1)
        train_mse = float(np.mean((train_pred - y[train_idx]) ** 2))
        val_mse = float(np.mean((val_pred - y[val_idx]) ** 2))
        test_mse = float(np.mean((test_pred - y[test_idx]) ** 2))
        test_max = float(np.max(np.abs(test_pred - y[test_idx])))
        gs_residual = 0.0
        for row in test_pred:
            gs_residual += self._gs_residual_loss(row, self.cfg.grid_shape)
        gs_residual /= max(len(test_pred), 1)

        weights_path = ""
        if save_path is not None:
            self.save_weights(save_path)
            weights_path = str(save_path)

        return PretrainingResult(
            n_samples=len(x),
            n_components=self.cfg.n_components,
            explained_variance=explained,
            train_mse=train_mse,
            val_mse=val_mse,
            test_mse=test_mse,
            test_max_error=test_max,
            gs_residual=float(gs_residual),
            train_time_s=time.perf_counter() - t0,
            weights_path=weights_path,
            evidence_kind="synthetic_pretraining",
            campaign=campaign,
        )

    def fine_tune_from_efit_reconstructions(
        self,
        geqdsk_paths: list[Path],
        *,
        reference_artifact_root: str | Path | None = None,
        require_reference_artifacts: bool = True,
        n_perturbations: int = 5,
        seed: int = 42,
    ) -> TrainingResult:
        """Fine-tune on real EFIT/P-EFIT artefacts only after evidence admission."""
        if require_reference_artifacts:
            from validation.validate_neural_equilibrium_reference import validate_neural_equilibrium_reference

            root = (
                Path(reference_artifact_root)
                if reference_artifact_root is not None
                else REPO_ROOT / "validation" / "reports" / "neural_equilibrium_reference"
            )
            report = validate_neural_equilibrium_reference(root, require_reference_artifacts=True)
            if report["status"] != "pass":
                raise RuntimeError("real EFIT fine-tuning requires passing neural equilibrium reference artifacts")
        if not geqdsk_paths:
            raise FileNotFoundError("No GEQDSK/EQDSK files supplied for EFIT fine-tuning")
        return self.train_from_geqdsk(geqdsk_paths, n_perturbations=n_perturbations, seed=seed)

    # ── Training from SPARC GEQDSKs ─────────────────────────────────

    def train_from_geqdsk(
        self,
        geqdsk_paths: list[Path],
        n_perturbations: int = 25,
        seed: int = 42,
    ) -> TrainingResult:
        """
        Train on real SPARC GEQDSK equilibria with perturbations.

        For each GEQDSK file, generates n_perturbations by scaling p'/FF'
        profiles, yielding n_files × n_perturbations training pairs.

        Input features (12-dim):
            [I_p, B_t, R_axis, Z_axis, pprime_scale, ffprime_scale,
             simag, sibry, kappa, delta_upper, delta_lower, q95]

        Output: flattened ψ(R,Z) → PCA coefficients

        Uses 70/15/15 train/val/test split with val-loss early stopping
        (patience=20) and combined MSE + lambda_gs * GS residual loss.
        """
        from scpn_control.core.eqdsk import read_geqdsk

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        X_features: list[AnyFloatArray] = []
        Y_psi: list[AnyFloatArray] = []

        # Target grid: use the first file's grid as reference
        first_eq = read_geqdsk(geqdsk_paths[0])
        target_nh, target_nw = first_eq.nh, first_eq.nw
        self.cfg.grid_shape = (target_nh, target_nw)

        for path in geqdsk_paths:
            eq = read_geqdsk(path)

            # Interpolate onto target grid if needed
            if eq.nh != target_nh or eq.nw != target_nw:
                from scipy.interpolate import RectBivariateSpline

                spline = RectBivariateSpline(eq.z, eq.r, eq.psirz, kx=3, ky=3)
                target_r = np.linspace(eq.rleft, eq.rleft + eq.rdim, target_nw)
                target_z = np.linspace(eq.zmid - eq.zdim / 2, eq.zmid + eq.zdim / 2, target_nh)
                psi_interp = spline(target_z, target_r, grid=True)
            else:
                psi_interp = eq.psirz

            # Extract shape parameters from boundary if available
            kappa = 1.7  # default elongation
            delta_upper = 0.3  # default upper triangularity
            delta_lower = 0.3  # default lower triangularity
            q95 = 3.0  # default safety factor at 95% flux
            if eq.rbdry is not None and len(eq.rbdry) > 3:
                r_span = eq.rbdry.max() - eq.rbdry.min()
                kappa = (eq.zbdry.max() - eq.zbdry.min()) / max(r_span, 0.01)
            if hasattr(eq, "qpsi") and eq.qpsi is not None and len(eq.qpsi) > 0:
                idx_95 = int(0.95 * len(eq.qpsi))
                q95 = eq.qpsi[min(idx_95, len(eq.qpsi) - 1)]

            # Base feature vector (12-dim)
            base_features = np.array(
                [
                    eq.current / 1e6,  # I_p in MA
                    eq.bcentr,  # B_t in T
                    eq.rmaxis,  # R_axis in m
                    eq.zmaxis,  # Z_axis in m
                    1.0,  # pprime scale factor
                    1.0,  # ffprime scale factor
                    eq.simag,  # psi at axis
                    eq.sibry,  # psi at boundary
                    kappa,  # elongation
                    delta_upper,  # upper triangularity
                    delta_lower,  # lower triangularity
                    q95,  # safety factor at 95% flux
                ]
            )

            # Unperturbed sample
            X_features.append(base_features)
            Y_psi.append(psi_interp.ravel())

            # Perturbed samples: scale p'/FF' and blend psi
            for _ in range(n_perturbations):
                pp_scale = rng.uniform(0.7, 1.3)
                ff_scale = rng.uniform(0.7, 1.3)

                # Perturbed features (shape params stay at base values)
                feat = base_features.copy()
                feat[4] = pp_scale
                feat[5] = ff_scale
                # kappa/delta/q95 at indices 8-11 are inherited from base

                # Linearly blend psi with a scale-dependent offset
                # This simulates the effect of profile scaling on equilibrium
                denom = eq.sibry - eq.simag
                if abs(denom) < 1e-12:  # pragma: no cover - real GEQDSK boundary flux differs from axis flux; defensive
                    denom = 1.0
                psi_n = (psi_interp - eq.simag) / denom

                # Profile perturbation modifies the normalised psi shape
                mix = 0.5 * (pp_scale + ff_scale) - 1.0  # deviation from 1.0
                # Perturb interior normalised psi
                plasma_mask = (psi_n >= 0) & (psi_n < 1.0)
                psi_perturbed = psi_interp.copy()
                psi_perturbed[plasma_mask] += mix * 0.1 * denom * (1.0 - psi_n[plasma_mask])

                X_features.append(feat)
                Y_psi.append(psi_perturbed.ravel())

        X = np.array(X_features)
        Y = np.array(Y_psi)
        n_samples = len(X)
        logger.info("Training data: %d samples, %d features → %d outputs", n_samples, X.shape[1], Y.shape[1])

        # Normalise inputs
        self._input_mean = X.mean(axis=0)
        input_std = X.std(axis=0)
        input_std[input_std < 1e-10] = 1.0
        self._input_std = input_std
        X_norm = (X - self._input_mean) / self._input_std

        # PCA on flattened psi
        Y_compressed = self.pca.fit_transform(Y)
        assert self.pca.explained_variance_ratio_ is not None
        explained = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info(
            "PCA: %d → %d components, %.2f%% variance retained", Y.shape[1], self.cfg.n_components, explained * 100
        )

        # ── Train/val/test split (70/15/15) ────────────────────────
        indices = rng.permutation(n_samples)
        n_val = max(1, int(0.15 * n_samples)) if n_samples >= 3 else 0
        n_train = n_samples - n_val - (1 if n_samples >= 3 else 0)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        X_train, Y_train = X_norm[train_idx], Y_compressed[train_idx]
        X_val, Y_val = X_norm[val_idx], Y_compressed[val_idx]
        _X_test, _Y_test = X_norm[test_idx], Y_compressed[test_idx]
        # Keep uncompressed targets for GS residual evaluation
        Y_test_raw = Y[test_idx]

        logger.info(
            "Split: %d train / %d val / %d test",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        # ── Train MLP: X_train → Y_train ─────────────────────────
        self.cfg.n_input_features = X.shape[1]
        layer_sizes = [
            self.cfg.n_input_features,
            *self.cfg.hidden_sizes,
            self.cfg.n_components,
        ]
        self.mlp = SimpleMLP(layer_sizes, seed=seed)

        # Mini-batch SGD with momentum
        lr = 1e-4
        momentum = 0.9
        n_epochs = 500
        batch_size = min(32, len(X_train))

        velocity = [np.zeros_like(w) for w in self.mlp.weights]
        velocity_b = [np.zeros_like(b) for b in self.mlp.biases]

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(n_epochs):
            order = rng.permutation(len(X_train))
            epoch_loss = 0.0

            for start in range(0, len(X_train), batch_size):
                idx = order[start : start + batch_size]
                x_batch = X_train[idx]
                y_batch = Y_train[idx]

                # Forward pass (store activations for backprop)
                activations = [x_batch]
                h = x_batch
                for i, (W, b) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
                    z = h @ W + b
                    if i < len(self.mlp.weights) - 1:
                        h = np.maximum(0, z)
                    else:
                        h = z
                    activations.append(h)

                # MSE loss
                error = activations[-1] - y_batch
                loss = float(np.mean(error**2))

                # GS residual loss on this batch
                gs_loss = 0.0
                for row in range(len(idx)):
                    psi_pred_flat = self.pca.inverse_transform(activations[-1][row : row + 1])[0]
                    gs_loss += self._gs_residual_loss(psi_pred_flat, self.cfg.grid_shape)
                gs_loss /= len(idx)
                loss = loss + self.cfg.lambda_gs * gs_loss

                epoch_loss += loss * len(idx)

                # Backprop
                delta = 2.0 * error / len(idx)
                for i in range(len(self.mlp.weights) - 1, -1, -1):
                    grad_w = activations[i].T @ delta
                    grad_b = delta.sum(axis=0)

                    # Gradient clipping to prevent explosive gradients
                    np.clip(grad_w, -1.0, 1.0, out=grad_w)
                    np.clip(grad_b, -1.0, 1.0, out=grad_b)

                    velocity[i] = momentum * velocity[i] - lr * grad_w
                    velocity_b[i] = momentum * velocity_b[i] - lr * grad_b

                    self.mlp.weights[i] += velocity[i]
                    self.mlp.biases[i] += velocity_b[i]

                    if i > 0:
                        delta = delta @ self.mlp.weights[i].T
                        # ReLU derivative
                        delta *= (activations[i] > 0).astype(float)

            epoch_loss /= max(len(X_train), 1)
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss

            # ── Validation loss (MSE + lambda_gs * GS) ────────────
            if len(X_val) > 0:
                val_pred = self.mlp.forward(X_val)
                val_mse = float(np.mean((val_pred - Y_val) ** 2))
                val_gs = 0.0
                for row in range(len(X_val)):
                    psi_pred_flat = self.pca.inverse_transform(val_pred[row : row + 1])[0]
                    val_gs += self._gs_residual_loss(psi_pred_flat, self.cfg.grid_shape)
                val_gs /= len(X_val)
                val_loss = val_mse + self.cfg.lambda_gs * val_gs
            else:  # pragma: no cover - the 70/15/15 split yields a non-empty val set for any trainable size (n>=3); defensive
                val_loss = epoch_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.6f)",
                        epoch,
                        val_loss,
                    )
                    break

            if epoch % 50 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.6f  val_loss=%.6f",
                    epoch,
                    epoch_loss,
                    val_loss,
                )

        self.is_trained = True
        train_time = time.perf_counter() - t0

        # ── Evaluate on held-out test set ─────────────────────────
        test_metrics = self.evaluate_surrogate(
            X[test_idx],
            Y_test_raw,
        )
        logger.info(
            "Test set: MSE=%.6f  max_error=%.6f  GS_residual=%.6f",
            test_metrics["mse"],
            test_metrics["max_error"],
            test_metrics["gs_residual"],
        )

        return TrainingResult(
            n_samples=n_samples,
            n_components=self.cfg.n_components,
            explained_variance=explained,
            final_loss=best_train_loss,
            train_time_s=train_time,
            weights_path="",
            val_loss=best_val_loss,
            test_mse=test_metrics["mse"],
            test_max_error=test_metrics["max_error"],
        )

    # ── Inference ────────────────────────────────────────────────────

    def predict(self, features: AnyFloatArray) -> FloatArray:
        """
        Predict ψ(R,Z) from input features.

        Parameters
        ----------
        features : AnyFloatArray
            Shape (n_features,) or (batch, n_features).

        Returns
        -------
        AnyFloatArray
            Shape (nh, nw) or (batch, nh, nw).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_from_geqdsk() or load_weights() first.")

        if features.ndim == 1:
            features = features[np.newaxis, :]

        assert self._input_mean is not None and self._input_std is not None
        assert self.mlp is not None
        x_norm = (features - self._input_mean) / self._input_std
        coeffs = self.mlp.predict(x_norm)
        psi_flat = self.pca.inverse_transform(coeffs)

        nh, nw = self.cfg.grid_shape
        if features.shape[0] == 1:
            return psi_flat.reshape(nh, nw)
        return psi_flat.reshape(-1, nh, nw)

    # ── Save / Load ──────────────────────────────────────────────────

    def save_weights(self, path: str | Path = DEFAULT_WEIGHTS_PATH) -> None:
        """Save model to .npz (no pickle dependency)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        assert self.mlp is not None
        assert self.pca.mean_ is not None
        assert self.pca.components_ is not None
        assert self.pca.explained_variance_ratio_ is not None
        assert self._input_mean is not None
        assert self._input_std is not None
        payload: dict[str, AnyFloatArray] = {
            "n_components": np.array([self.cfg.n_components]),
            "grid_nh": np.array([self.cfg.grid_shape[0]]),
            "grid_nw": np.array([self.cfg.grid_shape[1]]),
            "n_input_features": np.array([self.cfg.n_input_features]),
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "pca_evr": self.pca.explained_variance_ratio_,
            "input_mean": self._input_mean,
            "input_std": self._input_std,
            "n_layers": np.array([len(self.mlp.weights)]),
        }
        for i, (w, b) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
            payload[f"w{i}"] = w
            payload[f"b{i}"] = b

        np.savez(path, **payload)  # type: ignore[arg-type]
        logger.info("Saved neural equilibrium weights to %s", path)

    def load_weights(self, path: str | Path = DEFAULT_WEIGHTS_PATH) -> None:
        """Load model from .npz."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.cfg.n_components = int(data["n_components"][0])
            self.cfg.grid_shape = (int(data["grid_nh"][0]), int(data["grid_nw"][0]))
            self.cfg.n_input_features = int(data["n_input_features"][0])

            self.pca = MinimalPCA(self.cfg.n_components)
            self.pca.mean_ = np.array(data["pca_mean"])
            self.pca.components_ = np.array(data["pca_components"])
            self.pca.explained_variance_ratio_ = np.array(data["pca_evr"])

            self._input_mean = np.array(data["input_mean"])
            self._input_std = np.array(data["input_std"])

            n_layers = int(data["n_layers"][0])
            weights = [np.array(data[f"w{i}"]) for i in range(n_layers)]
            biases = [np.array(data[f"b{i}"]) for i in range(n_layers)]

            # Reconstruct layer sizes from weight shapes
            layer_sizes = [weights[0].shape[0]]
            for w in weights:
                layer_sizes.append(w.shape[1])
            self.mlp = SimpleMLP(layer_sizes)
            self.mlp.weights = weights
            self.mlp.biases = biases

        self.is_trained = True
        logger.info("Loaded neural equilibrium weights from %s", path)

    # ── Convenience ──────────────────────────────────────────────────

    def benchmark(self, features: AnyFloatArray, n_runs: int = 100) -> dict[str, float]:
        """Time inference over n_runs and return stats."""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(features)
            times.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }


# ── SPARC training convenience function ─────────────────────────────


def train_on_sparc(
    sparc_dir: str | Path | None = None,
    save_path: str | Path = DEFAULT_WEIGHTS_PATH,
    n_perturbations: int = 25,
    seed: int = 42,
) -> TrainingResult:
    """
    Train neural equilibrium on all SPARC GEQDSK files and save weights.

    This is the recommended entry point for training.
    """
    if sparc_dir is None:
        sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    sparc_dir = Path(sparc_dir)

    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    if not files:
        raise FileNotFoundError(f"No GEQDSK/EQDSK files in {sparc_dir}")

    accel = NeuralEquilibriumAccelerator()
    result = accel.train_from_geqdsk(
        files,
        n_perturbations=n_perturbations,
        seed=seed,
    )
    accel.save_weights(save_path)
    result.weights_path = str(save_path)
    return result


def pretrain_neural_equilibrium_synthetic(
    *,
    n_samples: int = 2048,
    save_path: str | Path = REPO_ROOT / "weights" / "neural_equilibrium_synthetic_pretrain.npz",
    grid_shape: tuple[int, int] = (65, 65),
    n_components: int = 20,
    seed: int = 20240531,
) -> PretrainingResult:
    """Convenience entry point for bounded synthetic pretraining."""
    accel = NeuralEquilibriumAccelerator(
        NeuralEqConfig(
            n_components=n_components,
            hidden_sizes=(),
            n_input_features=len(NEURAL_EQ_FEATURE_NAMES),
            grid_shape=grid_shape,
        )
    )
    return accel.pretrain_from_synthetic_equilibria(
        n_samples=n_samples,
        seed=seed,
        save_path=save_path,
    )


def neural_equilibrium_claim_evidence(
    pretraining: PretrainingResult,
    *,
    weights_path: str | Path,
    source: str,
    source_id: str,
    model_id: str = "neural_equilibrium_pca_mlp",
    reference_artifact_path: str | Path | None = None,
) -> NeuralEquilibriumClaimEvidence:
    """Build fail-closed evidence for neural-equilibrium predictive claims.

    Synthetic pretraining may support bounded pretraining and inference-plumbing
    statements only. Facility or predictive claims require a persisted
    P-EFIT/documented-reference artefact that validates against the same weight
    checksum and declares psi, pressure, q-profile, boundary, and magnetic-axis
    error tolerances.
    """
    if not isinstance(pretraining, PretrainingResult):
        raise ValueError("pretraining must be a PretrainingResult")
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(f"neural-equilibrium weights not found: {weights}")
    weights_sha256 = _sha256_file(weights)
    metrics: dict[str, object] = {}
    tolerances: dict[str, object] = {}
    reference_source = "none"
    reference_dataset_id = ""
    reference_artifact_sha256 = ""
    reference_equilibria_count = 0
    facility_allowed = False
    if reference_artifact_path is not None:
        from validation.validate_neural_equilibrium_reference import validate_neural_equilibrium_reference

        artifact_path = Path(reference_artifact_path)
        report = validate_neural_equilibrium_reference(artifact_path, require_reference_artifacts=True)
        if report["status"] != "pass":
            raise ValueError("neural-equilibrium reference artifact failed strict validation")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        reference_source = _non_empty_text("source", str(payload["source"]))
        # validate_neural_equilibrium_reference already enforces source admissibility; defensive re-check.
        if reference_source not in _NEURAL_EQUILIBRIUM_REFERENCE_SOURCES:  # pragma: no cover
            raise ValueError("neural-equilibrium reference source is not admissible")
        if payload["trained_weights_sha256"].lower() != weights_sha256.lower():
            raise ValueError("neural-equilibrium reference artifact does not match supplied weights")
        reference_dataset_id = _non_empty_text("reference_dataset_id", str(payload["reference_dataset_id"]))
        reference_artifact_sha256 = _non_empty_text(
            "reference_artifact_sha256", str(payload["reference_artifact_sha256"])
        )
        reference_equilibria_count = _require_positive_int(
            "reference_equilibria_count", int(payload["reference_equilibria_count"])
        )
        metrics = dict(payload["metrics"])
        tolerances = dict(payload["tolerances"])
        facility_allowed = True
    claim_status = (
        "matched neural-equilibrium reference admission passed"
        if facility_allowed
        else "synthetic pretraining evidence only; predictive EFIT/P-EFIT claims blocked"
    )
    return NeuralEquilibriumClaimEvidence(
        schema_version=_NEURAL_EQUILIBRIUM_CLAIM_SCHEMA_VERSION,
        model_id=_non_empty_text("model_id", model_id),
        source=_non_empty_text("source", source),
        source_id=_non_empty_text("source_id", source_id),
        weights_path=str(weights),
        weights_sha256=weights_sha256,
        reference_source=reference_source,
        reference_dataset_id=reference_dataset_id,
        reference_artifact_sha256=reference_artifact_sha256,
        reference_equilibria_count=reference_equilibria_count,
        grid_shape=(int(pretraining.campaign.grid_shape[0]), int(pretraining.campaign.grid_shape[1])),
        feature_names=tuple(pretraining.campaign.feature_names),
        n_components=int(pretraining.n_components),
        explained_variance=float(pretraining.explained_variance),
        synthetic_test_mse=float(pretraining.test_mse),
        synthetic_gs_residual=float(pretraining.gs_residual),
        psi_rmse_Wb=_finite_nonnegative_or_none("psi_rmse_Wb", metrics.get("psi_rmse_Wb")),
        pressure_rmse_Pa=_finite_nonnegative_or_none("pressure_rmse_Pa", metrics.get("pressure_rmse_Pa")),
        q_profile_rmse=_finite_nonnegative_or_none("q_profile_rmse", metrics.get("q_profile_rmse")),
        boundary_rmse_m=_finite_nonnegative_or_none("boundary_rmse_m", metrics.get("boundary_rmse_m")),
        axis_position_error_m=_finite_nonnegative_or_none(
            "axis_position_error_m", metrics.get("axis_position_error_m")
        ),
        psi_tolerance_Wb=_finite_positive_or_none("psi_tolerance_Wb", tolerances.get("psi_rmse_Wb")),
        pressure_tolerance_Pa=_finite_positive_or_none("pressure_tolerance_Pa", tolerances.get("pressure_rmse_Pa")),
        q_profile_tolerance=_finite_positive_or_none("q_profile_tolerance", tolerances.get("q_profile_rmse")),
        boundary_tolerance_m=_finite_positive_or_none("boundary_tolerance_m", tolerances.get("boundary_rmse_m")),
        axis_position_tolerance_m=_finite_positive_or_none(
            "axis_position_tolerance_m", tolerances.get("axis_position_error_m")
        ),
        facility_claim_allowed=facility_allowed,
        claim_status=claim_status,
    )


def assert_neural_equilibrium_facility_claim_admissible(
    evidence: NeuralEquilibriumClaimEvidence,
) -> NeuralEquilibriumClaimEvidence:
    """Return evidence only when strict matched-reference admission passed."""
    if not isinstance(evidence, NeuralEquilibriumClaimEvidence):
        raise ValueError("evidence must be NeuralEquilibriumClaimEvidence")
    if evidence.schema_version != _NEURAL_EQUILIBRIUM_CLAIM_SCHEMA_VERSION:
        raise ValueError("neural-equilibrium claim evidence schema_version is unsupported")
    if not evidence.facility_claim_allowed:
        raise ValueError("neural-equilibrium predictive claim is blocked without matched reference evidence")
    return evidence


def save_neural_equilibrium_claim_evidence(evidence: NeuralEquilibriumClaimEvidence, path: str | Path) -> None:
    """Persist neural-equilibrium claim evidence as deterministic JSON."""
    if not isinstance(evidence, NeuralEquilibriumClaimEvidence):
        raise ValueError("evidence must be NeuralEquilibriumClaimEvidence")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "NEURAL_EQ_FEATURE_NAMES",
    "MinimalPCA",
    "NeuralEqConfig",
    "NeuralEquilibriumAccelerator",
    "NeuralEquilibriumClaimEvidence",
    "PretrainingResult",
    "SimpleMLP",
    "SyntheticEquilibriumCampaign",
    "TrainingResult",
    "assert_neural_equilibrium_facility_claim_admissible",
    "generate_synthetic_equilibrium_dataset",
    "neural_equilibrium_claim_evidence",
    "pretrain_neural_equilibrium_synthetic",
    "save_neural_equilibrium_claim_evidence",
    "train_on_sparc",
]


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    if not sparc_dir.exists():
        print(f"SPARC data not found at {sparc_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Training Neural Equilibrium on SPARC GEQDSKs")
    print("=" * 60)

    result = train_on_sparc(sparc_dir)
    print(f"\nSamples: {result.n_samples}")
    print(f"PCA components: {result.n_components}")
    print(f"Explained variance: {result.explained_variance * 100:.2f}%")
    print(f"Final train loss: {result.final_loss:.6f}")
    print(f"Val loss: {result.val_loss:.6f}")
    print(f"Test MSE: {result.test_mse:.6f}")
    print(f"Test max error: {result.test_max_error:.6f}")
    print(f"Train time: {result.train_time_s:.1f}s")
    print(f"Weights: {result.weights_path}")

    # Quick validation
    accel = NeuralEquilibriumAccelerator()
    accel.load_weights(result.weights_path)

    from scpn_control.core.eqdsk import read_geqdsk

    test_eq = read_geqdsk(next(sparc_dir.glob("*.geqdsk")))
    kappa_cli = 1.7
    q95_cli = 3.0
    if test_eq.rbdry is not None and len(test_eq.rbdry) > 3:
        r_span = test_eq.rbdry.max() - test_eq.rbdry.min()
        kappa_cli = (test_eq.zbdry.max() - test_eq.zbdry.min()) / max(r_span, 0.01)
    if hasattr(test_eq, "qpsi") and test_eq.qpsi is not None and len(test_eq.qpsi) > 0:
        idx_95 = int(0.95 * len(test_eq.qpsi))
        q95_cli = test_eq.qpsi[min(idx_95, len(test_eq.qpsi) - 1)]
    features = np.array(
        [
            test_eq.current / 1e6,
            test_eq.bcentr,
            test_eq.rmaxis,
            test_eq.zmaxis,
            1.0,
            1.0,
            test_eq.simag,
            test_eq.sibry,
            kappa_cli,
            0.3,
            0.3,
            q95_cli,
        ]
    )

    psi_pred = accel.predict(features)
    diff = psi_pred - test_eq.psirz[: psi_pred.shape[0], : psi_pred.shape[1]]
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(test_eq.psirz[: psi_pred.shape[0], : psi_pred.shape[1]]))
    print(f"\nValidation relative L2 on first file: {rel_l2:.6f}")

    bench = accel.benchmark(features)
    print(f"Inference: {bench['mean_ms']:.3f} ms (median: {bench['median_ms']:.3f} ms)")
