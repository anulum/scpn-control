# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neural Transport Surrogate
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Neural-network surrogate for turbulent transport coefficients.

Replaces the simple critical-gradient transport model with a trained
MLP that reproduces gyrokinetic-level predictions at millisecond
inference speeds.  When no trained weights are available the module
falls back to an analytic critical-gradient model, so existing code
keeps working without any neural network dependency.

Inspired by the QLKNN paradigm (van de Plassche et al.,
*Phys. Plasmas* 27, 022310, 2020).  The current implementation is a
compact MLP (10→128→64→3) that maps local plasma parameters to
turbulent fluxes; it does not replicate the full QLKNN-10D
multi-branch architecture.

Training data
-------------
The module is designed for the QLKNN-10D public dataset:

    https://doi.org/10.5281/zenodo.3700755

Download the dataset and run the training recipe in
``docs/NEURAL_TRANSPORT_TRAINING.md`` (to be created) to produce an
``.npz`` weight file that this module loads at construction time.

References
----------
.. [1] van de Plassche, K.L. et al. (2020). "Fast modeling of
       turbulent transport in fusion plasmas using neural networks."
       *Phys. Plasmas* 27, 022310. doi:10.1063/1.5134126
.. [2] Citrin, J. et al. (2015). "Real-time capable first-principles
       based modelling of tokamak turbulent transport." *Nucl. Fusion*
       55, 092001.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Weight file format version expected by this loader.
_WEIGHTS_FORMAT_VERSION = 1
_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "weights" / "neural_transport_qlknn.npz"


# ── Data containers ───────────────────────────────────────────────────


@dataclass
class TransportInputs:
    """Local plasma parameters at a single radial location.

    All quantities are in SI / conventional tokamak units.

    Parameters
    ----------
    rho : float
        Normalised toroidal flux coordinate (0 = axis, 1 = edge).
    te_kev : float
        Electron temperature [keV].
    ti_kev : float
        Ion temperature [keV].
    ne_19 : float
        Electron density [10^19 m^-3].
    grad_te : float
        Normalised electron temperature gradient R/L_Te.
    grad_ti : float
        Normalised ion temperature gradient R/L_Ti.
    grad_ne : float
        Normalised electron density gradient R/L_ne.
    q : float
        Safety factor.
    s_hat : float
        Magnetic shear s = (r/q)(dq/dr).
    beta_e : float
        Electron beta (kinetic pressure / magnetic pressure).
    """

    rho: float = 0.5
    te_kev: float = 5.0
    ti_kev: float = 5.0
    ne_19: float = 5.0
    grad_te: float = 6.0
    grad_ti: float = 6.0
    grad_ne: float = 2.0
    q: float = 1.5
    s_hat: float = 0.8
    beta_e: float = 0.01


@dataclass
class TransportFluxes:
    """Predicted turbulent transport fluxes.

    Parameters
    ----------
    chi_e : float
        Electron thermal diffusivity [m^2/s].
    chi_i : float
        Ion thermal diffusivity [m^2/s].
    d_e : float
        Particle diffusivity [m^2/s].
    channel : str
        Dominant instability channel ("ITG", "TEM", "ETG", or "stable").
    """

    chi_e: float = 0.0
    chi_i: float = 0.0
    d_e: float = 0.0
    channel: str = "stable"


# ── Analytic fallback (critical-gradient model) ──────────────────────

# Dimits et al., Phys. Plasmas 7, 969 (2000), Fig. 3
_CRIT_ITG = 4.0  # R/L_Ti threshold for ITG onset (Cyclone base-case)
_CRIT_TEM = 5.0  # R/L_Te threshold for TEM; simplified from TGLF Eq. (32)
# Sugama & Horton, Phys. Plasmas 4, 405 (1997), Eq. (1)
_CHI_GB = 1.0  # Gyro-Bohm normalisation [m^2/s] at reference B, T, n
# Garbet et al., Plasma Phys. Control. Fusion 46, B557 (2004), §3
_STIFFNESS = 2.0  # Transport stiffness exponent above critical gradient


def critical_gradient_model(inp: TransportInputs) -> TransportFluxes:
    """Analytic critical-gradient transport model (fallback).

    Implements a stiff critical-gradient model:

        chi_i = chi_GB * max(0, R/L_Ti - crit_ITG)^stiffness
        chi_e = chi_GB * max(0, R/L_Te - crit_TEM)^stiffness
        D_e   = chi_e / 3  (simplified Ware pinch)

    This is the same physics as the Rust ``TransportSolver`` but
    parameterised in terms of normalised gradients rather than raw
    temperature differences.

    Parameters
    ----------
    inp : TransportInputs
        Local plasma parameters.

    Returns
    -------
    TransportFluxes
        Predicted fluxes with dominant channel identification.
    """
    excess_itg = max(0.0, inp.grad_ti - _CRIT_ITG)
    excess_tem = max(0.0, inp.grad_te - _CRIT_TEM)

    chi_i = _CHI_GB * excess_itg**_STIFFNESS
    chi_e = _CHI_GB * excess_tem**_STIFFNESS
    d_e = chi_e / 3.0  # Ware pinch approximation; QLKNN includes full pinch

    if chi_i > chi_e and chi_i > 0:
        channel = "ITG"
    elif chi_e > 0:
        channel = "TEM"
    else:
        channel = "stable"

    return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)


# ── MLP inference engine ─────────────────────────────────────────────


@dataclass
class MLPWeights:
    """Stored weights for a simple feedforward MLP.

    Architecture: input(10) → hidden1 → hidden2 → output(3)
    Activation: ReLU on hidden layers, softplus on output (ensures chi > 0).
    """

    w1: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b1: FloatArray = field(default_factory=lambda: np.zeros(0))
    w2: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b2: FloatArray = field(default_factory=lambda: np.zeros(0))
    w3: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    b3: FloatArray = field(default_factory=lambda: np.zeros(0))
    input_mean: FloatArray = field(default_factory=lambda: np.zeros(10))
    input_std: FloatArray = field(default_factory=lambda: np.ones(10))
    output_scale: FloatArray = field(default_factory=lambda: np.ones(3))


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(0.0, x)


def _softplus(x: FloatArray) -> FloatArray:
    return np.asarray(np.log1p(np.exp(np.clip(x, -20.0, 20.0))))


def _mlp_forward(x: FloatArray, weights: MLPWeights) -> FloatArray:
    """Forward pass through the 3-layer MLP.

    Parameters
    ----------
    x : FloatArray
        Input vector of shape ``(10,)`` or ``(batch, 10)``.
    weights : MLPWeights
        Network parameters.

    Returns
    -------
    FloatArray
        Output vector of shape ``(3,)`` or ``(batch, 3)``
        representing ``[chi_e, chi_i, D_e]``.
    """
    # Normalise inputs
    x_norm = (x - weights.input_mean) / np.maximum(weights.input_std, 1e-8)

    h1 = _relu(x_norm @ weights.w1 + weights.b1)
    h2 = _relu(h1 @ weights.w2 + weights.b2)
    out = _softplus(h2 @ weights.w3 + weights.b3)

    return out * weights.output_scale


# ── Main transport surrogate ─────────────────────────────────────────


class NeuralTransportModel:
    """Neural transport surrogate with analytic fallback.

    On construction, attempts to load MLP weights from *weights_path*.
    If loading fails (file missing, wrong format), the model
    transparently falls back to :func:`critical_gradient_model`.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to a ``.npz`` file containing MLP weights.  The file must
        contain arrays: ``w1, b1, w2, b2, w3, b3, input_mean,
        input_std, output_scale``.

    Examples
    --------
    >>> model = NeuralTransportModel()          # fallback mode
    >>> inp = TransportInputs(grad_ti=8.0)
    >>> fluxes = model.predict(inp)
    >>> fluxes.chi_i > 0
    True
    >>> model.is_neural
    False
    """

    def __init__(self, weights_path: str | Path | None = None, *, auto_discover: bool = True) -> None:
        self._weights: MLPWeights | None = None
        self.is_neural: bool = False
        self.weights_path: Path | None = None
        self.weights_checksum: str | None = None

        if weights_path is not None:
            self.weights_path = Path(weights_path)
            self._try_load_weights()
        elif auto_discover and _DEFAULT_WEIGHTS_PATH.exists():
            self.weights_path = _DEFAULT_WEIGHTS_PATH
            self._try_load_weights()

    def _try_load_weights(self) -> None:
        """Attempt to load MLP weights from disk."""
        if self.weights_path is None or not self.weights_path.exists():
            logger.info(
                "Neural transport weights not found at %s — using critical-gradient fallback",
                self.weights_path,
            )
            return

        try:
            data = np.load(self.weights_path)
            required = ["w1", "b1", "w2", "b2", "w3", "b3", "input_mean", "input_std", "output_scale"]
            for key in required:
                if key not in data:
                    logger.warning("Weight file missing key '%s' — falling back", key)
                    return

            # Version check (optional key, defaults to 1)
            version = int(data["version"].item()) if "version" in data else 1
            if version != _WEIGHTS_FORMAT_VERSION:
                logger.warning(
                    "Weight file version %d != expected %d — falling back",
                    version,
                    _WEIGHTS_FORMAT_VERSION,
                )
                return

            self._weights = MLPWeights(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                w3=data["w3"],
                b3=data["b3"],
                input_mean=data["input_mean"],
                input_std=data["input_std"],
                output_scale=data["output_scale"],
            )
            self.is_neural = True

            # Compute checksum for reproducibility tracking
            raw = b"".join(data[k].tobytes() for k in sorted(data.files) if k != "version")
            self.weights_checksum = hashlib.sha256(raw).hexdigest()[:16]

            logger.info(
                "Loaded neural transport weights from %s (layers: %s→%s→%s→3, version=%d, sha256=%s)",
                self.weights_path,
                self._weights.w1.shape[0],
                self._weights.w1.shape[1],
                self._weights.w2.shape[1],
                version,
                self.weights_checksum,
            )
        except (OSError, KeyError, ValueError, TypeError):
            logger.exception("Failed to load neural transport weights")

    def predict(self, inp: TransportInputs) -> TransportFluxes:
        """Predict turbulent transport fluxes for given local parameters.

        Uses the neural MLP if weights are loaded, otherwise falls back
        to the analytic critical-gradient model.

        Parameters
        ----------
        inp : TransportInputs
            Local plasma parameters at a single radial point.

        Returns
        -------
        TransportFluxes
            Predicted heat and particle diffusivities.
        """
        if not self.is_neural or self._weights is None:
            return critical_gradient_model(inp)

        x = np.array(
            [
                inp.rho,
                inp.te_kev,
                inp.ti_kev,
                inp.ne_19,
                inp.grad_te,
                inp.grad_ti,
                inp.grad_ne,
                inp.q,
                inp.s_hat,
                inp.beta_e,
            ]
        )
        out = _mlp_forward(x, self._weights)

        chi_e = float(out[0])
        chi_i = float(out[1])
        d_e = float(out[2])

        if chi_i > chi_e and chi_i > 0:
            channel = "ITG"
        elif chi_e > 0:
            channel = "TEM"
        else:  # pragma: no cover — requires both chi_e, chi_i ≤ 0
            channel = "stable"

        return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)

    def predict_profile(
        self,
        rho: FloatArray,
        te: FloatArray,
        ti: FloatArray,
        ne: FloatArray,
        q_profile: FloatArray,
        s_hat_profile: FloatArray,
        r_major: float = 6.2,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict transport coefficients on the full radial profile.

        Computes normalised gradients from the profile arrays via
        central finite differences, then evaluates the surrogate at
        each radial point.  When the MLP is loaded the entire profile
        is evaluated in a single batched forward pass (no Python loop).

        Parameters
        ----------
        rho : FloatArray
            Normalised radius grid (0 to 1), shape ``(N,)``.
        te, ti : FloatArray
            Electron/ion temperature profiles [keV], shape ``(N,)``.
        ne : FloatArray
            Electron density profile [10^19 m^-3], shape ``(N,)``.
        q_profile : FloatArray
            Safety factor profile, shape ``(N,)``.
        s_hat_profile : FloatArray
            Magnetic shear profile, shape ``(N,)``.
        r_major : float
            Major radius [m] for gradient normalisation.

        Returns
        -------
        chi_e, chi_i, d_e : FloatArray
            Transport coefficient profiles, each shape ``(N,)``.
        """

        # Normalised gradients: R/L_X = -R * (1/X) * dX/dr
        def norm_grad(x: FloatArray) -> FloatArray:
            dx = np.gradient(x, rho)
            safe_x = np.maximum(np.abs(x), 1e-6)
            return np.asarray(-r_major * dx / safe_x)

        grad_te = np.clip(norm_grad(te), 0, 50)
        grad_ti = np.clip(norm_grad(ti), 0, 50)
        grad_ne = np.clip(norm_grad(ne), -10, 30)
        beta_e = 4.03e-3 * ne * te

        # ── Neural path: single batched forward pass ─────────────
        if self.is_neural and self._weights is not None:
            x_batch = np.column_stack(
                [
                    rho,
                    te,
                    ti,
                    ne,
                    grad_te,
                    grad_ti,
                    grad_ne,
                    q_profile,
                    s_hat_profile,
                    beta_e,
                ]
            )  # (N, 10)
            out = _mlp_forward(x_batch, self._weights)  # (N, 3)
            chi_e_out = out[:, 0]
            chi_i_out = out[:, 1]
            d_e_out = out[:, 2]
            return chi_e_out, chi_i_out, d_e_out

        # ── Fallback: vectorised critical-gradient model ─────────
        excess_itg = np.maximum(0.0, grad_ti - _CRIT_ITG)
        excess_tem = np.maximum(0.0, grad_te - _CRIT_TEM)

        chi_i_out = _CHI_GB * excess_itg**_STIFFNESS
        chi_e_out = _CHI_GB * excess_tem**_STIFFNESS
        d_e_out = chi_e_out / 3.0

        return chi_e_out, chi_i_out, d_e_out


def reference_transport_benchmark_cases() -> tuple[tuple[str, TransportInputs], ...]:
    """Deterministic analytic benchmark set for transport cross-validation.

    The cases are built around the cited critical-gradient thresholds so the
    benchmark spans stable, ITG-dominant, TEM-dominant, and mixed regimes.
    """
    return (
        ("stable_core", TransportInputs(rho=0.25, grad_ti=_CRIT_ITG - 1.0, grad_te=_CRIT_TEM - 1.0)),
        ("marginal_itg", TransportInputs(rho=0.35, grad_ti=_CRIT_ITG + 0.5, grad_te=_CRIT_TEM - 1.5)),
        ("marginal_tem", TransportInputs(rho=0.45, grad_ti=_CRIT_ITG - 1.5, grad_te=_CRIT_TEM + 0.5)),
        ("itg_dominant", TransportInputs(rho=0.55, grad_ti=_CRIT_ITG + 4.0, grad_te=_CRIT_TEM - 0.5)),
        ("tem_dominant", TransportInputs(rho=0.65, grad_ti=_CRIT_ITG - 0.5, grad_te=_CRIT_TEM + 4.0)),
        ("mixed_gradient", TransportInputs(rho=0.75, grad_ti=_CRIT_ITG + 3.0, grad_te=_CRIT_TEM + 3.0)),
        ("edge_pedestal", TransportInputs(rho=0.92, grad_ti=_CRIT_ITG + 6.0, grad_te=_CRIT_TEM + 5.0)),
    )


def _reference_transport_profile(
    n_points: int = 24,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    n = max(int(n_points), 8)
    rho = np.linspace(0.05, 0.95, n, dtype=np.float64)
    te = 12.0 * (1.0 - 0.85 * rho**2)
    ti = 11.0 * (1.0 - 0.80 * rho**2)
    ne = 9.0 * (1.0 - 0.45 * rho**2)
    q_profile = 1.0 + 2.4 * rho**2
    s_hat_profile = 0.4 + 1.8 * rho
    return rho, te, ti, ne, q_profile, s_hat_profile


def cross_validate_neural_transport(
    model: NeuralTransportModel | None = None,
    *,
    benchmark_cases: tuple[tuple[str, TransportInputs], ...] | None = None,
    profile_points: int = 24,
) -> dict[str, Any]:
    """Compare the current transport surrogate to the analytic reference.

    Returns pointwise and profile-level error metrics against the
    :func:`critical_gradient_model` benchmark. This is a reference check, not a
    claim of agreement with full QLKNN-10D or first-principles turbulence
    solvers.
    """
    active_model = NeuralTransportModel() if model is None else model
    cases = reference_transport_benchmark_cases() if benchmark_cases is None else benchmark_cases
    if len(cases) < 1:
        raise ValueError("benchmark_cases must contain at least one case.")

    predicted_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    case_names: list[str] = []
    channel_matches = 0
    max_abs_error = 0.0

    for name, inp in cases:
        predicted = active_model.predict(inp)
        reference = critical_gradient_model(inp)
        pred_vec = [predicted.chi_e, predicted.chi_i, predicted.d_e]
        ref_vec = [reference.chi_e, reference.chi_i, reference.d_e]
        predicted_rows.append(pred_vec)
        reference_rows.append(ref_vec)
        case_names.append(name)
        if predicted.channel == reference.channel:
            channel_matches += 1
        max_abs_error = max(max_abs_error, max(abs(p - r) for p, r in zip(pred_vec, ref_vec)))

    predicted_arr = np.asarray(predicted_rows, dtype=np.float64)
    reference_arr = np.asarray(reference_rows, dtype=np.float64)
    error_arr = predicted_arr - reference_arr
    rmse = np.sqrt(np.mean(error_arr**2, axis=0))
    mae = np.mean(np.abs(error_arr), axis=0)
    ref_rms = np.sqrt(np.mean(reference_arr**2, axis=0))
    rel_rmse = rmse / np.maximum(ref_rms, 1.0e-8)

    rho, te, ti, ne, q_profile, s_hat_profile = _reference_transport_profile(profile_points)
    reference_profile_model = NeuralTransportModel(auto_discover=False)
    ref_chi_e, ref_chi_i, ref_d_e = reference_profile_model.predict_profile(rho, te, ti, ne, q_profile, s_hat_profile)
    pred_chi_e, pred_chi_i, pred_d_e = active_model.predict_profile(rho, te, ti, ne, q_profile, s_hat_profile)
    profile_pred = np.column_stack([pred_chi_e, pred_chi_i, pred_d_e])
    profile_ref = np.column_stack([ref_chi_e, ref_chi_i, ref_d_e])
    profile_err = profile_pred - profile_ref
    profile_rmse = np.sqrt(np.mean(profile_err**2, axis=0))
    profile_ref_rms = np.sqrt(np.mean(profile_ref**2, axis=0))
    profile_rel_rmse = profile_rmse / np.maximum(profile_ref_rms, 1.0e-8)

    return {
        "mode": "neural" if active_model.is_neural else "analytic_fallback",
        "weights_path": str(active_model.weights_path) if active_model.weights_path is not None else None,
        "weights_checksum": active_model.weights_checksum,
        "n_cases": int(len(cases)),
        "benchmark_cases": tuple(case_names),
        "per_channel_rmse": [float(v) for v in rmse],
        "per_channel_mae": [float(v) for v in mae],
        "per_channel_relative_rmse": [float(v) for v in rel_rmse],
        "profile_per_channel_rmse": [float(v) for v in profile_rmse],
        "profile_per_channel_relative_rmse": [float(v) for v in profile_rel_rmse],
        "channel_agreement": float(channel_matches / len(cases)),
        "max_abs_error": float(max_abs_error),
    }


# Backward-compatible class name used by older interop/parity tests.
NeuralTransportSurrogate = NeuralTransportModel
