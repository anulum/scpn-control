# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Transport
"""
Neural-transport facade for turbulent transport coefficients.

The module provides a compact MLP weight loader and an analytic
critical-gradient fallback.  Quantitative QLKNN or QuaLiKiz claims are
admissible only when the loaded weights are paired with strict reference
artifacts; otherwise the public claim path remains the analytic fallback.

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
``docs/neural_transport_training.md`` to produce an ``.npz`` weight file
that this module loads at construction time.

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
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Weight file format version expected by this loader.
_WEIGHTS_FORMAT_VERSION = 1
_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "weights" / "neural_transport_qlknn.npz"
_NEURAL_TRANSPORT_CLAIM_SCHEMA_VERSION = 1
_NEURAL_TRANSPORT_REFERENCE_SOURCES = frozenset({"real_qualikiz", "documented_public_reference"})
_NEURAL_TRANSPORT_FEATURE_SCHEMA = (
    "R_LTi",
    "R_LTe",
    "R_Ln",
    "q",
    "s_hat",
    "alpha",
    "Ti_Te",
    "Zeff",
    "collisionality",
    "beta_e",
)


class _ChannelWeightsArray(np.ndarray[tuple[int, ...], np.dtype[np.float64]]):
    """Array view with reduction semantics that survive NumPy module reloads."""

    def sum(  # type: ignore[override]  # NumPy's overloads cannot express this reload-safe axis subset.
        self,
        axis: int | None = None,
        dtype: object = None,
        out: object = None,
        keepdims: bool = False,
        initial: float = 0.0,
        where: object = True,
    ) -> object:
        """Return sums without using ndarray's stale reload-sensitive defaults."""
        if dtype is not None or out is not None or keepdims or where is not True:
            raise ValueError("channel_weights.sum supports default dtype, out, keepdims, and where only")
        base = np.asarray(self, dtype=np.float64)
        if axis is None:
            return float(np.einsum("ij->", base, optimize=True)) + initial
        if axis == 0:
            return cast(FloatArray, np.einsum("ij->j", base, optimize=True) + initial)
        if axis == 1:
            return cast(FloatArray, np.einsum("ij->i", base, optimize=True) + initial)
        raise ValueError("channel_weights.sum supports axis None, 0, or 1")


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


@dataclass(frozen=True)
class NeuralTransportClosureResult:
    """Profile transport closure with provenance for controller coupling.

    Parameters
    ----------
    chi_e, chi_i, d_e : FloatArray
        Electron heat, ion heat, and particle diffusivity profiles.
    channel_weights : FloatArray
        Per-radius normalised contribution weights for ``chi_e``, ``chi_i``,
        and ``d_e`` with shape ``(3, N)``.
    source : str
        ``"neural"`` when loaded weights are active, otherwise
        ``"analytic_fallback"``.
    weights_checksum : str or None
        Loaded neural-weight checksum, or ``None`` for analytic fallback.
    """

    chi_e: FloatArray
    chi_i: FloatArray
    d_e: FloatArray
    channel_weights: FloatArray
    source: str
    weights_checksum: str | None


@dataclass(frozen=True)
class NeuralTransportClaimEvidence:
    """Serialisable admission evidence for neural-transport surrogate claims."""

    schema_version: int
    model_id: str
    source: str
    source_id: str
    surrogate_mode: str
    weights_path: str
    weights_sha256: str
    internal_weights_checksum: str
    feature_schema: tuple[str, ...]
    reference_source: str
    reference_dataset_id: str
    reference_artifact_sha256: str
    reference_sample_count: int
    local_case_count: int
    local_max_abs_error: float
    local_channel_agreement: float
    local_per_channel_relative_rmse: tuple[float, float, float]
    local_profile_per_channel_relative_rmse: tuple[float, float, float]
    chi_i_rmse_m2_s: float | None
    chi_e_rmse_m2_s: float | None
    d_e_rmse_m2_s: float | None
    chi_i_relative_mae: float | None
    unstable_branch_accuracy: float | None
    chi_i_rmse_tolerance_m2_s: float | None
    chi_e_rmse_tolerance_m2_s: float | None
    d_e_rmse_tolerance_m2_s: float | None
    chi_i_relative_mae_tolerance: float | None
    unstable_branch_accuracy_min: float | None
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


def _three_float_tuple(name: str, values: object) -> tuple[float, float, float]:
    if not isinstance(values, list | tuple) or len(values) != 3:
        raise ValueError(f"{name} must contain three channel values")
    out = tuple(float(v) for v in values)
    if not all(np.isfinite(v) and v >= 0.0 for v in out):
        raise ValueError(f"{name} must contain finite non-negative channel values")
    return cast(tuple[float, float, float], out)


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
        D_e   = chi_e · f(R/L_ne, magnetic shear) using a bounded density-channel
                response calibrated to stay inside the analytic fallback domain.

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
    d_e = cast(float, _fallback_particle_diffusivity(chi_e, inp.grad_ne, inp.s_hat))

    if chi_i > chi_e and chi_i > 0:
        channel = "ITG"
    elif chi_e > 0:
        channel = "TEM"
    else:
        channel = "stable"

    return TransportFluxes(chi_e=chi_e, chi_i=chi_i, d_e=d_e, channel=channel)


def _fallback_particle_diffusivity(
    chi_e: float | FloatArray,
    grad_ne: float | FloatArray,
    s_hat: float | FloatArray,
) -> float | FloatArray:
    """Bounded analytic particle-channel response for fallback transport."""
    ratio = 0.18 + 0.045 * np.maximum(np.asarray(grad_ne), 0.0) + 0.03 * np.abs(np.asarray(s_hat))
    ratio = np.clip(ratio, 0.05, 0.65)
    result = np.asarray(chi_e) * ratio
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


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

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        auto_discover: bool = True,
        allow_weight_load_fallback: bool = False,
        allow_legacy_weight_load_fallback: bool = False,
    ) -> None:
        if allow_weight_load_fallback and not allow_legacy_weight_load_fallback:
            raise ValueError(
                "allow_weight_load_fallback=True requires allow_legacy_weight_load_fallback=True; "
                "legacy neural-transport weight-load fallback is disabled by default."
            )
        self._weights: MLPWeights | None = None
        self.is_neural: bool = False
        self.weights_path: Path | None = None
        self.weights_checksum: str | None = None
        self._allow_weight_load_fallback = bool(allow_weight_load_fallback)
        self._allow_legacy_weight_load_fallback = bool(allow_legacy_weight_load_fallback)

        if weights_path is not None:
            self.weights_path = Path(weights_path)
            self._try_load_weights(strict=True)
        elif auto_discover and _DEFAULT_WEIGHTS_PATH.exists():
            self.weights_path = _DEFAULT_WEIGHTS_PATH
            self._try_load_weights(strict=False)

    def _try_load_weights(self, *, strict: bool) -> None:
        """Attempt to load MLP weights from disk."""
        if self.weights_path is None or not self.weights_path.exists():
            if strict and not self._allow_weight_load_fallback:
                raise FileNotFoundError(
                    f"Neural transport weights not found at {self.weights_path}. "
                    "Provide a valid weights_path or set both "
                    "allow_weight_load_fallback=True and "
                    "allow_legacy_weight_load_fallback=True for explicit degraded-mode operation."
                )
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
                    if strict and not self._allow_weight_load_fallback:
                        raise ValueError(
                            f"Neural transport weight file is missing required key '{key}'. "
                            "Provide a valid weights artifact or set both "
                            "allow_weight_load_fallback=True and "
                            "allow_legacy_weight_load_fallback=True for explicit degraded-mode operation."
                        )
                    logger.warning("Weight file missing key '%s' — falling back", key)
                    return

            # Version check (optional key, defaults to 1)
            version = int(data["version"].item()) if "version" in data else 1
            if version != _WEIGHTS_FORMAT_VERSION:
                if strict and not self._allow_weight_load_fallback:
                    raise ValueError(
                        f"Neural transport weight file version {version} != expected {_WEIGHTS_FORMAT_VERSION}. "
                        "Provide a compatible weights artifact or set both "
                        "allow_weight_load_fallback=True and "
                        "allow_legacy_weight_load_fallback=True for explicit degraded-mode operation."
                    )
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
        except (OSError, KeyError, TypeError, ValueError):
            if strict and not self._allow_weight_load_fallback:
                raise RuntimeError(
                    "Failed to load explicit neural transport weights. "
                    "Provide a valid weights artifact or set both "
                    "allow_weight_load_fallback=True and "
                    "allow_legacy_weight_load_fallback=True for explicit degraded-mode operation."
                )
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
        else:
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
        d_e_out = np.asarray(_fallback_particle_diffusivity(chi_e_out, grad_ne, s_hat_profile), dtype=float)

        return chi_e_out, chi_i_out, d_e_out


def _profile_array(name: str, values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional profile")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least three radial points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _normalised_channel_weights(chi_e: FloatArray, chi_i: FloatArray, d_e: FloatArray) -> FloatArray:
    import numpy as current_np

    raw = current_np.vstack([chi_e, chi_i, d_e])
    totals = current_np.einsum("ij->j", raw, optimize=True)
    weights = current_np.empty_like(raw)
    active = totals > 1e-14
    weights[:, active] = raw[:, active] / totals[active]
    weights[:, ~active] = 1.0 / 3.0
    return cast(FloatArray, weights.view(_ChannelWeightsArray))


def neural_transport_closure_profiles(
    rho: FloatArray,
    te: FloatArray,
    ti: FloatArray,
    ne: FloatArray,
    q_profile: FloatArray,
    s_hat_profile: FloatArray,
    *,
    model: NeuralTransportModel | None = None,
    r_major: float = 6.2,
    require_neural: bool = False,
    allow_fallback: bool = False,
    allow_legacy_fallback: bool = False,
) -> NeuralTransportClosureResult:
    """Return bounded neural-transport closure profiles with provenance.

    The function packages :meth:`NeuralTransportModel.predict_profile` for
    controller and differentiable-transport coupling.  It validates the radial
    grid and plasma profiles before evaluation, fails closed when neural weights
    are required but unavailable, and records whether the result came from the
    neural surrogate or the analytic critical-gradient fallback.
    """
    rho_arr = _profile_array("rho", rho)
    te_arr = _profile_array("te", te)
    ti_arr = _profile_array("ti", ti)
    ne_arr = _profile_array("ne", ne)
    q_arr = _profile_array("q_profile", q_profile)
    s_hat_arr = _profile_array("s_hat_profile", s_hat_profile)

    shape = rho_arr.shape
    for name, arr in (
        ("te", te_arr),
        ("ti", ti_arr),
        ("ne", ne_arr),
        ("q_profile", q_arr),
        ("s_hat_profile", s_hat_arr),
    ):
        if arr.shape != shape:
            raise ValueError(f"rho, te, ti, ne, q_profile, and s_hat_profile must have the same shape; {name} differs")
    if not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho must be strictly increasing")
    for name, arr in (("te", te_arr), ("ti", ti_arr), ("ne", ne_arr), ("q_profile", q_arr)):
        if not np.all(arr > 0.0):
            raise ValueError(f"{name} must be strictly positive")
    if not np.isfinite(r_major) or r_major <= 0.0:
        raise ValueError("r_major must be a positive finite major radius")

    active_model = model
    if active_model is None:
        active_model = NeuralTransportModel(
            auto_discover=True,
            allow_weight_load_fallback=allow_fallback,
            allow_legacy_weight_load_fallback=allow_legacy_fallback,
        )

    degraded_mode_allowed = allow_fallback and allow_legacy_fallback
    if require_neural and not active_model.is_neural and not degraded_mode_allowed:
        raise RuntimeError(
            "neural transport closure requires loaded neural transport weights; "
            "set both allow_fallback=True and allow_legacy_fallback=True to run the analytic fallback explicitly."
        )

    chi_e, chi_i, d_e = active_model.predict_profile(
        rho_arr,
        te_arr,
        ti_arr,
        ne_arr,
        q_arr,
        s_hat_arr,
        r_major=r_major,
    )
    chi_e_arr = np.asarray(chi_e, dtype=np.float64)
    chi_i_arr = np.asarray(chi_i, dtype=np.float64)
    d_e_arr = np.asarray(d_e, dtype=np.float64)

    for name, arr in (("chi_e", chi_e_arr), ("chi_i", chi_i_arr), ("d_e", d_e_arr)):
        if arr.shape != shape:
            raise RuntimeError(f"transport closure produced invalid {name} shape")
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"transport closure produced non-finite {name}")
        if np.any(arr < -1e-12):
            raise RuntimeError(f"transport closure produced negative {name}")

    chi_e_arr = np.maximum(chi_e_arr, 0.0)
    chi_i_arr = np.maximum(chi_i_arr, 0.0)
    d_e_arr = np.maximum(d_e_arr, 0.0)
    source = "neural" if active_model.is_neural else "analytic_fallback"

    return NeuralTransportClosureResult(
        chi_e=chi_e_arr,
        chi_i=chi_i_arr,
        d_e=d_e_arr,
        channel_weights=_normalised_channel_weights(chi_e_arr, chi_i_arr, d_e_arr),
        source=source,
        weights_checksum=active_model.weights_checksum if active_model.is_neural else None,
    )


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


def neural_transport_claim_evidence(
    validation_result: dict[str, Any],
    *,
    source: str,
    source_id: str,
    model_id: str = "neural_transport_qlknn_facade",
    weights_path: str | Path | None = None,
    reference_artifact_path: str | Path | None = None,
) -> NeuralTransportClaimEvidence:
    """Build fail-closed evidence for neural-transport quantitative claims."""
    mode = _non_empty_text("validation_result['mode']", str(validation_result.get("mode", "")))
    local_cases = int(validation_result.get("n_cases", 0))
    if local_cases < 1:
        raise ValueError("validation_result must contain at least one local benchmark case")
    max_abs_error = float(validation_result.get("max_abs_error", float("nan")))
    channel_agreement = float(validation_result.get("channel_agreement", float("nan")))
    if not np.isfinite(max_abs_error) or max_abs_error < 0.0:
        raise ValueError("validation_result max_abs_error must be finite and non-negative")
    if not np.isfinite(channel_agreement) or not 0.0 <= channel_agreement <= 1.0:
        raise ValueError("validation_result channel_agreement must be finite in [0, 1]")

    weights = Path(weights_path) if weights_path is not None else None
    weights_sha256 = ""
    if weights is not None:
        if not weights.is_file():
            raise FileNotFoundError(f"neural-transport weights not found: {weights}")
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
            raise ValueError("reference admission requires the exact neural-transport weights_path")
        from validation.validate_neural_transport_reference import validate_neural_transport_reference

        artifact_path = Path(reference_artifact_path)
        report = validate_neural_transport_reference(artifact_path, require_reference_artifacts=True)
        if report["status"] != "pass":
            raise ValueError("neural-transport reference artifact failed strict validation")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        reference_source = _non_empty_text("source", str(payload["source"]))
        if payload["trained_weights_sha256"].lower() != weights_sha256.lower():
            raise ValueError("neural-transport reference artifact does not match supplied weights")
        reference_dataset_id = _non_empty_text("reference_dataset_id", str(payload["reference_dataset_id"]))
        reference_artifact_sha256 = _non_empty_text(
            "reference_artifact_sha256", str(payload["reference_artifact_sha256"])
        )
        reference_sample_count = int(payload["reference_sample_count"])
        metrics = dict(payload["metrics"])
        tolerances = dict(payload["tolerances"])
        claim_allowed = True

    claim_status = (
        "matched neural-transport reference admission passed"
        if claim_allowed
        else "local surrogate regression evidence only; quantitative QuaLiKiz claims blocked"
    )
    return NeuralTransportClaimEvidence(
        schema_version=_NEURAL_TRANSPORT_CLAIM_SCHEMA_VERSION,
        model_id=_non_empty_text("model_id", model_id),
        source=_non_empty_text("source", source),
        source_id=_non_empty_text("source_id", source_id),
        surrogate_mode=mode,
        weights_path=str(weights) if weights is not None else "",
        weights_sha256=weights_sha256,
        internal_weights_checksum=str(validation_result.get("weights_checksum") or ""),
        feature_schema=_NEURAL_TRANSPORT_FEATURE_SCHEMA,
        reference_source=reference_source,
        reference_dataset_id=reference_dataset_id,
        reference_artifact_sha256=reference_artifact_sha256,
        reference_sample_count=reference_sample_count,
        local_case_count=local_cases,
        local_max_abs_error=max_abs_error,
        local_channel_agreement=channel_agreement,
        local_per_channel_relative_rmse=_three_float_tuple(
            "per_channel_relative_rmse", validation_result.get("per_channel_relative_rmse")
        ),
        local_profile_per_channel_relative_rmse=_three_float_tuple(
            "profile_per_channel_relative_rmse", validation_result.get("profile_per_channel_relative_rmse")
        ),
        chi_i_rmse_m2_s=_finite_nonnegative_or_none("chi_i_rmse_m2_s", metrics.get("chi_i_rmse_m2_s")),
        chi_e_rmse_m2_s=_finite_nonnegative_or_none("chi_e_rmse_m2_s", metrics.get("chi_e_rmse_m2_s")),
        d_e_rmse_m2_s=_finite_nonnegative_or_none("D_e_rmse_m2_s", metrics.get("D_e_rmse_m2_s")),
        chi_i_relative_mae=_finite_nonnegative_or_none("chi_i_relative_mae", metrics.get("chi_i_relative_mae")),
        unstable_branch_accuracy=_unit_interval_or_none(
            "unstable_branch_accuracy", metrics.get("unstable_branch_accuracy")
        ),
        chi_i_rmse_tolerance_m2_s=_finite_positive_or_none(
            "chi_i_rmse_tolerance_m2_s", tolerances.get("chi_i_rmse_m2_s")
        ),
        chi_e_rmse_tolerance_m2_s=_finite_positive_or_none(
            "chi_e_rmse_tolerance_m2_s", tolerances.get("chi_e_rmse_m2_s")
        ),
        d_e_rmse_tolerance_m2_s=_finite_positive_or_none("D_e_rmse_tolerance_m2_s", tolerances.get("D_e_rmse_m2_s")),
        chi_i_relative_mae_tolerance=_finite_positive_or_none(
            "chi_i_relative_mae_tolerance", tolerances.get("chi_i_relative_mae")
        ),
        unstable_branch_accuracy_min=_unit_interval_or_none(
            "unstable_branch_accuracy_min", tolerances.get("unstable_branch_accuracy_min")
        ),
        quantitative_claim_allowed=claim_allowed,
        claim_status=claim_status,
    )


def assert_neural_transport_quantitative_claim_admissible(
    evidence: NeuralTransportClaimEvidence,
) -> NeuralTransportClaimEvidence:
    """Return evidence only when strict matched-reference admission passed."""
    if not isinstance(evidence, NeuralTransportClaimEvidence):
        raise ValueError("evidence must be NeuralTransportClaimEvidence")
    if evidence.schema_version != _NEURAL_TRANSPORT_CLAIM_SCHEMA_VERSION:
        raise ValueError("neural-transport claim evidence schema_version is unsupported")
    if not evidence.quantitative_claim_allowed:
        raise ValueError("neural-transport quantitative claim is blocked without matched reference evidence")
    return evidence


def save_neural_transport_claim_evidence(evidence: NeuralTransportClaimEvidence, path: str | Path) -> None:
    """Persist neural-transport claim evidence as deterministic JSON."""
    if not isinstance(evidence, NeuralTransportClaimEvidence):
        raise ValueError("evidence must be NeuralTransportClaimEvidence")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Backward-compatible class name used by older interop/parity tests.
NeuralTransportSurrogate = NeuralTransportModel
