# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Disruption-prediction training, loading, and inference helpers with guarded optional dependencies."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Disruption prediction reference boundary: Kates-Harbeck et al. 2019,
# Nature 568, 526, validates FRNN disruption prediction on JET/DIII-D.
# This module's default public score is not that trained model; it is the
# fixed-weight heuristic declared by disruption_risk_claim_boundary().
#
# Locked-mode disruption precursor: de Vries et al. 2011, Nucl. Fusion 51,
# 053018 — cross-machine disruption database; locked modes dominate.
#
# Warning-time requirement: τ_warning > τ_TQ + τ_mitigation ≈ 10–30 ms for
# ITER; Lehnen et al. 2015, J. Nucl. Mater. 463, 39.
#
# Input features (locked-mode amplitude, P_rad fraction, q95, β_N, l_i,
# Greenwald fraction): Rea et al. 2019, Nucl. Fusion 59, 096016, Table I.

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import numpy as np

from scpn_control._typing import FloatArray

try:
    import torch  # pragma: no cover - optional torch model path
    import torch.nn as nn  # pragma: no cover - optional torch model path
    import torch.optim as optim  # pragma: no cover - optional torch model path
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None

from scpn_control.core._validators import (
    require_int as _require_int,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_FILENAME = "disruption_model.pth"
PROBABILISTIC_SIGMA_LEVELS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
TOROIDAL_PERTURB_FRACTION = 0.08
MIN_PROBABILISTIC_NOISE_SCALE = 1.0e-4
from scpn_control.control.disruption_fault_campaigns import (
    HybridAnomalyDetector as HybridAnomalyDetector,
)
from scpn_control.control.disruption_fault_campaigns import (
    _normalize_fault_campaign_inputs as _normalize_fault_campaign_inputs,
)
from scpn_control.control.disruption_fault_campaigns import (
    _synthetic_control_signal as _synthetic_control_signal,
)
from scpn_control.control.disruption_fault_campaigns import (
    apply_bit_flip_fault as apply_bit_flip_fault,
)
from scpn_control.control.disruption_fault_campaigns import (
    run_anomaly_alarm_campaign as run_anomaly_alarm_campaign,
)
from scpn_control.control.disruption_fault_campaigns import (
    run_fault_noise_campaign as run_fault_noise_campaign,
)
from scpn_control.control.disruption_physics_proxies import (
    _linear_percentile as _linear_percentile,
)
from scpn_control.control.disruption_physics_proxies import (
    build_disruption_feature_vector as build_disruption_feature_vector,
)
from scpn_control.control.disruption_physics_proxies import (
    disruption_warning_time as disruption_warning_time,
)
from scpn_control.control.disruption_physics_proxies import (
    predict_disruption_risk as predict_disruption_risk,
)
from scpn_control.control.disruption_physics_proxies import (
    simulate_tearing_mode as simulate_tearing_mode,
)
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_FEATURE_CONTRACT as DISRUPTION_FEATURE_CONTRACT,
)
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_HEURISTIC_REQUIRED_ACTION as DISRUPTION_HEURISTIC_REQUIRED_ACTION,
)
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_HEURISTIC_SCORE_SOURCE as DISRUPTION_HEURISTIC_SCORE_SOURCE,
)
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_HEURISTIC_TRAINING_PROVENANCE as DISRUPTION_HEURISTIC_TRAINING_PROVENANCE,
)
from scpn_control.control.disruption_risk_claims import (
    DISRUPTION_HEURISTIC_VALIDATION_PROVENANCE as DISRUPTION_HEURISTIC_VALIDATION_PROVENANCE,
)
from scpn_control.control.disruption_risk_claims import (
    LOCKED_MODE_ALARM_THRESHOLD as LOCKED_MODE_ALARM_THRESHOLD,
)
from scpn_control.control.disruption_risk_claims import (
    TAU_WARNING_MIN_S as TAU_WARNING_MIN_S,
)
from scpn_control.control.disruption_risk_claims import (
    DisruptionRiskClaimBoundary as DisruptionRiskClaimBoundary,
)
from scpn_control.control.disruption_risk_claims import (
    _attach_disruption_claim_boundary as _attach_disruption_claim_boundary,
)
from scpn_control.control.disruption_risk_claims import (
    disruption_risk_claim_boundary as disruption_risk_claim_boundary,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    """Return the default trained-model artifact path."""
    return _repo_root() / "artifacts" / DEFAULT_MODEL_FILENAME


class DisruptionCheckpointIntegrityError(RuntimeError):
    """Raised when a disruption-model checkpoint fails its weights-hash check.

    Raised when a pinned digest mismatches, when a sidecar is malformed, or when
    ``require_pin`` is set but no digest is available. It is a hard, fail-closed
    error and is *not* downgraded to the heuristic fallback (unlike a corrupt or
    unreadable file). The strong "never load unverified weights" guarantee holds
    only when a digest is pinned or ``require_pin=True``; an unpinned load is
    RCE-safe (``weights_only=True``) and records the digest for provenance, but
    does not verify the weights against a known-good reference.
    """


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_checkpoint_digest(path: Path, explicit: str | None) -> str | None:
    """Resolve the expected checkpoint digest from an explicit value or sidecar.

    Precedence: an ``explicit`` digest wins; otherwise a ``<checkpoint>.sha256``
    sidecar file (first whitespace-delimited token, as written by ``sha256sum``)
    is used if present. Returns ``None`` when neither pins a digest.
    """
    if explicit is not None:
        token = explicit.strip().split()[0] if explicit.strip() else ""
        if len(token) != 64 or not _is_hex(token):
            raise ValueError("expected_sha256 must be a 64-character hex SHA-256 digest")
        return token.lower()
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return None
    raw = sidecar.read_text(encoding="utf-8").strip()
    token = raw.split()[0] if raw else ""
    if len(token) != 64 or not _is_hex(token):
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint sidecar {sidecar.name} does not contain a valid SHA-256 digest"
        )
    return token.lower()


def _is_hex(text: str) -> bool:
    """Return whether every character in ``text`` is a hexadecimal digit."""
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def verify_checkpoint_integrity(path: Path, expected_sha256: str | None = None, *, require_pin: bool = False) -> str:
    """Return a checkpoint's SHA-256, enforcing an expected digest when pinned.

    When ``expected_sha256`` (or a ``<checkpoint>.sha256`` sidecar) pins a digest,
    a mismatch raises :class:`DisruptionCheckpointIntegrityError`. With nothing
    pinned the digest is returned for provenance without gating the load — unless
    ``require_pin`` is set, in which case an unpinned load is itself a fail-closed
    :class:`DisruptionCheckpointIntegrityError` (the strong "never load unverified
    weights" posture for safety-critical use).
    """
    expected = _expected_checkpoint_digest(path, expected_sha256)
    if expected is None and require_pin:
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint {path.name} has no pinned SHA-256 digest but require_pin is set"
        )
    actual = _sha256_file(path)
    if expected is not None and actual.lower() != expected:
        raise DisruptionCheckpointIntegrityError(
            f"checkpoint {path.name} SHA-256 {actual} does not match the pinned digest {expected}"
        )
    return actual


def _normalize_seq_len(seq_len: int) -> int:
    return _require_int("seq_len", seq_len, 8)


def _prepare_signal_window(signal: Any, seq_len: int) -> FloatArray:
    seq_len = _normalize_seq_len(seq_len)
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size == 0:
        raise ValueError("signal must contain at least one sample")
    if flat.size >= seq_len:
        return flat[:seq_len]
    return np.pad(flat, (0, seq_len - flat.size), mode="edge")


def _estimate_signal_noise_scale(signal: Any) -> float:
    flat = np.asarray(signal, dtype=float).reshape(-1)
    if flat.size < 2:
        return MIN_PROBABILISTIC_NOISE_SCALE
    diff_scale = float(np.std(np.diff(flat)))
    return max(diff_scale, MIN_PROBABILISTIC_NOISE_SCALE)


def _sigma_point_pattern(length: int) -> FloatArray:
    n = _require_int("length", length, 1)
    if n == 1:
        return np.ones(1, dtype=float)
    pattern = np.linspace(-1.0, 1.0, n, dtype=float)
    rms = float(np.sqrt(np.mean(pattern * pattern)))
    return pattern / max(rms, 1.0e-12)


def _perturb_toroidal_observables(
    toroidal_observables: dict[str, float] | None,
    sigma_level: float,
) -> dict[str, float] | None:
    if toroidal_observables is None:
        return None
    scale = 1.0 + TOROIDAL_PERTURB_FRACTION * float(sigma_level)
    return {key: float(value) * scale for key, value in toroidal_observables.items()}


def _summarize_risk_samples(
    samples: Any,
    *,
    center_risk: float,
    method: str,
) -> dict[str, Any]:
    clipped = np.clip(np.asarray(samples, dtype=float).reshape(-1), 0.0, 1.0)
    if clipped.size < 1:
        clipped = np.asarray([center_risk], dtype=float)
    q05, q50, q95 = np.quantile(clipped, [0.05, 0.50, 0.95])
    return {
        "probabilistic_output": True,
        "probabilistic_method": method,
        "risk_mean": float(np.mean(clipped)),
        "risk_std": float(np.std(clipped)),
        "risk_p05": float(q05),
        "risk_p50": float(q50),
        "risk_p95": float(q95),
        "risk_interval": [float(q05), float(q95)],
        "risk_interval_width": float(q95 - q05),
        "risk_samples_used": int(clipped.size),
        "risk_center": float(center_risk),
    }


def _deterministic_risk_samples(
    signal: Any,
    toroidal_observables: dict[str, float] | None,
) -> FloatArray:
    flat = np.asarray(signal, dtype=float).reshape(-1)
    noise_scale = _estimate_signal_noise_scale(flat)
    pattern = _sigma_point_pattern(max(flat.size, 1))
    samples = [
        predict_disruption_risk(
            flat + float(level) * noise_scale * pattern,
            _perturb_toroidal_observables(toroidal_observables, float(level)),
        )
        for level in PROBABILISTIC_SIGMA_LEVELS
    ]
    return np.asarray(samples, dtype=float)


def _model_risk_samples(  # pragma: no cover - requires torch
    model: Any,
    signal: Any,
    *,
    seq_len: int,
) -> FloatArray:
    if torch is None:
        raise RuntimeError("Torch is required for model-based risk samples.")
    prepared = _prepare_signal_window(signal, seq_len)
    noise_scale = _estimate_signal_noise_scale(prepared)
    pattern = _sigma_point_pattern(prepared.size)
    samples: list[float] = []
    model.eval()
    for level in PROBABILISTIC_SIGMA_LEVELS:
        perturbed = prepared + float(level) * noise_scale * pattern
        input_tensor = torch.tensor(perturbed, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            samples.append(float(np.clip(float(model(input_tensor).item()), 0.0, 1.0)))
    return np.asarray(samples, dtype=float)


if TYPE_CHECKING:

    class _TorchModule:
        """Static subset of ``torch.nn.Module`` used by this optional model."""

        def __init__(self) -> None: ...

        def train(self, mode: bool = True) -> Any: ...

        def eval(self) -> Any: ...

        def parameters(self) -> Any: ...

        def state_dict(self) -> Any: ...

        def load_state_dict(self, state_dict: Any) -> Any: ...

        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _DisruptionTransformerImpl:  # pragma: no cover - requires torch for full model
    """Transformer encoder for disruption prediction with MC dropout.

    Architecture follows Kates-Harbeck et al. 2019, Nature 568, 526 (FRNN)
    adapted to a single-channel time series with positional encoding.
    MC dropout uncertainty: Gal & Ghahramani 2016, ICML.
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN, dropout: float = 0.1) -> None:
        if torch is None or nn is None:
            raise RuntimeError("Torch is required for DisruptionTransformer.")
        super().__init__()
        self.seq_len = _normalize_seq_len(seq_len)
        self.embedding = nn.Linear(1, 32)
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.seq_len, 32))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, src: Any) -> Any:
        """Forward pass of the disruption transformer.

        Parameters
        ----------
        src
            Input sequence batch, shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        Any
            The per-sequence disruption score tensor.
        """
        if src.ndim != 3:
            raise ValueError(f"Input tensor must have shape [batch, seq, 1]; got rank {src.ndim}.")
        if src.shape[1] < 1:
            raise ValueError("Input sequence length must be >= 1.")
        if src.shape[2] != 1:
            raise ValueError(f"Input feature dimension must be 1; got {src.shape[2]}.")
        if src.shape[1] > self.seq_len:
            raise ValueError(f"Input sequence length {src.shape[1]} exceeds configured seq_len {self.seq_len}.")
        x = self.embedding(src) + self.pos_encoder[:, : src.shape[1], :]
        output = self.transformer(x)
        last_step = output[:, -1, :]
        return self.sigmoid(self.classifier(last_step))

    def predict_with_uncertainty(self, src: Any, n_samples: int = 10) -> tuple[float, float]:
        """MC dropout inference returning mean and standard deviation."""
        if torch is None:
            raise RuntimeError("Torch is required for DisruptionTransformer.")
        cast(Any, self).train()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(float(self.forward(src).item()))
        return float(np.mean(samples)), float(np.std(samples))

    def predict(self, seq: Any) -> float:
        """Return a bounded disruption risk for one input sequence.

        Parameters
        ----------
        seq
            One sequence as ``(seq,)``, ``(seq, 1)``, or ``(1, seq, 1)``. The
            input is converted to a ``float32`` tensor and validated by
            ``forward`` before inference.

        Returns
        -------
        float
            Disruption risk clipped to ``[0, 1]``.

        Raises
        ------
        RuntimeError
            If torch is unavailable.
        ValueError
            If ``seq`` is not a single sequence accepted by ``forward``.
        """
        if torch is None:
            raise RuntimeError("Torch is required for DisruptionTransformer.")
        tensor = torch.as_tensor(seq, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1, 1)
        elif tensor.ndim == 2:
            if tensor.shape[1] != 1:
                raise ValueError(f"Input feature dimension must be 1; got {tensor.shape[1]}.")
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(f"Input sequence must have rank 1, 2, or 3; got rank {tensor.ndim}.")

        cast(Any, self).eval()
        with torch.no_grad():
            score = self.forward(tensor)
        if score.numel() != 1:
            raise ValueError("predict expects exactly one sequence.")
        risk = float(score.reshape(-1)[0].item())
        return min(1.0, max(0.0, risk))


if TYPE_CHECKING:

    class DisruptionTransformer(_DisruptionTransformerImpl, _TorchModule):
        """Typed public transformer surface for static analysis."""

elif nn is not None:

    class DisruptionTransformer(_DisruptionTransformerImpl, nn.Module):  # pragma: no cover - requires torch
        """Runtime transformer surface backed by ``torch.nn.Module``."""

else:

    class DisruptionTransformer(_DisruptionTransformerImpl):  # pragma: no cover - optional dependency fallback
        """Runtime transformer surface that raises until torch is installed."""


def train_predictor(  # pragma: no cover - requires torch+matplotlib
    seq_len: int = DEFAULT_SEQ_LEN,
    n_shots: int = 500,
    epochs: int = 50,
    model_path: str | Path | None = None,
    seed: int = 42,
    save_plot: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Train a disruption-transformer predictor on synthetic shot sequences.

    Parameters
    ----------
    seq_len
        Input sequence length.
    n_shots
        Number of synthetic shots to generate.
    epochs
        Training epochs.
    model_path
        Optional output path for the trained model.
    seed
        Random seed.
    save_plot
        Whether to save a training-curve plot.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        The trained model and a training-history/metrics mapping.

    Raises
    ------
    RuntimeError
        If torch is unavailable.
    """
    if torch is None or optim is None:
        raise RuntimeError("Torch is required for train_predictor().")

    seq_len = _normalize_seq_len(seq_len)
    n_shots = _require_int("n_shots", n_shots, 8)
    epochs = _require_int("epochs", epochs, 1)
    seed = _require_int("seed", seed, 0)
    torch.manual_seed(seed)
    data_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1000003)

    model_path = Path(model_path) if model_path is not None else default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("--- SCPN SAFETY AI: Disruption Prediction (Transformer) ---")
    logger.info(f"Sequence length: {seq_len} | Shots: {n_shots} | Epochs: {epochs}")

    logger.info("Generating synthetic shots (Rutherford Physics)...")
    X_train = []
    y_train = []

    sim_steps = max(1000, seq_len + 16)
    for _ in range(n_shots):
        sig, label, _ = simulate_tearing_mode(steps=sim_steps, rng=data_rng)
        sig_window = _prepare_signal_window(sig, seq_len)
        X_train.append(sig_window.reshape(-1, 1))
        y_train.append(label)

    X_tensor = torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = DisruptionTransformer(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    logger.info("Training Transformer...")
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")

    logger.info("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode(steps=sim_steps, rng=eval_rng)
    input_sig = _prepare_signal_window(test_sig, seq_len)
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

    model.eval()
    with torch.no_grad():
        risk = model(input_tensor).item()

    logger.info(f"Test Shot Ground Truth: {'DISRUPTIVE' if test_lbl else 'SAFE'}")
    logger.info(f"AI Prediction Risk: {risk * 100:.1f}%")

    torch.save({"state_dict": model.state_dict(), "seq_len": int(seq_len)}, model_path)
    logger.info(f"Saved model: {model_path}")

    plot_path = _repo_root() / "artifacts" / "Disruption_AI_Result.png"
    if save_plot and HAS_MPL:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(losses)
            ax1.set_title("Transformer Training Loss")
            ax1.set_xlabel("Epoch")
            ax2.plot(test_sig, "r-" if test_lbl else "g-")
            ax2.set_title(f"Diagnostic Signal (AI Risk: {risk:.2f})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Saved: {plot_path}")
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Skipping disruption training plot: %s", exc)

    return model, _attach_disruption_claim_boundary(
        {
            "seq_len": int(seq_len),
            "shots": int(n_shots),
            "epochs": int(epochs),
            "model_path": str(model_path),
            "risk": float(risk),
            "test_label": int(test_lbl),
            "training_data": "synthetic_shots",
            "facility_roc_validated": False,
        }
    )


def load_or_train_predictor(
    model_path: str | Path | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    force_retrain: bool = False,
    train_kwargs: dict[str, Any] | None = None,
    train_if_missing: bool = True,
    allow_fallback: bool = False,
    allow_legacy_fallback: bool = False,
    expected_sha256: str | None = None,
    require_pin: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a trained predictor, training one if missing.

    When ``expected_sha256`` (or a ``<checkpoint>.sha256`` sidecar) pins a digest,
    the checkpoint's weights hash is verified before it is deserialised; a
    mismatch raises :class:`DisruptionCheckpointIntegrityError` and is never
    downgraded to the heuristic fallback. Set ``require_pin=True`` for
    safety-critical use to also reject an unpinned checkpoint (fail-closed when no
    digest is available). The resolved digest is always recorded as
    ``weights_sha256`` in the returned metadata for provenance.

    Parameters
    ----------
    model_path
        Optional path to a saved model.
    seq_len
        Input sequence length.
    force_retrain
        Retrain even when a saved model exists.
    train_kwargs
        Optional keyword arguments forwarded to :func:`train_predictor`.
    train_if_missing
        Train a new model when none is found.
    allow_fallback
        Permit the legacy deterministic fallback (requires
        ``allow_legacy_fallback``).
    allow_legacy_fallback
        Enable the legacy deterministic fallback path.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        The model and its metadata mapping.

    Raises
    ------
    ValueError
        If ``allow_fallback`` is set without ``allow_legacy_fallback``.
    """
    if allow_fallback and not allow_legacy_fallback:
        raise ValueError(
            "allow_fallback=True requires allow_legacy_fallback=True; "
            "legacy deterministic fallback is disabled by default."
        )

    if torch is None:
        if not allow_fallback:
            raise RuntimeError("Torch is required for load_or_train_predictor().")
        return None, _attach_disruption_claim_boundary(
            {
                "trained": False,
                "fallback": True,
                "reason": "torch_unavailable",
                "model_path": str(model_path) if model_path is not None else str(default_model_path()),
                "seq_len": int(_normalize_seq_len(seq_len)),
            }
        )

    path = Path(model_path) if model_path is not None else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = _normalize_seq_len(seq_len)
    kwargs = dict(train_kwargs or {})

    if path.exists() and not force_retrain:  # pragma: no cover - requires torch checkpoint
        # Fail-closed weights-integrity gate BEFORE deserialisation: a pinned
        # digest mismatch (or a missing pin under require_pin) raises hard and is
        # never swallowed by allow_fallback.
        weights_sha256 = verify_checkpoint_integrity(path, expected_sha256, require_pin=require_pin)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                loaded_seq_len = _normalize_seq_len(checkpoint.get("seq_len", seq_len))
            else:
                state_dict = checkpoint
                loaded_seq_len = seq_len

            model = DisruptionTransformer(seq_len=loaded_seq_len)
            model.load_state_dict(state_dict)
            model.eval()
            return model, _attach_disruption_claim_boundary(
                {
                    "trained": False,
                    "fallback": False,
                    "model_path": str(path),
                    "seq_len": int(loaded_seq_len),
                    "training_data": "checkpoint_metadata_unavailable",
                    "facility_roc_validated": False,
                    "weights_sha256": weights_sha256,
                }
            )
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            if not allow_fallback:
                raise
            return None, _attach_disruption_claim_boundary(
                {
                    "trained": False,
                    "fallback": True,
                    "reason": f"checkpoint_load_failed:{exc.__class__.__name__}",
                    "model_path": str(path),
                    "seq_len": int(seq_len),
                }
            )

    if not train_if_missing and not force_retrain:  # pragma: no cover - requires torch
        if allow_fallback:
            return None, _attach_disruption_claim_boundary(
                {
                    "trained": False,
                    "fallback": True,
                    "reason": "checkpoint_missing",
                    "model_path": str(path),
                    "seq_len": int(seq_len),
                }
            )
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    kwargs.setdefault("seq_len", seq_len)
    kwargs.setdefault("model_path", path)
    try:
        model, info = train_predictor(**kwargs)  # pragma: no cover - requires torch
    except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover - requires torch
        if not allow_fallback:
            raise
        return None, _attach_disruption_claim_boundary(
            {
                "trained": False,
                "fallback": True,
                "reason": f"train_failed:{exc.__class__.__name__}",
                "model_path": str(path),
                "seq_len": int(seq_len),
            }
        )
    info["trained"] = True  # pragma: no cover - requires torch
    info["fallback"] = False  # pragma: no cover - requires torch
    info["facility_roc_validated"] = False  # pragma: no cover - requires torch
    info.setdefault(
        "claim_boundary", disruption_risk_claim_boundary().to_metadata()
    )  # pragma: no cover - requires torch
    return model, info  # pragma: no cover - requires torch


def predict_disruption_risk_safe(
    signal: Any,
    toroidal_observables: dict[str, float] | None = None,
    *,
    model_path: str | Path | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    train_if_missing: bool = False,
    mc_samples: int = 10,
    allow_legacy_fallback: bool = False,
    allow_inference_fallback: bool = False,
    allow_legacy_inference_fallback: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Predict disruption risk with MC dropout uncertainty if model is available.

    Returns
    -------
    risk, metadata
        ``risk`` is the MC mean risk in ``[0, 1]``.
        ``metadata`` includes ``risk_std`` (epistemic uncertainty).
    """
    base_risk = float(np.clip(predict_disruption_risk(signal, toroidal_observables), 0.0, 1.0))
    if allow_inference_fallback and not allow_legacy_inference_fallback:
        raise ValueError(
            "allow_inference_fallback=True requires allow_legacy_inference_fallback=True; "
            "legacy inference fallback is disabled by default."
        )

    model, meta = load_or_train_predictor(
        model_path=model_path,
        seq_len=seq_len,
        force_retrain=False,
        train_kwargs={"seq_len": _normalize_seq_len(seq_len), "save_plot": False},
        train_if_missing=bool(train_if_missing),
        allow_fallback=bool(allow_legacy_fallback),
        allow_legacy_fallback=bool(allow_legacy_fallback),
    )

    if model is None or torch is None:
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        out_meta.update(
            _summarize_risk_samples(
                _deterministic_risk_samples(signal, toroidal_observables),
                center_risk=base_risk,
                method="deterministic_sigma_points",
            )
        )
        return base_risk, out_meta

    try:  # pragma: no cover - requires torch model
        model_seq_len = int(meta.get("seq_len", _normalize_seq_len(seq_len)))
        input_sig = _prepare_signal_window(signal, model_seq_len)
        input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

        if hasattr(model, "predict_with_uncertainty"):
            mean_risk, std_risk = model.predict_with_uncertainty(input_tensor, n_samples=mc_samples)
        else:
            model.eval()
            with torch.no_grad():
                mean_risk = float(model(input_tensor).item())
            std_risk = 0.0

        out_meta = dict(meta)
        out_meta["mode"] = "checkpoint"
        out_meta["risk_source"] = "transformer_mc_dropout"
        out_meta["risk_std"] = std_risk
        out_meta["risk_mean"] = mean_risk

        input_samples = _deterministic_risk_samples(signal, toroidal_observables)
        combined_samples = np.append(input_samples, [mean_risk])

        out_meta.update(
            _summarize_risk_samples(
                combined_samples,
                center_risk=mean_risk,
                method="transformer_mc_plus_sigma_points",
            )
        )
        return mean_risk, out_meta
    except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover - requires torch model
        if not allow_inference_fallback:
            raise RuntimeError(
                "Transformer inference failed and legacy inference fallback is disabled. "
                "Set allow_inference_fallback=True and allow_legacy_inference_fallback=True "
                "for explicit degraded-mode operation."
            ) from exc
        out_meta = dict(meta)
        out_meta["mode"] = "fallback"
        out_meta["risk_source"] = "predict_disruption_risk"
        out_meta["reason"] = f"inference_failed:{exc.__class__.__name__}"
        out_meta.update(
            _summarize_risk_samples(
                _deterministic_risk_samples(signal, toroidal_observables),
                center_risk=base_risk,
                method="deterministic_sigma_points",
            )
        )
        return base_risk, out_meta


def evaluate_predictor(
    model: Any,
    X_test: Any,
    y_test: Any,
    times_test: Any = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate disruption predictor on test set.

    Returns accuracy, precision, recall, F1, confusion matrix,
    and recall@T for T in [10, 20, 30, 50, 100] ms.
    """
    pred_list: list[int] = []
    for seq in X_test:
        pred = model.predict(seq)
        pred_list.append(1 if pred > threshold else 0)
    predictions = np.array(pred_list)
    y_true = np.array(y_test)

    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    if times_test is not None:
        for T_ms in [10, 20, 30, 50, 100]:
            T_s = T_ms / 1000.0
            early_enough = np.array(times_test) >= T_s
            mask = (y_true == 1) & early_enough
            mask_items = [bool(item) for item in mask.tolist()]
            mask_count = sum(1 for item in mask_items if item)
            if mask_count > 0:
                predicted_positive = [bool(item) for item in (predictions == 1).tolist()]
                true_positive_hits = sum(
                    1
                    for predicted, included in zip(predicted_positive, mask_items, strict=True)
                    if included and predicted
                )
                recall_at_t = true_positive_hits / mask_count
            else:
                recall_at_t = 0.0
            result[f"recall_at_{T_ms}ms"] = float(recall_at_t)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or load disruption Transformer predictor.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    _, meta = load_or_train_predictor(
        model_path=args.model_path,
        seq_len=args.seq_len,
        force_retrain=bool(args.force_retrain),
        train_kwargs={
            "seq_len": args.seq_len,
            "n_shots": args.shots,
            "epochs": args.epochs,
            "seed": args.seed,
            "save_plot": not args.no_plot,
        },
    )
    print(
        f"Predictor ready | trained={meta.get('trained')} | seq_len={meta.get('seq_len')} "
        f"| model_path={meta.get('model_path')}"
    )
