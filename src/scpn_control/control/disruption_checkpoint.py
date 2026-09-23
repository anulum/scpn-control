# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption checkpoint integrity and train/load helpers

"""Checkpoint path, weights-hash integrity, and train/load orchestration.

This leaf owns the fail-closed SHA-256 integrity gate, default model path
helpers, and the train/load entry points that deserialise or synthesise a
predictor. The optional torch model class remains on the
owner; claim boundaries and physics proxies live in sibling leaves.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.control.disruption_checkpoint_integrity import (
    DisruptionCheckpointIntegrityError as DisruptionCheckpointIntegrityError,
)
from scpn_control.control.disruption_checkpoint_integrity import (
    _expected_checkpoint_digest as _expected_checkpoint_digest,
)
from scpn_control.control.disruption_checkpoint_integrity import (
    _is_hex as _is_hex,
)
from scpn_control.control.disruption_checkpoint_integrity import (
    _sha256_file as _sha256_file,
)
from scpn_control.control.disruption_checkpoint_integrity import (
    verified_checkpoint_snapshot,
)
from scpn_control.control.disruption_checkpoint_integrity import (
    verify_checkpoint_integrity as verify_checkpoint_integrity,
)
from scpn_control.control.disruption_physics_proxies import simulate_tearing_mode
from scpn_control.control.disruption_risk_claims import (
    _attach_disruption_claim_boundary,
    disruption_risk_claim_boundary,
)
from scpn_control.core._validators import (
    require_int as _require_int,
)

logger = logging.getLogger(__name__)

# Matches owner DEFAULT_SEQ_LEN; kept local so this leaf does not import the
# owner at module load time (owner re-exports this module).
_DEFAULT_SEQ_LEN = 100
DEFAULT_MODEL_FILENAME = "disruption_model.pth"

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional plot path
    _HAS_MPL = False

try:
    import torch  # pragma: no cover - optional torch model path
    import torch.nn as nn  # pragma: no cover - optional torch model path
    import torch.optim as optim  # pragma: no cover - optional torch model path
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None


def _repo_root() -> Path:
    """Return the repository root for artifact path resolution."""
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    """Return the default trained-model artifact path."""
    return _repo_root() / "artifacts" / DEFAULT_MODEL_FILENAME


def _owner_signal_helpers() -> tuple[Any, Any, Any]:
    """Lazy-load owner helpers to avoid import cycles with the torch model class."""
    from scpn_control.control import disruption_predictor as owner

    return owner.DisruptionTransformer, owner._normalize_seq_len, owner._prepare_signal_window


def train_predictor(
    seq_len: int = _DEFAULT_SEQ_LEN,
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
    if torch is None or optim is None or nn is None:
        raise RuntimeError("Torch is required for train_predictor().")

    disruption_transformer_cls, normalize_seq_len, prepare_signal_window = _owner_signal_helpers()

    seq_len = normalize_seq_len(seq_len)
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
    x_train = []
    y_train = []

    sim_steps = max(1000, seq_len + 16)
    for _ in range(n_shots):
        sig, label, _ = simulate_tearing_mode(steps=sim_steps, rng=data_rng)
        sig_window = prepare_signal_window(sig, seq_len)
        x_train.append(sig_window.reshape(-1, 1))
        y_train.append(label)

    x_tensor = torch.tensor(np.asarray(x_train, dtype=np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = disruption_transformer_cls(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    logger.info("Training Transformer...")
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")

    logger.info("Validating on a new shot...")
    test_sig, test_lbl, _ = simulate_tearing_mode(steps=sim_steps, rng=eval_rng)
    input_sig = prepare_signal_window(test_sig, seq_len)
    input_tensor = torch.tensor(input_sig, dtype=torch.float32).reshape(1, -1, 1)

    model.eval()
    with torch.no_grad():
        risk = model(input_tensor).item()

    logger.info(f"Test Shot Ground Truth: {'DISRUPTIVE' if test_lbl else 'SAFE'}")
    logger.info(f"AI Prediction Risk: {risk * 100:.1f}%")

    torch.save({"state_dict": model.state_dict(), "seq_len": int(seq_len)}, model_path)
    logger.info(f"Saved model: {model_path}")

    plot_path = _repo_root() / "artifacts" / "Disruption_AI_Result.png"
    if save_plot and _HAS_MPL:
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
    seq_len: int = _DEFAULT_SEQ_LEN,
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

    disruption_transformer_cls, normalize_seq_len, _prepare = _owner_signal_helpers()

    if torch is None:
        if not allow_fallback:
            raise RuntimeError("Torch is required for load_or_train_predictor().")
        return None, _attach_disruption_claim_boundary(
            {
                "trained": False,
                "fallback": True,
                "reason": "torch_unavailable",
                "model_path": str(model_path) if model_path is not None else str(default_model_path()),
                "seq_len": int(normalize_seq_len(seq_len)),
            }
        )

    path = Path(model_path) if model_path is not None else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = normalize_seq_len(seq_len)
    kwargs = dict(train_kwargs or {})

    if path.exists() and not force_retrain:
        # Copy, hash and load the same bytes. A path swap between verification
        # and deserialisation cannot substitute different checkpoint weights.
        with verified_checkpoint_snapshot(path, expected_sha256, require_pin=require_pin) as (
            checkpoint_stream,
            weights_sha256,
        ):
            try:
                checkpoint = torch.load(checkpoint_stream, map_location="cpu", weights_only=True)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    loaded_seq_len = normalize_seq_len(checkpoint.get("seq_len", seq_len))
                else:
                    state_dict = checkpoint
                    loaded_seq_len = seq_len

                model = disruption_transformer_cls(seq_len=loaded_seq_len)
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
            except (RuntimeError, ValueError, KeyError, OSError, pickle.UnpicklingError) as exc:
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

    if not train_if_missing and not force_retrain:
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
        model, info = train_predictor(**kwargs)
    except (RuntimeError, ValueError, OSError) as exc:
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
    info["trained"] = True
    info["fallback"] = False
    info["facility_roc_validated"] = False
    info.setdefault("claim_boundary", disruption_risk_claim_boundary().to_metadata())
    return model, info
