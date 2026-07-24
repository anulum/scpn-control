# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real fallback contracts for disruption train/load/safe predict

"""Fallback and guard contracts on real production entry points.

Torch-absent cases clear the leaf ``torch`` attribute only (optional-dep
contract). Train-failure uses a real unwritable path so ``train_predictor``
actually fails I/O. Missing-checkpoint safe-predict uses real filesystem
absence — no FakeTorch and no patched train/load APIs.
"""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.disruption_checkpoint as leaf
from scpn_control.control.disruption_predictor import (
    load_or_train_predictor,
    predict_disruption_risk_safe,
    train_predictor,
)

torch = pytest.importorskip("torch")


class TestTrainPredictorTorchGuard:
    def test_torch_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """train_predictor without torch raises RuntimeError (optional-dep contract)."""
        monkeypatch.setattr(leaf, "torch", None)
        monkeypatch.setattr(leaf, "optim", None)
        monkeypatch.setattr(leaf, "nn", None)
        with pytest.raises(RuntimeError, match="Torch is required"):
            train_predictor(n_shots=8, epochs=1, save_plot=False)


class TestLoadOrTrainFallback:
    def test_train_failure_with_fallback_real_io(self, tmp_path) -> None:
        """Directory-as-model-path makes real train I/O fail; fallback returns None."""
        blocked = tmp_path / "cannot_save_here"
        blocked.mkdir()
        model, info = load_or_train_predictor(
            model_path=blocked,
            force_retrain=True,
            allow_fallback=True,
            allow_legacy_fallback=True,
            train_if_missing=True,
            train_kwargs={"seq_len": 16, "n_shots": 8, "epochs": 1, "save_plot": False, "seed": 2},
        )
        assert model is None
        assert info["fallback"] is True
        assert "train_failed" in info["reason"]

    def test_train_failure_without_fallback_raises(self, tmp_path) -> None:
        blocked = tmp_path / "cannot_save_here"
        blocked.mkdir()
        with pytest.raises((RuntimeError, ValueError, OSError, IsADirectoryError)):
            load_or_train_predictor(
                model_path=blocked,
                force_retrain=True,
                allow_fallback=False,
                train_if_missing=True,
                train_kwargs={"seq_len": 16, "n_shots": 8, "epochs": 1, "save_plot": False, "seed": 3},
            )


class TestPredictDisruptionRiskSafeFallback:
    def test_inference_failure_returns_base_risk(self, tmp_path) -> None:
        """Missing checkpoint with legacy opt-in falls back to heuristic risk."""
        signal = np.random.default_rng(42).normal(0.0, 0.1, size=100)
        risk, meta = predict_disruption_risk_safe(
            signal,
            model_path=tmp_path / "nonexistent.pt",
            train_if_missing=False,
            allow_legacy_fallback=True,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"
