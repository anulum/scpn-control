# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface load_or_train + predict_disruption_risk_safe tests

"""Production contracts for load/train and safe prediction (CTL-G07 R7-S4).

Real torch is required for checkpoint and inference paths. Optional-dependency
tests only clear the leaf/owner ``torch`` module attributes to exercise the
genuine ``torch is None`` branches — no FakeTorch, no patched train/load APIs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import scpn_control.control.disruption_checkpoint as checkpoint_mod
import scpn_control.control.disruption_predictor as dp_mod
from scpn_control.control.disruption_predictor import (
    load_or_train_predictor,
    predict_disruption_risk_safe,
    train_predictor,
)

torch = pytest.importorskip("torch")


def _patch_torch_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate optional torch missing on the production import sites."""
    monkeypatch.setattr(checkpoint_mod, "torch", None)
    monkeypatch.setattr(checkpoint_mod, "optim", None)
    monkeypatch.setattr(checkpoint_mod, "nn", None)
    monkeypatch.setattr(dp_mod, "torch", None)


def _train_small(tmp_path, name: str = "safe_api_model.pth"):
    path = tmp_path / name
    model, info = train_predictor(
        seq_len=16,
        n_shots=8,
        epochs=1,
        model_path=path,
        seed=1,
        save_plot=False,
    )
    digest = checkpoint_mod.verify_checkpoint_integrity(path)
    return path, model, info, digest


class TestLoadOrTrainOptionalTorchAbsent:
    """Optional-dependency contract only (torch attribute cleared)."""

    def test_legacy_fallback_requires_explicit_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        with pytest.raises(ValueError, match="allow_legacy_fallback=True"):
            load_or_train_predictor(allow_fallback=True)

    def test_returns_none_model_when_torch_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        model, meta = load_or_train_predictor(allow_fallback=True, allow_legacy_fallback=True)
        assert model is None
        assert meta["fallback"] is True
        assert meta["reason"] == "torch_unavailable"

    def test_no_fallback_raises_when_torch_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        with pytest.raises(RuntimeError, match="Torch is required"):
            load_or_train_predictor(allow_fallback=False)

    def test_custom_model_path_in_meta(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        _patch_torch_absent(monkeypatch)
        _, meta = load_or_train_predictor(
            model_path=str(tmp_path / "custom.pth"),
            allow_fallback=True,
            allow_legacy_fallback=True,
        )
        assert "custom.pth" in meta["model_path"]

    def test_seq_len_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        _, meta = load_or_train_predictor(seq_len=64, allow_fallback=True, allow_legacy_fallback=True)
        assert meta["seq_len"] == 64

    def test_none_path_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        _, meta = load_or_train_predictor(model_path=None, allow_fallback=True, allow_legacy_fallback=True)
        assert "disruption_model.pth" in meta["model_path"]


class TestLoadOrTrainMissingCheckpointRealTorch:
    def test_missing_checkpoint_no_train_no_fallback_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_or_train_predictor(
                model_path=str(tmp_path / "nonexistent.pth"),
                train_if_missing=False,
                allow_fallback=False,
            )

    def test_missing_checkpoint_train_if_missing_false_fallback(self, tmp_path) -> None:
        model, meta = load_or_train_predictor(
            model_path=str(tmp_path / "nonexistent.pth"),
            train_if_missing=False,
            allow_fallback=True,
            allow_legacy_fallback=True,
        )
        assert model is None
        assert meta["fallback"] is True
        assert meta["reason"] == "checkpoint_missing"

    def test_missing_with_torch_absent_reports_torch_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """When torch is absent, that contract wins over missing-file detail."""
        _patch_torch_absent(monkeypatch)
        _, meta = load_or_train_predictor(
            model_path=str(tmp_path / "nonexistent.pth"),
            train_if_missing=False,
            allow_fallback=True,
            allow_legacy_fallback=True,
        )
        assert meta["reason"] == "torch_unavailable"


class TestPredictDisruptionRiskSafeFallback:
    def test_returns_bounded_risk_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        signal = np.ones(100) * 0.5
        risk, meta = predict_disruption_risk_safe(signal, allow_legacy_fallback=True)
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"
        assert meta["risk_source"] == "predict_disruption_risk"

    def test_with_toroidal_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        signal = np.linspace(0.1, 5.0, 100)
        toroidal = {"toroidal_n1_amp": 1.0, "toroidal_n2_amp": 0.5, "toroidal_n3_amp": 0.3}
        risk, meta = predict_disruption_risk_safe(signal, toroidal, allow_legacy_fallback=True)
        assert 0.0 <= risk <= 1.0
        assert meta["fallback"] is True

    def test_custom_seq_len_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        signal = np.ones(200)
        risk, meta = predict_disruption_risk_safe(signal, seq_len=64, allow_legacy_fallback=True)
        assert 0.0 <= risk <= 1.0
        assert meta["seq_len"] == 64

    def test_fallback_returns_probabilistic_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_torch_absent(monkeypatch)
        signal = np.linspace(0.2, 1.8, 96)
        risk, meta = predict_disruption_risk_safe(signal, allow_legacy_fallback=True)
        assert 0.0 <= risk <= 1.0
        assert meta["probabilistic_output"] is True
        assert meta["probabilistic_method"] == "deterministic_sigma_points"
        assert 0.0 <= meta["risk_p05"] <= meta["risk_p95"] <= 1.0
        assert meta["risk_interval"][0] == pytest.approx(meta["risk_p05"])
        assert meta["risk_interval"][1] == pytest.approx(meta["risk_p95"])
        assert meta["risk_samples_used"] == len(dp_mod.PROBABILISTIC_SIGMA_LEVELS)
        assert meta["risk_std"] >= 0.0

    def test_missing_checkpoint_falls_back_with_legacy_opt_in(self, tmp_path) -> None:
        """Real missing weights + allow_legacy_fallback → heuristic safe path."""
        signal = np.random.default_rng(42).normal(0.0, 0.1, size=100)
        risk, meta = predict_disruption_risk_safe(
            signal,
            model_path=tmp_path / "nonexistent.pt",
            train_if_missing=False,
            allow_legacy_fallback=True,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"


class TestPredictDisruptionRiskSafeWithRealCheckpoint:
    """End-to-end: real train → load → safe predict in checkpoint mode."""

    def test_returns_bounded_risk_from_real_checkpoint(self, tmp_path) -> None:
        path, _model, _info, digest = _train_small(tmp_path)
        signal = np.ones(32) * 0.5
        risk, meta = predict_disruption_risk_safe(
            signal,
            model_path=path,
            seq_len=16,
            train_if_missing=False,
            allow_legacy_fallback=False,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "checkpoint"
        assert meta["risk_source"] == "transformer_mc_dropout"
        assert meta["weights_sha256"] == digest

    def test_repeated_safe_predict_stays_in_checkpoint_mode(self, tmp_path) -> None:
        """Repeated real loads stay in checkpoint mode (MC dropout may vary slightly)."""
        path, _model, _info, _digest = _train_small(tmp_path, name="det.pth")
        signal = np.linspace(0.5, 1.5, 32)
        r1, m1 = predict_disruption_risk_safe(
            signal, model_path=path, seq_len=16, train_if_missing=False, allow_legacy_fallback=False
        )
        r2, m2 = predict_disruption_risk_safe(
            signal, model_path=path, seq_len=16, train_if_missing=False, allow_legacy_fallback=False
        )
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0
        assert m1["mode"] == m2["mode"] == "checkpoint"
        assert m1["weights_sha256"] == m2["weights_sha256"]

    def test_checkpoint_path_returns_probabilistic_metadata(self, tmp_path) -> None:
        path, _model, _info, digest = _train_small(tmp_path, name="prob.pth")
        signal = np.linspace(0.1, 1.1, 32)
        risk, meta = predict_disruption_risk_safe(
            signal,
            model_path=path,
            seq_len=16,
            train_if_missing=False,
            allow_legacy_fallback=False,
        )
        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "checkpoint"
        assert meta["risk_source"] == "transformer_mc_dropout"
        assert meta["probabilistic_output"] is True
        assert meta["probabilistic_method"] == "transformer_mc_plus_sigma_points"
        assert 0.0 <= meta["risk_p05"] <= meta["risk_p95"] <= 1.0
        assert meta["risk_samples_used"] >= len(dp_mod.PROBABILISTIC_SIGMA_LEVELS) + 1
        assert meta["risk_std"] >= 0.0
        assert meta["weights_sha256"] == digest
        # Pin mismatch cannot silently degrade to heuristic when pin is required at load.
        with pytest.raises(checkpoint_mod.DisruptionCheckpointIntegrityError):
            load_or_train_predictor(
                model_path=path,
                train_if_missing=False,
                expected_sha256="f" * 64,
                require_pin=True,
            )

    def test_load_or_train_round_trip_digest(self, tmp_path) -> None:
        path, _model, _info, digest = _train_small(tmp_path, name="roundtrip.pth")
        loaded, meta = load_or_train_predictor(
            model_path=path,
            seq_len=16,
            train_if_missing=False,
            expected_sha256=digest,
            require_pin=True,
        )
        assert loaded is not None
        assert meta["weights_sha256"] == digest
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
