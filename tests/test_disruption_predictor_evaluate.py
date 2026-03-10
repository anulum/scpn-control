# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption predictor evaluate + inference fallback tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for run_anomaly_alarm_campaign positive label+alarm (453-456),
load_or_train allow_fallback=False re-raise (706),
predict_disruption_risk_safe model inference failure (764-769)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from scpn_control.control.disruption_predictor import (
    load_or_train_predictor,
    predict_disruption_risk_safe,
    run_anomaly_alarm_campaign,
)

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestAnomalyCampaignPositiveLabel:
    def test_campaign_covers_positive_alarm(self):
        """run_anomaly_alarm_campaign with mock forcing true positives (453-456)."""

        def _mock_tearing(steps=1000, *, rng=None):
            # Return a high-amplitude disruptive signal (label=1)
            sig = np.ones(steps) * 5.0  # high anomaly signal
            return sig, 1, np.zeros(steps)

        with patch(
            "scpn_control.control.disruption_predictor.simulate_tearing_mode",
            side_effect=_mock_tearing,
        ):
            result = run_anomaly_alarm_campaign(
                window=50,
                episodes=10,
                seed=42,
                threshold=0.1,
            )
        assert result["true_positive_rate"] > 0.0


class TestLoadOrTrainNoFallback:
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_train_failure_no_fallback_raises(self, tmp_path):
        """Training failure with allow_fallback=False re-raises (line 706)."""
        with (
            patch(
                "scpn_control.control.disruption_predictor.train_predictor",
                side_effect=RuntimeError("mock train failure"),
            ),
            pytest.raises(RuntimeError, match="mock train failure"),
        ):
            load_or_train_predictor(
                model_path=tmp_path / "x.pt",
                force_retrain=True,
                allow_fallback=False,
                train_if_missing=True,
            )


class TestPredictSafeInferenceFailure:
    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
    def test_model_inference_failure_falls_back(self):
        """Model that raises on inference triggers fallback (lines 764-769)."""

        class _BrokenModel:
            def eval(self):
                pass

            def __call__(self, x):
                raise RuntimeError("inference exploded")

        signal = np.random.default_rng(42).normal(0.0, 0.1, size=100)
        with patch(
            "scpn_control.control.disruption_predictor.load_or_train_predictor",
            return_value=(_BrokenModel(), {"seq_len": 50, "trained": True}),
        ):
            risk, meta = predict_disruption_risk_safe(signal)

        assert 0.0 <= risk <= 1.0
        assert meta["mode"] == "fallback"
        assert "inference_failed" in meta["reason"]
