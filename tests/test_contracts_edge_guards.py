# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Contracts edge guard path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for extract_features missing key (136), non-finite target (144),
and passthrough non-finite value (157)."""

from __future__ import annotations


import pytest

from scpn_control.scpn.contracts import (
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
    extract_features,
)


class TestExtractFeaturesMissingKey:
    def test_missing_obs_key_raises(self):
        """Missing observation key raises KeyError (line 136)."""
        targets = ControlTargets()
        scales = ControlScales()
        with pytest.raises(KeyError, match="Missing observation key for feature extraction"):
            extract_features({}, targets, scales)


class TestExtractFeaturesNonFinite:
    def test_non_finite_target_raises(self):
        """Non-finite target raises ValueError (line 144)."""
        axes = [
            FeatureAxisSpec(
                obs_key="x",
                target=float("inf"),
                scale=1.0,
                pos_key="x_pos",
                neg_key="x_neg",
            )
        ]
        with pytest.raises(ValueError, match="target must be finite"):
            extract_features({"x": 1.0}, ControlTargets(), ControlScales(), feature_axes=axes)

    def test_non_finite_scale_raises(self):
        """Non-finite scale raises ValueError (line 147)."""
        axes = [
            FeatureAxisSpec(
                obs_key="x",
                target=1.0,
                scale=float("nan"),
                pos_key="x_pos",
                neg_key="x_neg",
            )
        ]
        with pytest.raises(ValueError, match="scale must be finite"):
            extract_features({"x": 1.0}, ControlTargets(), ControlScales(), feature_axes=axes)


class TestPassthroughNonFinite:
    def test_passthrough_non_finite_value_raises(self):
        """Non-finite passthrough observation raises ValueError (line 159-160)."""
        targets = ControlTargets()
        scales = ControlScales()
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "extra": float("inf")}
        with pytest.raises(ValueError, match="Passthrough observation value must be finite"):
            extract_features(obs, targets, scales, passthrough_keys=["extra"])

    def test_passthrough_missing_key_raises(self):
        """Missing passthrough key raises KeyError (line 157)."""
        targets = ControlTargets()
        scales = ControlScales()
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        with pytest.raises(KeyError, match="Missing observation key for passthrough"):
            extract_features(obs, targets, scales, passthrough_keys=["nonexistent"])
