# ──────────────────────────────────────────────────────────────────────
# SCPN Control — control/__init__.py Tests
# ──────────────────────────────────────────────────────────────────────
"""Tests for control package init: normalize_bounds, get_nengo_controller."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control import normalize_bounds, get_nengo_controller


class TestNormalizeBounds:
    def test_valid_pair(self):
        lo, hi = normalize_bounds((1.0, 5.0), "test")
        assert lo == 1.0
        assert hi == 5.0

    def test_rejects_lo_ge_hi(self):
        with pytest.raises(ValueError, match="test"):
            normalize_bounds((5.0, 5.0), "test")

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="inf_bounds"):
            normalize_bounds((float("-inf"), 1.0), "inf_bounds")

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="nan_bounds"):
            normalize_bounds((0.0, float("nan")), "nan_bounds")


class TestGetNengoController:
    def test_returns_class(self):
        cls = get_nengo_controller()
        assert cls is not None
        assert hasattr(cls, "__init__")
