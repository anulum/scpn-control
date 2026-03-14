# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Validator Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Parametrized tests for core/_validators.py."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core._validators import (
    require_1d_array,
    require_finite_float,
    require_fraction,
    require_int,
    require_non_negative_float,
    require_positive_float,
    require_range,
)


# ── require_finite_float ─────────────────────────────────────────────


@pytest.mark.parametrize("val,expected", [(1.0, 1.0), (0, 0.0), (-3.5, -3.5), (np.float32(2.5), 2.5)])
def test_finite_float_accepts(val, expected):
    assert require_finite_float("x", val) == pytest.approx(expected)


@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
def test_finite_float_rejects_nonfinite(val):
    with pytest.raises(ValueError, match="finite"):
        require_finite_float("x", val)


def test_finite_float_rejects_string():
    with pytest.raises((ValueError, TypeError)):
        require_finite_float("x", "abc")


# ── require_positive_float ───────────────────────────────────────────


@pytest.mark.parametrize("val", [0.001, 1.0, 100.0])
def test_positive_float_accepts(val):
    assert require_positive_float("x", val) == pytest.approx(val)


@pytest.mark.parametrize("val", [0.0, -1.0, -0.001])
def test_positive_float_rejects_non_positive(val):
    with pytest.raises(ValueError, match="> 0"):
        require_positive_float("x", val)


@pytest.mark.parametrize("val", [float("nan"), float("inf")])
def test_positive_float_rejects_nonfinite(val):
    with pytest.raises(ValueError, match="finite"):
        require_positive_float("x", val)


# ── require_non_negative_float ───────────────────────────────────────


@pytest.mark.parametrize("val", [0.0, 0.5, 100.0])
def test_non_negative_accepts(val):
    assert require_non_negative_float("x", val) == pytest.approx(val)


@pytest.mark.parametrize("val", [-0.001, -100.0])
def test_non_negative_rejects_negative(val):
    with pytest.raises(ValueError, match=">= 0"):
        require_non_negative_float("x", val)


@pytest.mark.parametrize("val", [float("nan"), float("inf")])
def test_non_negative_rejects_nonfinite(val):
    with pytest.raises(ValueError, match="finite"):
        require_non_negative_float("x", val)


# ── require_int ──────────────────────────────────────────────────────


@pytest.mark.parametrize("val,minimum,expected", [(5, None, 5), (0, 0, 0), (np.int64(3), 1, 3)])
def test_int_accepts(val, minimum, expected):
    assert require_int("x", val, minimum=minimum) == expected


def test_int_rejects_bool():
    with pytest.raises(ValueError, match="integer"):
        require_int("x", True)


@pytest.mark.parametrize("val", [1.5, "3", None])
def test_int_rejects_non_int(val):
    with pytest.raises(ValueError, match="integer"):
        require_int("x", val)


def test_int_rejects_below_minimum():
    with pytest.raises(ValueError, match=">= 5"):
        require_int("x", 3, minimum=5)


# ── require_fraction ─────────────────────────────────────────────────


@pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
def test_fraction_accepts(val):
    assert require_fraction("x", val) == pytest.approx(val)


@pytest.mark.parametrize("val", [-0.01, 1.01, 2.0, -1.0])
def test_fraction_rejects_outside_bounds(val):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        require_fraction("x", val)


@pytest.mark.parametrize("val", [float("nan"), float("inf")])
def test_fraction_rejects_nonfinite(val):
    with pytest.raises(ValueError, match="finite"):
        require_fraction("x", val)


# ── require_range ────────────────────────────────────────────────────


def test_range_accepts_valid():
    assert require_range("x", (1.0, 5.0)) == (1.0, 5.0)


def test_range_accepts_with_min_allowed():
    assert require_range("x", (0.0, 10.0), min_allowed=0.0) == (0.0, 10.0)


def test_range_rejects_low_below_min():
    with pytest.raises(ValueError, match=">="):
        require_range("x", (-1.0, 5.0), min_allowed=0.0)


def test_range_rejects_high_le_low():
    with pytest.raises(ValueError, match="low < high"):
        require_range("x", (5.0, 5.0))


def test_range_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        require_range("x", (float("nan"), 5.0))


# ── require_1d_array ─────────────────────────────────────────────────


def test_1d_array_accepts_list():
    arr = require_1d_array("x", [1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    assert arr.dtype == np.float64


def test_1d_array_accepts_numpy():
    arr = require_1d_array("x", np.array([1.0, 2.0]))
    assert arr.shape == (2,)


def test_1d_array_rejects_2d():
    with pytest.raises(ValueError, match="1D"):
        require_1d_array("x", np.ones((3, 2)))


def test_1d_array_rejects_empty():
    with pytest.raises(ValueError, match="at least 1"):
        require_1d_array("x", [])


def test_1d_array_rejects_too_small():
    with pytest.raises(ValueError, match="at least 3"):
        require_1d_array("x", [1.0, 2.0], minimum_size=3)


def test_1d_array_expected_size_ok():
    arr = require_1d_array("x", [1.0, 2.0, 3.0], expected_size=3)
    assert arr.size == 3


def test_1d_array_expected_size_wrong():
    with pytest.raises(ValueError, match="4 samples"):
        require_1d_array("x", [1.0, 2.0, 3.0], expected_size=4)


def test_1d_array_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        require_1d_array("x", [1.0, float("nan"), 3.0])


def test_1d_array_rejects_inf():
    with pytest.raises(ValueError, match="finite"):
        require_1d_array("x", [1.0, float("inf")])
