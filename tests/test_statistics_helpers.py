# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Statistics helper tests.
"""Regression tests for small statistical helpers."""

from __future__ import annotations

import pytest

from scpn_control.core._statistics import linear_percentile


def test_linear_percentile_returns_single_sample() -> None:
    """A one-sample history returns its only finite value."""
    assert linear_percentile([4.2], 95.0) == pytest.approx(4.2)


def test_linear_percentile_interpolates_between_ordered_samples() -> None:
    """The helper matches NumPy's default linear percentile definition."""
    assert linear_percentile([1.0, 2.0, 3.0], 50.0) == pytest.approx(2.0)
    assert linear_percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)


def test_linear_percentile_rejects_empty_sample() -> None:
    """Empty runtime histories cannot produce a percentile."""
    with pytest.raises(ValueError, match="must not be empty"):
        linear_percentile([], 95.0)


def test_linear_percentile_rejects_out_of_range_percentile() -> None:
    """Percentiles outside the closed [0, 100] interval are invalid."""
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        linear_percentile([1.0, 2.0], 101.0)


def test_linear_percentile_rejects_nonfinite_sample() -> None:
    """Non-finite runtime histories fail closed."""
    with pytest.raises(ValueError, match="finite"):
        linear_percentile([1.0, float("nan")], 95.0)
