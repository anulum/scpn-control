# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Contract kernel helper tests.
"""Focused tests for shared contract numeric kernels."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.contracts import decode_action_vector, feature_error_components


def test_feature_error_components_rejects_bad_shapes() -> None:
    """Feature component vectors require aligned one-dimensional inputs."""
    with pytest.raises(ValueError, match="one-dimensional"):
        feature_error_components(np.asarray([[1.0]], dtype=np.float64), [1.0], [1.0])
    with pytest.raises(ValueError, match="equal shapes"):
        feature_error_components([1.0], [1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="axis_names"):
        feature_error_components([1.0], [1.0], [1.0], axis_names=())
    with pytest.raises(ValueError, match="out_pos and out_neg"):
        feature_error_components([1.0], [1.0], [1.0], out_pos=np.zeros(2, dtype=np.float64))


def test_feature_error_components_reports_named_nonfinite_values() -> None:
    """Finite-value errors identify the offending feature axis."""
    with pytest.raises(ValueError, match="Observation value.*R_axis_m"):
        feature_error_components([float("nan")], [1.0], [1.0], axis_names=["R_axis_m"])
    with pytest.raises(ValueError, match="Feature axis target.*Z_axis_m"):
        feature_error_components([1.0], [float("inf")], [1.0], axis_names=["Z_axis_m"])
    with pytest.raises(ValueError, match="Feature axis scale.*beta_n"):
        feature_error_components([1.0], [1.0], [float("nan")], axis_names=["beta_n"])


def test_decode_action_vector_rejects_invalid_inputs() -> None:
    """Action-vector decoding validates shapes and bounds before mutation."""
    previous = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError, match="dt must be finite"):
        decode_action_vector([0.0], [0], [0], [1.0], [1.0], [1.0], 0.0, previous)
    with pytest.raises(ValueError, match="marking must be one-dimensional"):
        decode_action_vector(np.asarray([[0.0]], dtype=np.float64), [0], [0], [1.0], [1.0], [1.0], 0.1, previous)
    with pytest.raises(ValueError, match="equal shapes"):
        decode_action_vector([0.0], [0], [0], [], [1.0], [1.0], 0.1, previous)
    with pytest.raises(ValueError, match="work must match"):
        decode_action_vector([0.0], [0], [0], [1.0], [1.0], [1.0], 0.1, previous, work=np.zeros(2))


def test_decode_action_vector_returns_previous_for_empty_actions() -> None:
    """An empty action vector is a stable no-op over the previous vector."""
    previous = np.zeros(0, dtype=np.float64)
    result = decode_action_vector([], [], [], [], [], [], 0.1, previous)
    assert result is previous
    assert result.shape == (0,)
