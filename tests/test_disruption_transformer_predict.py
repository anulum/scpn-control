# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — DisruptionTransformer predict() regression tests.
"""Regression tests for the public DisruptionTransformer.predict surface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch: Any = pytest.importorskip("torch")

from scpn_control.control.disruption_predictor import DisruptionTransformer, evaluate_predictor


def test_predict_accepts_single_sequence_shapes() -> None:
    """DisruptionTransformer.predict accepts the shapes used by evaluators."""
    model = DisruptionTransformer(seq_len=32)

    one_dimensional = model.predict(np.zeros(32, dtype=np.float32))
    two_dimensional = model.predict(np.zeros((32, 1), dtype=np.float32))
    three_dimensional = model.predict(torch.zeros(1, 32, 1))

    assert 0.0 <= one_dimensional <= 1.0
    assert 0.0 <= two_dimensional <= 1.0
    assert 0.0 <= three_dimensional <= 1.0


def test_predict_rejects_feature_matrix_without_single_channel() -> None:
    """Feature matrices must carry exactly one input channel."""
    model = DisruptionTransformer(seq_len=32)
    with pytest.raises(ValueError, match="feature dimension must be 1"):
        model.predict(np.zeros((32, 2), dtype=np.float32))


def test_evaluate_predictor_accepts_disruption_transformer_directly() -> None:
    """The evaluator no longer needs an external predict wrapper."""
    model = DisruptionTransformer(seq_len=32)
    x_test = [np.zeros((32, 1), dtype=np.float32), np.ones((32, 1), dtype=np.float32)]
    y_test = [0, 1]

    result = evaluate_predictor(model, x_test, y_test)

    assert set(result["confusion_matrix"]) == {"tp", "fp", "tn", "fn"}
    assert 0.0 <= result["accuracy"] <= 1.0
