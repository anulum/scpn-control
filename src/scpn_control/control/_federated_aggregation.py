# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated aggregation.
"""Validate model updates and compute dataset-weighted FedAvg."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control._federated_state import model_weights_from_state


def fedavg_aggregate(client_updates: list[dict[str, Any]]) -> dict[str, FloatArray]:
    """Average finite eight-input MLP models with positive integer sample counts.

    The weighting follows McMahan et al., AISTATS 2017. Input validation is
    applied to the public aggregation boundary before any result is returned.
    """
    if not isinstance(client_updates, list) or not client_updates:
        raise ValueError("aggregate requires at least one client update")
    parsed: list[tuple[int, dict[str, FloatArray]]] = []
    for index, update in enumerate(client_updates):
        if not isinstance(update, dict) or "n_samples" not in update or "weights" not in update:
            raise ValueError(f"aggregate update {index} requires weights and n_samples")
        count = update["n_samples"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"aggregate requires positive client sample counts; update {index} n_samples is invalid")
        parsed.append((count, model_weights_from_state(update["weights"])))

    total = sum(count for count, _ in parsed)
    average = {key: np.zeros_like(value) for key, value in parsed[0][1].items()}
    with np.errstate(over="ignore", invalid="ignore"):
        for count, weights in parsed:
            fraction = count / total
            for key, value in weights.items():
                average[key] += value * fraction
    return model_weights_from_state(average)
