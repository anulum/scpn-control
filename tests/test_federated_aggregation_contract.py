# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated aggregation contract tests.
"""Public aggregation must reject malformed or nonfinite client updates."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from scpn_control.control.federated_disruption import (
    FederatedConfig,
    FederatedServer,
    _init_mlp_weights,
    create_facility_clients_from_arrays,
)


@pytest.mark.parametrize("sample_count", [-1, 0, 1.5, True])
def test_public_aggregate_rejects_invalid_sample_count(sample_count: object) -> None:
    """A nonpositive or noninteger facility weight cannot influence FedAvg."""
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=9)
    update: dict[str, Any] = {"weights": _init_mlp_weights(np.random.default_rng(9)), "n_samples": sample_count}
    with pytest.raises(ValueError, match="n_samples"):
        server.aggregate([update])


def test_public_aggregate_rejects_nonfinite_model() -> None:
    """One invalid client weight cannot turn a global parameter into NaN."""
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=9)
    weights = _init_mlp_weights(np.random.default_rng(9))
    weights["w1"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="weights.w1"):
        server.aggregate([{"weights": weights, "n_samples": 3}])


def test_public_aggregate_rejects_missing_sample_count() -> None:
    """An update missing its dataset size cannot enter weighted averaging."""
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=9)
    weights = _init_mlp_weights(np.random.default_rng(9))
    with pytest.raises(ValueError, match="n_samples"):
        server.aggregate([{"weights": weights}])


def test_public_aggregate_rejects_wrong_parameter_shape() -> None:
    """A syntactically named update cannot bypass MLP dimension checks."""
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=9)
    weights = _init_mlp_weights(np.random.default_rng(9))
    weights["w1"] = np.ones((1, 1))
    with pytest.raises(ValueError, match="weights.w1"):
        server.aggregate([{"weights": weights, "n_samples": 3}])


def test_nonfinite_local_training_rolls_back_round() -> None:
    """Extreme finite facility data cannot commit nonfinite model state."""
    data = {
        machine: {
            "X_train": np.full((4, 8), 1e200),
            "y_train": np.array([0.0, 1.0, 0.0, 1.0]),
            "X_test": np.full((2, 8), 1e200),
            "y_test": np.array([0.0, 1.0]),
        }
        for machine in ("DIII-D", "JET")
    }
    clients = create_facility_clients_from_arrays(data)
    server = FederatedServer(FederatedConfig(local_epochs=2, machines=["DIII-D", "JET"]), seed=9)
    before = copy.deepcopy(server.get_state())

    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="weights"):
        server.run_round(clients)

    assert server.get_state() == before
    assert all(not client._weights for client in clients)
