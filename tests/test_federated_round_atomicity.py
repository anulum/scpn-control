# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated round failure atomicity tests.
"""Public round failure must preserve the previously committed model state."""

from __future__ import annotations

import copy
import json

import pytest

from scpn_control.control.federated_disruption import (
    DifferentialPrivacyConfig,
    FederatedConfig,
    FederatedServer,
    create_machine_clients,
)


def test_failed_aggregate_restores_client_and_privacy_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed round leaves every client and the DP stream ready to retry."""
    clients = create_machine_clients([{"machine": "DIII-D"}, {"machine": "JET"}], seed=31)
    config = FederatedConfig(
        local_epochs=1,
        machines=["DIII-D", "JET"],
        dp_config=DifferentialPrivacyConfig(seed=17),
    )
    server = FederatedServer(config, seed=29)
    before_state = server.get_state()
    before_rng = copy.deepcopy(server.dp_rng.bit_generator.state)
    before_client_weights = [{key: value.copy() for key, value in client._weights.items()} for client in clients]

    def fail_aggregate(_updates: object) -> object:
        raise RuntimeError("aggregate failed")

    monkeypatch.setattr(server, "aggregate", fail_aggregate)
    with pytest.raises(RuntimeError, match="aggregate failed"):
        server.run_round(clients)

    assert server.get_state() == before_state
    assert json.dumps(server.dp_rng.bit_generator.state, sort_keys=True) == json.dumps(before_rng, sort_keys=True)
    for client, prior in zip(clients, before_client_weights, strict=True):
        assert client._weights.keys() == prior.keys()


def test_failed_client_evaluation_restores_prior_local_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later client failure rolls back earlier local training changes."""
    clients = create_machine_clients([{"machine": "DIII-D"}, {"machine": "JET"}], seed=31)
    server = FederatedServer(FederatedConfig(local_epochs=1, machines=["DIII-D", "JET"]), seed=29)
    original_state = server.get_state()

    def fail_evaluate(_weights: object) -> object:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(clients[1], "local_evaluate", fail_evaluate)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        server.run_round(clients)

    assert server.get_state() == original_state
    assert all(not client._weights for client in clients)
