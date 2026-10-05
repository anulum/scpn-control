# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated client contract tests.
"""Public federation must count distinct declared facilities only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_control.control.federated_disruption import FederatedConfig, FederatedServer, create_machine_clients


@pytest.mark.parametrize("machines", [[], ["DIII-D", "DIII-D"]])
def test_config_rejects_empty_or_duplicate_facilities(machines: list[str]) -> None:
    """A declared federation contains at least one distinct machine."""
    with pytest.raises(ValueError, match="machines"):
        FederatedConfig(machines=machines)


def test_config_rejects_invalid_optional_policy_values() -> None:
    """Boolean proximal weight and unparsed privacy policy cannot enter rounds."""
    boolean_mu: Any = True
    with pytest.raises(ValueError, match="mu_proximal"):
        FederatedConfig(mu_proximal=boolean_mu)
    invalid_dp: Any = "enabled"
    with pytest.raises(ValueError, match="dp_config"):
        FederatedConfig(dp_config=invalid_dp)


def test_config_copies_caller_machine_list() -> None:
    """Later caller list mutation cannot silently expand the federation."""
    machines = ["DIII-D", "JET"]
    config = FederatedConfig(machines=machines)
    machines.append("KSTAR")
    assert config.machines == ["DIII-D", "JET"]


def test_round_rejects_duplicate_facility_without_mutation() -> None:
    """Two client objects from one facility cannot satisfy min_clients=2."""
    clients = create_machine_clients([{"machine": "DIII-D"}, {"machine": "DIII-D"}], seed=73)
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=31)
    before = server.get_state()

    with pytest.raises(ValueError, match="distinct"):
        server.run_round(clients)

    assert server.get_state() == before
    assert all(not client._weights for client in clients)


def test_round_rejects_undeclared_facility_without_mutation() -> None:
    """A client outside the config cannot change the global model."""
    clients = create_machine_clients([{"machine": "DIII-D"}, {"machine": "KSTAR"}], seed=73)
    server = FederatedServer(FederatedConfig(machines=["DIII-D", "JET"]), seed=31)
    before = server.get_state()

    with pytest.raises(ValueError, match="not configured"):
        server.run_round(clients)

    assert server.get_state() == before
    assert all(not client._weights for client in clients)


def test_server_learning_rate_changes_public_round() -> None:
    """FederatedConfig.learning_rate controls local optimisation in a round."""
    specs = [{"machine": "DIII-D"}, {"machine": "JET"}]
    slow_clients = create_machine_clients(specs, seed=13)
    fast_clients = create_machine_clients(specs, seed=13)
    slow = FederatedServer(FederatedConfig(local_epochs=2, learning_rate=0.001, machines=["DIII-D", "JET"]), seed=7)
    fast = FederatedServer(FederatedConfig(local_epochs=2, learning_rate=0.1, machines=["DIII-D", "JET"]), seed=7)

    slow.run_round(slow_clients)
    fast.run_round(fast_clients)

    differences = [np.max(np.abs(slow.global_weights[key] - fast.global_weights[key])) for key in slow.global_weights]
    assert max(differences) > 1e-4
