# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated state contract tests.
"""Public state restoration must validate and faithfully resume a server."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from scpn_control.control.federated_disruption import (
    DifferentialPrivacyConfig,
    FederatedConfig,
    FederatedServer,
    MachineClient,
    create_machine_clients,
)


def _server_after_round() -> tuple[FederatedServer, list[MachineClient]]:
    """Build a seeded DP server after one public federated round."""
    clients = create_machine_clients([{"machine": "DIII-D"}, {"machine": "JET"}], seed=91)
    server = FederatedServer(
        FederatedConfig(
            local_epochs=1,
            machines=["DIII-D", "JET"],
            dp_config=DifferentialPrivacyConfig(max_update_norm=0.05, noise_multiplier=2.0, seed=21),
        ),
        seed=19,
    )
    server.run_round(clients)
    return server, clients


def test_json_state_resumes_the_same_noisy_round() -> None:
    """Restoring a JSON snapshot continues the exact DP random stream."""
    server, clients = _server_after_round()
    restored = FederatedServer.from_state(json.loads(json.dumps(server.get_state())))

    server.run_round(clients)
    restored.run_round(clients)

    for key, value in server.global_weights.items():
        np.testing.assert_array_equal(restored.global_weights[key], value)
    assert restored.privacy_ledger == server.privacy_ledger


def test_state_rejects_malformed_model_shape() -> None:
    """Restoration rejects a model that the public forward pass cannot use."""
    server, _ = _server_after_round()
    state = server.get_state()
    state["weights"]["w1"] = [[1.0]]

    with pytest.raises(ValueError, match="weights.w1"):
        FederatedServer.from_state(state)


def test_state_rejects_forged_privacy_totals() -> None:
    """Caller-written totals cannot be admitted by a matching config alone."""
    server, _ = _server_after_round()
    state = server.get_state()
    state["privacy_ledger"][0]["cumulative_epsilon"] = 0.0

    with pytest.raises(ValueError, match="cumulative_epsilon"):
        FederatedServer.from_state(state)


def test_state_without_random_stream_is_not_resumable() -> None:
    """A legacy snapshot missing random state must fail closed for replay."""
    server, _ = _server_after_round()
    state = server.get_state()
    state.pop("dp_rng_state", None)

    with pytest.raises(ValueError, match="dp_rng_state"):
        FederatedServer.from_state(state)


@pytest.fixture(scope="module")
def dp_snapshot() -> dict[str, Any]:
    """Provide a complete public state with one real completed DP round."""
    server, _ = _server_after_round()
    payload = json.loads(json.dumps(server.get_state()))
    if not isinstance(payload, dict):
        raise AssertionError("get_state did not serialize to a mapping")
    return payload


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"w1": [[1.0]]}, "weights.w1"),
        ({"w1": [["bad"]]}, "weights.w1"),
        ({"w1": [[float("inf")]]}, "weights.w1"),
    ],
)
def test_model_state_refuses_incomplete_or_nonnumeric_weights(
    dp_snapshot: dict[str, Any], weights: dict[str, Any], message: str
) -> None:
    """A restored model cannot bypass the public MLP parameter contract."""
    state = copy.deepcopy(dp_snapshot)
    state["weights"].update(weights)
    with pytest.raises(ValueError, match=message):
        FederatedServer.from_state(state)


def test_model_state_rejects_missing_parameter(dp_snapshot: dict[str, Any]) -> None:
    """A caller cannot omit an MLP parameter from a resumable model."""
    state = copy.deepcopy(dp_snapshot)
    del state["weights"]["b3"]
    with pytest.raises(ValueError, match="six MLP parameters"):
        FederatedServer.from_state(state)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("round_index", 2, "round_index"),
        ("participating_clients", 0, "participating_clients"),
        ("participating_clients", 3, "participating_clients"),
        ("delta", 0.1, "delta"),
        ("noise_multiplier", "2.0", "noise_multiplier"),
        ("max_update_norm", float("nan"), "max_update_norm"),
        ("epsilon_spent", 0.0, "epsilon_spent"),
        ("epsilon_spent", 10**1000, "epsilon_spent"),
        ("clipped_clients", ["DIII-D", "DIII-D"], "clipped_clients"),
        ("clipped_clients", ["SPARC"], "clipped_clients"),
    ],
)
def test_privacy_state_refuses_inconsistent_public_records(
    dp_snapshot: dict[str, Any], field: str, value: object, message: str
) -> None:
    """The restored ledger must agree with its config and completed rounds."""
    state = copy.deepcopy(dp_snapshot)
    state["privacy_ledger"][0][field] = value
    with pytest.raises(ValueError, match=message):
        FederatedServer.from_state(state)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rng_state", "invalid", "rng_state"),
        ("dp_rng_state", {"bit_generator": "MT19937"}, "dp_rng_state"),
        ("dp_rng_state", {"bit_generator": "PCG64"}, "dp_rng_state"),
        ("privacy_ledger", {}, "privacy_ledger"),
        ("config", [], "config"),
    ],
)
def test_state_refuses_broken_replay_or_envelope(
    dp_snapshot: dict[str, Any], field: str, value: object, message: str
) -> None:
    """Missing or malformed replay data cannot be treated as a live server."""
    state = copy.deepcopy(dp_snapshot)
    state[field] = value
    with pytest.raises(ValueError, match=message):
        FederatedServer.from_state(state)


def test_nonprivate_server_rejects_privacy_records(dp_snapshot: dict[str, Any]) -> None:
    """A ledger cannot be restored under a configuration with no noise policy."""
    state = copy.deepcopy(dp_snapshot)
    state["config"]["dp_config"] = None
    with pytest.raises(ValueError, match="privacy_ledger requires dp_config"):
        FederatedServer.from_state(state)


def test_privacy_state_rejects_missing_record_field(dp_snapshot: dict[str, Any]) -> None:
    """An incomplete record cannot be mistaken for a completed round."""
    state = copy.deepcopy(dp_snapshot)
    del state["privacy_ledger"][0]["delta"]
    with pytest.raises(ValueError, match="privacy_ledger.*invalid fields"):
        FederatedServer.from_state(state)


def test_state_rejects_non_mapping_dp_config(dp_snapshot: dict[str, Any]) -> None:
    """An invalid DP policy cannot enter the resumed training path."""
    state = copy.deepcopy(dp_snapshot)
    state["config"]["dp_config"] = "disabled"
    with pytest.raises(ValueError, match="dp_config"):
        FederatedServer.from_state(state)


def test_state_rejects_undeclared_config_fields(dp_snapshot: dict[str, Any]) -> None:
    """Unexpected configuration keys cannot silently change replay meaning."""
    state = copy.deepcopy(dp_snapshot)
    state["config"]["operator_note"] = "trusted"
    with pytest.raises(ValueError, match="config has invalid fields"):
        FederatedServer.from_state(state)
