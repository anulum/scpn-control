# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.
"""Bounded synthetic multi-facility disruption benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from scpn_control.control._federated_clients import create_machine_clients
from scpn_control.control._federated_config import FederatedConfig
from scpn_control.control._federated_privacy import DifferentialPrivacyConfig
from scpn_control.control._federated_server import FederatedServer


@dataclass(frozen=True)
class FacilityBenchmarkSummary:
    """Deterministic benchmark summary for a federated disruption campaign."""

    aggregation: str
    machines: tuple[str, ...]
    n_rounds: int
    mean_accuracy: float
    mean_loss: float
    per_machine_accuracy: dict[str, float]
    privacy_epsilon: float | None
    privacy_delta: float | None
    evidence_kind: str


def run_synthetic_multifacility_benchmark(
    *,
    machines: list[str] | tuple[str, ...] = ("DIII-D", "JET", "KSTAR", "EAST"),
    n_rounds: int = 4,
    local_epochs: int = 3,
    aggregation: str = "fedprox",
    dp_config: DifferentialPrivacyConfig | None = None,
    seed: int = 20240531,
) -> FacilityBenchmarkSummary:
    """Run a deterministic synthetic multi-facility disruption benchmark.

    This benchmark exercises the production federation, heterogeneity, and
    nominal update-noise accounting. It is not measured cross-facility validation.
    """
    machine_list = list(machines)
    client_specs = [
        {
            "machine": machine,
            "n_train": 180,
            "n_test": 60,
            "disruption_fraction": 0.25 + 0.05 * (idx % 4),
            "learning_rate": 0.015,
        }
        for idx, machine in enumerate(machine_list)
    ]
    clients = create_machine_clients(client_specs, seed=seed)
    cfg = FederatedConfig(
        n_rounds=n_rounds,
        local_epochs=local_epochs,
        learning_rate=0.015,
        aggregation=aggregation,
        mu_proximal=0.05,
        min_clients=max(2, min(3, len(machine_list))),
        machines=machine_list,
        dp_config=dp_config,
    )
    server = FederatedServer(cfg, seed=seed)
    history = server.train(clients, n_rounds)
    final_metrics = history[-1]["client_metrics"]
    per_machine_accuracy = {str(metric["machine"]): float(metric["accuracy"]) for metric in final_metrics}
    privacy = server.privacy_summary()
    return FacilityBenchmarkSummary(
        aggregation=aggregation,
        machines=tuple(machine_list),
        n_rounds=n_rounds,
        mean_accuracy=float(history[-1]["mean_accuracy"]),
        mean_loss=float(history[-1]["mean_loss"]),
        per_machine_accuracy=per_machine_accuracy,
        privacy_epsilon=None if privacy["epsilon"] is None else float(privacy["epsilon"]),
        privacy_delta=None if privacy["delta"] is None else float(privacy["delta"]),
        evidence_kind="synthetic_multi_facility",
    )
