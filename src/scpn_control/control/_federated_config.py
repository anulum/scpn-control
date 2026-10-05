# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated configuration.
"""Validate server settings and named facility membership."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scpn_control.control._federated_clients import MACHINE_PROFILES
from scpn_control.control._federated_privacy import (
    DifferentialPrivacyConfig,
    _require_positive_float,
    _require_positive_int,
)


@dataclass
class FederatedConfig:
    """Configuration for federated disruption prediction training."""

    n_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    aggregation: str = "fedavg"
    mu_proximal: float = 0.01  # FedProx proximal term weight; Li et al. MLSys 2020
    min_clients: int = 2
    machines: list[str] = field(default_factory=lambda: ["DIII-D", "JET", "KSTAR"])
    dp_config: DifferentialPrivacyConfig | None = None

    def __post_init__(self) -> None:
        self.n_rounds = _require_positive_int("n_rounds", self.n_rounds)
        self.local_epochs = _require_positive_int("local_epochs", self.local_epochs)
        self.learning_rate = _require_positive_float("learning_rate", self.learning_rate)
        if self.aggregation not in ("fedavg", "fedprox"):
            raise ValueError(f"aggregation must be 'fedavg' or 'fedprox', got {self.aggregation!r}")
        if isinstance(self.mu_proximal, bool) or not isinstance(
            self.mu_proximal, (int, float, np.integer, np.floating)
        ):
            raise ValueError("mu_proximal must be finite and >= 0")
        self.mu_proximal = float(self.mu_proximal)
        if not np.isfinite(self.mu_proximal) or self.mu_proximal < 0:
            raise ValueError("mu_proximal must be finite and >= 0")
        self.min_clients = _require_positive_int("min_clients", self.min_clients)
        if self.dp_config is not None and not isinstance(self.dp_config, DifferentialPrivacyConfig):
            raise ValueError("dp_config must be a DifferentialPrivacyConfig or null")
        if (
            not isinstance(self.machines, list)
            or not self.machines
            or not all(isinstance(machine, str) for machine in self.machines)
        ):
            raise ValueError("machines must be a nonempty list of facility names")
        if len(set(self.machines)) != len(self.machines):
            raise ValueError("machines must name distinct facilities")
        for machine in self.machines:
            if machine not in MACHINE_PROFILES:
                raise ValueError(f"Unknown machine {machine!r}; available: {sorted(MACHINE_PROFILES)}")
        self.machines = list(self.machines)
