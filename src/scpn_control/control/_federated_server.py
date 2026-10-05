# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated Disruption.
"""Federated aggregation, rounds, and state persistence."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control._federated_aggregation import fedavg_aggregate
from scpn_control.control._federated_clients import MachineClient
from scpn_control.control._federated_config import FederatedConfig
from scpn_control.control._federated_model import _apply_weight_delta, _init_mlp_weights, _l2_norm, _weight_delta
from scpn_control.control._federated_privacy import (
    DifferentialPrivacyConfig,
    PrivacyLedgerEntry,
    gaussian_mechanism_epsilon,
)
from scpn_control.control._federated_state import (
    STATE_SCHEMA_VERSION,
    model_weights_from_state,
    privacy_ledger_from_state,
    random_state_from_payload,
)

logger = logging.getLogger("scpn_control.control.federated_disruption")


class FederatedServer:
    """Orchestrates federated training across machine clients."""

    def __init__(self, config: FederatedConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dp_rng = np.random.default_rng(config.dp_config.seed if config.dp_config is not None else seed + 1)
        self.global_weights = _init_mlp_weights(self.rng)
        self.privacy_ledger: list[PrivacyLedgerEntry] = []

    def aggregate(self, client_updates: list[dict[str, Any]]) -> dict[str, FloatArray]:
        """FedAvg: weighted average of model weights by dataset size.

        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
        """
        return fedavg_aggregate(client_updates)

    def _privatise_client_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        round_index: int,
    ) -> list[dict[str, Any]]:
        """Clip and noise full client model deltas for facility-level DP."""
        dp = self.config.dp_config
        if dp is None:
            return updates

        privatized: list[dict[str, Any]] = []
        clipped_clients: list[str] = []
        for update in updates:
            machine = str(update["machine"])
            delta = _weight_delta(update["weights"], self.global_weights)
            norm = _l2_norm(delta)
            clip = min(1.0, dp.max_update_norm / max(norm, 1.0e-12))
            if clip < 1.0:
                clipped_clients.append(machine)

            noised_delta: dict[str, FloatArray] = {}
            for key, value in delta.items():
                noise = self.dp_rng.normal(
                    0.0,
                    dp.noise_multiplier * dp.max_update_norm,
                    size=value.shape,
                )
                noised_delta[key] = value * clip + noise
            privatized.append(
                {
                    "weights": _apply_weight_delta(self.global_weights, noised_delta),
                    "n_samples": update["n_samples"],
                    "machine": machine,
                }
            )

        epsilon = gaussian_mechanism_epsilon(dp.noise_multiplier, dp.delta)
        cumulative = epsilon + (self.privacy_ledger[-1].cumulative_epsilon if self.privacy_ledger else 0.0)
        self.privacy_ledger.append(
            PrivacyLedgerEntry(
                round_index=round_index,
                participating_clients=len(updates),
                epsilon_spent=float(epsilon),
                cumulative_epsilon=float(cumulative),
                delta=float(dp.delta),
                max_update_norm=float(dp.max_update_norm),
                noise_multiplier=float(dp.noise_multiplier),
                clipped_clients=tuple(clipped_clients),
            )
        )
        return privatized

    def fedprox_aggregate(
        self,
        client_updates: list[dict[str, Any]],
        global_weights: dict[str, FloatArray],
        mu: float,
    ) -> dict[str, FloatArray]:
        """FedProx aggregation with proximal regularisation.

        The proximal term is applied during local training (not aggregation),
        so aggregation itself is weighted averaging — the difference from
        FedAvg is in the client-side gradient update. This method exists
        for API symmetry; the mu parameter documents the proximal weight used.
        """
        _ = mu  # applied during local_train, not aggregation
        return self.aggregate(client_updates)

    def run_round(self, clients: list[MachineClient]) -> dict[str, Any]:
        """Train and aggregate one round, restoring state if any step fails."""
        if len(clients) < self.config.min_clients:
            raise ValueError(f"Need >= {self.config.min_clients} clients, got {len(clients)}")
        machines = [client.machine for client in clients]
        if len(set(machines)) != len(machines):
            raise ValueError("run_round requires distinct client facilities")
        undeclared = set(machines) - set(self.config.machines)
        if undeclared:
            raise ValueError(f"client facilities not configured: {sorted(undeclared)}")

        prior_weights = {key: value.copy() for key, value in self.global_weights.items()}
        prior_client_weights = [
            (client, {key: value.copy() for key, value in client._weights.items()}) for client in clients
        ]
        prior_ledger = list(self.privacy_ledger)
        prior_dp_rng = copy.deepcopy(self.dp_rng.bit_generator.state)
        mu = self.config.mu_proximal if self.config.aggregation == "fedprox" else 0.0
        updates: list[dict[str, Any]] = []
        client_metrics: list[dict[str, Any]] = []

        try:
            for client in clients:
                local_w = model_weights_from_state(
                    client.local_train(
                        self.global_weights,
                        self.config.local_epochs,
                        mu,
                        learning_rate=self.config.learning_rate,
                    )
                )
                metrics = client.local_evaluate(local_w)
                updates.append({"weights": local_w, "n_samples": client.get_data_size(), "machine": client.machine})
                client_metrics.append({"machine": client.machine, **metrics})

            updates = self._privatise_client_updates(updates, round_index=len(self.privacy_ledger))
            if self.config.aggregation == "fedprox":
                self.global_weights = self.fedprox_aggregate(updates, self.global_weights, mu)
            else:
                self.global_weights = self.aggregate(updates)
        except Exception:
            self.global_weights = prior_weights
            self.privacy_ledger = prior_ledger
            self.dp_rng.bit_generator.state = prior_dp_rng
            for client, weights in prior_client_weights:
                client._weights = weights
            raise

        result: dict[str, Any] = {"client_metrics": client_metrics}
        if self.config.dp_config is not None:
            result["privacy"] = self.privacy_ledger[-1]
        return result

    def train(self, clients: list[MachineClient], n_rounds: int | None = None) -> list[dict[str, Any]]:
        """Full federated training loop.

        Returns per-round metrics including per-client accuracy, loss, n_samples.
        """
        rounds = n_rounds if n_rounds is not None else self.config.n_rounds
        history: list[dict[str, Any]] = []

        for r in range(rounds):
            round_result = self.run_round(clients)
            mean_loss = float(np.mean([m["loss"] for m in round_result["client_metrics"]]))
            mean_acc = float(np.mean([m["accuracy"] for m in round_result["client_metrics"]]))
            round_result["round"] = r
            round_result["mean_loss"] = mean_loss
            round_result["mean_accuracy"] = mean_acc
            history.append(round_result)
            logger.info("round %d  mean_loss=%.4f  mean_acc=%.3f", r, mean_loss, mean_acc)

        return history

    def privacy_summary(self) -> dict[str, float | int | None]:
        """Return nominal Gaussian accounting values for completed rounds."""
        if self.config.dp_config is None:
            return {"epsilon": None, "delta": None, "rounds": 0}
        epsilon = self.privacy_ledger[-1].cumulative_epsilon if self.privacy_ledger else 0.0
        return {
            "epsilon": float(epsilon),
            "delta": float(self.config.dp_config.delta),
            "rounds": len(self.privacy_ledger),
        }

    def get_state(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible snapshot for exact continuation."""
        dp_config = None
        if self.config.dp_config is not None:
            dp_config = {
                "max_update_norm": self.config.dp_config.max_update_norm,
                "noise_multiplier": self.config.dp_config.noise_multiplier,
                "delta": self.config.dp_config.delta,
                "seed": self.config.dp_config.seed,
            }
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "config": {
                "n_rounds": self.config.n_rounds,
                "local_epochs": self.config.local_epochs,
                "learning_rate": self.config.learning_rate,
                "aggregation": self.config.aggregation,
                "mu_proximal": self.config.mu_proximal,
                "min_clients": self.config.min_clients,
                "machines": list(self.config.machines),
                "dp_config": dp_config,
            },
            "weights": {k: v.tolist() for k, v in self.global_weights.items()},
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "dp_rng_state": copy.deepcopy(self.dp_rng.bit_generator.state),
            "privacy_ledger": [
                {
                    "round_index": entry.round_index,
                    "participating_clients": entry.participating_clients,
                    "epsilon_spent": entry.epsilon_spent,
                    "cumulative_epsilon": entry.cumulative_epsilon,
                    "delta": entry.delta,
                    "max_update_norm": entry.max_update_norm,
                    "noise_multiplier": entry.noise_multiplier,
                    "clipped_clients": list(entry.clipped_clients),
                }
                for entry in self.privacy_ledger
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FederatedServer:
        """Validate and resume a versioned snapshot without authenticating it."""
        required = {"schema_version", "config", "weights", "rng_state", "dp_rng_state", "privacy_ledger"}
        if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION or set(state) != required:
            raise ValueError("state requires schema_version 2 and all replay fields, including dp_rng_state")
        if not isinstance(state["config"], dict):
            raise ValueError("config must be a mapping")
        cfg_payload = dict(state["config"])
        dp_payload = cfg_payload.get("dp_config")
        if isinstance(dp_payload, dict):
            cfg_payload["dp_config"] = DifferentialPrivacyConfig(**dp_payload)
        elif dp_payload is not None:
            raise ValueError("dp_config must be a mapping or null")
        try:
            cfg = FederatedConfig(**cfg_payload)
        except TypeError as exc:
            raise ValueError("config has invalid fields") from exc
        weights = model_weights_from_state(state["weights"])
        ledger = privacy_ledger_from_state(state["privacy_ledger"], cfg.dp_config, cfg.machines, cfg.min_clients)
        rng_state = random_state_from_payload("rng_state", state["rng_state"])
        dp_rng_state = random_state_from_payload("dp_rng_state", state["dp_rng_state"])
        server = cls(cfg)
        server.global_weights = weights
        server.privacy_ledger = ledger
        server.rng.bit_generator.state = rng_state
        server.dp_rng.bit_generator.state = dp_rng_state
        return server
