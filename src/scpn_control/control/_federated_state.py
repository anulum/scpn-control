# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated state validation.
"""Validate replayable federated model state and nominal privacy records."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control._federated_privacy import (
    DifferentialPrivacyConfig,
    PrivacyLedgerEntry,
    gaussian_mechanism_epsilon,
)

STATE_SCHEMA_VERSION = 2
WEIGHT_SHAPES = {
    "w1": (8, 32),
    "b1": (32,),
    "w2": (32, 16),
    "b2": (16,),
    "w3": (16, 1),
    "b3": (1,),
}
LEDGER_FIELDS = {
    "round_index",
    "participating_clients",
    "epsilon_spent",
    "cumulative_epsilon",
    "delta",
    "max_update_norm",
    "noise_multiplier",
    "clipped_clients",
}


def _finite_number(name: str, value: object) -> float:
    """Parse one finite JSON number without accepting booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_integer(name: str, value: object) -> int:
    """Parse one positive JSON integer without coercing booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def model_weights_from_state(payload: object) -> dict[str, FloatArray]:
    """Accept only finite weights for the declared eight-input MLP."""
    if not isinstance(payload, dict) or set(payload) != set(WEIGHT_SHAPES):
        raise ValueError("weights must contain exactly the six MLP parameters")
    weights: dict[str, FloatArray] = {}
    for name, shape in WEIGHT_SHAPES.items():
        try:
            raw = np.asarray(payload[name])
            if raw.dtype.kind not in "iuf":
                raise ValueError("nonnumeric")
            value = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"weights.{name} must be finite numeric values") from exc
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"weights.{name} must be finite with shape {shape}")
        weights[name] = value.copy()
    return weights


def random_state_from_payload(name: str, payload: object) -> dict[str, Any]:
    """Validate a JSON-compatible default NumPy generator state for replay."""
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a NumPy generator state")
    generator = np.random.default_rng(0)
    if payload.get("bit_generator") != type(generator.bit_generator).__name__:
        raise ValueError(f"{name} has an unsupported bit generator")
    try:
        generator.bit_generator.state = copy.deepcopy(payload)
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError(f"{name} contains an invalid generator state") from exc
    return copy.deepcopy(dict(generator.bit_generator.state))


def privacy_ledger_from_state(
    payload: object,
    dp_config: DifferentialPrivacyConfig | None,
    machines: list[str],
    min_clients: int,
) -> list[PrivacyLedgerEntry]:
    """Check nominal arithmetic and declared facility identities, not provenance."""
    if not isinstance(payload, list):
        raise ValueError("privacy_ledger must be a list")
    if dp_config is None:
        if payload:
            raise ValueError("privacy_ledger requires dp_config")
        return []
    records: list[PrivacyLedgerEntry] = []
    cumulative = 0.0
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or set(raw) != LEDGER_FIELDS:
            raise ValueError(f"privacy_ledger[{index}] has invalid fields")
        round_index = raw["round_index"]
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index != index:
            raise ValueError(f"privacy_ledger[{index}].round_index is not sequential")
        participating = _positive_integer("participating_clients", raw["participating_clients"])
        if participating < min_clients or participating > len(machines):
            raise ValueError(f"privacy_ledger[{index}].participating_clients is outside the federation")
        for field, expected in (
            ("delta", dp_config.delta),
            ("max_update_norm", dp_config.max_update_norm),
            ("noise_multiplier", dp_config.noise_multiplier),
        ):
            if _finite_number(field, raw[field]) != float(expected):
                raise ValueError(f"privacy_ledger[{index}].{field} differs from dp_config")
        epsilon = gaussian_mechanism_epsilon(dp_config.noise_multiplier, dp_config.delta)
        cumulative += epsilon
        if not math.isclose(_finite_number("epsilon_spent", raw["epsilon_spent"]), epsilon, rel_tol=1e-12):
            raise ValueError(f"privacy_ledger[{index}].epsilon_spent is inconsistent")
        if not math.isclose(_finite_number("cumulative_epsilon", raw["cumulative_epsilon"]), cumulative, rel_tol=1e-12):
            raise ValueError(f"privacy_ledger[{index}].cumulative_epsilon is inconsistent")
        clipped = raw["clipped_clients"]
        if (
            not isinstance(clipped, list)
            or not all(isinstance(machine, str) for machine in clipped)
            or len(set(clipped)) != len(clipped)
            or len(clipped) > participating
            or not set(clipped).issubset(machines)
        ):
            raise ValueError(f"privacy_ledger[{index}].clipped_clients is invalid")
        records.append(
            PrivacyLedgerEntry(
                round_index=index,
                participating_clients=participating,
                epsilon_spent=epsilon,
                cumulative_epsilon=cumulative,
                delta=float(dp_config.delta),
                max_update_norm=float(dp_config.max_update_norm),
                noise_multiplier=float(dp_config.noise_multiplier),
                clipped_clients=tuple(clipped),
            )
        )
    return records
