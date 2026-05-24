# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Checkpoint and resume API
"""
Utilities for saving and restoring simulation and campaign states.

Enables resuming long-running stress campaigns and persisting solver
hot-starts across sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1


def _validate_episode(episode: int) -> int:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("episode must be a non-negative integer")
    return episode


def _validate_finite_tree(name: str, obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        if not np.all(np.isfinite(obj)):
            raise ValueError(f"{name} must contain only finite values")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            _validate_finite_tree(f"{name}.{key}", value)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            _validate_finite_tree(f"{name}[{idx}]", value)
        return
    if isinstance(obj, (float, int)) and not isinstance(obj, bool):
        if not np.isfinite(float(obj)):
            raise ValueError(f"{name} must contain only finite values")


def _validate_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("checkpoint payload must be an object")
    for key in ("schema_version", "episode", "solver_state", "metrics"):
        if key not in data:
            raise ValueError(f"checkpoint payload missing {key}")
    if data["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CHECKPOINT_SCHEMA_VERSION}")
    if not isinstance(data["solver_state"], dict):
        raise ValueError("solver_state must be an object")
    if not isinstance(data["metrics"], dict):
        raise ValueError("metrics must be an object")
    _validate_episode(data["episode"])
    _validate_finite_tree("solver_state", data["solver_state"])
    _validate_finite_tree("metrics", data["metrics"])
    return data


def save_checkpoint(
    path: Path | str,
    solver_state: dict[str, Any],
    episode: int,
    metrics: dict[str, Any],
) -> None:
    """Save simulation state to a JSON checkpoint.

    Parameters
    ----------
    path : Path or str
        Destination file path.
    solver_state : dict
        Current state of the physics solvers (e.g. Psi, profiles).
    episode : int
        Current episode counter.
    metrics : dict
        Accumulated performance metrics.
    """
    episode = _validate_episode(episode)
    if not isinstance(solver_state, dict):
        raise ValueError("solver_state must be an object")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    _validate_finite_tree("solver_state", solver_state)
    _validate_finite_tree("metrics", metrics)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def _serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serializable(v) for v in obj]
        return obj

    data = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "episode": episode,
        "solver_state": _serializable(solver_state),
        "metrics": _serializable(metrics),
        "timestamp": float(np.datetime64("now").astype(float)),  # Simple epoch
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: Path | str) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Restore simulation state from a JSON checkpoint.

    Parameters
    ----------
    path : Path or str
        Source file path.

    Returns
    -------
    tuple
        (solver_state, episode, metrics)
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint must contain valid JSON") from exc
    data = _validate_payload(data)

    # Convert lists back to numpy arrays where appropriate
    def _restore(obj: Any) -> Any:
        if isinstance(obj, list):
            # Heuristic: convert lists of numbers to arrays
            if obj and isinstance(obj[0], (int, float)):
                return np.array(obj)
            return [_restore(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        return obj

    solver_state = _restore(data["solver_state"])
    episode = int(data["episode"])
    metrics = _restore(data["metrics"])

    return solver_state, episode, metrics
