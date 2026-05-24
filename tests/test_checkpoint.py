# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Checkpoint tests
"""
Tests for the save/resume checkpoint API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    """Verify that state is preserved exactly across save/load."""
    path = tmp_path / "test_ckpt.json"

    solver_state = {"Psi": np.array([[1.0, 2.0], [3.0, 4.0]]), "nested": {"arr": np.array([0.1, 0.2])}}
    episode = 42
    metrics = {"accuracy": 0.99, "history": [0.1, 0.5, 0.8]}

    save_checkpoint(path, solver_state, episode, metrics)

    s_restored, e_restored, m_restored = load_checkpoint(path)

    assert e_restored == episode
    assert m_restored["accuracy"] == metrics["accuracy"]
    np.testing.assert_allclose(s_restored["Psi"], solver_state["Psi"])
    np.testing.assert_allclose(s_restored["nested"]["arr"], solver_state["nested"]["arr"])


def test_checkpoint_resume_simulation(tmp_path: Path):
    """Verify that a simulation can be resumed from a checkpoint."""
    path = tmp_path / "resume_ckpt.json"

    # 1. Run "Phase 1" of a campaign
    initial_metrics = {"total_reward": 100.0}
    state_at_ep_10 = {"last_val": 5.0}

    save_checkpoint(path, state_at_ep_10, 10, initial_metrics)

    # 2. Resume "Phase 2"
    state, ep, metrics = load_checkpoint(path)
    assert ep == 10

    # Simulate further work
    ep += 1
    metrics["total_reward"] += 50.0

    assert ep == 11
    assert metrics["total_reward"] == 150.0


def test_checkpoint_rejects_nonfinite_state_and_metrics(tmp_path: Path):
    """Checkpoint writer rejects NaN/Inf values before persistence."""
    path = tmp_path / "bad_ckpt.json"

    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(path, {"Psi": np.array([1.0, np.nan])}, 1, {"loss": 0.1})

    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(path, {"Psi": np.array([1.0, 2.0])}, 1, {"loss": float("inf")})


def test_checkpoint_rejects_invalid_episode_and_schema(tmp_path: Path):
    """Checkpoint writer and reader enforce episode and schema-version boundaries."""
    path = tmp_path / "bad_schema.json"

    with pytest.raises(ValueError, match="episode"):
        save_checkpoint(path, {"Psi": np.array([1.0])}, -1, {"loss": 0.1})

    path.write_text('{"episode": 1, "solver_state": {}, "metrics": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_checkpoint(path)


def test_checkpoint_rejects_corrupt_or_malformed_payload(tmp_path: Path):
    """Checkpoint reader fails closed on corrupt JSON and malformed required sections."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_checkpoint(corrupt)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version": 1, "episode": 1, "solver_state": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_checkpoint(malformed)
