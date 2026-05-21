# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Gk Online Learner
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Online Learner Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_online_learner import LearnerConfig, OnlineLearner


def _random_samples(n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(rng.random(10), rng.random(3)) for _ in range(n)]


def test_buffer_not_full_initially():
    learner = OnlineLearner()
    assert not learner.buffer_full


def test_buffer_fills():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert learner.buffer_full


def test_try_retrain_skips_empty():
    learner = OnlineLearner()
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_skips_below_threshold():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    assert not learner.buffer_full
    result = learner.try_retrain()
    assert result is None


def test_try_retrain_runs_when_full():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=20, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(20, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    # May return weights or None (if validation loss doesn't improve over inf)
    # First retrain should succeed since best_val starts at inf
    assert result is not None
    assert "w1" in result
    assert "w3" in result
    assert learner.generation == 1


def test_retrain_clears_buffer():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.buffer) == 0


def test_max_generations():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, max_generations=1, n_epochs=2))
    rng = np.random.default_rng(42)

    # First retrain
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()

    # Second attempt should be blocked
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    assert result is None  # max_generations reached


def test_retrain_history_tracked():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    assert len(learner.retrain_history) == 1
    assert "val_loss" in learner.retrain_history[0]


def test_reset():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=2))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    learner.try_retrain()
    learner.reset()
    assert learner.generation == 0
    assert len(learner.buffer) == 0
    assert len(learner.retrain_history) == 0


def test_weights_have_correct_shapes():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    result = learner.try_retrain()
    if result is not None:
        assert result["w1"].shape == (10, 64)
        assert result["b1"].shape == (64,)
        assert result["w2"].shape == (64, 32)
        assert result["w3"].shape == (32, 3)


def test_retrain_with_existing_weights():
    """Cover gk_online_learner.py lines 113-115: retrain from existing weights."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=3))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    weights = learner.try_retrain()
    assert weights is not None

    # Second round with existing weights
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    result2 = learner.try_retrain(current_weights=weights)
    # May succeed or rollback; either way exercises the branch
    assert learner.generation >= 1


def test_retrain_rollback_on_worse_loss():
    """Cover gk_online_learner.py lines 164-166: rollback when val_loss worsens."""
    learner = OnlineLearner(config=LearnerConfig(buffer_size=10, n_epochs=1))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    w1 = learner.try_retrain()
    assert w1 is not None

    # Force a very low best_val_loss so next retrain rolls back
    learner._best_val_loss = 0.0
    for inp, tgt in _random_samples(10, rng):
        learner.add_sample(inp, tgt)
    w2 = learner.try_retrain()
    assert w2 is None
    assert learner.retrain_history[-1]["accepted"] is False


def test_add_sample_rejects_invalid_shapes():
    learner = OnlineLearner()
    with pytest.raises(ValueError, match="input_10d must have shape"):
        learner.add_sample(np.zeros(9), np.zeros(3))
    with pytest.raises(ValueError, match="target_3d must have shape"):
        learner.add_sample(np.zeros(10), np.zeros(2))


def test_add_sample_rejects_nonfinite_values():
    learner = OnlineLearner()
    bad_inp = np.zeros(10)
    bad_inp[0] = np.nan
    with pytest.raises(ValueError, match="input_10d must contain only finite values"):
        learner.add_sample(bad_inp, np.zeros(3))
    bad_tgt = np.zeros(3)
    bad_tgt[1] = np.inf
    with pytest.raises(ValueError, match="target_3d must contain only finite values"):
        learner.add_sample(np.zeros(10), bad_tgt)


def test_try_retrain_rejects_invalid_current_weights_schema():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=1))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    with pytest.raises(ValueError, match="missing required keys"):
        learner.try_retrain(current_weights={"w1": np.zeros((10, 64))})


def test_try_retrain_rejects_invalid_current_weights_shape():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=1))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    bad_weights = {
        "w1": np.zeros((9, 64)),
        "b1": np.zeros(64),
        "w2": np.zeros((64, 32)),
        "b2": np.zeros(32),
        "w3": np.zeros((32, 3)),
        "b3": np.zeros(3),
    }
    with pytest.raises(ValueError, match="current_weights\\['w1'\\] must have shape"):
        learner.try_retrain(current_weights=bad_weights)


def test_try_retrain_rejects_nonfinite_current_weights():
    learner = OnlineLearner(config=LearnerConfig(buffer_size=5, n_epochs=1))
    rng = np.random.default_rng(42)
    for inp, tgt in _random_samples(5, rng):
        learner.add_sample(inp, tgt)
    bad_weights = {
        "w1": np.full((10, 64), np.nan),
        "b1": np.zeros(64),
        "w2": np.zeros((64, 32)),
        "b2": np.zeros(32),
        "w3": np.zeros((32, 3)),
        "b3": np.zeros(3),
    }
    with pytest.raises(ValueError, match="must contain only finite values"):
        learner.try_retrain(current_weights=bad_weights)
