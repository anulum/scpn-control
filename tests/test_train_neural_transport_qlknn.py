# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Legacy QLKNN trainer input and no-learning CLI contracts.
"""Exercise real dataset preparation and CLI refusals without model training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tools.train_neural_transport_qlknn as training


def test_default_proxy_generator_is_reproducible_and_preserves_global_rng() -> None:
    """Replay the real default 5,000-row proxy without changing global RNG state."""
    before = np.random.get_state()
    features, targets = training.generate_synthetic_qlknn_data()
    replay_features, replay_targets = training.generate_synthetic_qlknn_data()
    after = np.random.get_state()
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert features.shape == (5000, 10) and targets.shape == (5000, 3)
    assert features.dtype == targets.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(features, replay_features)
    np.testing.assert_array_equal(targets, replay_targets)
    assert np.all(np.isfinite(features)) and np.all(np.isfinite(targets))
    assert np.all(targets >= 0)
    np.testing.assert_allclose(features[:, 9], 4.03e-3 * features[:, 3] * features[:, 1], rtol=0, atol=0)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_proxy_generator_returns_empty_shapes_without_training() -> None:
    """Preserve zero-row preparation behavior without calling the learner."""
    features, targets = training.generate_synthetic_qlknn_data(n_samples=0)
    assert features.shape == (0, 10) and targets.shape == (0, 3)


@pytest.mark.parametrize(
    ("input_key", "output_key"), [("inputs", "outputs"), ("inputs", "Y"), ("X", "outputs"), ("X", "Y")]
)
def test_npz_loader_preserves_stored_arrays_and_precedes_csv(input_key: str, output_key: str, tmp_path: Path) -> None:
    """Read actual archive keys without converting dtype or falling through to CSV."""
    features = np.arange(30, dtype=np.int64).reshape(3, 10)
    targets = np.arange(9, dtype=np.float32).reshape(3, 3)
    archive_arrays: dict[str, training.Array] = {input_key: features, output_key: targets}
    np.savez(tmp_path / "data.npz", **archive_arrays)
    (tmp_path / "invalid.csv").write_text("header\nnot,numeric\n")
    loaded_features, loaded_targets = training.load_qlknn_data(tmp_path)
    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_targets, targets)
    assert loaded_features.dtype == features.dtype and loaded_targets.dtype == targets.dtype


def test_csv_loader_slices_columns_after_skipping_one_header(tmp_path: Path) -> None:
    """Read actual CSV rows, retain10/3columns and ignore additional fields."""
    rows = np.arange(42, dtype=np.float64).reshape(3, 14)
    path = tmp_path / "data.csv"
    path.write_text("labels\n" + "\n".join(",".join(str(value) for value in row) for row in rows) + "\n")
    features, targets = training.load_qlknn_data(tmp_path)
    np.testing.assert_array_equal(features, rows[:, :10])
    np.testing.assert_array_equal(targets, rows[:, 10:13])


def test_npz_primary_keys_take_precedence_over_fallbacks(tmp_path: Path) -> None:
    """Prefer both explicit input/output arrays over actual alternative keys."""
    features = np.ones((2, 10))
    targets = np.ones((2, 3))
    np.savez(tmp_path / "data.npz", inputs=features, outputs=targets, X=features * 2, Y=targets * 3)
    loaded_features, loaded_targets = training.load_qlknn_data(tmp_path)
    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_targets, targets)


def test_loader_refuses_missing_dataset_without_training(tmp_path: Path) -> None:
    """Propagate the actual absent-file refusal before the learner is reached."""
    with pytest.raises(FileNotFoundError, match="No .npz or .csv files"):
        training.load_qlknn_data(tmp_path)


def test_loader_refuses_archive_without_feature_keys(tmp_path: Path) -> None:
    """Keep NumPy's missing-key refusal for an actual invalid archive."""
    np.savez(tmp_path / "data.npz", unrelated=np.ones(2))
    with pytest.raises(KeyError):
        training.load_qlknn_data(tmp_path)


def test_loader_refuses_object_arrays_without_pickle_loading(tmp_path: Path) -> None:
    """Reject a real object-dtype archive under NumPy's default secure reader."""
    np.savez(tmp_path / "data.npz", X=np.array([{"value": 1}], dtype=object), Y=np.ones((1, 3)))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        training.load_qlknn_data(tmp_path)


@pytest.mark.parametrize("mode", ["help", "no_input", "missing_data"])
def test_trainer_cli_help_and_missing_input_create_no_weights(mode: str, tmp_path: Path) -> None:
    """Exercise actual CLI help/refusal paths from another cwd without learning."""
    output = tmp_path / "uncreated" / "model.npz"
    argv = [sys.executable, str(Path(training.__file__).resolve()), "--output", str(output)]
    if mode == "help":
        argv.append("--help")
    elif mode == "missing_data":
        argv.extend(["--data-dir", str(tmp_path / "absent-data")])
    result = subprocess.run(argv, cwd=tmp_path, text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == (0 if mode == "help" else 1)
    if mode == "help":
        assert "--synthetic" in result.stdout and "--data-dir" in result.stdout
    elif mode == "no_input":
        assert "Specify --data-dir or --synthetic" in result.stdout
    else:
        assert "FileNotFoundError" in result.stderr
    assert not output.parent.exists()
    assert not output.exists() and not output.with_suffix(".metrics.json").exists()
