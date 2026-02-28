# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neural Equilibrium Training Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for train_from_geqdsk, train_on_sparc, and the __main__ CLI block."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    train_on_sparc,
)

SPARC_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "sparc"
DIIID_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "diiid"


def _geqdsk_files(directory: Path, limit: int = 3) -> list[Path]:
    files = sorted(directory.glob("*.geqdsk")) + sorted(directory.glob("*.eqdsk"))
    return files[:limit]


@pytest.fixture()
def sparc_files():
    files = _geqdsk_files(SPARC_DIR, limit=2)
    if not files:
        pytest.skip("No SPARC GEQDSK files found")
    return files


@pytest.fixture()
def diiid_files():
    files = _geqdsk_files(DIIID_DIR, limit=2)
    if not files:
        pytest.skip("No DIII-D GEQDSK files found")
    return files


class TestTrainFromGeqdsk:
    def test_trains_and_returns_result(self, sparc_files):
        accel = NeuralEquilibriumAccelerator(NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
        ))
        result = accel.train_from_geqdsk(sparc_files, n_perturbations=3, seed=0)
        assert result.n_samples > 0
        assert result.n_components == 5
        assert 0.0 < result.explained_variance <= 1.0
        assert result.final_loss >= 0.0
        assert result.train_time_s > 0.0
        assert np.isfinite(result.val_loss)
        assert np.isfinite(result.test_mse)

    def test_predict_after_geqdsk_training(self, sparc_files):
        accel = NeuralEquilibriumAccelerator(NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
        ))
        accel.train_from_geqdsk(sparc_files, n_perturbations=3, seed=0)
        assert accel.is_trained

        features = np.zeros(12)
        features[0] = 1.0  # I_p placeholder
        features[1] = 5.0  # B_t placeholder
        features[4] = 1.0  # pprime scale
        features[5] = 1.0  # ffprime scale
        psi = accel.predict(features)
        assert psi.ndim == 2
        assert psi.shape[0] > 0 and psi.shape[1] > 0

    def test_save_load_after_geqdsk(self, sparc_files, tmp_path):
        accel = NeuralEquilibriumAccelerator(NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
        ))
        accel.train_from_geqdsk(sparc_files, n_perturbations=3, seed=42)

        path = tmp_path / "geqdsk_weights.npz"
        accel.save_weights(path)
        assert path.exists()

        accel2 = NeuralEquilibriumAccelerator()
        accel2.load_weights(path)
        assert accel2.is_trained

        features = np.ones(12)
        psi1 = accel.predict(features)
        psi2 = accel2.predict(features)
        np.testing.assert_allclose(psi2, psi1, atol=1e-10)

    def test_diiid_files_different_grid(self, diiid_files):
        """DIII-D files may have different grid sizes — tests interpolation path."""
        accel = NeuralEquilibriumAccelerator(NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
        ))
        result = accel.train_from_geqdsk(diiid_files, n_perturbations=2, seed=7)
        assert result.n_samples > 0
        assert accel.is_trained

    def test_evaluate_after_geqdsk(self, sparc_files):
        accel = NeuralEquilibriumAccelerator(NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
        ))
        accel.train_from_geqdsk(sparc_files, n_perturbations=3, seed=0)

        from scpn_control.core.eqdsk import read_geqdsk
        eq = read_geqdsk(sparc_files[0])
        X = np.ones((2, 12))
        Y = np.tile(eq.psirz.ravel(), (2, 1))
        metrics = accel.evaluate_surrogate(X, Y)
        assert "mse" in metrics
        assert "max_error" in metrics
        assert "gs_residual" in metrics


class TestTrainOnSparc:
    def test_convenience_function(self, tmp_path):
        files = _geqdsk_files(SPARC_DIR, limit=2)
        if not files:
            pytest.skip("No SPARC GEQDSK files found")

        result = train_on_sparc(
            sparc_dir=SPARC_DIR,
            save_path=tmp_path / "sparc_weights.npz",
            n_perturbations=2,
            seed=0,
        )
        assert result.n_samples > 0
        assert (tmp_path / "sparc_weights.npz").exists()
        assert result.weights_path == str(tmp_path / "sparc_weights.npz")

    def test_missing_dir_raises(self, tmp_path):
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No GEQDSK"):
            train_on_sparc(sparc_dir=empty)

    def test_default_sparc_dir(self):
        """train_on_sparc with sparc_dir=None uses the repo default."""
        files = _geqdsk_files(SPARC_DIR, limit=1)
        if not files:
            pytest.skip("No SPARC GEQDSK files found")
        # Just verify the path resolution — don't run full training
        from scpn_control.core.neural_equilibrium import REPO_ROOT
        default_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
        assert default_dir.exists()
