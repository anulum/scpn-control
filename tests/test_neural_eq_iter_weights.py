# ──────────────────────────────────────────────────────────────────────
# SCPN Control — ITER Neural Equilibrium Weights Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Validate ITER pre-trained neural equilibrium weights.

Loads ``weights/neural_equilibrium_iter.npz``, runs inference at
ITER-like parameters, and checks against Solov'ev analytic ground
truth and GS solver output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.fusion_kernel import FusionKernel
from scpn_control.core.neural_equilibrium import NeuralEquilibriumAccelerator

REPO_ROOT = Path(__file__).resolve().parents[1]
ITER_WEIGHTS = REPO_ROOT / "weights" / "neural_equilibrium_iter.npz"
sys.path.insert(0, str(REPO_ROOT / "tools"))

# ITER reference — Nucl. Fusion 39 (1999)
R0 = 6.2
A_MINOR = 2.0
B0 = 5.3
IP_MA = 15.0
KAPPA = 1.7
DELTA = 0.33
Q95 = 3.0
GRID_N = 129
R_MIN = R0 - 2 * A_MINOR
R_MAX = R0 + 2 * A_MINOR
Z_MIN = -2 * A_MINOR * KAPPA
Z_MAX = 2 * A_MINOR * KAPPA

pytestmark = pytest.mark.skipif(
    not ITER_WEIGHTS.exists(),
    reason="ITER weights not generated yet",
)


def _solovev_reference() -> tuple[np.ndarray, np.ndarray]:
    """Generate a Solov'ev equilibrium at nominal ITER params and its features."""
    from train_neural_equilibrium_iter import _solovev_psi

    R_arr = np.linspace(R_MIN, R_MAX, GRID_N)
    Z_arr = np.linspace(Z_MIN, Z_MAX, GRID_N)
    RR, ZZ = np.meshgrid(R_arr, Z_arr)
    # Nominal Solov'ev coefficients (mid-range of training distribution)
    psi = _solovev_psi(RR, ZZ, R0, A_MINOR, KAPPA, A1=-0.05, A2=-0.035)
    idx = np.unravel_index(psi.argmax(), psi.shape)
    feat = np.array(
        [
            IP_MA,
            B0,
            float(R_arr[idx[1]]),
            float(Z_arr[idx[0]]),
            1.0,
            1.0,
            float(psi.max()),
            float(psi.min()),
            KAPPA,
            DELTA,
            DELTA,
            Q95,
        ]
    )
    return psi, feat


def _gs_reference() -> np.ndarray:
    """Solve GS at nominal ITER parameters."""
    cfg = {
        "reactor_name": "ITER-Test",
        "grid_resolution": [GRID_N, GRID_N],
        "dimensions": {"R_min": R_MIN, "R_max": R_MAX, "Z_min": Z_MIN, "Z_max": Z_MAX},
        "physics": {
            "plasma_current_target": IP_MA,
            "vacuum_permeability": 1.0,
            "profiles": {
                "mode": "h-mode",
                "p_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3},
                "ff_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3},
            },
        },
        "coils": [
            {"name": "PF1", "r": R0 - 3.2, "z": 7.6, "current": 8.0},
            {"name": "PF2", "r": R0 + 2.0, "z": 6.7, "current": -1.32},
            {"name": "PF3", "r": R0 + 5.8, "z": 2.7, "current": -3.02},
            {"name": "PF4", "r": R0 + 6.4, "z": -2.3, "current": -2.98},
            {"name": "PF5", "r": R0 + 2.2, "z": -6.7, "current": -1.37},
            {"name": "PF6", "r": R0 - 1.9, "z": -7.6, "current": -0.39},
            {"name": "CS", "r": R0 - 4.5, "z": 0.0, "current": 0.15},
        ],
        "solver": {
            "solver_method": "sor",
            "max_iterations": 80,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.15,
            "sor_omega": 1.7,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        f.flush()
        tmp = f.name
    try:
        kernel = FusionKernel(tmp)
        return kernel.solve_fixed_boundary()["psi"]
    finally:
        Path(tmp).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def accel() -> NeuralEquilibriumAccelerator:
    a = NeuralEquilibriumAccelerator()
    a.load_weights(ITER_WEIGHTS)
    return a


@pytest.fixture(scope="module")
def solovev_ref() -> tuple[np.ndarray, np.ndarray]:
    return _solovev_reference()


@pytest.fixture(scope="module")
def gs_psi() -> np.ndarray:
    return _gs_reference()


class TestIterWeightsLoad:
    def test_weights_file_exists(self):
        assert ITER_WEIGHTS.exists()
        assert ITER_WEIGHTS.stat().st_size > 0

    def test_load_succeeds(self, accel):
        assert accel.is_trained
        assert accel.mlp is not None

    def test_grid_shape_matches(self, accel):
        assert accel.cfg.grid_shape == (GRID_N, GRID_N)

    def test_input_features_12(self, accel):
        assert accel.cfg.n_input_features == 12

    def test_pca_components(self, accel):
        assert accel.cfg.n_components == 20
        assert accel.pca.components_ is not None
        assert accel.pca.components_.shape == (20, GRID_N * GRID_N)

    def test_weight_shapes_match_sparc(self, accel):
        """ITER and SPARC weights use identical architecture."""
        sparc_path = REPO_ROOT / "weights" / "neural_equilibrium_sparc.npz"
        if not sparc_path.exists():
            pytest.skip("SPARC weights not available")
        sparc = NeuralEquilibriumAccelerator()
        sparc.load_weights(sparc_path)
        assert accel.cfg.n_input_features == sparc.cfg.n_input_features
        assert accel.cfg.grid_shape == sparc.cfg.grid_shape
        assert len(accel.mlp.weights) == len(sparc.mlp.weights)
        for i, (w_iter, w_sparc) in enumerate(zip(accel.mlp.weights, sparc.mlp.weights)):
            assert w_iter.shape == w_sparc.shape, f"Layer {i} shape mismatch"


class TestIterInference:
    def test_predict_shape(self, accel):
        feat = np.array([IP_MA, B0, R0, 0.0, 1.0, 1.0, 1.0, 0.0, KAPPA, DELTA, DELTA, Q95])
        psi = accel.predict(feat)
        assert psi.shape == (GRID_N, GRID_N)

    def test_predict_batch(self, accel):
        feat = np.tile(
            [IP_MA, B0, R0, 0.0, 1.0, 1.0, 1.0, 0.0, KAPPA, DELTA, DELTA, Q95],
            (3, 1),
        )
        psi = accel.predict(feat)
        assert psi.shape == (3, GRID_N, GRID_N)

    def test_no_nan_inf(self, accel):
        feat = np.array([IP_MA, B0, R0, 0.0, 1.0, 1.0, 1.0, 0.0, KAPPA, DELTA, DELTA, Q95])
        psi = accel.predict(feat)
        assert np.all(np.isfinite(psi))

    def test_rmse_vs_solovev(self, accel, solovev_ref):
        """Surrogate vs Solov'ev ground truth: relative L2 < 0.15."""
        psi_true, feat = solovev_ref
        psi_pred = accel.predict(feat)
        ref_norm = float(np.linalg.norm(psi_true))
        rel_l2 = float(np.linalg.norm(psi_pred - psi_true)) / max(ref_norm, 1e-10)
        assert rel_l2 < 0.15, f"Relative L2 {rel_l2:.4f} exceeds 0.15"

    def test_nrmse_vs_gs_solver(self, accel, gs_psi):
        """Normalised RMSE vs GS solver must be below 0.5 (cross-domain)."""
        R_arr = np.linspace(R_MIN, R_MAX, GRID_N)
        Z_arr = np.linspace(Z_MIN, Z_MAX, GRID_N)
        idx = np.unravel_index(gs_psi.argmax(), gs_psi.shape)
        feat = np.array(
            [
                IP_MA,
                B0,
                float(R_arr[idx[1]]),
                float(Z_arr[idx[0]]),
                1.0,
                1.0,
                float(gs_psi.max()),
                float(gs_psi.min()),
                KAPPA,
                DELTA,
                DELTA,
                Q95,
            ]
        )
        psi_pred = accel.predict(feat)
        psi_range = float(gs_psi.max() - gs_psi.min())
        rmse = float(np.sqrt(np.mean((psi_pred - gs_psi) ** 2)))
        nrmse = rmse / max(psi_range, 1e-10)
        assert nrmse < 0.5, f"NRMSE {nrmse:.4f} exceeds 0.5"

    def test_benchmark_under_10ms(self, accel):
        feat = np.array([IP_MA, B0, R0, 0.0, 1.0, 1.0, 1.0, 0.0, KAPPA, DELTA, DELTA, Q95])
        stats = accel.benchmark(feat, n_runs=50)
        assert stats["median_ms"] < 10.0, f"Median {stats['median_ms']:.1f}ms exceeds 10ms"

    def test_psi_has_interior_maximum(self, accel, solovev_ref):
        """Predicted psi should have a maximum inside the grid (magnetic axis)."""
        _, feat = solovev_ref
        psi = accel.predict(feat)
        idx = np.unravel_index(psi.argmax(), psi.shape)
        # Axis should not be on the boundary
        assert 5 < idx[0] < GRID_N - 5
        assert 5 < idx[1] < GRID_N - 5
