# ──────────────────────────────────────────────────────────────────────
# SCPN Control — ITER Neural Equilibrium Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Generate pre-trained PCA+MLP weights for ITER-like equilibria.

Generates training data via two strategies:
  1. Solov'ev analytic equilibria (fast, exact GS solutions)
  2. GS solver perturbations (optional, slower but more realistic)

ITER parameters: R0=6.2m, a=2.0m, B0=5.3T, Ip=15MA, kappa=1.7.

Usage:
    python tools/train_neural_equilibrium_iter.py [--gs-only] [--n-samples N]

Reference: Solov'ev, Sov. Phys. JETP 26 (1968) 400
           ITER Physics Basis, Nucl. Fusion 39 (1999)
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_control.core.neural_equilibrium import (  # noqa: E402
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    SimpleMLP,
)

logger = logging.getLogger(__name__)

# ITER reference — ITER Physics Basis, Nucl. Fusion 39 (1999)
R0 = 6.2  # major radius [m]
A_MINOR = 2.0  # minor radius [m]
B0 = 5.3  # toroidal field [T]
IP_MA = 15.0  # plasma current [MA]
KAPPA = 1.7  # elongation
DELTA = 0.33  # triangularity
Q95 = 3.0  # safety factor at 95% flux

GRID_N = 129
WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_equilibrium_iter.npz"

R_MIN = R0 - 2.0 * A_MINOR
R_MAX = R0 + 2.0 * A_MINOR
Z_MIN = -2.0 * A_MINOR * KAPPA
Z_MAX = 2.0 * A_MINOR * KAPPA


def _solovev_psi(
    RR: np.ndarray,
    ZZ: np.ndarray,
    R0: float,
    a: float,
    kappa: float,
    A1: float,
    A2: float,
) -> np.ndarray:
    """Solov'ev analytic equilibrium: psi(R,Z).

    Exact solution to R^2 Delta*psi = -mu0 R^2 dp/dpsi - F dF/dpsi
    with linear p'(psi) and FF'(psi). Solov'ev (1968).

    psi = A1/8 * R^4 + A2 * (R^2 * Z^2 / 2)
          + c1 + c2*R^2 + c3*(R^4 - 4*R^2*Z^2)

    Coefficients c1,c2,c3 set by boundary conditions:
      psi(R0+a, 0) = 0, psi(R0-a, 0) = 0, psi(R0, kappa*a) = 0.
    """
    # Normalise
    x = RR / R0
    y = ZZ / R0
    eps = a / R0

    # Homogeneous solutions
    h1 = np.ones_like(x)
    h2 = x**2
    h3 = x**4 - 4.0 * x**2 * y**2

    # Particular solutions
    p1 = x**4 / 8.0
    p2 = x**2 * y**2 / 2.0

    # Boundary conditions: psi=0 at (1+eps,0), (1-eps,0), (1, kappa*eps)
    Rp = 1.0 + eps
    Rm = 1.0 - eps
    Rz = 1.0
    keps = kappa * eps

    # 3x3 system for c1, c2, c3
    M = np.array(
        [
            [1.0, Rp**2, Rp**4],
            [1.0, Rm**2, Rm**4],
            [1.0, Rz**2, Rz**4 - 4.0 * Rz**2 * keps**2],
        ]
    )
    rhs_1 = -np.array([A1 * Rp**4 / 8.0, A1 * Rm**4 / 8.0, A1 * Rz**4 / 8.0])
    rhs_2 = -np.array([0.0, 0.0, A2 * Rz**2 * keps**2 / 2.0])
    rhs = rhs_1 + rhs_2

    c = np.linalg.solve(M, rhs)

    psi = A1 * p1 + A2 * p2 + c[0] * h1 + c[1] * h2 + c[2] * h3
    return psi * R0**2  # restore physical units


def generate_solovev_dataset(
    n_samples: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training pairs from Solov'ev equilibria with varied parameters."""
    rng = np.random.default_rng(seed)
    R_arr = np.linspace(R_MIN, R_MAX, GRID_N)
    Z_arr = np.linspace(Z_MIN, Z_MAX, GRID_N)
    RR, ZZ = np.meshgrid(R_arr, Z_arr)

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    for _ in range(n_samples):
        ip = rng.uniform(12.0, 18.0)
        kappa_s = rng.uniform(1.5, 1.9)
        a_s = rng.uniform(1.7, 2.3)
        # Solov'ev source amplitudes scale with Ip
        A1 = rng.uniform(-0.08, -0.02) * (ip / 15.0)
        A2 = rng.uniform(-0.06, -0.01) * (ip / 15.0)
        pp_scale = rng.uniform(0.6, 1.5)
        ff_scale = rng.uniform(0.6, 1.5)

        psi = _solovev_psi(RR, ZZ, R0, a_s, kappa_s, A1, A2)

        if np.isnan(psi).any() or np.isinf(psi).any():
            continue

        psi_axis = float(psi.max())
        psi_bry = float(psi.min())
        idx_max = np.unravel_index(psi.argmax(), psi.shape)
        r_axis = float(R_arr[idx_max[1]])
        z_axis = float(Z_arr[idx_max[0]])

        feat = np.array(
            [
                ip,
                B0,
                r_axis,
                z_axis,
                pp_scale,
                ff_scale,
                psi_axis,
                psi_bry,
                kappa_s,
                DELTA,
                DELTA,
                Q95,
            ]
        )

        X_list.append(feat)
        Y_list.append(psi.ravel())

    logger.info("Solov'ev dataset: %d samples", len(X_list))
    return np.array(X_list), np.array(Y_list)


def generate_gs_dataset(
    n_samples: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training pairs from the full GS solver (slower, more realistic)."""
    from scpn_control.core.fusion_kernel import FusionKernel

    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    R_arr = np.linspace(R_MIN, R_MAX, GRID_N)
    Z_arr = np.linspace(Z_MIN, Z_MAX, GRID_N)

    t0 = time.perf_counter()

    for i in range(n_samples):
        ip = rng.uniform(12.0, 18.0)
        pp_scale = rng.uniform(0.6, 1.5)
        ff_scale = rng.uniform(0.6, 1.5)
        core_a_p = rng.uniform(0.1, 0.6)
        core_a_ff = rng.uniform(0.1, 0.6)

        cfg = {
            "reactor_name": "ITER-NeuralEq-Train",
            "grid_resolution": [GRID_N, GRID_N],
            "dimensions": {"R_min": R_MIN, "R_max": R_MAX, "Z_min": Z_MIN, "Z_max": Z_MAX},
            "physics": {
                "plasma_current_target": ip,
                "vacuum_permeability": 1.0,
                "profiles": {
                    "mode": "h-mode",
                    "p_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": pp_scale, "core_alpha": core_a_p},
                    "ff_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": ff_scale, "core_alpha": core_a_ff},
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
            tmp_path = f.name

        try:
            kernel = FusionKernel(tmp_path)
            result = kernel.solve_fixed_boundary()
            psi = result["psi"]
        except Exception as exc:
            logger.warning("GS sample %d failed: %s", i, exc)
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if np.isnan(psi).any() or np.isinf(psi).any():
            continue

        psi_axis = float(psi.max())
        psi_bry = float(psi.min())
        idx_max = np.unravel_index(psi.argmax(), psi.shape)
        r_axis = float(R_arr[idx_max[1]])
        z_axis = float(Z_arr[idx_max[0]])

        feat = np.array(
            [
                ip,
                B0,
                r_axis,
                z_axis,
                pp_scale,
                ff_scale,
                psi_axis,
                psi_bry,
                KAPPA,
                DELTA,
                DELTA,
                Q95,
            ]
        )
        X_list.append(feat)
        Y_list.append(psi.ravel())

        if (i + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  GS %d/%d done (%.1fs)", i + 1, n_samples, elapsed)

    logger.info("GS dataset: %d samples", len(X_list))
    if len(X_list) == 0:
        return np.empty((0, 12)), np.empty((0, GRID_N * GRID_N))
    return np.array(X_list), np.array(Y_list)


def train_iter_weights(
    n_solovev: int = 80,
    n_gs: int = 0,
    n_components: int = 20,
    hidden_sizes: tuple[int, ...] = (128, 64, 32),
    n_epochs: int = 500,
    seed: int = 42,
) -> None:
    """Full pipeline: generate data, train PCA+MLP, save weights."""
    t0 = time.perf_counter()

    # Solov'ev data (fast)
    X_sol, Y_sol = generate_solovev_dataset(n_samples=n_solovev, seed=seed)

    # GS solver data (slow, optional)
    if n_gs > 0:
        X_gs, Y_gs = generate_gs_dataset(n_samples=n_gs, seed=seed + 1000)
        if len(X_gs) > 0:
            X = np.concatenate([X_sol, X_gs], axis=0)
            Y = np.concatenate([Y_sol, Y_gs], axis=0)
        else:
            X, Y = X_sol, Y_sol
    else:
        X, Y = X_sol, Y_sol

    if len(X) < 5:
        raise RuntimeError(f"Only {len(X)} samples — need at least 5")

    logger.info("Total training data: %d samples", len(X))

    cfg = NeuralEqConfig(
        n_components=min(n_components, len(X) - 1),
        hidden_sizes=hidden_sizes,
        n_input_features=12,
        grid_shape=(GRID_N, GRID_N),
    )
    accel = NeuralEquilibriumAccelerator(cfg)

    accel._input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-10] = 1.0
    accel._input_std = input_std
    X_norm = (X - accel._input_mean) / accel._input_std

    Y_compressed = accel.pca.fit_transform(Y)
    explained = float(np.sum(accel.pca.explained_variance_ratio_))
    logger.info("PCA: %d components, %.2f%% variance", cfg.n_components, explained * 100)

    # Train/val split (80/20)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = max(1, int(0.2 * len(X)))
    n_train = len(X) - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, Y_train = X_norm[train_idx], Y_compressed[train_idx]
    X_val, Y_val = X_norm[val_idx], Y_compressed[val_idx]

    layer_sizes = [12, *hidden_sizes, cfg.n_components]
    accel.mlp = SimpleMLP(layer_sizes, seed=seed)

    lr = 1e-4
    momentum = 0.9
    batch_size = min(32, n_train)

    velocity = [np.zeros_like(w) for w in accel.mlp.weights]
    velocity_b = [np.zeros_like(b) for b in accel.mlp.biases]

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 30

    for epoch in range(n_epochs):
        order = rng.permutation(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            x_batch = X_train[idx]
            y_batch = Y_train[idx]

            activations = [x_batch]
            h = x_batch
            for i, (W, b) in enumerate(zip(accel.mlp.weights, accel.mlp.biases)):
                z = h @ W + b
                h = np.maximum(0, z) if i < len(accel.mlp.weights) - 1 else z
                activations.append(h)

            error = activations[-1] - y_batch
            loss = float(np.mean(error**2))
            epoch_loss += loss * len(idx)

            delta = 2.0 * error / len(idx)
            for i in range(len(accel.mlp.weights) - 1, -1, -1):
                grad_w = activations[i].T @ delta
                grad_b = delta.sum(axis=0)
                np.clip(grad_w, -1.0, 1.0, out=grad_w)
                np.clip(grad_b, -1.0, 1.0, out=grad_b)

                velocity[i] = momentum * velocity[i] - lr * grad_w
                velocity_b[i] = momentum * velocity_b[i] - lr * grad_b
                accel.mlp.weights[i] += velocity[i]
                accel.mlp.biases[i] += velocity_b[i]

                if i > 0:
                    delta = delta @ accel.mlp.weights[i].T
                    delta *= (activations[i] > 0).astype(float)

        epoch_loss /= max(n_train, 1)

        val_pred = accel.mlp.forward(X_val)
        val_loss = float(np.mean((val_pred - Y_val) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (val_loss=%.6f)", epoch, val_loss)
                break

        if epoch % 50 == 0:
            logger.info("Epoch %d: train=%.6f  val=%.6f", epoch, epoch_loss, val_loss)

    accel.is_trained = True
    accel.cfg.n_input_features = 12

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    accel.save_weights(WEIGHTS_PATH)

    psi_pred = accel.predict(X[0])
    psi_true = Y[0].reshape(GRID_N, GRID_N)
    rmse = float(np.sqrt(np.mean((psi_pred - psi_true) ** 2)))
    rel_l2 = float(np.linalg.norm(psi_pred - psi_true) / max(np.linalg.norm(psi_true), 1e-15))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1fs | PCA %.1f%% | val_loss %.6f | RMSE %.6f | rel_L2 %.6f",
        elapsed,
        explained * 100,
        best_val_loss,
        rmse,
        rel_l2,
    )
    logger.info("Weights: %s", WEIGHTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    use_gs = "--gs-only" in sys.argv
    n_samples = 80

    for arg in sys.argv[1:]:
        if arg.startswith("--n-samples="):
            n_samples = int(arg.split("=")[1])

    if use_gs:
        train_iter_weights(n_solovev=0, n_gs=n_samples)
    else:
        train_iter_weights(n_solovev=n_samples, n_gs=0)
