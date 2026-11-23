# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ITER-like PCA+MLP neural equilibrium training.
"""Train the PCA+MLP neural-equilibrium weights for an ITER-like 129 x 129 grid.

Training data come from two sources:

1. Analytic Solov'ev equilibria (the default). Both source terms of
   :func:`_solovev_psi` satisfy ``Delta* psi = x^2`` in normalised
   coordinates, so every sample is a pressure-driven Solov'ev equilibrium
   with ``FF' = 0``. Their difference is the homogeneous solution ``h3 / 8``,
   so only ``A1 + A2`` matters. The family therefore varies one source
   amplitude, the minor radius and the elongation.
2. Fixed-boundary Grad-Shafranov solves of the repository kernel
   (``--gs-only``), slower and with H-mode profiles. Samples whose flux
   maximum is not an interior O-point are skipped.

Every sample carries the accelerator's 12-feature vector:
``(Ip, B0, R_axis, Z_axis, pp_scale, ff_scale, psi_max, psi_min, kappa,
delta, delta, q95)``. For Solov'ev samples, ``pp_scale`` and ``ff_scale``
are random labels that do not enter ``psi``. ``B0``, ``delta`` and ``q95``
are constants, ``psi_max`` and ``psi_min`` are the extrema over the whole
box, and the axis is the grid location of ``psi_max``.

The MLP is trained on the PCA coefficients with mini-batch momentum
descent. Gradients come from exact backpropagation through the pre-update
weights, then each element is clipped to [-1, 1]. Early stopping restores
the weights of the best validation epoch. Accuracy is reported on the
held-out validation samples.

Usage::

    python tools/train_neural_equilibrium_iter.py [--gs-only] [--n-samples N] [--output PATH]

References: Solov'ev, Sov. Phys. JETP 26 (1968) 400; ITER Physics Basis,
Nucl. Fusion 39 (1999).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scpn_control.core.magnetic_axis import locate_magnetic_axis  # noqa: E402
from scpn_control.core.neural_equilibrium import (  # noqa: E402
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    SimpleMLP,
)

FloatArray = NDArray[np.float64]

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
N_FEATURES = 12
WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_equilibrium_iter.npz"

R_MIN = R0 - 2.0 * A_MINOR
R_MAX = R0 + 2.0 * A_MINOR
Z_MIN = -2.0 * A_MINOR * KAPPA
Z_MAX = 2.0 * A_MINOR * KAPPA


def _grid() -> tuple[FloatArray, FloatArray]:
    """Return the R and Z coordinates of the training grid."""
    return np.linspace(R_MIN, R_MAX, GRID_N, dtype=np.float64), np.linspace(Z_MIN, Z_MAX, GRID_N, dtype=np.float64)


def _solovev_psi(
    RR: FloatArray,
    ZZ: FloatArray,
    R0: float,
    a: float,
    kappa: float,
    A1: float,
    A2: float,
) -> FloatArray:
    """Return a Solov'ev flux map on the given grid.

    In normalised coordinates ``x = R / R0`` and ``y = Z / R0``,

        psi / R0^2 = A1 x^4 / 8 + A2 x^2 y^2 / 2 + c1 + c2 x^2 + c3 (x^4 - 4 x^2 y^2),

    and ``c1``, ``c2``, ``c3`` make ``psi`` vanish at ``(1 + eps, 0)``,
    ``(1 - eps, 0)`` and ``(1, kappa eps)`` with ``eps = a / R0``. Both source
    terms satisfy ``Delta* = x^2``, so ``Delta* psi = (A1 + A2) x^2 R0^2``
    exactly: a pressure-driven equilibrium with ``FF' = 0``. Only ``A1 + A2``
    changes the map, because the difference of the two terms is the
    homogeneous ``(x^4 - 4 x^2 y^2) / 8``.

    Parameters
    ----------
    RR, ZZ : FloatArray
        Meshgrid coordinates [m].
    R0, a, kappa : float
        Major radius [m], minor radius [m] and elongation of the three zero points.
    A1, A2 : float
        Source amplitudes; negative values give a flux maximum inside the zero points.

    Returns
    -------
    FloatArray
        The flux map, shaped like ``RR``.
    """
    x = RR / R0
    y = ZZ / R0
    eps = a / R0

    h1 = np.ones_like(x)
    h2 = x**2
    h3 = x**4 - 4.0 * x**2 * y**2
    p1 = x**4 / 8.0
    p2 = x**2 * y**2 / 2.0

    Rp = 1.0 + eps
    Rm = 1.0 - eps
    keps = kappa * eps
    M = np.array(
        [
            [1.0, Rp**2, Rp**4],
            [1.0, Rm**2, Rm**4],
            [1.0, 1.0, 1.0 - 4.0 * keps**2],
        ]
    )
    rhs = -np.array([A1 * Rp**4 / 8.0, A1 * Rm**4 / 8.0, A1 / 8.0 + A2 * keps**2 / 2.0])
    c = np.linalg.solve(M, rhs)

    psi: FloatArray = (A1 * p1 + A2 * p2 + c[0] * h1 + c[1] * h2 + c[2] * h3) * R0**2
    return psi


def _features(ip: float, psi: FloatArray, pp_scale: float, ff_scale: float, kappa: float) -> FloatArray:
    """Return the 12-feature vector of one flux map; the axis is the grid location of its maximum."""
    r_arr, z_arr = _grid()
    iz, ir = np.unravel_index(int(np.argmax(psi)), psi.shape)
    return np.array(
        [
            ip,
            B0,
            float(r_arr[ir]),
            float(z_arr[iz]),
            pp_scale,
            ff_scale,
            float(psi.max()),
            float(psi.min()),
            kappa,
            DELTA,
            DELTA,
            Q95,
        ]
    )


def _stack(features: list[FloatArray], targets: list[FloatArray]) -> tuple[FloatArray, FloatArray]:
    """Stack samples into ``(n, 12)`` features and ``(n, GRID_N**2)`` targets, also when empty."""
    if not features:
        return np.empty((0, N_FEATURES)), np.empty((0, GRID_N * GRID_N))
    return np.array(features), np.array(targets)


def generate_solovev_dataset(n_samples: int = 80, seed: int = 42) -> tuple[FloatArray, FloatArray]:
    """Return features and flattened flux maps of ``n_samples`` random Solov'ev equilibria.

    Parameters
    ----------
    n_samples : int
        Number of draws. Draws with a non-finite map are dropped.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Features ``(n, 12)`` and targets ``(n, GRID_N**2)``. Ip is drawn from
        [12, 18] MA, kappa from [1.5, 1.9] and a from [1.7, 2.3] m. A1 and A2 are
        drawn from [-0.08, -0.02] and [-0.06, -0.01], both scaled by Ip / 15.
        pp_scale and ff_scale come from [0.6, 1.5]; they are labels only.
    """
    rng = np.random.default_rng(seed)
    r_arr, z_arr = _grid()
    RR, ZZ = np.meshgrid(r_arr, z_arr)
    features: list[FloatArray] = []
    targets: list[FloatArray] = []
    for _ in range(n_samples):
        ip = rng.uniform(12.0, 18.0)
        kappa_s = rng.uniform(1.5, 1.9)
        a_s = rng.uniform(1.7, 2.3)
        A1 = rng.uniform(-0.08, -0.02) * (ip / 15.0)
        A2 = rng.uniform(-0.06, -0.01) * (ip / 15.0)
        pp_scale = rng.uniform(0.6, 1.5)
        ff_scale = rng.uniform(0.6, 1.5)
        psi = _solovev_psi(RR, ZZ, R0, a_s, kappa_s, A1, A2)
        if not np.isfinite(psi).all():
            continue
        features.append(_features(ip, psi, pp_scale, ff_scale, kappa_s))
        targets.append(psi.ravel())
    logger.info("Solov'ev dataset: %d samples", len(features))
    return _stack(features, targets)


def _gs_config(ip: float, pp_scale: float, ff_scale: float, core_a_p: float, core_a_ff: float) -> dict[str, object]:
    """Return the fixed-boundary kernel configuration of one ITER-like GS sample."""
    return {
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


def generate_gs_dataset(n_samples: int = 20, seed: int = 42) -> tuple[FloatArray, FloatArray]:
    """Return features and flattened flux maps of fixed-boundary kernel solves.

    Parameters
    ----------
    n_samples : int
        Number of draws. A draw is skipped, with a warning, when the kernel
        raises, the map is not finite, or its maximum is not an interior O-point.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Features ``(n, 12)`` and targets ``(n, GRID_N**2)``, possibly empty.
        Elongation is the constant :data:`KAPPA` for these samples.
    """
    from scpn_control.core.fusion_kernel import FusionKernel

    rng = np.random.default_rng(seed)
    r_arr, z_arr = _grid()
    features: list[FloatArray] = []
    targets: list[FloatArray] = []
    t0 = time.perf_counter()
    for i in range(n_samples):
        ip = rng.uniform(12.0, 18.0)
        pp_scale = rng.uniform(0.6, 1.5)
        ff_scale = rng.uniform(0.6, 1.5)
        cfg = _gs_config(ip, pp_scale, ff_scale, rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(cfg, handle)
            tmp_path = handle.name
        try:
            psi = np.asarray(FusionKernel(tmp_path).solve_fixed_boundary()["psi"], dtype=np.float64)
        except Exception as exc:  # A failed draw is skipped; the dataset records only solved maps.
            logger.warning("GS sample %d failed: %s", i, exc)
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        if not np.isfinite(psi).all() or not locate_magnetic_axis(psi, r_arr, z_arr).interior:
            logger.warning("GS sample %d skipped: no finite map with an interior O-point", i)
            continue
        features.append(_features(ip, psi, pp_scale, ff_scale, KAPPA))
        targets.append(psi.ravel())
        if (i + 1) % 5 == 0:
            logger.info("  GS %d/%d done (%.1fs)", i + 1, n_samples, time.perf_counter() - t0)
    logger.info("GS dataset: %d samples", len(features))
    return _stack(features, targets)


def _gradients(
    weights: list[FloatArray], biases: list[FloatArray], x: FloatArray, y: FloatArray
) -> tuple[float, list[FloatArray], list[FloatArray]]:
    """Return the batch objective and its exact gradients for every layer.

    The objective is the batch mean of the squared error summed over the PCA
    coefficients. Hidden layers use ReLU (derivative 0 at 0) and the output is
    linear. All gradients are taken through the same, unmodified weights.
    """
    activations = [x]
    h = x
    for i, (W, b) in enumerate(zip(weights, biases, strict=True)):
        z = h @ W + b
        h = np.maximum(0.0, z) if i < len(weights) - 1 else z
        activations.append(h)
    error = activations[-1] - y
    loss = float(np.mean(np.sum(error**2, axis=1)))
    delta = 2.0 * error / len(x)
    grad_w: list[FloatArray] = [np.empty(0)] * len(weights)
    grad_b: list[FloatArray] = [np.empty(0)] * len(weights)
    for i in range(len(weights) - 1, -1, -1):
        grad_w[i] = activations[i].T @ delta
        grad_b[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ weights[i].T) * (activations[i] > 0.0)
    return loss, grad_w, grad_b


def train_iter_weights(
    n_solovev: int = 80,
    n_gs: int = 0,
    n_components: int = 20,
    hidden_sizes: tuple[int, ...] = (128, 64, 32),
    n_epochs: int = 500,
    seed: int = 42,
    output: Path = WEIGHTS_PATH,
) -> dict[str, float | int | str]:
    """Generate the data, train the PCA+MLP surrogate and save its weights.

    Parameters
    ----------
    n_solovev, n_gs : int
        Numbers of Solov'ev and GS draws; GS draws use ``seed + 1000``.
    n_components : int
        PCA components, capped at one less than the sample count.
    hidden_sizes : tuple[int, ...]
        Hidden layer widths of the MLP.
    n_epochs : int
        Epoch limit; training stops after 30 epochs without a validation gain.
    seed : int
        Seed of the data, the split, the batches and the initial weights.
    output : pathlib.Path
        Weight file to write; its parent is created.

    Returns
    -------
    dict[str, float | int | str]
        Sample counts, PCA explained variance, the best validation objective in
        PCA space (as in :func:`_gradients`), its epoch, the mean relative L2 error and the RMSE of the
        reconstructed flux on the held-out samples, and the output path.

    Raises
    ------
    RuntimeError
        If fewer than five samples survive generation.
    """
    t0 = time.perf_counter()
    X_sol, Y_sol = generate_solovev_dataset(n_samples=n_solovev, seed=seed) if n_solovev > 0 else _stack([], [])
    X_gs, Y_gs = generate_gs_dataset(n_samples=n_gs, seed=seed + 1000) if n_gs > 0 else _stack([], [])
    X = np.concatenate([X_sol, X_gs], axis=0)
    Y = np.concatenate([Y_sol, Y_gs], axis=0)
    if len(X) < 5:
        raise RuntimeError(f"Only {len(X)} samples — need at least 5")
    logger.info("Total training data: %d samples", len(X))

    cfg = NeuralEqConfig(
        n_components=min(n_components, len(X) - 1),
        hidden_sizes=hidden_sizes,
        n_input_features=N_FEATURES,
        grid_shape=(GRID_N, GRID_N),
    )
    accel = NeuralEquilibriumAccelerator(cfg)
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-10] = 1.0
    accel._input_mean = input_mean
    accel._input_std = input_std
    X_norm = (X - input_mean) / input_std
    Y_compressed = accel.pca.fit_transform(Y)
    explained = float(np.sum(np.asarray(accel.pca.explained_variance_ratio_)))
    logger.info("PCA: %d components, %.2f%% variance", cfg.n_components, explained * 100)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = max(1, int(0.2 * len(X)))
    n_train = len(X) - n_val
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, Y_train = X_norm[train_idx], Y_compressed[train_idx]
    X_val, Y_val = X_norm[val_idx], Y_compressed[val_idx]

    mlp = SimpleMLP([N_FEATURES, *hidden_sizes, cfg.n_components], seed=seed)
    accel.mlp = mlp
    lr, momentum, patience = 1e-4, 0.9, 30
    batch_size = min(32, n_train)
    velocity = [np.zeros_like(w) for w in mlp.weights]
    velocity_b = [np.zeros_like(b) for b in mlp.biases]
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = [w.copy() for w in mlp.weights]
    best_biases = [b.copy() for b in mlp.biases]
    stale_epochs = 0

    for epoch in range(n_epochs):
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            float_weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
            float_biases = [np.asarray(b, dtype=np.float64) for b in mlp.biases]
            loss, grad_w, grad_b = _gradients(float_weights, float_biases, X_train[idx], Y_train[idx])
            epoch_loss += loss * len(idx)
            for i in range(len(mlp.weights)):
                velocity[i] = momentum * velocity[i] - lr * np.clip(grad_w[i], -1.0, 1.0)
                velocity_b[i] = momentum * velocity_b[i] - lr * np.clip(grad_b[i], -1.0, 1.0)
                mlp.weights[i] = mlp.weights[i] + velocity[i]
                mlp.biases[i] = mlp.biases[i] + velocity_b[i]
        val_loss = float(np.mean(np.sum((mlp.forward(X_val) - Y_val) ** 2, axis=1)))
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            best_weights = [w.copy() for w in mlp.weights]
            best_biases = [b.copy() for b in mlp.biases]
        else:
            stale_epochs += 1
        if epoch % 50 == 0:
            logger.info("Epoch %d: train=%.6f  val=%.6f", epoch, epoch_loss / n_train, val_loss)
        if stale_epochs >= patience:
            logger.info("Early stopping at epoch %d; best epoch %d", epoch, best_epoch)
            break

    mlp.weights, mlp.biases = best_weights, best_biases
    accel.is_trained = True
    output.parent.mkdir(parents=True, exist_ok=True)
    accel.save_weights(output)

    predicted = np.asarray(accel.predict(X[val_idx]), dtype=np.float64).reshape(len(val_idx), -1)
    truth = Y[val_idx]
    rel_l2 = np.linalg.norm(predicted - truth, axis=1) / np.maximum(np.linalg.norm(truth, axis=1), 1e-15)
    metrics: dict[str, float | int | str] = {
        "n_samples": len(X),
        "n_validation": int(n_val),
        "pca_explained_variance": explained,
        "best_val_loss_pca": best_val_loss,
        "best_epoch": best_epoch,
        "validation_mean_rel_l2": float(rel_l2.mean()),
        "validation_rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
        "output": str(output),
    }
    logger.info("Done in %.1fs | %s", time.perf_counter() - t0, metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Train from the command line and print the held-out metrics as JSON.

    Parameters
    ----------
    argv : list[str], optional
        Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 after the weights are written.
    """
    parser = argparse.ArgumentParser(description="Train ITER-like PCA+MLP neural-equilibrium weights.")
    parser.add_argument("--gs-only", action="store_true", help="train on GS kernel solves instead of Solov'ev maps")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.gs_only:
        metrics = train_iter_weights(n_solovev=0, n_gs=args.n_samples, output=args.output)
    else:
        metrics = train_iter_weights(n_solovev=args.n_samples, n_gs=0, output=args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
