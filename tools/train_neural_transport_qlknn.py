#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — QLKNN-10D Neural Transport Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Train a neural transport MLP on QLKNN-10D-style data.

Usage:
    python tools/train_neural_transport_qlknn.py --data-dir data/qlknn10d/
    python tools/train_neural_transport_qlknn.py --synthetic  # CI-friendly synthetic data

Dataset provenance (real data):
    van de Plassche, K.L. et al. (2020). "Fast modeling of turbulent
    transport in fusion plasmas using neural networks."
    Phys. Plasmas 27, 022310. doi:10.1063/1.5134126
    Data: https://doi.org/10.5281/zenodo.3700755

Architecture: [10] → [128] → [64] → [3] with ReLU hidden, softplus output.
Input features: [rho, Te, Ti, ne, R/L_Te, R/L_Ti, R/L_ne, q, s_hat, beta_e]
Output targets: [chi_e, chi_i, D_e] in m^2/s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "neural_transport_qlknn.npz"
DEFAULT_METRICS = REPO_ROOT / "weights" / "neural_transport_qlknn.metrics.json"


def generate_synthetic_qlknn_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate synthetic training data mimicking QLKNN-10D structure.

    Uses the critical-gradient model as ground truth with added noise,
    providing a reasonable proxy for CI testing without the real dataset.

    Returns (X, Y) with X shape (n, 10) and Y shape (n, 3).
    """
    rng = np.random.default_rng(seed)

    rho = rng.uniform(0.1, 0.95, n_samples)
    te = rng.uniform(0.5, 20.0, n_samples)
    ti = rng.uniform(0.5, 20.0, n_samples)
    ne = rng.uniform(1.0, 15.0, n_samples)
    grad_te = rng.uniform(0.0, 30.0, n_samples)
    grad_ti = rng.uniform(0.0, 30.0, n_samples)
    grad_ne = rng.uniform(-5.0, 20.0, n_samples)
    q = rng.uniform(1.0, 6.0, n_samples)
    s_hat = rng.uniform(0.0, 3.0, n_samples)
    beta_e = 4.03e-3 * ne * te

    X = np.column_stack([rho, te, ti, ne, grad_te, grad_ti, grad_ne, q, s_hat, beta_e])

    # Dimits et al. 2000, Garbet et al. 2004
    crit_itg, crit_tem, stiffness = 4.0, 5.0, 2.0
    chi_gb = 1.0

    chi_i = chi_gb * np.maximum(0.0, grad_ti - crit_itg) ** stiffness
    chi_e = chi_gb * np.maximum(0.0, grad_te - crit_tem) ** stiffness
    d_e = chi_e / 3.0

    # Add 10% multiplicative noise to simulate gyrokinetic scatter
    chi_i *= 1.0 + 0.1 * rng.standard_normal(n_samples)
    chi_e *= 1.0 + 0.1 * rng.standard_normal(n_samples)
    d_e *= 1.0 + 0.1 * rng.standard_normal(n_samples)

    chi_i = np.maximum(chi_i, 0.0)
    chi_e = np.maximum(chi_e, 0.0)
    d_e = np.maximum(d_e, 0.0)

    Y = np.column_stack([chi_e, chi_i, d_e])
    return X, Y


def load_qlknn_data(data_dir: Path) -> tuple[NDArray, NDArray]:
    """Load QLKNN-10D data from Zenodo download.

    Expects CSV or NPZ files in data_dir with columns matching the
    10-dim input and 3-dim output specification.
    """
    npz_files = list(data_dir.glob("*.npz"))
    csv_files = list(data_dir.glob("*.csv"))

    if npz_files:
        data = np.load(npz_files[0])
        X = data["inputs"] if "inputs" in data else data["X"]
        Y = data["outputs"] if "outputs" in data else data["Y"]
        return X, Y

    if csv_files:
        import csv

        rows: list[list[float]] = []
        with open(csv_files[0]) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                rows.append([float(v) for v in row])
        arr = np.array(rows)
        return arr[:, :10], arr[:, 10:13]

    raise FileNotFoundError(f"No .npz or .csv files in {data_dir}")


def train_mlp(
    X: NDArray,
    Y: NDArray,
    hidden_sizes: tuple[int, int] = (128, 64),
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, NDArray]:
    """Train 3-layer MLP: input(10) → hidden1 → hidden2 → output(3).

    Returns dict of weight arrays ready for np.savez.
    """
    rng = np.random.default_rng(seed)

    # Z-score normalisation
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-8] = 1.0
    X_norm = (X - input_mean) / input_std

    # Output scale: normalise targets to ~unit range for stable training
    output_scale = np.maximum(Y.max(axis=0), 1e-8)
    Y_norm = Y / output_scale

    # 80/10/10 split
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    n_train = n - n_val - n_test

    X_train, Y_train = X_norm[idx[:n_train]], Y_norm[idx[:n_train]]
    X_val, Y_val = X_norm[idx[n_train : n_train + n_val]], Y_norm[idx[n_train : n_train + n_val]]
    X_test, Y_test = X_norm[idx[n_train + n_val :]], Y_norm[idx[n_train + n_val :]]

    # He initialisation
    h1, h2 = hidden_sizes
    w1 = rng.normal(0, np.sqrt(2.0 / 10), (10, h1))
    b1 = np.zeros(h1)
    w2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2))
    b2 = np.zeros(h2)
    w3 = rng.normal(0, np.sqrt(2.0 / h2), (h2, 3))
    b3 = np.zeros(3)

    # Momentum SGD
    momentum = 0.9
    vw1, vb1 = np.zeros_like(w1), np.zeros_like(b1)
    vw2, vb2 = np.zeros_like(w2), np.zeros_like(b2)
    vw3, vb3 = np.zeros_like(w3), np.zeros_like(b3)

    best_val_loss = float("inf")
    patience, patience_counter = 30, 0

    def softplus(x: NDArray) -> NDArray:
        return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))

    def softplus_grad(x: NDArray) -> NDArray:
        ex = np.exp(np.clip(x, -20.0, 20.0))
        return ex / (1.0 + ex)

    for epoch in range(n_epochs):
        order = rng.permutation(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            bi = order[start : start + batch_size]
            xb, yb = X_train[bi], Y_train[bi]
            bs = len(bi)

            # Forward
            z1 = xb @ w1 + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ w2 + b2
            a2 = np.maximum(0, z2)
            z3 = a2 @ w3 + b3
            out = softplus(z3)

            # MSE loss
            err = out - yb
            loss = float(np.mean(err**2))
            epoch_loss += loss * bs

            # Backprop through softplus
            d3 = 2.0 * err * softplus_grad(z3) / bs
            gw3 = a2.T @ d3
            gb3 = d3.sum(axis=0)

            d2 = (d3 @ w3.T) * (z2 > 0).astype(float)
            gw2 = a1.T @ d2
            gb2 = d2.sum(axis=0)

            d1 = (d2 @ w2.T) * (z1 > 0).astype(float)
            gw1 = xb.T @ d1
            gb1 = d1.sum(axis=0)

            # Clip gradients
            for g in [gw1, gb1, gw2, gb2, gw3, gb3]:
                np.clip(g, -5.0, 5.0, out=g)

            # Update with momentum
            vw1 = momentum * vw1 - lr * gw1
            vb1 = momentum * vb1 - lr * gb1
            vw2 = momentum * vw2 - lr * gw2
            vb2 = momentum * vb2 - lr * gb2
            vw3 = momentum * vw3 - lr * gw3
            vb3 = momentum * vb3 - lr * gb3

            w1 += vw1
            b1 += vb1
            w2 += vw2
            b2 += vb2
            w3 += vw3
            b3 += vb3

        epoch_loss /= max(n_train, 1)

        # Validation
        z1v = X_val @ w1 + b1
        a1v = np.maximum(0, z1v)
        z2v = a1v @ w2 + b2
        a2v = np.maximum(0, z2v)
        val_out = softplus(a2v @ w3 + b3)
        val_loss = float(np.mean((val_out - Y_val) ** 2))

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

    # Test evaluation
    z1t = X_test @ w1 + b1
    a1t = np.maximum(0, z1t)
    z2t = a1t @ w2 + b2
    a2t = np.maximum(0, z2t)
    test_out = softplus(a2t @ w3 + b3) * output_scale
    Y_test_raw = Y_test * output_scale
    test_rmse = float(np.sqrt(np.mean((test_out - Y_test_raw) ** 2)))
    per_channel_rmse = np.sqrt(np.mean((test_out - Y_test_raw) ** 2, axis=0))

    logger.info("Test RMSE: %.4f  per-channel: %s", test_rmse, per_channel_rmse)

    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_scale": output_scale,
        "version": np.array([1]),
        "_test_rmse": np.array([test_rmse]),
        "_per_channel_rmse": per_channel_rmse,
        "_n_train": np.array([n_train]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QLKNN-10D neural transport model")
    parser.add_argument("--data-dir", type=Path, help="Directory with QLKNN-10D data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (CI mode)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    if args.synthetic:
        logger.info("Generating synthetic QLKNN-10D data (5000 samples)")
        X, Y = generate_synthetic_qlknn_data(5000, args.seed)
    elif args.data_dir:
        logger.info("Loading QLKNN-10D data from %s", args.data_dir)
        X, Y = load_qlknn_data(args.data_dir)
    else:
        print("Specify --data-dir or --synthetic")
        sys.exit(1)

    logger.info("Data shape: X=%s Y=%s", X.shape, Y.shape)

    t0 = time.perf_counter()
    weights = train_mlp(X, Y, n_epochs=args.epochs, seed=args.seed)
    elapsed = time.perf_counter() - t0

    # Save weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_keys = {k: v for k, v in weights.items() if not k.startswith("_")}
    np.savez(args.output, **save_keys)
    logger.info("Saved weights to %s", args.output)

    # Save metrics
    metrics = {
        "test_rmse": float(weights["_test_rmse"][0]),
        "per_channel_rmse": weights["_per_channel_rmse"].tolist(),
        "n_train": int(weights["_n_train"][0]),
        "n_total": len(X),
        "train_time_s": elapsed,
        "synthetic": args.synthetic,
        "channels": ["chi_e", "chi_i", "D_e"],
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
