# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ITER-like PCA+MLP neural equilibrium training.
"""The ITER trainer's Solov'ev family, exact gradients, best-epoch restore and dataset filters."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core import fusion_kernel
from scpn_control.core.neural_equilibrium import NeuralEquilibriumAccelerator
from tools import train_neural_equilibrium_iter as trainer

FloatArray = NDArray[np.float64]


def _mesh() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return the training grid axes and its meshgrid."""
    r_arr, z_arr = trainer._grid()
    RR, ZZ = np.meshgrid(r_arr, z_arr)
    return r_arr, z_arr, RR, ZZ


def test_solovev_map_vanishes_at_its_three_points_and_has_a_constant_source() -> None:
    """The map is zero at the three conditions and Delta* psi / R^2 equals the constant (A1 + A2) / R0^2."""
    r_arr, z_arr, RR, ZZ = _mesh()
    a, kappa, A1, A2 = 2.0, 1.7, -0.05, -0.035
    points = np.array([[trainer.R0 + a, 0.0], [trainer.R0 - a, 0.0], [trainer.R0, kappa * a]])
    at_points = trainer._solovev_psi(points[:, 0], points[:, 1], trainer.R0, a, kappa, A1, A2)
    np.testing.assert_allclose(at_points, 0.0, atol=1e-12)

    psi = trainer._solovev_psi(RR, ZZ, trainer.R0, a, kappa, A1, A2)
    dR, dZ = r_arr[1] - r_arr[0], z_arr[1] - z_arr[0]
    inner = psi[1:-1, 1:-1]
    d2R = (psi[1:-1, 2:] - 2 * inner + psi[1:-1, :-2]) / dR**2
    dRc = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dR)
    d2Z = (psi[2:, 1:-1] - 2 * inner + psi[:-2, 1:-1]) / dZ**2
    source = (d2R - dRc / RR[1:-1, 1:-1] + d2Z) / RR[1:-1, 1:-1] ** 2
    # Central differences of a quartic keep an O(h^2) truncation error of about 1e-4.
    np.testing.assert_allclose(source, (A1 + A2) / trainer.R0**2, rtol=1e-3)


def test_only_the_sum_of_the_source_amplitudes_changes_the_map() -> None:
    """The two source terms differ by a homogeneous solution, so equal sums give equal maps."""
    _, _, RR, ZZ = _mesh()
    first = trainer._solovev_psi(RR, ZZ, trainer.R0, 2.0, 1.7, -0.05, -0.035)
    same_sum = trainer._solovev_psi(RR, ZZ, trainer.R0, 2.0, 1.7, -0.07, -0.015)
    other_sum = trainer._solovev_psi(RR, ZZ, trainer.R0, 2.0, 1.7, -0.05, -0.05)
    np.testing.assert_allclose(same_sum, first, atol=1e-12)
    assert float(np.abs(other_sum - first).max()) > 0.1


def test_solovev_dataset_layout_is_deterministic() -> None:
    """Features follow the 12-entry layout, the axis is the grid maximum, and equal seeds repeat."""
    r_arr, z_arr, _, _ = _mesh()
    X, Y = trainer.generate_solovev_dataset(n_samples=4, seed=7)
    assert X.shape == (4, 12) and Y.shape == (4, trainer.GRID_N**2)
    for features, flat in zip(X, Y, strict=True):
        psi = flat.reshape(trainer.GRID_N, trainer.GRID_N)
        iz, ir = np.unravel_index(int(np.argmax(psi)), psi.shape)
        assert features[2] == r_arr[ir] and features[3] == z_arr[iz]
        assert features[6] == psi.max() and features[7] == psi.min()
        assert (features[1], features[9], features[10], features[11]) == (
            trainer.B0,
            trainer.DELTA,
            trainer.DELTA,
            trainer.Q95,
        )
    X2, Y2 = trainer.generate_solovev_dataset(n_samples=4, seed=7)
    np.testing.assert_array_equal(X2, X)
    np.testing.assert_array_equal(Y2, Y)


def test_non_finite_solovev_draws_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draw whose map is not finite is not a training sample."""
    monkeypatch.setattr(trainer, "_solovev_psi", lambda RR, *_: np.full(RR.shape, np.nan))
    X, Y = trainer.generate_solovev_dataset(n_samples=3)
    assert X.shape == (0, 12) and Y.shape == (0, trainer.GRID_N**2)


def test_gradients_are_exact_for_every_layer() -> None:
    """Every layer's gradient matches central finite differences of the batch objective."""
    rng = np.random.default_rng(3)
    sizes = [4, 6, 5, 3]
    weights = [rng.normal(0, 0.5, (sizes[i], sizes[i + 1])) for i in range(3)]
    biases = [rng.normal(0, 0.1, sizes[i + 1]) for i in range(3)]
    x, y = rng.normal(size=(7, 4)), rng.normal(size=(7, 3))
    _, grad_w, grad_b = trainer._gradients(weights, biases, x, y)
    step = 1e-6
    for layer in range(3):
        for params, grads in ((weights, grad_w), (biases, grad_b)):
            probe = (1, 1) if params[layer].ndim == 2 else (1,)
            plus = [p.copy() for p in params]
            minus = [p.copy() for p in params]
            plus[layer][probe] += step
            minus[layer][probe] -= step
            other = biases if params is weights else weights
            if params is weights:
                up, down = trainer._gradients(plus, other, x, y)[0], trainer._gradients(minus, other, x, y)[0]
            else:
                up, down = trainer._gradients(other, plus, x, y)[0], trainer._gradients(other, minus, x, y)[0]
            assert grads[layer][probe] == pytest.approx((up - down) / (2 * step), rel=1e-5, abs=1e-9)


def test_training_saves_the_best_validation_epoch(tmp_path: Path) -> None:
    """The written weights reproduce the reported best validation objective and held-out errors."""
    output = tmp_path / "w" / "iter.npz"
    metrics = trainer.train_iter_weights(
        n_solovev=8, n_components=3, hidden_sizes=(8,), n_epochs=40, seed=5, output=output
    )
    assert metrics["n_samples"] == 8 and metrics["n_validation"] == 1 and metrics["output"] == str(output)
    assert 0 <= int(metrics["best_epoch"]) < 40

    X, Y = trainer.generate_solovev_dataset(n_samples=8, seed=5)
    rng = np.random.default_rng(5)
    val = rng.permutation(8)[7:]
    accel = NeuralEquilibriumAccelerator()
    accel.load_weights(output)
    assert accel.mlp is not None and accel.cfg.n_input_features == 12
    x_val = (X[val] - accel._input_mean) / accel._input_std
    y_val = accel.pca.transform(Y[val])
    objective = float(np.mean(np.sum((accel.mlp.forward(x_val) - y_val) ** 2, axis=1)))
    assert objective == pytest.approx(float(metrics["best_val_loss_pca"]), rel=1e-9)
    predicted = np.asarray(accel.predict(X[val])).reshape(1, -1)
    rel = float(np.linalg.norm(predicted - Y[val]) / np.linalg.norm(Y[val]))
    assert rel == pytest.approx(float(metrics["validation_mean_rel_l2"]), rel=1e-9)


def test_an_epoch_limit_without_early_stopping_keeps_the_best_epoch(tmp_path: Path) -> None:
    """When the epoch limit ends training first, the best of those epochs is still the one saved."""
    metrics = trainer.train_iter_weights(
        n_solovev=6, n_components=2, hidden_sizes=(4,), n_epochs=2, seed=9, output=tmp_path / "short.npz"
    )
    assert metrics["best_epoch"] in (0, 1)
    assert (tmp_path / "short.npz").stat().st_size > 0


def test_too_few_samples_are_refused(tmp_path: Path) -> None:
    """Fewer than five surviving samples cannot be split and trained."""
    with pytest.raises(RuntimeError, match="Only 3 samples"):
        trainer.train_iter_weights(n_solovev=3, output=tmp_path / "x.npz")
    assert not (tmp_path / "x.npz").exists()


class _StubKernel:
    """Kernel stand-in that replays a scripted outcome per solve."""

    outcomes: list[str] = []
    configs: list[dict[str, Any]] = []

    def __init__(self, path: str) -> None:
        """Record the configuration the trainer wrote for this draw."""
        self.config = json.loads(Path(path).read_text(encoding="utf-8"))
        _StubKernel.configs.append(self.config)

    def solve_fixed_boundary(self) -> dict[str, FloatArray]:
        """Return or raise the next scripted outcome."""
        outcome = _StubKernel.outcomes.pop(0)
        _, _, RR, ZZ = _mesh()
        if outcome == "raise":
            raise RuntimeError("scripted solver failure")
        if outcome == "nan":
            return {"psi": np.full(RR.shape, np.nan)}
        if outcome == "edge":
            return {"psi": RR.copy()}
        ip = float(self.config["physics"]["plasma_current_target"])
        return {"psi": trainer._solovev_psi(RR, ZZ, trainer.R0, 2.0, 1.7, -0.05 * ip / 15.0, -0.035)}


def test_gs_dataset_skips_failed_non_finite_and_edge_maximum_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only solved, finite maps with an interior O-point become samples; temporary configs are removed."""
    monkeypatch.setattr(fusion_kernel, "FusionKernel", _StubKernel)
    monkeypatch.setattr(_StubKernel, "outcomes", ["raise", "nan", "edge", "good", "good"])
    monkeypatch.setattr(_StubKernel, "configs", [])
    X, Y = trainer.generate_gs_dataset(n_samples=5, seed=11)
    assert X.shape == (2, 12) and Y.shape == (2, trainer.GRID_N**2)
    assert np.all(X[:, 8] == trainer.KAPPA)
    assert [cfg["physics"]["plasma_current_target"] for cfg in _StubKernel.configs][3:] == list(X[:, 0])
    assert all(cfg["grid_resolution"] == [trainer.GRID_N, trainer.GRID_N] for cfg in _StubKernel.configs)

    monkeypatch.setattr(_StubKernel, "outcomes", ["raise"])
    X_empty, _ = trainer.generate_gs_dataset(n_samples=1)
    assert X_empty.shape == (0, 12)


def test_command_trains_from_either_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command parses --n-samples N, trains Solov'ev or GS data and prints the metrics."""
    calls: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> dict[str, float | int | str]:
        """Record the keyword arguments and return metrics naming the output."""
        calls.append(kwargs)
        return {"output": str(kwargs["output"])}

    monkeypatch.setattr(trainer, "train_iter_weights", fake_train)
    assert trainer.main(["--n-samples", "12", "--output", str(tmp_path / "a.npz")]) == 0
    assert trainer.main(["--gs-only", "--n-samples", "6", "--output", str(tmp_path / "b.npz")]) == 0
    assert calls == [
        {"n_solovev": 12, "n_gs": 0, "output": tmp_path / "a.npz"},
        {"n_solovev": 0, "n_gs": 6, "output": tmp_path / "b.npz"},
    ]
    assert json.loads(capsys.readouterr().out.split("}\n")[0] + "}") == {"output": str(tmp_path / "a.npz")}


def test_script_entry_trains_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executed as a script, the module trains a small Solov'ev set and writes the weights."""
    script = Path(trainer.__file__)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(script), "--n-samples", "6", "--output", str(tmp_path / "s.npz")])
    with pytest.raises(SystemExit) as script_exit:
        runpy.run_path(str(script), run_name="__main__")
    assert script_exit.value.code == 0
    assert (tmp_path / "s.npz").stat().st_size > 0
    assert sys.path[0] == str(trainer.REPO_ROOT / "src")
