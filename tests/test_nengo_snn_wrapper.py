# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Nengo SNN Wrapper Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for nengo_snn_wrapper with mocked nengo dependency."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Build a mock nengo module that satisfies all attribute accesses the wrapper
# performs during build_network() and step().
# ---------------------------------------------------------------------------


def _make_mock_nengo():
    mock = MagicMock()
    mock.LIF = MagicMock(return_value=MagicMock())
    mock.dists.Uniform = MagicMock(return_value=MagicMock())

    network_ctx = MagicMock()
    network_ctx.__enter__ = MagicMock(return_value=network_ctx)
    network_ctx.__exit__ = MagicMock(return_value=False)
    network_ctx.all_connections = []
    mock.Network.return_value = network_ctx

    node = MagicMock()
    node.__getitem__ = MagicMock(return_value=MagicMock())
    mock.Node.return_value = node

    mock.Ensemble.return_value = MagicMock()
    mock.Connection.return_value = MagicMock()

    probe = MagicMock()
    mock.Probe.return_value = probe

    sim = MagicMock()
    sim.data = {probe: np.zeros((1, 2))}
    sim.step = MagicMock()
    sim.reset = MagicMock()
    mock.Simulator.return_value = sim

    return mock, sim, probe


@pytest.fixture()
def _patch_nengo(monkeypatch):
    """Patch the wrapper module so _nengo_available=True and _nengo is our mock."""
    mock_nengo, sim, probe = _make_mock_nengo()
    import scpn_control.control.nengo_snn_wrapper as mod

    monkeypatch.setattr(mod, "_nengo", mock_nengo)
    monkeypatch.setattr(mod, "_nengo_available", True)
    return SimpleNamespace(mock=mock_nengo, sim=sim, probe=probe, mod=mod)


# ── NengoSNNConfig ───────────────────────────────────────────────────


def test_config_defaults():
    from scpn_control.control.nengo_snn_wrapper import NengoSNNConfig

    cfg = NengoSNNConfig()
    assert cfg.n_neurons == 200
    assert cfg.n_channels == 2
    assert cfg.dt == 0.001


def test_config_custom():
    from scpn_control.control.nengo_snn_wrapper import NengoSNNConfig

    cfg = NengoSNNConfig(n_neurons=100, n_channels=4, gain=10.0)
    assert cfg.n_neurons == 100
    assert cfg.n_channels == 4
    assert cfg.gain == 10.0


# ── nengo_available() ────────────────────────────────────────────────


def test_nengo_available_reflects_flag():
    from scpn_control.control.nengo_snn_wrapper import nengo_available

    # The real module has _nengo_available set at import; just verify it's callable.
    assert isinstance(nengo_available(), bool)


# ── NengoSNNController ──────────────────────────────────────────────


def test_construction_without_nengo():
    """ImportError when nengo is unavailable."""
    from scpn_control.control.nengo_snn_wrapper import NengoSNNController

    with (
        patch("scpn_control.control.nengo_snn_wrapper._nengo_available", False),
        pytest.raises(ImportError, match="Nengo is required"),
    ):
        NengoSNNController()


@pytest.mark.usefixtures("_patch_nengo")
def test_construction_with_mock(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    assert ctrl._built is True
    assert ctrl._step_count == 0


@pytest.mark.usefixtures("_patch_nengo")
def test_step_returns_array(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    error = np.array([0.1, -0.2])
    out = ctrl.step(error)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


@pytest.mark.usefixtures("_patch_nengo")
def test_step_increments_count(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    ctrl.step(np.zeros(2))
    ctrl.step(np.zeros(2))
    assert ctrl._step_count == 2


@pytest.mark.usefixtures("_patch_nengo")
def test_reset_clears_state(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    ctrl.step(np.ones(2))
    ctrl.reset()
    assert ctrl._step_count == 0
    assert np.all(ctrl._last_output == 0.0)


@pytest.mark.usefixtures("_patch_nengo")
def test_get_spike_data_keys(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    ctrl.step(np.zeros(2))
    data = ctrl.get_spike_data()
    assert "output" in data


@pytest.mark.usefixtures("_patch_nengo")
def test_export_weights_returns_dict(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    weights = ctrl.export_weights()
    assert isinstance(weights, dict)


@pytest.mark.usefixtures("_patch_nengo")
def test_export_fpga_weights(tmp_path, _patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    out = tmp_path / "fpga_weights.npz"
    ctrl.export_fpga_weights(out)
    assert out.exists()
    loaded = np.load(str(out))
    assert "n_neurons" in loaded
    assert "n_channels" in loaded


@pytest.mark.usefixtures("_patch_nengo")
def test_benchmark_returns_stats(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    stats = ctrl.benchmark(n_steps=10)
    assert "mean_us" in stats
    assert "p95_us" in stats
    assert stats["mean_us"] >= 0.0


# ── NengoSNNControllerStub ──────────────────────────────────────────


def test_stub_raises():
    from scpn_control.control.nengo_snn_wrapper import NengoSNNControllerStub

    with pytest.raises(ImportError, match="Nengo is required"):
        NengoSNNControllerStub()


@pytest.mark.usefixtures("_patch_nengo")
def test_export_loihi_raises_without_loihi(_patch_nengo):
    ctrl = _patch_nengo.mod.NengoSNNController()
    with pytest.raises(ImportError, match="nengo_loihi"):
        ctrl.export_loihi("out.npz")
