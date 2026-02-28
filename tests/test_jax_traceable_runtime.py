# ──────────────────────────────────────────────────────────────────────
# SCPN Control — JAX Traceable Runtime Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Full coverage for jax_traceable_runtime: validation, backends, parity."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.jax_traceable_runtime import (
    TraceableRuntimeSpec,
    _resolve_backend,
    _resolve_backend_set,
    _simulate_numpy,
    _simulate_numpy_batch,
    _validate_batch_commands,
    _validate_commands,
    _validate_spec,
    available_traceable_backends,
    run_traceable_control_batch,
    run_traceable_control_loop,
    validate_traceable_backend_parity,
)


# ── Spec validation ──────────────────────────────────────────────────

class TestValidateSpec:
    def test_default_spec_passes(self):
        _validate_spec(TraceableRuntimeSpec())

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            _validate_spec(TraceableRuntimeSpec(dt_s=-1.0))

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            _validate_spec(TraceableRuntimeSpec(dt_s=0.0))

    def test_inf_dt_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            _validate_spec(TraceableRuntimeSpec(dt_s=float("inf")))

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="tau_s"):
            _validate_spec(TraceableRuntimeSpec(tau_s=-1.0))

    def test_nan_gain_raises(self):
        with pytest.raises(ValueError, match="gain"):
            _validate_spec(TraceableRuntimeSpec(gain=float("nan")))

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError, match="command_limit"):
            _validate_spec(TraceableRuntimeSpec(command_limit=-0.5))


class TestValidateCommands:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty 1D"):
            _validate_commands(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="non-empty 1D"):
            _validate_commands(np.ones((2, 3)))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _validate_commands(np.array([1.0, float("nan")]))

    def test_valid_passes(self):
        _validate_commands(np.ones(10))


class TestValidateBatchCommands:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="batch, steps"):
            _validate_batch_commands(np.ones(5))

    def test_zero_batch_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            _validate_batch_commands(np.ones((0, 5)))

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            _validate_batch_commands(np.ones((3, 0)))

    def test_inf_raises(self):
        cmd = np.ones((2, 5))
        cmd[0, 0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            _validate_batch_commands(cmd)


# ── Backend resolution ───────────────────────────────────────────────

class TestResolveBackend:
    def test_numpy(self):
        assert _resolve_backend("numpy") == "numpy"

    def test_auto_picks_something(self):
        b = _resolve_backend("auto")
        assert b in {"numpy", "jax", "torchscript"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            _resolve_backend("cuda")


class TestResolveBackendSet:
    def test_none_returns_available(self):
        result = _resolve_backend_set(None)
        assert "numpy" in result

    def test_invalid_backend_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _resolve_backend_set(["bogus"])

    def test_deduplication(self):
        result = _resolve_backend_set(["numpy", "numpy"])
        assert result == ["numpy"]


class TestAvailableBackends:
    def test_numpy_always_present(self):
        assert "numpy" in available_traceable_backends()


# ── Single rollout ───────────────────────────────────────────────────

class TestSimulateNumpy:
    def test_output_shape(self):
        cmd = np.ones(20)
        out = _simulate_numpy(cmd, 0.0, TraceableRuntimeSpec())
        assert out.shape == (20,)

    def test_zero_commands_decay_to_zero(self):
        cmd = np.zeros(100)
        out = _simulate_numpy(cmd, 1.0, TraceableRuntimeSpec())
        assert abs(out[-1]) < 0.05


class TestRunTraceableControlLoop:
    def test_numpy_backend(self):
        cmd = np.sin(np.linspace(0, 2 * np.pi, 64))
        result = run_traceable_control_loop(cmd, backend="numpy")
        assert result.backend_used == "numpy"
        assert not result.compiled
        assert result.state_history.shape == (64,)

    def test_jax_backend(self):
        pytest.importorskip("jax")
        cmd = np.sin(np.linspace(0, 2 * np.pi, 64))
        result = run_traceable_control_loop(cmd, backend="jax")
        assert result.backend_used == "jax"
        assert result.compiled

    def test_torchscript_backend(self):
        pytest.importorskip("torch")
        cmd = np.sin(np.linspace(0, 2 * np.pi, 64))
        result = run_traceable_control_loop(cmd, backend="torchscript")
        assert result.backend_used == "torchscript"
        assert result.compiled

    def test_auto_backend(self):
        cmd = np.ones(32)
        result = run_traceable_control_loop(cmd, backend="auto")
        assert result.backend_used in {"numpy", "jax", "torchscript"}

    def test_nonfinite_initial_state_raises(self):
        with pytest.raises(ValueError, match="initial_state"):
            run_traceable_control_loop(np.ones(10), initial_state=float("nan"))

    def test_custom_spec(self):
        spec = TraceableRuntimeSpec(dt_s=2e-3, tau_s=10e-3, gain=0.5, command_limit=0.8)
        result = run_traceable_control_loop(np.ones(32), spec=spec, backend="numpy")
        assert result.state_history.shape == (32,)


# ── Batch rollout ────────────────────────────────────────────────────

class TestSimulateNumpyBatch:
    def test_output_shape(self):
        cmd = np.ones((4, 20))
        x0 = np.zeros(4)
        out = _simulate_numpy_batch(cmd, x0, TraceableRuntimeSpec())
        assert out.shape == (4, 20)


class TestRunTraceableControlBatch:
    def test_numpy_batch(self):
        cmd = np.random.default_rng(0).normal(size=(4, 32))
        result = run_traceable_control_batch(cmd, backend="numpy")
        assert result.backend_used == "numpy"
        assert result.state_history.shape == (4, 32)

    def test_jax_batch(self):
        pytest.importorskip("jax")
        cmd = np.random.default_rng(0).normal(size=(4, 32))
        result = run_traceable_control_batch(cmd, backend="jax")
        assert result.backend_used == "jax"
        assert result.state_history.shape == (4, 32)

    def test_torchscript_batch(self):
        pytest.importorskip("torch")
        cmd = np.random.default_rng(0).normal(size=(4, 32))
        result = run_traceable_control_batch(cmd, backend="torchscript")
        assert result.backend_used == "torchscript"
        assert result.state_history.shape == (4, 32)

    def test_scalar_initial_state_broadcast(self):
        cmd = np.ones((3, 16))
        result = run_traceable_control_batch(cmd, initial_state=0.5, backend="numpy")
        assert result.state_history.shape == (3, 16)

    def test_none_initial_state_defaults_zero(self):
        cmd = np.zeros((2, 16))
        result = run_traceable_control_batch(cmd, initial_state=None, backend="numpy")
        np.testing.assert_allclose(result.state_history, 0.0, atol=1e-12)

    def test_wrong_x0_length_raises(self):
        cmd = np.ones((3, 16))
        with pytest.raises(ValueError, match="batch dimension"):
            run_traceable_control_batch(cmd, initial_state=np.zeros(5), backend="numpy")

    def test_nonfinite_x0_raises(self):
        cmd = np.ones((2, 16))
        with pytest.raises(ValueError, match="finite"):
            run_traceable_control_batch(
                cmd, initial_state=np.array([0.0, float("nan")]), backend="numpy",
            )


# ── Backend parity ───────────────────────────────────────────────────

class TestValidateTraceableBackendParity:
    def test_numpy_only(self):
        reports = validate_traceable_backend_parity(
            steps=32, batch=4, backends=["numpy"],
        )
        assert "numpy" in reports
        assert reports["numpy"].single_max_abs_err == 0.0
        assert reports["numpy"].batch_max_abs_err == 0.0

    def test_jax_parity(self):
        pytest.importorskip("jax")
        # JAX may use float32 unless JAX_ENABLE_X64 is set; use loose tol
        reports = validate_traceable_backend_parity(
            steps=32, batch=4, backends=["jax"], atol=1e-4,
        )
        assert reports["jax"].single_within_tol

    def test_torchscript_parity(self):
        pytest.importorskip("torch")
        reports = validate_traceable_backend_parity(
            steps=32, batch=4, backends=["torchscript"],
        )
        assert reports["torchscript"].single_within_tol

    def test_all_backends(self):
        reports = validate_traceable_backend_parity(steps=32, batch=4)
        assert len(reports) >= 1

    def test_invalid_steps_raises(self):
        with pytest.raises(ValueError, match="steps"):
            validate_traceable_backend_parity(steps=0)

    def test_invalid_batch_raises(self):
        with pytest.raises(ValueError, match="batch"):
            validate_traceable_backend_parity(batch=0)

    def test_negative_atol_raises(self):
        with pytest.raises(ValueError, match="atol"):
            validate_traceable_backend_parity(atol=-1.0)

    def test_custom_spec(self):
        spec = TraceableRuntimeSpec(dt_s=5e-3, tau_s=20e-3)
        reports = validate_traceable_backend_parity(
            steps=16, batch=2, spec=spec, backends=["numpy"],
        )
        assert "numpy" in reports
