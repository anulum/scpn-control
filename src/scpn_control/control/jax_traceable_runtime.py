# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX Traceable Runtime
"""Optional JAX-traceable control-loop utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None
    jnp = cast(Any, None)  # optional-dep fallback (keeps jnp.* annotations typed)
    _HAS_JAX = False

try:
    import torch  # pragma: no cover - optional TorchScript backend path

    _HAS_TORCH = True  # pragma: no cover - optional TorchScript backend path
except ImportError:
    torch = None
    _HAS_TORCH = False


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TraceableRuntimeSpec:
    """Configuration for reduced traceable first-order actuator dynamics."""

    dt_s: float = 1.0e-3
    tau_s: float = 5.0e-3
    gain: float = 1.0
    command_limit: float = 1.0


@dataclass(frozen=True)
class TraceableRuntimeResult:
    """Result of a traceable control-loop rollout."""

    state_history: FloatArray
    backend_used: str
    compiled: bool


@dataclass(frozen=True)
class TraceableRuntimeBatchResult:
    """Result of batched traceable control-loop rollout."""

    state_history: FloatArray
    backend_used: str
    compiled: bool


@dataclass(frozen=True)
class TraceableBackendParityReport:
    """Parity metrics against NumPy reference backend."""

    backend: str
    single_max_abs_err: float
    batch_max_abs_err: float
    single_within_tol: bool
    batch_within_tol: bool


def _validate_spec(spec: TraceableRuntimeSpec) -> None:
    if not np.isfinite(spec.dt_s) or spec.dt_s <= 0.0:
        raise ValueError("dt_s must be finite and > 0.")
    if not np.isfinite(spec.tau_s) or spec.tau_s <= 0.0:
        raise ValueError("tau_s must be finite and > 0.")
    if not np.isfinite(spec.gain):
        raise ValueError("gain must be finite.")
    if not np.isfinite(spec.command_limit) or spec.command_limit <= 0.0:
        raise ValueError("command_limit must be finite and > 0.")


def _validate_commands(commands: FloatArray) -> None:
    if commands.ndim != 1 or commands.size == 0:
        raise ValueError("commands must be a non-empty 1D array.")
    if not np.all(np.isfinite(commands)):
        raise ValueError("commands must contain only finite values.")


def _validate_batch_commands(commands: FloatArray) -> None:
    if commands.ndim != 2 or commands.shape[0] == 0 or commands.shape[1] == 0:
        raise ValueError("commands must have shape (batch, steps) with non-zero sizes.")
    if not np.all(np.isfinite(commands)):
        raise ValueError("commands must contain only finite values.")


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return int(value)


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _coerce_scalar_initial_state(initial_state: float) -> float:
    arr = np.asarray(initial_state, dtype=np.float64)
    if arr.ndim != 0:
        raise ValueError("initial_state must be scalar for single-loop rollout.")
    value = float(arr)
    if not np.isfinite(value):
        raise ValueError("initial_state must be finite.")
    return value


def _resolve_backend(
    backend: str,
    *,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> str:
    b = str(backend).strip().lower()
    if b not in {"auto", "numpy", "jax", "torchscript"}:
        raise ValueError("backend must be one of: auto, numpy, jax, torchscript.")
    if allow_numpy_fallback and not allow_legacy_numpy_fallback:
        raise ValueError(
            "allow_numpy_fallback=True requires allow_legacy_numpy_fallback=True; "
            "legacy NumPy auto-backend fallback is disabled by default."
        )
    if b == "auto":
        if _HAS_JAX:
            return "jax"
        if _HAS_TORCH:
            return "torchscript"
        if not allow_numpy_fallback:
            raise RuntimeError(
                "No compiled backend is available for backend='auto'. "
                "Set backend='numpy' explicitly or set "
                "allow_numpy_fallback=True and allow_legacy_numpy_fallback=True "
                "for explicit degraded-mode operation."
            )
        return "numpy"
    return b


def available_traceable_backends() -> list[str]:
    """Return available runtime backends on this machine."""
    out = ["numpy"]
    if _HAS_JAX:
        out.append("jax")
    if _HAS_TORCH:
        out.append("torchscript")
    return out


def _resolve_backend_set(backends: list[str] | tuple[str, ...] | None) -> list[str]:
    available = available_traceable_backends()
    if backends is None:
        return available
    out: list[str] = []
    seen: set[str] = set()
    for raw in backends:
        name = str(raw).strip().lower()
        if name not in {"numpy", "jax", "torchscript"}:
            raise ValueError(f"Unsupported backend '{raw}'. Allowed: numpy, jax, torchscript.")
        if name not in available:
            raise ValueError(f"Requested backend '{name}' is not available on this host.")
        if name not in seen:
            out.append(name)
            seen.add(name)
    if not out:
        raise ValueError("backends must contain at least one backend when provided.")
    return out


def _simulate_numpy(commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec) -> FloatArray:
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    state = float(initial_state)
    out = np.empty_like(commands, dtype=np.float64)
    for i, cmd in enumerate(commands):
        cmd_clipped = float(np.clip(cmd, -spec.command_limit, spec.command_limit))
        state = state + alpha * ((spec.gain * cmd_clipped) - state)
        out[i] = state
    return out


def _jax_float_dtype() -> Any:
    if not _HAS_JAX:
        raise RuntimeError("JAX backend requested but JAX is not installed.")
    assert jnp is not None
    assert jax is not None
    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
    return jnp.float64 if x64_enabled else jnp.float32


def _simulate_jax(commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec) -> FloatArray:
    if not _HAS_JAX:
        raise RuntimeError("JAX backend requested but JAX is not installed.")
    assert jnp is not None
    assert jax is not None

    dtype = _jax_float_dtype()
    cmd = jnp.asarray(commands, dtype=dtype)
    alpha = jnp.asarray(spec.dt_s / (spec.tau_s + spec.dt_s), dtype=dtype)
    gain = jnp.asarray(spec.gain, dtype=dtype)
    limit = jnp.asarray(spec.command_limit, dtype=dtype)

    def _step(state: Any, u: Any) -> tuple[Any, Any]:
        u_clip = jnp.clip(u, -limit, limit)
        next_state = state + alpha * ((gain * u_clip) - state)
        return next_state, next_state

    @jax.jit
    def _rollout(x0: Any, u: Any) -> Any:
        _, hist = jax.lax.scan(_step, x0, u)
        return hist

    hist = _rollout(jnp.asarray(initial_state, dtype=dtype), cmd)
    return np.asarray(hist, dtype=np.float64)


_torchscript_rollout = None
_torchscript_rollout_batch = None


def _compile_torchscript_rollouts() -> tuple[Any, Any]:  # pragma: no cover - torch optional, absent on CI
    """Compile optional TorchScript kernels lazily to keep module import warning-clean."""
    global _torchscript_rollout, _torchscript_rollout_batch
    if not _HAS_TORCH:
        raise RuntimeError("TorchScript backend requested but torch is not installed.")
    if _torchscript_rollout is not None and _torchscript_rollout_batch is not None:
        return _torchscript_rollout, _torchscript_rollout_batch
    assert torch is not None

    # torch.jit.script compiles the following bodies to TorchScript IR; the
    # torchscript backend tests exercise them, but coverage.py cannot observe
    # JIT-compiled execution, so the kernel bodies are marked no cover.
    def _rollout_impl(  # pragma: no cover - optional TorchScript backend path
        cmd: torch.Tensor,
        initial_state: float,
        alpha: float,
        gain: float,
        limit: float,
    ) -> torch.Tensor:
        n = cmd.numel()
        out = torch.empty((n,), dtype=cmd.dtype, device=cmd.device)
        state = torch.tensor(initial_state, dtype=cmd.dtype, device=cmd.device)
        for i in range(n):
            u = torch.clamp(cmd[i], -limit, limit)
            state = state + alpha * ((gain * u) - state)
            out[i] = state
        return out

    def _rollout_batch_impl(  # pragma: no cover - optional TorchScript backend path
        cmd: torch.Tensor,
        initial_state: torch.Tensor,
        alpha: float,
        gain: float,
        limit: float,
    ) -> torch.Tensor:
        batch = cmd.size(0)
        steps = cmd.size(1)
        out = torch.empty((batch, steps), dtype=cmd.dtype, device=cmd.device)
        state = initial_state.clone()
        for t in range(steps):
            u = torch.clamp(cmd[:, t], -limit, limit)
            state = state + alpha * ((gain * u) - state)
            out[:, t] = state
        return out

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated.*",
            category=DeprecationWarning,
        )
        _torchscript_rollout = torch.jit.script(_rollout_impl)
        _torchscript_rollout_batch = torch.jit.script(_rollout_batch_impl)
    return _torchscript_rollout, _torchscript_rollout_batch


def _simulate_torchscript(
    commands: FloatArray, initial_state: float, spec: TraceableRuntimeSpec
) -> FloatArray:  # pragma: no cover - optional TorchScript backend path
    rollout, _ = _compile_torchscript_rollouts()

    cmd = torch.as_tensor(commands, dtype=torch.float64)
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    hist = rollout(
        cmd,
        float(initial_state),
        alpha,
        float(spec.gain),
        float(spec.command_limit),
    )
    return np.asarray(hist.detach().cpu().numpy(), dtype=np.float64)


def _simulate_numpy_batch(commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec) -> FloatArray:
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    state = np.asarray(initial_state, dtype=np.float64).copy()
    out = np.empty_like(commands, dtype=np.float64)
    for t in range(commands.shape[1]):
        u = np.clip(commands[:, t], -spec.command_limit, spec.command_limit)
        state = state + alpha * ((spec.gain * u) - state)
        out[:, t] = state
    return out


def _simulate_jax_batch(commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec) -> FloatArray:
    if not _HAS_JAX:
        raise RuntimeError("JAX backend requested but JAX is not installed.")
    assert jnp is not None
    assert jax is not None

    dtype = _jax_float_dtype()
    cmd = jnp.asarray(commands, dtype=dtype)
    x0 = jnp.asarray(initial_state, dtype=dtype)
    alpha = jnp.asarray(spec.dt_s / (spec.tau_s + spec.dt_s), dtype=dtype)
    gain = jnp.asarray(spec.gain, dtype=dtype)
    limit = jnp.asarray(spec.command_limit, dtype=dtype)

    def _step(state: Any, u_t: Any) -> tuple[Any, Any]:
        u_clip = jnp.clip(u_t, -limit, limit)
        next_state = state + alpha * ((gain * u_clip) - state)
        return next_state, next_state

    @jax.jit
    def _rollout_batch(batch_x0: Any, batch_u: Any) -> Any:
        _, hist_tb = jax.lax.scan(_step, batch_x0, jnp.swapaxes(batch_u, 0, 1))
        return jnp.swapaxes(hist_tb, 0, 1)

    hist = _rollout_batch(x0, cmd)
    return np.asarray(hist, dtype=np.float64)


def _simulate_torchscript_batch(  # pragma: no cover - optional TorchScript backend path
    commands: FloatArray, initial_state: FloatArray, spec: TraceableRuntimeSpec
) -> FloatArray:
    _, rollout_batch = _compile_torchscript_rollouts()

    cmd = torch.as_tensor(commands, dtype=torch.float64)
    x0 = torch.as_tensor(initial_state, dtype=torch.float64)
    alpha = float(spec.dt_s / (spec.tau_s + spec.dt_s))
    hist = rollout_batch(
        cmd,
        x0,
        alpha,
        float(spec.gain),
        float(spec.command_limit),
    )
    return np.asarray(hist.detach().cpu().numpy(), dtype=np.float64)


def run_traceable_control_loop(
    commands: FloatArray,
    *,
    initial_state: float = 0.0,
    spec: TraceableRuntimeSpec | None = None,
    backend: str = "auto",
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> TraceableRuntimeResult:
    """
    Run a reduced control loop suitable for optional JAX tracing/JIT.

    `backend` can be `auto`, `numpy`, `jax`, or `torchscript`.
    """
    cmd_arr = np.asarray(commands, dtype=np.float64)
    _validate_commands(cmd_arr)
    x0 = _coerce_scalar_initial_state(initial_state)

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)

    b = _resolve_backend(
        backend,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
    )

    if b == "jax":
        return TraceableRuntimeResult(
            state_history=_simulate_jax(cmd_arr, x0, runtime_spec),
            backend_used="jax",
            compiled=True,
        )

    if b == "torchscript":  # pragma: no cover - optional TorchScript backend path
        return TraceableRuntimeResult(
            state_history=_simulate_torchscript(cmd_arr, x0, runtime_spec),
            backend_used="torchscript",
            compiled=True,
        )

    return TraceableRuntimeResult(
        state_history=_simulate_numpy(cmd_arr, x0, runtime_spec),
        backend_used="numpy",
        compiled=False,
    )


def run_traceable_control_batch(
    commands: FloatArray,
    *,
    initial_state: FloatArray | float | None = None,
    spec: TraceableRuntimeSpec | None = None,
    backend: str = "auto",
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> TraceableRuntimeBatchResult:
    """
    Run batched reduced control loops with optional JAX/TorchScript backends.

    `commands` shape: (batch, steps)
    """
    cmd_arr = np.asarray(commands, dtype=np.float64)
    _validate_batch_commands(cmd_arr)

    batch = int(cmd_arr.shape[0])
    if initial_state is None:
        x0 = np.zeros(batch, dtype=np.float64)
    else:
        arr = np.asarray(initial_state, dtype=np.float64)
        if arr.ndim == 0:
            x0 = np.full(batch, float(arr), dtype=np.float64)
        else:
            x0 = arr.reshape(-1)
        if x0.size != batch:
            raise ValueError("initial_state length must match commands batch dimension.")
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must contain only finite values.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)
    b = _resolve_backend(
        backend,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
    )

    if b == "jax":
        return TraceableRuntimeBatchResult(
            state_history=_simulate_jax_batch(cmd_arr, x0, runtime_spec),
            backend_used="jax",
            compiled=True,
        )
    if b == "torchscript":  # pragma: no cover - optional TorchScript backend path
        return TraceableRuntimeBatchResult(
            state_history=_simulate_torchscript_batch(cmd_arr, x0, runtime_spec),
            backend_used="torchscript",
            compiled=True,
        )
    return TraceableRuntimeBatchResult(
        state_history=_simulate_numpy_batch(cmd_arr, x0, runtime_spec),
        backend_used="numpy",
        compiled=False,
    )


def validate_traceable_backend_parity(
    *,
    steps: int = 64,
    batch: int = 8,
    seed: int = 42,
    spec: TraceableRuntimeSpec | None = None,
    atol: float = 1e-8,
    backends: list[str] | tuple[str, ...] | None = None,
) -> dict[str, TraceableBackendParityReport]:
    """Compare available compiled backends to NumPy for single and batch rollouts."""
    steps_i = _require_positive_int("steps", steps)
    batch_i = _require_positive_int("batch", batch)
    seed_i = _require_int("seed", seed)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and >= 0.")

    runtime_spec = spec if spec is not None else TraceableRuntimeSpec()
    _validate_spec(runtime_spec)

    rng = np.random.default_rng(seed_i)
    single_cmd = np.asarray(rng.normal(0.0, 1.0, size=steps_i), dtype=np.float64)
    batch_cmd = np.asarray(rng.normal(0.0, 1.0, size=(batch_i, steps_i)), dtype=np.float64)
    batch_x0 = np.asarray(rng.normal(0.0, 0.2, size=batch_i), dtype=np.float64)
    x0 = float(rng.normal(0.0, 0.2))

    ref_single = run_traceable_control_loop(
        single_cmd, initial_state=x0, spec=runtime_spec, backend="numpy"
    ).state_history
    ref_batch = run_traceable_control_batch(
        batch_cmd, initial_state=batch_x0, spec=runtime_spec, backend="numpy"
    ).state_history

    reports: dict[str, TraceableBackendParityReport] = {}
    backend_list = _resolve_backend_set(backends)
    for backend in backend_list:
        out_single = run_traceable_control_loop(
            single_cmd, initial_state=x0, spec=runtime_spec, backend=backend
        ).state_history
        out_batch = run_traceable_control_batch(
            batch_cmd, initial_state=batch_x0, spec=runtime_spec, backend=backend
        ).state_history

        s_err = float(np.max(np.abs(out_single - ref_single)))
        b_err = float(np.max(np.abs(out_batch - ref_batch)))
        reports[backend] = TraceableBackendParityReport(
            backend=backend,
            single_max_abs_err=s_err,
            batch_max_abs_err=b_err,
            single_within_tol=bool(s_err <= atol),
            batch_within_tol=bool(b_err <= atol),
        )
    return reports
