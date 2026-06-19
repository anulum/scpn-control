# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust Compat
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust Compat
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Backward compatibility layer: imports from Rust (scpn_control_rs) if available,
falls back to pure-Python implementations.

Usage:
    from scpn_control.core._rust_compat import FusionKernel, RUST_BACKEND
"""

from __future__ import annotations

import json
import os
import tempfile
import weakref
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.fusion_kernel import (
    _fusion_kernel_config_dump,
    _parse_fusion_kernel_config,
    _psi_gradient_fields,
    _psi_hessian_determinant,
    _reject_duplicate_json_keys,
    _select_x_point_index,
    _x_point_search_mask,
)

try:
    from scpn_control_rs import (
        PyEquilibriumResult,  # noqa: F401
        PyFusionKernel,
        PyUdpTransportBridge,
        shafranov_bv,
        solve_coil_currents,
    )

    _RUST_AVAILABLE = True  # pragma: no cover
except ImportError:
    _RUST_AVAILABLE = False


def _rust_available() -> bool:
    """Check if the Rust backend is loadable."""
    return _RUST_AVAILABLE


def _remove_normalised_config(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _normalise_rust_config_path(config_path: str) -> tuple[dict[str, Any], str, str | None]:
    with open(config_path, encoding="utf-8") as f:
        raw_config = json.load(f, object_pairs_hook=_reject_duplicate_json_keys)
    normalised_config = _fusion_kernel_config_dump(_parse_fusion_kernel_config(raw_config))
    physics_config_any = normalised_config.setdefault("physics", {})
    if not isinstance(physics_config_any, dict):
        raise ValueError("physics configuration must be an object")
    physics_config: dict[str, Any] = physics_config_any
    physics_config.setdefault("vacuum_permeability", 1.0)
    solver_config_any = normalised_config.setdefault("solver", {})
    if not isinstance(solver_config_any, dict):
        raise ValueError("solver configuration must be an object")
    solver_config: dict[str, Any] = solver_config_any
    solver_config.setdefault("solver_method", "sor")
    solver_config.setdefault("max_iterations", 1000)
    solver_config.setdefault("convergence_threshold", 1.0e-4)
    solver_config.setdefault("sor_omega", 1.8)
    if raw_config == normalised_config:
        return normalised_config, config_path, None

    handle, normalised_path = tempfile.mkstemp(prefix="scpn_control_rust_cfg_", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as f:
        json.dump(normalised_config, f)
    return normalised_config, normalised_path, normalised_path


class RustAcceleratedKernel:
    """
    Drop-in wrapper around Rust PyFusionKernel that mirrors the Python
    FusionKernel attribute interface (.Psi, .R, .Z, .RR, .ZZ, .cfg, etc.).

    Delegates equilibrium solve to Rust for ~20x speedup while keeping
    all attribute accesses compatible with downstream code.
    """

    def __init__(self, config_path: str | os.PathLike[str]) -> None:
        self._config_path = str(config_path)
        self._normalised_config_finalizer: Any | None = None
        self.cfg, rust_config_path, cleanup_path = _normalise_rust_config_path(self._config_path)
        if cleanup_path is not None:
            self._config_path = rust_config_path
            self._normalised_config_finalizer = weakref.finalize(self, _remove_normalised_config, cleanup_path)
        self._rust = PyFusionKernel(rust_config_path)

        self.R = np.asarray(self._rust.get_r())
        self.Z = np.asarray(self._rust.get_z())
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        self.Psi = np.asarray(self._rust.get_psi())
        try:
            self.J_phi = np.asarray(self._rust.get_j_phi())
        except AttributeError:
            self.J_phi = np.zeros((self.NZ, self.NR))

        self.B_R = np.zeros((self.NZ, self.NR))
        self.B_Z = np.zeros((self.NZ, self.NR))

    def solve_equilibrium(self) -> Any:
        """Solve Grad-Shafranov equilibrium via Rust backend."""
        result = self._rust.solve_equilibrium()

        # Sync arrays back to Python attributes
        self.Psi = np.asarray(self._rust.get_psi())
        try:
            self.J_phi = np.asarray(self._rust.get_j_phi())
        except AttributeError:
            pass

        # Compute B-field from Psi (matching Python FusionKernel.compute_b_field)
        self.compute_b_field()

        return result

    def sample_psi_at(self, r: float, z: float) -> float:
        """Interpolate psi at a single (R, Z) point via Rust bilinear interpolation."""
        return float(self._rust.sample_psi_at(float(r), float(z)))

    def sample_psi_at_probes(self, probes: list[tuple[float, float]]) -> FloatArray:
        """Interpolate psi at multiple (R, Z) probe positions.

        Parameters
        ----------
        probes : list of (float, float)
            Probe positions as (R, Z) pairs.

        Returns
        -------
        ndarray, shape (len(probes),)
        """
        if hasattr(self._rust, "sample_psi_at_probes"):
            return np.asarray(self._rust.sample_psi_at_probes(probes))
        return np.array([self.sample_psi_at(r, z) for r, z in probes])

    def compute_b_field(self) -> None:
        """Compute (B_R, B_Z) from current psi, delegating to Rust when available."""
        try:
            br, bz = self._rust.compute_b_field()
            self.B_R = np.asarray(br)
            self.B_Z = np.asarray(bz)
        except (AttributeError, RuntimeError):
            dPsi_dR, dPsi_dZ = _psi_gradient_fields(self.Psi, self.dR, self.dZ)
            R_safe = np.maximum(self.RR, 1e-6)
            self.B_R = -(1.0 / R_safe) * dPsi_dZ
            self.B_Z = (1.0 / R_safe) * dPsi_dR

    def find_x_point(self, Psi: AnyFloatArray) -> tuple[tuple[Any, Any], Any]:
        """
        Locate the null point (B=0) using local minimization.
        Matches Python FusionKernel.find_x_point() interface.
        """
        dPsi_dR, dPsi_dZ = _psi_gradient_fields(np.asarray(Psi, dtype=float), self.dR, self.dZ)
        gradient_norm = np.hypot(dPsi_dR, dPsi_dZ)
        hessian_det = _psi_hessian_determinant(np.asarray(Psi, dtype=float), self.dR, self.dZ)
        mask_divertor = _x_point_search_mask(self.ZZ, float(self.cfg["dimensions"]["Z_min"]))
        iz, ir, _ = _select_x_point_index(gradient_norm, mask_divertor, hessian_det)

        if iz >= 0 and ir >= 0:
            return (self.R[ir], self.Z[iz]), Psi[iz, ir]
        return (0, 0), np.min(Psi)

    def calculate_vacuum_field(self) -> Any:
        """Compute vacuum field via Python FusionKernel (not yet in PyO3)."""
        from scpn_control.core.fusion_kernel import FusionKernel as _PyFK

        fk = _PyFK(self._config_path)
        return fk.calculate_vacuum_field()

    def set_solver_method(self, method: str) -> None:
        """Set inner linear solver: 'sor' or 'multigrid'."""
        if hasattr(self._rust, "set_solver_method"):
            self._rust.set_solver_method(method)

    def solver_method(self) -> str:
        """Get current solver method name."""
        if hasattr(self._rust, "solver_method"):
            return str(self._rust.solver_method())
        return "sor"  # pragma: no cover

    def calculate_thermodynamics(self, p_aux_mw: float = 50.0) -> dict[str, Any]:  # pragma: no cover
        """D-T fusion thermodynamics from current equilibrium (Rust backend)."""
        return dict(self._rust.calculate_thermodynamics(p_aux_mw))

    def save_results(self, filename: str = "equilibrium_nonlinear.npz") -> None:
        """Save current state to .npz file."""
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)


# ─── Public API ─────────────────────────────────────────────────────

FusionKernel: Any

if _RUST_AVAILABLE:  # pragma: no cover
    FusionKernel = RustAcceleratedKernel
    RUST_BACKEND = True
else:
    RUST_BACKEND = False


# Re-export Rust-only helpers (with compatibility shims where needed)
if _RUST_AVAILABLE:  # pragma: no cover

    def rust_shafranov_bv(*args: Any, **kwargs: Any) -> Any:
        """Compatibility wrapper for legacy config-path invocation.

        Supported call forms:
        - rust_shafranov_bv(r_geo, a_min, ip_ma) -> tuple[float, float, float]
        - rust_shafranov_bv(config_path) -> vacuum Psi array
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], (str, os.PathLike)):
            from scpn_control.core.fusion_kernel import FusionKernel as _PyFusionKernel

            fk = _PyFusionKernel(str(args[0]))
            return fk.calculate_vacuum_field()
        return shafranov_bv(*args, **kwargs)

    rust_solve_coil_currents = solve_coil_currents

    def rust_simulate_tearing_mode(steps: int, seed: int | None = None) -> Any:
        """Python tearing-mode simulation (no Rust implementation exists)."""
        from scpn_control.control.disruption_predictor import (
            simulate_tearing_mode as _py_tearing,
        )

        if seed is None:
            return _py_tearing(steps=int(steps))
        rng = np.random.default_rng(seed=int(seed))
        return _py_tearing(steps=int(steps), rng=rng)
else:

    def rust_shafranov_bv(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("scpn_control_rs not installed. Run: maturin develop")

    def rust_solve_coil_currents(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("scpn_control_rs not installed. Run: maturin develop")

    def rust_simulate_tearing_mode(steps: int, seed: int | None = None) -> Any:
        """Python tearing-mode simulation fallback."""
        from scpn_control.control.disruption_predictor import (
            simulate_tearing_mode as _py_tearing,
        )

        if seed is None:
            return _py_tearing(steps=int(steps))
        rng = np.random.default_rng(seed=int(seed))
        return _py_tearing(steps=int(steps), rng=rng)


def rust_bosch_hale_dt(t_kev: float) -> float:
    """Bosch-Hale D-T reaction rate [m³/s] at temperature t_kev [keV]."""
    if not _RUST_AVAILABLE:
        raise ImportError("scpn_control_rs not installed. Run: maturin develop")
    from scpn_control_rs import bosch_hale_dt  # pragma: no cover

    return float(bosch_hale_dt(float(t_kev)))  # pragma: no cover


class RustSnnPool:
    """Python wrapper for Rust SpikingControllerPool (LIF neuron population).

    Falls back with ImportError if the Rust extension is not compiled.

    Parameters
    ----------
    n_neurons : int
        Number of LIF neurons per sub-population (positive/negative).
    gain : float
        Output scaling factor.
    window_size : int
        Sliding window length for rate-code averaging.
    """

    def __init__(self, n_neurons: int = 50, gain: float = 10.0, window_size: int = 20):
        from scpn_control_rs import PySnnPool  # pragma: no cover

        self._inner = PySnnPool(n_neurons, gain, window_size)  # pragma: no cover

    def step(self, error: float) -> float:
        """Process *error* through SNN pool and return scalar control output."""
        return float(self._inner.step(error))

    @property
    def n_neurons(self) -> int:
        return int(self._inner.n_neurons)

    @property
    def gain(self) -> float:
        return float(self._inner.gain)

    def __repr__(self) -> str:
        return f"RustSnnPool(n_neurons={self.n_neurons}, gain={self.gain})"


class RustSnnController:
    """Python wrapper for Rust NeuroCyberneticController (dual R+Z SNN pools).

    Falls back with ImportError if the Rust extension is not compiled.

    Parameters
    ----------
    target_r : float
        Target major-radius position [m].
    target_z : float
        Target vertical position [m].
    """

    def __init__(self, target_r: float = 6.2, target_z: float = 0.0):
        from scpn_control_rs import PySnnController  # pragma: no cover

        self._inner = PySnnController(target_r, target_z)  # pragma: no cover

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        """Process measured (R, Z) position and return (ctrl_R, ctrl_Z)."""
        r, z = self._inner.step(measured_r, measured_z)
        return float(r), float(z)

    @property
    def target_r(self) -> float:
        return float(self._inner.target_r)

    @property
    def target_z(self) -> float:
        return float(self._inner.target_z)

    def __repr__(self) -> str:
        return f"RustSnnController(target_r={self.target_r}, target_z={self.target_z})"


if _RUST_AVAILABLE:  # pragma: no cover

    class _NativeRustUdpTransportBridge:
        """Python wrapper for zero-copy Rust UDP transport publisher."""

        def __init__(
            self,
            endpoint: str = "239.0.0.1",
            port: int = 5555,
            ttl: int = 1,
            max_queue: int = 4,
            backend: str = "std",
            heartbeat_port: int = 0,
            heartbeat_timeout_ms: int = 3,
        ):
            self._inner = PyUdpTransportBridge(
                str(endpoint),
                int(port),
                int(ttl),
                int(max_queue),
                str(backend),
                int(heartbeat_port),
                int(heartbeat_timeout_ms),
            )  # pragma: no cover

        def start(self) -> None:
            """Start the background UDP publisher worker."""
            self._inner.start()

        def publish(
            self,
            r_error: float,
            z_error: float,
            r_command: float,
            z_command: float,
            acados_time_ns: int,
            snn_time_ns: int,
            status: int = 0,
        ) -> bool:
            """Publish one transport snapshot. Returns False if publisher queue is full."""
            return bool(
                self._inner.publish(
                    r_error,
                    z_error,
                    r_command,
                    z_command,
                    int(acados_time_ns),
                    int(snn_time_ns),
                    int(status),
                )
            )

        def stop(self) -> None:
            """Stop publisher thread and close the bridge."""
            self._inner.stop()

        def is_running(self) -> bool:
            return bool(self._inner.is_running())

        def stopped(self) -> bool:
            return bool(self._inner.stopped())

        def heartbeat_age_ns(self) -> int:
            return int(self._inner.heartbeat_age_ns())

        def heartbeat_expired(self) -> bool:
            return bool(self._inner.heartbeat_expired())

        def payload_bytes(self) -> int:
            return int(self._inner.payload_bytes())

        def backend(self) -> str:
            return str(self._inner.backend())

else:

    class _FallbackRustUdpTransportBridge:
        """Fallback when compiled Rust extension is unavailable."""

        def __init__(
            self,
            endpoint: str = "239.0.0.1",
            port: int = 5555,
            ttl: int = 1,
            max_queue: int = 4,
            backend: str = "std",
            heartbeat_port: int = 0,
            heartbeat_timeout_ms: int = 3,
        ):
            raise ImportError("scpn_control_rs not installed. Run: maturin develop")

        def start(self) -> None:
            raise ImportError("scpn_control_rs not installed. Run: maturin develop")

        def publish(self, *args: Any, **kwargs: Any) -> bool:
            raise ImportError("scpn_control_rs not installed. Run: maturin develop")

        def stop(self) -> None:
            raise ImportError("scpn_control_rs not installed. Run: maturin develop")

        def is_running(self) -> bool:
            return False

        def stopped(self) -> bool:
            return True

        def heartbeat_age_ns(self) -> int:
            return 0

        def heartbeat_expired(self) -> bool:
            return False

        def payload_bytes(self) -> int:
            return 0

        def backend(self) -> str:
            return ""


RustUdpTransportBridge: type[_NativeRustUdpTransportBridge] | type[_FallbackRustUdpTransportBridge]
if _RUST_AVAILABLE:  # pragma: no cover
    RustUdpTransportBridge = _NativeRustUdpTransportBridge
else:
    RustUdpTransportBridge = _FallbackRustUdpTransportBridge


class RustPIDController:
    """Rust PID controller (kp, ki, kd gains with finite-input validation).

    Parameters
    ----------
    kp, ki, kd : float
        Proportional / integral / derivative gains.
    """

    class _PurePythonPID:
        """Pure Python fallback for environments where PyPIDController is absent."""

        __slots__ = ("_kp", "_ki", "_kd", "_integral", "_prev_error")

        def __init__(self, kp: float, ki: float, kd: float) -> None:
            self._kp = float(kp)
            self._ki = float(ki)
            self._kd = float(kd)
            self._integral = 0.0
            self._prev_error = 0.0

        def step(self, error: float) -> float:
            err = float(error)
            derivative = err - self._prev_error
            self._prev_error = err
            self._integral += err
            return self._kp * err + self._ki * self._integral + self._kd * derivative

        def reset(self) -> None:
            self._integral = 0.0
            self._prev_error = 0.0

        @property
        def kp(self) -> float:
            return self._kp

        @property
        def ki(self) -> float:
            return self._ki

        @property
        def kd(self) -> float:
            return self._kd

    def __init__(self, kp: float, ki: float, kd: float):
        try:
            from scpn_control_rs import PyPIDController  # pragma: no cover

            self._inner = PyPIDController(kp, ki, kd)  # pragma: no cover
            self._mode = "rust"
        except (ImportError, AttributeError):  # pragma: no cover
            self._inner = self._PurePythonPID(kp, ki, kd)  # pragma: no cover
            self._mode = "fallback"

    @classmethod
    def radial(cls) -> "RustPIDController":
        obj = cls.__new__(cls)  # pragma: no cover
        try:
            from scpn_control_rs import PyPIDController  # pragma: no cover

            obj._inner = PyPIDController.radial()  # pragma: no cover
            obj._mode = "rust"
        except (ImportError, AttributeError):  # pragma: no cover
            obj._inner = cls._PurePythonPID(1.0, 0.1, 0.01)  # pragma: no cover
            obj._mode = "fallback"
        return obj  # pragma: no cover

    @classmethod
    def vertical(cls) -> "RustPIDController":
        obj = cls.__new__(cls)  # pragma: no cover
        try:
            from scpn_control_rs import PyPIDController  # pragma: no cover

            obj._inner = PyPIDController.vertical()  # pragma: no cover
            obj._mode = "rust"
        except (ImportError, AttributeError):  # pragma: no cover
            obj._inner = cls._PurePythonPID(1.0, 0.1, 0.01)  # pragma: no cover
            obj._mode = "fallback"
        return obj  # pragma: no cover

    def step(self, error: float) -> float:
        return float(self._inner.step(error))

    def reset(self) -> None:
        self._inner.reset()

    @property
    def kp(self) -> float:
        return float(self._inner.kp)

    @property
    def ki(self) -> float:
        return float(self._inner.ki)

    @property
    def kd(self) -> float:
        return float(self._inner.kd)

    def __repr__(self) -> str:
        return f"RustPIDController(mode={self._mode}, kp={self.kp}, ki={self.ki}, kd={self.kd})"


class RustIsoFluxController:
    """Rust iso-flux controller (decoupled R + Z PID).

    Parameters
    ----------
    target_r, target_z : float
        Target R, Z position [m].
    """

    def __init__(self, target_r: float, target_z: float):
        try:
            from scpn_control_rs import PyIsoFluxController  # pragma: no cover

            self._inner = PyIsoFluxController(target_r, target_z)  # pragma: no cover
            self._mode = "rust"  # pragma: no cover
        except (ImportError, AttributeError):  # pragma: no cover
            from scpn_control_rs import PySnnController  # pragma: no cover

            self._inner = PySnnController(float(target_r), float(target_z))  # pragma: no cover
            self._mode = "fallback"

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        r, z = self._inner.step(measured_r, measured_z)
        return float(r), float(z)

    @property
    def target_r(self) -> float:
        return float(self._inner.target_r)

    @property
    def target_z(self) -> float:
        return float(self._inner.target_z)

    def __repr__(self) -> str:
        return f"RustIsoFluxController(mode={self._mode}, target_r={self.target_r}, target_z={self.target_z})"


class RustHInfController:
    """Rust H-infinity observer-based controller for vertical stability.

    Uses LQR-approximated gains for the 2-state VDE plant
    (full DARE solver pending ndarray-linalg).

    Parameters
    ----------
    gamma_growth : float
        Unstable growth rate [1/s].
    damping : float
        Passive damping coefficient.
    gamma : float
        H-infinity performance level.
    u_max : float
        Actuator saturation limit [A].
    dt : float
        Nominal timestep [s].
    """

    def __init__(
        self,
        gamma_growth: float = 100.0,
        damping: float = 10.0,
        gamma: float = 1.0,
        u_max: float = 10.0,
        dt: float = 1e-3,
    ):
        from scpn_control_rs import PyHInfController  # pragma: no cover

        a = np.array([[0.0, 1.0], [gamma_growth**2, -damping]], dtype=np.float64)  # pragma: no cover
        b2 = np.array([[0.0], [1.0]], dtype=np.float64)  # pragma: no cover
        c2 = np.array([[1.0, 0.0]], dtype=np.float64)  # pragma: no cover
        self._inner = PyHInfController(a, b2, c2, gamma, dt)  # pragma: no cover
        self._u_max = u_max  # pragma: no cover

    def step(self, y: float, dt: float) -> float:
        """Measurement y → control u (observer-based, saturation-limited)."""
        u = self._inner.step(y, dt)
        return float(max(-self._u_max, min(self._u_max, u)))

    def reset(self) -> None:
        pass

    @property
    def gamma(self) -> float:
        return float(self._inner.gamma)

    @property
    def u_max(self) -> float:
        return float(self._u_max)

    def __repr__(self) -> str:
        return f"RustHInfController(gamma={self.gamma}, u_max={self.u_max})"


def rust_multigrid_vcycle(
    source: AnyFloatArray,
    psi_bc: AnyFloatArray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    tol: float = 1e-6,
    max_cycles: int = 500,
) -> tuple[FloatArray, float, int, bool]:
    """Multigrid V-cycle GS solver. Uses Rust backend when available,
    falls back to FusionKernel's Python multigrid.

    Returns
    -------
    tuple of (psi, residual, n_cycles, converged)
    """
    try:
        from scpn_control_rs import multigrid_vcycle as _rust_mg  # pragma: no cover

        result = _rust_mg(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)  # pragma: no cover
        return (np.asarray(result[0]), float(result[1]), int(result[2]), bool(result[3]))  # pragma: no cover
    except ImportError:
        return _python_multigrid_vcycle(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)


def _python_multigrid_vcycle(
    source: AnyFloatArray,
    psi_bc: AnyFloatArray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    tol: float,
    max_cycles: int,
) -> tuple[FloatArray, float, int, bool]:
    """Pure-Python multigrid V-cycle via FusionKernel methods."""
    from scpn_control.core.fusion_kernel import FusionKernel

    source = np.ascontiguousarray(source, dtype=np.float64).copy()
    psi = np.ascontiguousarray(psi_bc, dtype=np.float64).copy()

    R = np.linspace(r_min, r_max, nr)
    Z = np.linspace(z_min, z_max, nz)
    RR, _ = np.meshgrid(R, Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    fk = object.__new__(FusionKernel)

    residual_norm = float("inf")
    for cycle in range(max_cycles):
        psi = fk._multigrid_vcycle(psi, source, RR, dR, dZ)
        residual = fk._mg_residual(psi, source, RR, dR, dZ)
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm < tol:
            return psi, residual_norm, cycle + 1, True

    return psi, residual_norm, max_cycles, False


# SPI fallback constants — match Rust spi.rs exactly
_SPI_DT = 1e-5  # [s]
_SPI_T_TOTAL = 0.05  # [s]
_SPI_T_MIX = 0.002  # [s] assimilation cutoff
_SPI_TQ_THRESHOLD = 0.1  # [keV]
_SPI_P_RAD_COEFF = 1e10  # [W·keV^{-0.5}]
_SPI_TE_FLOOR = 0.01  # [keV]
_SPI_L_PLASMA = 1e-6  # [H]


class RustSPIMitigation:
    """SPI disruption-mitigation simulator. Uses Rust backend when available,
    falls back to pure-Python implementation matching Rust constants.

    Parameters
    ----------
    w_th_mj : float
        Initial stored thermal energy [MJ].
    ip_ma : float
        Initial plasma current [MA].
    te_kev : float
        Initial electron temperature [keV].
    """

    def __init__(self, w_th_mj: float = 300.0, ip_ma: float = 15.0, te_kev: float = 20.0):
        for name, val in [("w_th_mj", w_th_mj), ("ip_ma", ip_ma), ("te_kev", te_kev)]:
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {val}")

        try:
            from scpn_control_rs import PySPIMitigation  # pragma: no cover

            self._inner = PySPIMitigation(w_th_mj, ip_ma, te_kev)  # pragma: no cover
            self._use_rust = True  # pragma: no cover
        except ImportError:
            self._use_rust = False
            self._w_th = w_th_mj * 1e6  # J
            self._ip = ip_ma * 1e6  # A
            self._te = te_kev

    def run(self) -> list[dict[str, Any]]:
        """Run full SPI simulation and return snapshot history."""
        if self._use_rust:  # pragma: no cover
            return list(self._inner.run())

        w_th, ip, te = self._w_th, self._ip, self._te
        n_steps = int(_SPI_T_TOTAL / _SPI_DT)
        history: list[dict[str, Any]] = []
        t = 0.0
        phase = "Assimilation"

        for _ in range(n_steps):
            history.append(
                {
                    "time": t,
                    "w_th_mj": w_th / 1e6,
                    "ip_ma": ip / 1e6,
                    "te_kev": te,
                    "phase": phase,
                }
            )

            if t > _SPI_T_MIX:
                if phase == "Assimilation":
                    phase = "ThermalQuench"

                if phase == "ThermalQuench":
                    p_rad = _SPI_P_RAD_COEFF * np.sqrt(te)
                    w_old = w_th
                    w_th = max(0.0, w_th - p_rad * _SPI_DT)
                    if w_old > 0.0:
                        te = max(_SPI_TE_FLOOR, te * (w_th / w_old))
                    if te < _SPI_TQ_THRESHOLD:
                        phase = "CurrentQuench"

                if phase == "CurrentQuench":
                    r_plasma = 1e-6 / (te**1.5)
                    ip = max(0.0, ip - (r_plasma / _SPI_L_PLASMA) * ip * _SPI_DT)

            t += _SPI_DT

        return history


def rust_svd_optimal_correction(
    response_matrix: AnyFloatArray,
    error: AnyFloatArray,
    gain: float = 0.8,
) -> FloatArray:
    """SVD-based coil current correction. Uses Rust backend when available,
    falls back to NumPy SVD pseudoinverse.

    Parameters
    ----------
    response_matrix : ndarray, shape (m, n)
        Plant response Jacobian (typically 2 x n_coils).
    error : ndarray, shape (m,)
        Position error vector [R_err, Z_err].
    gain : float
        Correction gain factor.

    Returns
    -------
    ndarray, shape (n,)
        Coil current deltas.
    """
    try:
        from scpn_control_rs import svd_optimal_correction as _rust_svd  # pragma: no cover

        return np.asarray(  # pragma: no cover
            _rust_svd(
                np.ascontiguousarray(response_matrix, dtype=np.float64),
                np.ascontiguousarray(error, dtype=np.float64),
                float(gain),
            )
        )
    except ImportError:
        return _python_svd_optimal_correction(response_matrix, error, gain)


def _python_svd_optimal_correction(
    response_matrix: AnyFloatArray,
    error: AnyFloatArray,
    gain: float,
    sv_cutoff: float = 1e-6,
) -> FloatArray:
    """Truncated SVD pseudoinverse with singular value cutoff.

    Matches Rust linalg.rs svd_optimal_correction algorithm.
    """
    J = np.ascontiguousarray(response_matrix, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64).ravel()

    if J.ndim != 2:
        raise ValueError(f"response_matrix must be 2D, got {J.ndim}D")
    if e.shape[0] != J.shape[0]:
        raise ValueError(f"error length {e.shape[0]} != matrix rows {J.shape[0]}")

    u, sigma, vt = np.linalg.svd(J, full_matrices=False)
    sigma_inv = np.where(sigma > sv_cutoff, 1.0 / sigma, 0.0)
    return np.asarray(gain * (vt.T @ (sigma_inv * (u.T @ e))), dtype=np.float64)
