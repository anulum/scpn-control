# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust Compat
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Backward compatibility layer: imports from Rust (scpn_control_rs) if available,
falls back to pure-Python implementations.

Usage:
    from scpn_control.core._rust_compat import FusionKernel, RUST_BACKEND
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from scpn_control_rs import (
        PyEquilibriumResult,  # noqa: F401
        PyFusionKernel,
        shafranov_bv,
        solve_coil_currents,
    )

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _rust_available() -> bool:
    """Check if the Rust backend is loadable."""
    return _RUST_AVAILABLE


class RustAcceleratedKernel:
    """
    Drop-in wrapper around Rust PyFusionKernel that mirrors the Python
    FusionKernel attribute interface (.Psi, .R, .Z, .RR, .ZZ, .cfg, etc.).

    Delegates equilibrium solve to Rust for ~20x speedup while keeping
    all attribute accesses compatible with downstream code.
    """

    def __init__(self, config_path: str | os.PathLike[str]) -> None:
        self._config_path = str(config_path)
        self._rust = PyFusionKernel(self._config_path)

        import json

        with open(config_path, "r") as f:
            self.cfg = json.load(f)

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

    def sample_psi_at_probes(self, probes: list) -> np.ndarray:
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
            dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
            R_safe = np.maximum(self.RR, 1e-6)
            self.B_R = -(1.0 / R_safe) * dPsi_dZ
            self.B_Z = (1.0 / R_safe) * dPsi_dR

    def find_x_point(self, Psi: np.ndarray) -> tuple[tuple[Any, Any], Any]:
        """
        Locate the null point (B=0) using local minimization.
        Matches Python FusionKernel.find_x_point() interface.
        """
        dPsi_dR, dPsi_dZ = np.gradient(Psi, self.dR, self.dZ)
        B_mag = np.sqrt(dPsi_dR**2 + dPsi_dZ**2)

        mask_divertor = self.ZZ < (self.cfg["dimensions"]["Z_min"] * 0.5)

        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, 1e9)
            idx_min = np.argmin(masked_B)
            iz, ir = np.unravel_index(idx_min, Psi.shape)
            return (self.R[ir], self.Z[iz]), Psi[iz, ir]
        else:
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
        return "sor"

    def calculate_thermodynamics(self, p_aux_mw: float = 50.0) -> dict[str, Any]:
        """D-T fusion thermodynamics from current equilibrium (Rust backend)."""
        return dict(self._rust.calculate_thermodynamics(p_aux_mw))

    def save_results(self, filename: str = "equilibrium_nonlinear.npz") -> None:
        """Save current state to .npz file."""
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)


# ─── Public API ─────────────────────────────────────────────────────

FusionKernel: Any

if _RUST_AVAILABLE:
    FusionKernel = RustAcceleratedKernel
    RUST_BACKEND = True
else:
    RUST_BACKEND = False


# Re-export Rust-only helpers (with compatibility shims where needed)
if _RUST_AVAILABLE:

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
    from scpn_control_rs import bosch_hale_dt

    return float(bosch_hale_dt(float(t_kev)))


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
        from scpn_control_rs import PySnnPool  # type: ignore[import-untyped]

        self._inner = PySnnPool(n_neurons, gain, window_size)

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
        from scpn_control_rs import PySnnController  # type: ignore[import-untyped]

        self._inner = PySnnController(target_r, target_z)

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


class RustPIDController:
    """Rust PID controller (kp, ki, kd gains with finite-input validation).

    Parameters
    ----------
    kp, ki, kd : float
        Proportional / integral / derivative gains.
    """

    def __init__(self, kp: float, ki: float, kd: float):
        from scpn_control_rs import PyPIDController  # type: ignore[import-untyped]

        self._inner = PyPIDController(kp, ki, kd)

    @classmethod
    def radial(cls) -> "RustPIDController":
        from scpn_control_rs import PyPIDController  # type: ignore[import-untyped]

        obj = cls.__new__(cls)
        obj._inner = PyPIDController.radial()
        return obj

    @classmethod
    def vertical(cls) -> "RustPIDController":
        from scpn_control_rs import PyPIDController  # type: ignore[import-untyped]

        obj = cls.__new__(cls)
        obj._inner = PyPIDController.vertical()
        return obj

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
        return f"RustPIDController(kp={self.kp}, ki={self.ki}, kd={self.kd})"


class RustIsoFluxController:
    """Rust iso-flux controller (decoupled R + Z PID).

    Parameters
    ----------
    target_r, target_z : float
        Target R, Z position [m].
    """

    def __init__(self, target_r: float, target_z: float):
        from scpn_control_rs import PyIsoFluxController  # type: ignore[import-untyped]

        self._inner = PyIsoFluxController(target_r, target_z)

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
        return f"RustIsoFluxController(target_r={self.target_r}, target_z={self.target_z})"


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
        from scpn_control_rs import PyHInfController  # type: ignore[import-untyped]

        # 2-state VDE plant: x = [z, dz/dt]
        a = np.array([[0.0, 1.0], [gamma_growth**2, -damping]], dtype=np.float64)
        b2 = np.array([[0.0], [1.0]], dtype=np.float64)
        c2 = np.array([[1.0, 0.0]], dtype=np.float64)
        self._inner = PyHInfController(a, b2, c2, gamma, dt)
        self._u_max = u_max

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
    source: np.ndarray,
    psi_bc: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    tol: float = 1e-6,
    max_cycles: int = 500,
) -> tuple[np.ndarray, float, int, bool]:
    """Multigrid V-cycle GS solver. Uses Rust backend when available,
    falls back to FusionKernel's Python multigrid.

    Returns
    -------
    tuple of (psi, residual, n_cycles, converged)
    """
    try:
        from scpn_control_rs import multigrid_vcycle as _rust_mg  # type: ignore[import-untyped]

        result = _rust_mg(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)
        return (np.asarray(result[0]), float(result[1]), int(result[2]), bool(result[3]))
    except ImportError:
        return _python_multigrid_vcycle(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)


def _python_multigrid_vcycle(
    source: np.ndarray,
    psi_bc: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    tol: float,
    max_cycles: int,
) -> tuple[np.ndarray, float, int, bool]:
    """Pure-Python multigrid V-cycle via FusionKernel methods."""
    from scpn_control.core.fusion_kernel import FusionKernel

    source = np.ascontiguousarray(source, dtype=np.float64)
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
            from scpn_control_rs import PySPIMitigation  # type: ignore[import-untyped]

            self._inner = PySPIMitigation(w_th_mj, ip_ma, te_kev)
            self._use_rust = True
        except ImportError:
            self._use_rust = False
            self._w_th = w_th_mj * 1e6  # J
            self._ip = ip_ma * 1e6  # A
            self._te = te_kev

    def run(self) -> list[dict]:
        """Run full SPI simulation and return snapshot history."""
        if self._use_rust:
            return list(self._inner.run())

        w_th, ip, te = self._w_th, self._ip, self._te
        n_steps = int(_SPI_T_TOTAL / _SPI_DT)
        history: list[dict] = []
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
    response_matrix: np.ndarray,
    error: np.ndarray,
    gain: float = 0.8,
) -> np.ndarray:
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
        from scpn_control_rs import svd_optimal_correction as _rust_svd  # type: ignore[import-untyped]

        return np.asarray(
            _rust_svd(
                np.ascontiguousarray(response_matrix, dtype=np.float64),
                np.ascontiguousarray(error, dtype=np.float64),
                float(gain),
            )
        )
    except ImportError:
        return _python_svd_optimal_correction(response_matrix, error, gain)


def _python_svd_optimal_correction(
    response_matrix: np.ndarray,
    error: np.ndarray,
    gain: float,
    sv_cutoff: float = 1e-6,
) -> np.ndarray:
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
