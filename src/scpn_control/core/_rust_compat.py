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
import os
from typing import Any, Optional

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


def _rust_available():
    """Check if the Rust backend is loadable."""
    return _RUST_AVAILABLE


class RustAcceleratedKernel:
    """
    Drop-in wrapper around Rust PyFusionKernel that mirrors the Python
    FusionKernel attribute interface (.Psi, .R, .Z, .RR, .ZZ, .cfg, etc.).

    Delegates equilibrium solve to Rust for ~20x speedup while keeping
    all attribute accesses compatible with downstream code.
    """

    def __init__(self, config_path):
        self._config_path = str(config_path)
        # Load via Rust (PyO3 expects str, not Path)
        self._rust = PyFusionKernel(self._config_path)

        # Also load JSON config for attribute access (bridges read .cfg directly)
        import json
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

        # Mirror grid attributes
        nr, nz = self._rust.grid_shape()
        self.NR = nr
        self.NZ = nz
        self.R = np.asarray(self._rust.get_r())
        self.Z = np.asarray(self._rust.get_z())
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        # Initialize Psi and J_phi from Rust state
        self.Psi = np.asarray(self._rust.get_psi())
        self.J_phi = np.asarray(self._rust.get_j_phi())

        # B-field placeholders (computed after solve)
        self.B_R = np.zeros((self.NZ, self.NR))
        self.B_Z = np.zeros((self.NZ, self.NR))

    def solve_equilibrium(self):
        """Solve Grad-Shafranov equilibrium via Rust backend."""
        result = self._rust.solve_equilibrium()

        # Sync arrays back to Python attributes
        self.Psi = np.asarray(self._rust.get_psi())
        self.J_phi = np.asarray(self._rust.get_j_phi())

        # Compute B-field from Psi (matching Python FusionKernel.compute_b_field)
        self.compute_b_field()

        return result

    def compute_b_field(self):
        """Compute magnetic field components from Psi gradient."""
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z = (1.0 / R_safe) * dPsi_dR

    def find_x_point(self, Psi):
        """
        Locate the null point (B=0) using local minimization.
        Matches Python FusionKernel.find_x_point() interface.
        """
        dPsi_dR, dPsi_dZ = np.gradient(Psi, self.dR, self.dZ)
        B_mag = np.sqrt(dPsi_dR**2 + dPsi_dZ**2)

        mask_divertor = self.ZZ < (self.cfg['dimensions']['Z_min'] * 0.5)

        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, 1e9)
            idx_min = np.argmin(masked_B)
            iz, ir = np.unravel_index(idx_min, Psi.shape)
            return (self.R[ir], self.Z[iz]), Psi[iz, ir]
        else:
            return (0, 0), np.min(Psi)

    def calculate_vacuum_field(self):
        """Compute vacuum field via Python FusionKernel (not yet in PyO3)."""
        from scpn_control.core.fusion_kernel import FusionKernel as _PyFK
        fk = _PyFK(self._config_path)
        return fk.calculate_vacuum_field()

    def set_solver_method(self, method: str) -> None:
        """Set inner linear solver: 'sor' or 'multigrid'."""
        self._rust.set_solver_method(method)

    def solver_method(self) -> str:
        """Get current solver method name."""
        return self._rust.solver_method()

    def calculate_thermodynamics(self, p_aux_mw: float = 50.0) -> dict:
        """D-T fusion thermodynamics from current equilibrium (Rust backend)."""
        return self._rust.calculate_thermodynamics(p_aux_mw)

    def save_results(self, filename="equilibrium_nonlinear.npz"):
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
    def rust_shafranov_bv(*args, **kwargs):
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

    def rust_simulate_tearing_mode(steps: int, seed: Optional[int] = None):
        """Python tearing-mode simulation (no Rust implementation exists)."""
        from scpn_control.control.disruption_predictor import (
            simulate_tearing_mode as _py_tearing,
        )
        if seed is None:
            return _py_tearing(steps=int(steps))
        rng = np.random.default_rng(seed=int(seed))
        return _py_tearing(steps=int(steps), rng=rng)
else:
    def rust_shafranov_bv(*args, **kwargs):
        raise ImportError("scpn_control_rs not installed. Run: maturin develop")

    def rust_solve_coil_currents(*args, **kwargs):
        raise ImportError("scpn_control_rs not installed. Run: maturin develop")

    def rust_simulate_tearing_mode(steps: int, seed: Optional[int] = None):
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
    return bosch_hale_dt(float(t_kev))


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
        return self._inner.step(error)

    @property
    def n_neurons(self) -> int:
        return self._inner.n_neurons

    @property
    def gain(self) -> float:
        return self._inner.gain

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
        return self._inner.step(measured_r, measured_z)

    @property
    def target_r(self) -> float:
        return self._inner.target_r

    @property
    def target_z(self) -> float:
        return self._inner.target_z

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
        return self._inner.step(error)

    def reset(self) -> None:
        self._inner.reset()

    @property
    def kp(self) -> float:
        return self._inner.kp

    @property
    def ki(self) -> float:
        return self._inner.ki

    @property
    def kd(self) -> float:
        return self._inner.kd

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
        return self._inner.step(measured_r, measured_z)

    @property
    def target_r(self) -> float:
        return self._inner.target_r

    @property
    def target_z(self) -> float:
        return self._inner.target_z

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

        self._inner = PyHInfController(gamma_growth, damping, gamma, u_max, dt)

    def step(self, y: float, dt: float) -> float:
        """Measurement y → control u (observer-based, saturation-limited)."""
        return self._inner.step(y, dt)

    def reset(self) -> None:
        self._inner.reset()

    @property
    def gamma(self) -> float:
        return self._inner.gamma

    @property
    def u_max(self) -> float:
        return self._inner.u_max

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
    """Call Rust multigrid V-cycle if available, else raise ImportError.

    Returns
    -------
    tuple of (psi, residual, n_cycles, converged)
    """
    from scpn_control_rs import multigrid_vcycle as _rust_mg  # type: ignore[import-untyped]
    return _rust_mg(source, psi_bc, r_min, r_max, z_min, z_max, nr, nz, tol, max_cycles)


class RustSPIMitigation:
    """Rust SPI disruption-mitigation simulator (10-20x faster than Python).

    Parameters
    ----------
    w_th_mj : float
        Initial stored thermal energy [MJ].
    ip_ma : float
        Initial plasma current [MA].
    te_kev : float
        Initial electron temperature [keV].
    """

    def __init__(
        self, w_th_mj: float = 300.0, ip_ma: float = 15.0, te_kev: float = 20.0
    ):
        from scpn_control_rs import PySPIMitigation  # type: ignore[import-untyped]

        self._inner = PySPIMitigation(w_th_mj, ip_ma, te_kev)

    def run(self) -> list[dict]:
        """Run full SPI simulation and return snapshot history."""
        return self._inner.run()


def rust_svd_optimal_correction(
    response_matrix: np.ndarray,
    error: np.ndarray,
    gain: float = 0.8,
) -> np.ndarray:
    """Rust SVD-based coil current correction (3-5x faster for 2xN).

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
    from scpn_control_rs import svd_optimal_correction as _rust_svd  # type: ignore[import-untyped]
    return _rust_svd(
        np.ascontiguousarray(response_matrix, dtype=np.float64),
        np.ascontiguousarray(error, dtype=np.float64),
        float(gain),
    )
