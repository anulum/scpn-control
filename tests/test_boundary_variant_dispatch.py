# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Boundary Variant Dispatch Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Ensure controller flows prefer the boundary-aware ``solve()`` API."""

from __future__ import annotations

import numpy as np

from scpn_control.control import solve_kernel
from scpn_control.control.director_interface import DirectorInterface
from scpn_control.control.fusion_control_room import run_control_room
from scpn_control.control.fusion_optimal_control import OptimalController
from scpn_control.control.fusion_sota_mpc import run_sota_simulation
from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController
from scpn_control.control.tokamak_flight_sim import run_flight_sim


class _VariantAwareKernel:
    """Kernel stand-in that exposes both solve entry points."""

    last_instance: "_VariantAwareKernel | None" = None

    def __init__(self, _config_file: str) -> None:
        type(self).last_instance = self
        self.solve_calls = 0
        self.solve_equilibrium_calls = 0
        self.cfg = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
                {"name": "PF5", "current": 0.0},
            ],
        }
        self.R = np.linspace(5.8, 6.4, 25)
        self.Z = np.linspace(-0.5, 0.5, 25)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self.J_phi = np.zeros_like(self.Psi)
        self._xp = (float(self.R[-2]), float(self.Z[1]))
        self.solve()

    def _update_state(self) -> None:
        currents = [float(c["current"]) for c in self.cfg["coils"]]
        radial_drive = 0.8 * currents[2] - 0.4 * currents[1] + 0.2 * currents[3]
        vertical_drive = 0.7 * currents[4] - 0.6 * currents[0] + 0.1 * currents[2]
        center_r = 6.1 + 0.08 * np.tanh(radial_drive / 8.0)
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 8.0)
        self.Psi = 1.0 - ((self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.4) ** 2)
        self.J_phi = np.exp(-((self.RR - center_r) ** 2 + ((self.ZZ - center_z) / 1.6) ** 2))
        xr = 5.0 + 0.03 * np.tanh((currents[1] - currents[0]) / 6.0)
        xz = -3.5 + 0.03 * np.tanh((currents[4] - currents[2]) / 6.0)
        self._xp = (float(xr), float(xz))

    def solve(self) -> dict[str, object]:
        self.solve_calls += 1
        self._update_state()
        return {"boundary_variant": "free_boundary", "converged": True}

    def solve_equilibrium(self) -> None:
        self.solve_equilibrium_calls += 1
        self._update_state()

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return self._xp, 0.0


class _VariantAwareBrain:
    def step(self, error: float) -> float:
        return -0.1 * float(error)


class _VariantAwareController:
    def __init__(self, config_path: str) -> None:
        self.kernel = _VariantAwareKernel(config_path)
        self.brain_R = _VariantAwareBrain()
        self.brain_Z = _VariantAwareBrain()

    def initialize_brains(self, use_quantum: bool = True) -> None:
        return None


def _assert_dispatch_used() -> _VariantAwareKernel:
    kernel = _VariantAwareKernel.last_instance
    assert kernel is not None
    assert kernel.solve_calls > 0
    assert kernel.solve_equilibrium_calls == 0
    return kernel


def test_solve_kernel_prefers_solve_method() -> None:
    kernel = _VariantAwareKernel("dummy.json")
    kernel.solve_calls = 0
    kernel.solve_equilibrium_calls = 0

    solve_kernel(kernel)

    assert kernel.solve_calls == 1
    assert kernel.solve_equilibrium_calls == 0


def test_neuro_cybernetic_controller_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    nc = NeuroCyberneticController(
        "dummy.json",
        seed=42,
        shot_duration=4,
        kernel_factory=_VariantAwareKernel,
    )

    summary = nc.run_shot(save_plot=False, verbose=False)

    assert summary["steps"] == 4
    _assert_dispatch_used()


def test_director_interface_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    di = DirectorInterface(
        "dummy.json",
        controller_factory=_VariantAwareController,
    )

    result = di.run_directed_mission(duration=6, save_plot=False, verbose=False)

    assert result["steps"] == 6
    _assert_dispatch_used()


def test_optimal_controller_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    pilot = OptimalController(
        "dummy.json",
        kernel_factory=_VariantAwareKernel,
        verbose=False,
        coil_current_limits=(-2.0, 2.0),
        current_target_limits=(6.5, 8.5),
    )

    summary = pilot.run_optimal_shot(
        shot_steps=5,
        target_r=6.1,
        target_z=0.0,
        identify_first=True,
        save_plot=False,
    )

    assert summary["steps"] == 5
    _assert_dispatch_used()


def test_run_flight_sim_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    summary = run_flight_sim(
        config_file="dummy.json",
        shot_duration=6,
        seed=7,
        save_plot=False,
        verbose=False,
        kernel_factory=_VariantAwareKernel,
    )

    assert summary["steps"] == 6
    _assert_dispatch_used()


def test_run_sota_simulation_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    summary = run_sota_simulation(
        config_file="dummy.json",
        shot_length=6,
        prediction_horizon=3,
        save_plot=False,
        verbose=False,
        kernel_factory=_VariantAwareKernel,
        action_limit=0.25,
        coil_current_limits=(-1.0, 1.0),
    )

    assert summary["steps"] == 6
    _assert_dispatch_used()


def test_run_control_room_prefers_variant_dispatch() -> None:
    _VariantAwareKernel.last_instance = None
    summary = run_control_room(
        sim_duration=6,
        seed=11,
        save_animation=False,
        save_report=False,
        verbose=False,
        kernel_factory=_VariantAwareKernel,
        config_file="dummy.json",
    )

    assert summary["steps"] == 6
    assert summary["psi_source"] == "kernel"
    _assert_dispatch_used()
