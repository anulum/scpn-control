"""Benchmark: FusionKernel.phase_sync_step() → SNN closed-loop hook.

Measures the full Python-side latency of:
  1. phase_sync_step() (Kuramoto + ζ sin(Ψ−θ))
  2. LIF SNN spike integration
  3. Rate decode → Ψ feedback

Run: pytest benches/bench_fusion_snn_hook.py -v
Requires: pytest-benchmark
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step

# ── Minimal LIF layer (mirrors SNN controller hot path) ──────────────

class LIFLayer:
    """Minimal LIF population for benchmarking."""

    __slots__ = ("n", "v", "v_rest", "v_th", "v_reset", "tau", "r_mem", "dt")

    def __init__(self, n: int, dt: float = 1e-3):
        self.n = n
        self.v = np.full(n, -65.0)
        self.v_rest = -65.0
        self.v_th = -55.0
        self.v_reset = -70.0
        self.tau = 10e-3
        self.r_mem = 1.0
        self.dt = dt

    def step(self, i_syn: np.ndarray) -> np.ndarray:
        dv = (-(self.v - self.v_rest) + self.r_mem * i_syn) / self.tau
        self.v += self.dt * dv
        spikes = self.v >= self.v_th
        self.v[spikes] = self.v_reset
        return spikes


def rate_to_psi(spikes: np.ndarray, nu_max: float = 100.0) -> float:
    """Decode spike rate to global phase Ψ ∈ (−π, π]."""
    nu = float(np.mean(spikes)) / 1e-3  # spikes per second (1 ms window)
    return np.pi * (2.0 * min(nu / nu_max, 1.0) - 1.0)


# ── Benchmarks ────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFusionSNNHookBench:

    @pytest.mark.benchmark(group="phase_sync_step")
    @pytest.mark.parametrize("N", [256, 1000, 4096])
    def test_phase_sync_step_only(self, benchmark, rng, N):
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 0.5, N)
        benchmark(
            kuramoto_sakaguchi_step,
            theta, omega,
            dt=1e-3, K=2.0, zeta=0.5,
            psi_driver=0.3, psi_mode="external",
        )

    @pytest.mark.benchmark(group="snn_lif_step")
    @pytest.mark.parametrize("N", [64, 256, 1024])
    def test_lif_step_only(self, benchmark, rng, N):
        lif = LIFLayer(N)
        i_syn = rng.normal(12.0, 2.0, N)
        benchmark(lif.step, i_syn)

    @pytest.mark.benchmark(group="full_closed_loop")
    @pytest.mark.parametrize("N_osc,N_lif", [(1000, 64), (4096, 256)])
    def test_full_closed_loop_single_tick(self, benchmark, rng, N_osc, N_lif):
        """Full tick: Kuramoto → LIF → rate decode → Ψ feedback."""
        theta = rng.uniform(-np.pi, np.pi, N_osc)
        omega = rng.normal(0, 0.5, N_osc)
        lif = LIFLayer(N_lif)
        psi = 0.0

        def closed_loop_tick():
            nonlocal theta, psi
            # Kuramoto step with current Ψ
            out = kuramoto_sakaguchi_step(
                theta, omega, dt=1e-3, K=2.0, zeta=0.5,
                psi_driver=psi, psi_mode="external",
            )
            theta = out["theta1"]
            R = out["R"]
            psi_r = out["Psi_r"]

            # Inject Kuramoto coherence into SNN synaptic current
            i_syn = 10.0 + 5.0 * R * np.cos(
                psi_r - np.linspace(0, 2 * np.pi, N_lif, endpoint=False)
            )
            spikes = lif.step(i_syn)

            # Decode spike rate → Ψ for next tick
            psi = rate_to_psi(spikes)
            return out["R"]

        benchmark(closed_loop_tick)

    @pytest.mark.benchmark(group="full_closed_loop_100step")
    @pytest.mark.parametrize("N_osc,N_lif", [(1000, 64), (4096, 256)])
    def test_full_closed_loop_100_ticks(self, benchmark, rng, N_osc, N_lif):
        """100 ticks of the closed loop (10 ms window @ 10 kHz)."""
        theta0 = rng.uniform(-np.pi, np.pi, N_osc)
        omega = rng.normal(0, 0.5, N_osc)

        def run_100():
            theta = theta0.copy()
            lif = LIFLayer(N_lif)
            psi = 0.0
            for _ in range(100):
                out = kuramoto_sakaguchi_step(
                    theta, omega, dt=1e-4, K=2.0, zeta=0.5,
                    psi_driver=psi, psi_mode="external",
                )
                theta = out["theta1"]
                i_syn = 10.0 + 5.0 * out["R"] * np.cos(
                    out["Psi_r"] - np.linspace(0, 2 * np.pi, N_lif, endpoint=False)
                )
                spikes = lif.step(i_syn)
                psi = rate_to_psi(spikes)
            return out["R"]

        benchmark(run_100)

    @pytest.mark.benchmark(group="fusion_kernel_hook")
    def test_fusion_kernel_phase_sync(self, benchmark, rng, tmp_path):
        """FusionKernel.phase_sync_step() with config-driven params."""
        cfg = {
            "reactor_name": "bench_diiid",
            "dimensions": {"R_min": 0.5, "R_max": 2.5, "Z_min": -1.5, "Z_max": 1.5},
            "grid_resolution": [9, 9],
            "coils": {"positions": [], "currents": [], "turns": []},
            "physics": {},
            "solver": {"method": "sor", "max_iterations": 10, "tol": 1e-4},
            "phase_sync": {"K": 2.0, "zeta": 0.5, "psi_mode": "external"},
        }
        cfg_path = tmp_path / "bench.json"
        cfg_path.write_text(json.dumps(cfg))
        from scpn_control.core.fusion_kernel import FusionKernel
        kernel = FusionKernel(str(cfg_path))

        N = 1000
        theta = rng.uniform(-np.pi, np.pi, N)
        omega = rng.normal(0, 0.5, N)

        benchmark(kernel.phase_sync_step, theta, omega, dt=1e-3, psi_driver=0.3)
