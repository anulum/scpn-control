# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Advanced SOC Fusion Learning
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
import numpy as np

from scpn_control.control import normalize_bounds
from scpn_control.core._validators import (
    require_bounded_float,
    require_fraction,
    require_non_negative_float,
)

# --- ADVANCED PHYSICS PARAMETERS ---
L = 60
TIME_STEPS = 10000
# Diamond & Hahm, Phys. Plasmas 2, 3640 (1995): critical gradient for SOC avalanche
Z_CRIT_BASE = 6.0
# Kim & Diamond, Phys. Rev. Lett. 90, 185006 (2003): predator-prey zonal flow coupling
FLOW_GENERATION = 0.2
FLOW_DAMPING = 0.05
# Biglari, Diamond & Terry, Phys. Fluids B 2, 1 (1990): shear suppression of turbulence
SHEAR_EFFICIENCY = 3.0

# --- Q-LEARNING PARAMETERS ---
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1
N_STATES_TURB = 5
N_STATES_FLOW = 5
N_ACTIONS = 3


class CoupledSandpileReactor:
    """
    Predator-prey sandpile approximation for turbulence/flow coupling.
    """

    def __init__(
        self,
        size: int = L,
        *,
        z_crit_base: float = Z_CRIT_BASE,
        flow_generation: float = FLOW_GENERATION,
        flow_damping: float = FLOW_DAMPING,
        shear_efficiency: float = SHEAR_EFFICIENCY,
        max_sub_steps: int = 50,
        flow_bounds: Tuple[float, float] = (0.0, 5.0),
    ) -> None:
        size = int(size)
        if size < 8:
            raise ValueError("size must be >= 8.")
        self.size = size
        self.z_crit_base = float(z_crit_base)
        flow_generation = require_non_negative_float("flow_generation", flow_generation)
        flow_damping = require_bounded_float("flow_damping", flow_damping, low=0.0, high=1.0, high_exclusive=True)
        shear_efficiency = require_non_negative_float("shear_efficiency", shear_efficiency)
        max_sub_steps = int(max_sub_steps)
        if max_sub_steps < 1:
            raise ValueError("max_sub_steps must be >= 1.")
        self.flow_generation = flow_generation
        self.flow_damping = flow_damping
        self.shear_efficiency = shear_efficiency
        self.max_sub_steps = max_sub_steps
        self.flow_bounds = normalize_bounds(flow_bounds, "flow_bounds")

        self.Z = np.zeros(self.size, dtype=np.float64)
        self.h = np.zeros(self.size, dtype=np.float64)
        self.flow = 0.0

    def drive(self, amount: float = 1.0) -> None:
        self.Z[0] += max(float(amount), 0.0)

    def step_physics(self, external_shear: float) -> tuple[int, float, float]:
        eff_shear = float(self.flow + float(external_shear))
        current_z_crit = float(self.z_crit_base + self.shear_efficiency * eff_shear)
        total_topple = 0

        for _ in range(self.max_sub_steps):
            sites_active = np.where(self.Z >= current_z_crit)[0]
            if sites_active.size == 0:
                break
            for i in sites_active:
                self.Z[i] -= 2.0
                if i + 1 < self.size:
                    self.Z[i + 1] += 1.0
                if i - 1 >= 0:
                    self.Z[i - 1] += 1.0
                total_topple += 1

        self.flow += (float(total_topple) * self.flow_generation) / float(self.size)
        self.flow *= 1.0 - self.flow_damping
        lo, hi = self.flow_bounds
        self.flow = float(np.clip(self.flow, lo, hi))
        return int(total_topple), float(self.flow), eff_shear

    def get_profile_energy(self) -> float:
        self.h[:] = np.cumsum(self.Z[::-1])[::-1]
        return float(self.h[0] if self.h.size else 0.0)


class FusionAIAgent:
    """
    Tabular Q-learning controller on discretized turbulence/flow states.
    """

    def __init__(
        self,
        *,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        epsilon: float = EPSILON,
        n_states_turb: int = N_STATES_TURB,
        n_states_flow: int = N_STATES_FLOW,
        n_actions: int = N_ACTIONS,
    ) -> None:
        alpha = require_fraction("alpha", alpha)
        gamma = require_fraction("gamma", gamma)
        epsilon = require_fraction("epsilon", epsilon)
        n_states_turb = int(n_states_turb)
        if n_states_turb < 1:
            raise ValueError("n_states_turb must be >= 1.")
        n_states_flow = int(n_states_flow)
        if n_states_flow < 1:
            raise ValueError("n_states_flow must be >= 1.")
        n_actions = int(n_actions)
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1.")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states_turb = n_states_turb
        self.n_states_flow = n_states_flow
        self.n_actions = n_actions
        self.q_table = np.zeros(
            (self.n_states_turb, self.n_states_flow, self.n_actions),
            dtype=np.float64,
        )
        self.total_reward = 0.0

    def discretize_state(self, turb: float, flow: float) -> tuple[int, int]:
        s_turb = min(int(np.log1p(max(float(turb), 0.0))), self.n_states_turb - 1)
        s_flow = min(int(max(float(flow), 0.0)), self.n_states_flow - 1)
        return s_turb, s_flow

    def choose_action(
        self,
        state: tuple[int, int],
        rng: np.random.Generator,
    ) -> int:
        if float(rng.random()) < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def learn(
        self,
        state: tuple[int, int],
        action: int,
        new_state: tuple[int, int],
        reward: float,
    ) -> float:
        old_q = float(self.q_table[state][int(action)])
        max_future_q = float(np.max(self.q_table[new_state]))
        new_q = old_q + self.alpha * (float(reward) + self.gamma * max_future_q - old_q)
        self.q_table[state][int(action)] = new_q
        self.total_reward += float(reward)
        return new_q


class FusionAI_Agent(FusionAIAgent):
    """Backward-compatible alias for older scripts."""


def _plot_learning(
    h_turb: np.ndarray,
    h_flow: np.ndarray,
    h_temp: np.ndarray,
    h_shear_ctrl: np.ndarray,
    q_table: np.ndarray,
    output_path: str,
) -> tuple[bool, str | None]:
    try:
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 3, 1)
        lookback = min(1000, int(h_turb.size))
        if lookback > 0:
            ax1.plot(h_turb[-lookback:], h_flow[-lookback:], "k-", alpha=0.3)
        lookback_recent = min(100, int(h_turb.size))
        if lookback_recent > 0:
            ax1.plot(
                h_turb[-lookback_recent:],
                h_flow[-lookback_recent:],
                "r-",
                linewidth=2,
                label="Last 100 steps",
            )
        ax1.set_title("Phase Space: L-H Transition")
        ax1.set_xlabel("Turbulence (Avalanche)")
        ax1.set_ylabel("Zonal Flow (Shear)")
        ax1.legend()

        ax2 = fig.add_subplot(2, 3, (2, 3))
        ax2.plot(h_temp, color="orange", label="Core Temperature")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(h_flow, color="blue", alpha=0.3, label="Internal Flow")  # type: ignore[attr-defined]
        ax2.set_title("Reactor Evolution: Temperature Growth")
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")  # type: ignore[attr-defined]

        ax3 = fig.add_subplot(2, 3, (4, 5))
        ax3.plot(h_turb, "r-", alpha=0.3, label="Turbulence")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(h_shear_ctrl, "g-", linewidth=2, label="AI Control Signal")  # type: ignore[attr-defined]
        ax3.set_title("AI Agent Response to Instability")
        ax3.legend(loc="upper left")
        ax3_twin.legend(loc="upper right")  # type: ignore[attr-defined]

        ax4 = fig.add_subplot(2, 3, 6)
        policy_map = np.argmax(q_table, axis=2)
        im = ax4.imshow(policy_map, origin="lower", cmap="viridis")
        ax4.set_title("Learned Policy (Q-Table)")
        ax4.set_xlabel("Flow State")
        ax4.set_ylabel("Turbulence State")
        plt.colorbar(im, ax=ax4, label="Action (0=Down, 2=Up)")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True, None
    except (OSError, ValueError) as exc:
        return False, str(exc)


def run_advanced_learning_sim(
    size: int = L,
    time_steps: int = TIME_STEPS,
    seed: int = 42,
    *,
    epsilon: float = EPSILON,
    noise_probability: float = 0.01,
    shear_step: float = 0.05,
    shear_bounds: Tuple[float, float] = (0.0, 1.0),
    save_plot: bool = True,
    output_path: str = "Advanced_SOC_Learning.png",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run deterministic SOC+Q-learning control simulation and return summary metrics.
    """
    steps = int(time_steps)
    if steps < 1:
        raise ValueError("time_steps must be >= 1.")
    lo_shear, hi_shear = normalize_bounds(shear_bounds, "shear_bounds")
    shear_step = require_non_negative_float("shear_step", shear_step)
    noise_probability = require_fraction("noise_probability", noise_probability)
    rng = np.random.default_rng(int(seed))

    if verbose:
        logger.info("--- SCPN MASTERPIECE: Predator-Prey Physics + Q-Learning Control ---")

    reactor = CoupledSandpileReactor(size=int(size))
    brain = FusionAIAgent(epsilon=float(epsilon))

    h_turb: list[float] = []
    h_flow: list[float] = []
    h_temp: list[float] = []
    h_shear_ctrl: list[float] = []
    h_shear_total: list[float] = []

    current_ext_shear = 0.0
    cadence = max(steps // 10, 1)
    for t in range(steps):
        if t == 0:
            turb_prev = 0.0
            flow_prev = 0.0
        else:
            turb_prev = h_turb[-1]
            flow_prev = h_flow[-1]

        state = brain.discretize_state(turb_prev, flow_prev)
        action_idx = brain.choose_action(state, rng)
        if action_idx == 0:
            current_ext_shear -= shear_step
        elif action_idx == 2:
            current_ext_shear += shear_step
        current_ext_shear = float(np.clip(current_ext_shear, lo_shear, hi_shear))

        reactor.drive()
        if float(rng.random()) < noise_probability:
            reactor.drive()

        av_size, flow_val, total_shear = reactor.step_physics(current_ext_shear)
        core_temp = reactor.get_profile_energy()

        reward = (core_temp * 0.1) - (float(av_size) * 0.5) - (current_ext_shear * 2.0)
        next_state = brain.discretize_state(float(av_size), flow_val)
        brain.learn(state, action_idx, next_state, reward)

        h_turb.append(float(av_size))
        h_flow.append(float(flow_val))
        h_temp.append(float(core_temp))
        h_shear_ctrl.append(float(current_ext_shear))
        h_shear_total.append(float(total_shear))

        if verbose and (t % cadence == 0 or t == steps - 1):
            logger.info(
                "Step %d: Temp=%.2f | Flow=%.3f | Turb=%s | AI_Shear=%.3f | Q-Avg=%.4f",
                t,
                core_temp,
                flow_val,
                av_size,
                current_ext_shear,
                float(np.mean(brain.q_table)),
            )

    turb_arr = np.asarray(h_turb, dtype=np.float64)
    flow_arr = np.asarray(h_flow, dtype=np.float64)
    temp_arr = np.asarray(h_temp, dtype=np.float64)
    shear_arr = np.asarray(h_shear_ctrl, dtype=np.float64)
    total_shear_arr = np.asarray(h_shear_total, dtype=np.float64)

    plot_saved = False
    plot_error: str | None = None
    if save_plot:
        plot_saved, plot_error = _plot_learning(
            turb_arr,
            flow_arr,
            temp_arr,
            shear_arr,
            brain.q_table,
            output_path,
        )
        if verbose and plot_saved:
            logger.info("Simulation complete. Analysis saved: %s", output_path)

    return {
        "seed": int(seed),
        "steps": int(steps),
        "final_core_temp": float(temp_arr[-1]) if temp_arr.size else 0.0,
        "final_flow": float(flow_arr[-1]) if flow_arr.size else 0.0,
        "final_external_shear": float(shear_arr[-1]) if shear_arr.size else 0.0,
        "mean_turbulence": float(np.mean(turb_arr)) if turb_arr.size else 0.0,
        "mean_flow": float(np.mean(flow_arr)) if flow_arr.size else 0.0,
        "mean_core_temp": float(np.mean(temp_arr)) if temp_arr.size else 0.0,
        "max_external_shear": float(np.max(shear_arr)) if shear_arr.size else 0.0,
        "mean_total_shear": float(np.mean(total_shear_arr)) if total_shear_arr.size else 0.0,
        "total_reward": float(brain.total_reward),
        "q_table_mean": float(np.mean(brain.q_table)),
        "q_table_max_abs": float(np.max(np.abs(brain.q_table))),
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


if __name__ == "__main__":
    run_advanced_learning_sim()
