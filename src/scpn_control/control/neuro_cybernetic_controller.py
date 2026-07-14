# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neuro Cybernetic Controller
"""Neuro-cybernetic controller primitives for symbolic and neural control loops."""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, cast

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False
import logging

import numpy as np

from scpn_control._typing import AnyFloatArray

logger = logging.getLogger(__name__)
from scpn_control.control import solve_kernel

try:
    # sc_neurocore is an optional dependency, uninstalled on the CI coverage job;
    # the success-path imports below are therefore unreachable there.
    from sc_neurocore.neurons.stochastic_lif import (
        StochasticLIFNeuron as _StochasticLIFNeuron,
    )  # pragma: no cover - optional sc-neurocore integration path
    from sc_neurocore.sources.quantum_entropy import (
        QuantumEntropySource as _QuantumEntropySource,
    )  # pragma: no cover - optional sc-neurocore integration path

    SC_NEUROCORE_AVAILABLE = True  # pragma: no cover - optional sc-neurocore integration path
    StochasticLIFNeuron: Any = _StochasticLIFNeuron
    QuantumEntropySource: Any = _QuantumEntropySource
except ImportError:  # pragma: no cover - optional dependency path
    SC_NEUROCORE_AVAILABLE = False
    StochasticLIFNeuron = None
    QuantumEntropySource = None

# ITER Physics Basis, NF 39 (1999) Table 1
SHOT_DURATION = 100  # time steps
TARGET_R = 6.2  # m, ITER major radius
TARGET_Z = 0.0  # m, midplane

_SAFETY_STATE_NORMAL = "normal"
_SAFETY_STATE_SHUTDOWN_RAMP = "safe_shutdown_ramp"
_SAFETY_STATE_SAFE_SHUTDOWN = "safe_shutdown"


def _resolve_fusion_kernel() -> Any:
    """Resolve FusionKernel lazily to keep pool-only paths dependency-light."""
    try:
        from scpn_control.core._rust_compat import (
            FusionKernel as _FusionKernel,
        )  # pragma: no cover - optional sc-neurocore integration path

        return _FusionKernel  # pragma: no cover - optional sc-neurocore integration path
    except ImportError:
        try:
            from scpn_control.core.fusion_kernel import FusionKernel as _FusionKernel

            return _FusionKernel
        except ImportError as exc:  # pragma: no cover - import-guard path
            raise ImportError(
                "Unable to import FusionKernel. Run with PYTHONPATH=src "
                "or use `python -m scpn_control.control.neuro_cybernetic_controller`."
            ) from exc


class SpikingControllerPool:
    """
    Push-pull spiking control population.

    Preferred backend is ``sc-neurocore``. If unavailable, a reduced NumPy LIF
    population is used so controller workflows remain executable in CI.
    """

    def __init__(
        self,
        n_neurons: int = 20,
        gain: float = 1.0,
        tau_window: int = 10,
        use_quantum: bool = False,
        *,
        seed: int = 42,
        allow_numpy_fallback: bool = False,
        allow_legacy_numpy_fallback: bool = False,
        dt_s: float = 1.0e-3,
        tau_mem_s: float = 15.0e-3,
        noise_std: float = 0.02,
    ) -> None:
        n_neurons = int(n_neurons)
        if n_neurons < 1:
            raise ValueError("n_neurons must be >= 1.")
        gain = float(gain)
        if not math.isfinite(gain):
            raise ValueError("gain must be finite.")
        tau_window = int(tau_window)
        if tau_window < 1:
            raise ValueError("tau_window must be >= 1.")
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0.")
        tau_mem_s = float(tau_mem_s)
        if not math.isfinite(tau_mem_s) or tau_mem_s <= 0.0:
            raise ValueError("tau_mem_s must be finite and > 0.")
        noise_std = float(noise_std)
        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std must be finite and >= 0.")

        self.n_neurons = n_neurons
        self.gain = gain
        self.window_size = tau_window
        self.use_quantum = bool(use_quantum)
        self._i_scale = 5.0
        self._i_bias = 0.1
        self.last_rate_pos = 0.0
        self.last_rate_neg = 0.0
        self.confidence = 1.0

        self.history_pos: deque[int] = deque(maxlen=self.window_size)
        self.history_neg: deque[int] = deque(maxlen=self.window_size)
        for _ in range(self.window_size):
            self.history_pos.append(0)
            self.history_neg.append(0)

        if SC_NEUROCORE_AVAILABLE:  # pragma: no cover - sc_neurocore optional dep, uninstalled on CI
            self.backend = "sc_neurocore"
            self.q_source = (
                QuantumEntropySource(n_qubits=4) if self.use_quantum and QuantumEntropySource is not None else None
            )
            self.pop_pos = [StochasticLIFNeuron(seed=i, entropy_source=self.q_source) for i in range(self.n_neurons)]
            self.pop_neg = [
                StochasticLIFNeuron(seed=i + 1000, entropy_source=self.q_source) for i in range(self.n_neurons)
            ]
            self._v_pos = None
            self._v_neg = None
            self._rng_pos = None
            self._rng_neg = None
            self._alpha = 0.0
            self._noise_std = 0.0
            self._v_threshold = 1.0
            self._v_reset = 0.0
            return

        if allow_numpy_fallback and not allow_legacy_numpy_fallback:
            raise ValueError(
                "allow_numpy_fallback=True requires allow_legacy_numpy_fallback=True; "
                "legacy NumPy fallback is disabled by default."
            )

        if not allow_numpy_fallback:
            raise RuntimeError("sc-neurocore is unavailable and allow_numpy_fallback=False.")

        self.backend = "numpy_lif"
        self.q_source = None
        self.pop_pos = []
        self.pop_neg = []
        self._rng_pos = np.random.default_rng(int(seed))
        self._rng_neg = np.random.default_rng(int(seed) + 100003)
        self._v_pos = np.zeros(self.n_neurons, dtype=np.float64)
        self._v_neg = np.zeros(self.n_neurons, dtype=np.float64)
        self._alpha = dt_s / tau_mem_s
        self._noise_std = noise_std
        # Reduced threshold keeps fallback lane responsive in low-current control
        # regimes while preserving deterministic push-pull polarity.
        self._v_threshold = 0.35
        self._v_reset = 0.0

    def _step_numpy_population(
        self,
        v: AnyFloatArray,
        rng: np.random.Generator,
        input_current: float,
    ) -> int:
        noise = rng.normal(0.0, self._noise_std, size=v.shape)
        v += self._alpha * (-v + float(input_current) + noise)
        fired = v >= self._v_threshold
        n_fired = int(np.count_nonzero(fired))
        if n_fired > 0:
            v[fired] = self._v_reset
        return n_fired

    def step(self, error_signal: float) -> float:
        """Advance the SNN populations one step and return the decoded command.

        The error is split into excitatory drives to the positive and negative
        neuron populations; the spike-rate difference decodes the command.

        Parameters
        ----------
        error_signal
            The scalar control error driving the populations.

        Returns
        -------
        float
            The decoded actuator command for this step.
        """
        input_pos = max(0.0, float(error_signal)) * self._i_scale
        input_neg = max(0.0, -float(error_signal)) * self._i_scale

        if self.backend == "sc_neurocore":  # pragma: no cover - sc_neurocore backend, uninstalled on CI
            spikes_pos = 0
            for neuron in self.pop_pos:
                if neuron.step(self._i_bias + input_pos):
                    spikes_pos += 1

            spikes_neg = 0
            for neuron in self.pop_neg:
                if neuron.step(self._i_bias + input_neg):
                    spikes_neg += 1
        else:
            assert self._v_pos is not None and self._rng_pos is not None
            assert self._v_neg is not None and self._rng_neg is not None
            spikes_pos = self._step_numpy_population(self._v_pos, self._rng_pos, self._i_bias + input_pos)
            spikes_neg = self._step_numpy_population(self._v_neg, self._rng_neg, self._i_bias + input_neg)

        self.history_pos.append(spikes_pos)
        self.history_neg.append(spikes_neg)

        self.last_rate_pos = float(sum(self.history_pos) / (self.window_size * self.n_neurons))
        self.last_rate_neg = float(sum(self.history_neg) / (self.window_size * self.n_neurons))
        self.confidence = float(
            np.clip(1.0 - abs(float(error_signal)) / 20.0 - max(self.last_rate_pos, self.last_rate_neg), 0.0, 1.0)
        )
        return float((self.last_rate_pos - self.last_rate_neg) * self.gain)


class NeuroCyberneticController:
    """Replaces PID loops with push-pull spiking populations."""

    def __init__(
        self,
        config_file: str,
        seed: int = 42,
        *,
        shot_duration: int = SHOT_DURATION,
        allow_numpy_fallback: bool = False,
        allow_legacy_numpy_fallback: bool = False,
        kernel_factory: Callable[[str], Any] | None = None,
        safety_confidence_threshold: float = 0.35,
        safe_shutdown_ramp_steps: int = 3,
    ) -> None:
        if int(shot_duration) <= 0:
            raise ValueError("shot_duration must be > 0")
        safety_confidence_threshold = float(safety_confidence_threshold)
        if not math.isfinite(safety_confidence_threshold) or not (0.0 <= safety_confidence_threshold <= 1.0):
            raise ValueError("safety_confidence_threshold must be finite in [0.0, 1.0]")
        safe_shutdown_ramp_steps = int(safe_shutdown_ramp_steps)
        if safe_shutdown_ramp_steps < 1:
            raise ValueError("safe_shutdown_ramp_steps must be >= 1")
        fusion_kernel_cls = kernel_factory if kernel_factory is not None else _resolve_fusion_kernel()
        self.kernel = fusion_kernel_cls(config_file)
        self.seed = int(seed)
        self.shot_duration = int(shot_duration)
        self.allow_numpy_fallback = bool(allow_numpy_fallback)
        self.allow_legacy_numpy_fallback = bool(allow_legacy_numpy_fallback)
        self.safety_confidence_threshold = safety_confidence_threshold
        self.safe_shutdown_ramp_steps = safe_shutdown_ramp_steps
        self.history: Dict[str, list[float]] = {}
        self.brain_R: SpikingControllerPool | None = None
        self.brain_Z: SpikingControllerPool | None = None
        self.safety_state = _SAFETY_STATE_NORMAL
        self.safety_reason = ""
        self._safe_shutdown_counter = 0
        self._safety_trigger_count = 0
        self._overflow_trap_count = 0
        self._min_confidence_r = 1.0
        self._min_confidence_z = 1.0
        self._reset_history()

    @property
    def safety_trigger_count(self) -> int:
        """Number of safety-FSM entries since the last reset."""
        return int(self._safety_trigger_count)

    @property
    def overflow_trap_count(self) -> int:
        """Number of non-finite command traps since the last reset."""
        return int(self._overflow_trap_count)

    def _reset_history(self) -> None:
        self.history = {
            "t": [],
            "Ip": [],
            "R_axis": [],
            "Z_axis": [],
            "Err_R": [],
            "Err_Z": [],
            "Control_R": [],
            "Control_Z": [],
            "Spike_Rates": [],
            "Confidence_R": [],
            "Confidence_Z": [],
            "Safety_State": [],
        }

    def _enter_safe_shutdown(self, reason: str) -> None:
        if self.safety_state == _SAFETY_STATE_SAFE_SHUTDOWN:
            return
        if self.safety_state == _SAFETY_STATE_SHUTDOWN_RAMP:
            if reason == "overflow_trap":
                self.safety_reason = reason
            return
        self.safety_state = _SAFETY_STATE_SHUTDOWN_RAMP
        self.safety_reason = reason
        self._safe_shutdown_counter = self.safe_shutdown_ramp_steps
        self._safety_trigger_count += 1

    def _safety_is_active(self) -> bool:
        return self.safety_state != _SAFETY_STATE_NORMAL

    def _safe_command(self, command: float) -> float:
        """Apply hardcoded safety-shutdown command shaping."""
        if self.safety_state == _SAFETY_STATE_NORMAL:
            return float(command)
        if self.safety_state == _SAFETY_STATE_SAFE_SHUTDOWN:
            return 0.0
        if self._safe_shutdown_counter <= 0:
            self.safety_state = _SAFETY_STATE_SAFE_SHUTDOWN
            return 0.0
        factor = float(self._safe_shutdown_counter) / float(self.safe_shutdown_ramp_steps)
        return float(command * factor)

    def _advance_safe_state(self) -> None:
        if self.safety_state != _SAFETY_STATE_SHUTDOWN_RAMP:
            return
        if self._safe_shutdown_counter > 0:
            self._safe_shutdown_counter -= 1
        if self._safe_shutdown_counter <= 0:
            self._safe_shutdown_counter = 0
            self.safety_state = _SAFETY_STATE_SAFE_SHUTDOWN

    def _reset_safe_fsm(self) -> None:
        """Reset safety override FSM to deterministic normal operation."""
        self.safety_state = _SAFETY_STATE_NORMAL
        self.safety_reason = ""
        self._safe_shutdown_counter = 0
        self._safety_trigger_count = 0
        self._overflow_trap_count = 0
        self._min_confidence_r = 1.0
        self._min_confidence_z = 1.0

    def initialize_brains(self, use_quantum: bool = False) -> None:
        """Reset and initialise the SNN populations and the safety FSM.

        Parameters
        ----------
        use_quantum
            Whether to initialise the quantum-hybrid entropy source instead of
            the classical generator.
        """
        self._reset_safe_fsm()

        self.brain_R = SpikingControllerPool(
            n_neurons=50,
            gain=10.0,
            tau_window=20,
            use_quantum=use_quantum,
            seed=self.seed + 1,
            allow_numpy_fallback=self.allow_numpy_fallback,
            allow_legacy_numpy_fallback=self.allow_legacy_numpy_fallback,
        )
        self.brain_Z = SpikingControllerPool(
            n_neurons=50,
            gain=20.0,
            tau_window=20,
            use_quantum=use_quantum,
            seed=self.seed + 2,
            allow_numpy_fallback=self.allow_numpy_fallback,
            allow_legacy_numpy_fallback=self.allow_legacy_numpy_fallback,
        )

    def run_shot(
        self,
        *,
        save_plot: bool = True,
        verbose: bool = True,
        output_path: str | None = None,
    ) -> Dict[str, Any]:
        """Run a classical-SNN control shot end to end.

        Parameters
        ----------
        save_plot
            Whether to save a history plot.
        verbose
            Whether to log progress.
        output_path
            Optional output path for the plot.

        Returns
        -------
        Dict[str, Any]
            The simulation history and summary metrics.
        """
        self.initialize_brains(use_quantum=False)
        return self._execute_simulation(
            "Neuro-Cybernetic (Classical SNN)",
            mode="classical",
            save_plot=save_plot,
            verbose=verbose,
            output_path=output_path,
        )

    def run_quantum_shot(
        self,
        *,
        save_plot: bool = True,
        verbose: bool = True,
        output_path: str | None = None,
    ) -> Dict[str, Any]:
        """Run a quantum-neuro-hybrid control shot end to end.

        Parameters
        ----------
        save_plot
            Whether to save a history plot.
        verbose
            Whether to log progress.
        output_path
            Optional output path for the plot.

        Returns
        -------
        Dict[str, Any]
            The simulation history and summary metrics.
        """
        self.initialize_brains(use_quantum=True)
        return self._execute_simulation(
            "Quantum-Neuro Hybrid (QNN)",
            mode="quantum",
            save_plot=save_plot,
            verbose=verbose,
            output_path=output_path,
        )

    def _execute_simulation(
        self,
        title: str,
        *,
        mode: str,
        save_plot: bool,
        verbose: bool,
        output_path: str | None,
    ) -> Dict[str, Any]:
        assert self.brain_R is not None and self.brain_Z is not None
        if verbose:
            logger.info(f"--- {title.upper()} PLASMA INTERFACE ---")
            logger.info("Initializing Stochastic Neural Network (SNN)...")
            logger.info(f"Neurons: {self.brain_R.n_neurons * 4} (Push-Pull Configuration)")

        self._reset_history()

        solve_kernel(self.kernel)

        physics_cfg = self.kernel.cfg.setdefault("physics", {})
        coils = self.kernel.cfg.setdefault("coils", [{} for _ in range(5)])
        while len(coils) < 5:
            coils.append({})
        for coil in coils:
            coil.setdefault("current", 0.0)

        for t in range(self.shot_duration):
            target_ip = 5.0 + (10.0 * t / self.shot_duration)
            physics_cfg["plasma_current_target"] = target_ip

            idx_max = int(np.argmax(self.kernel.Psi))
            iz, ir = np.unravel_index(idx_max, self.kernel.Psi.shape)
            curr_r = float(self.kernel.R[ir])
            curr_z = float(self.kernel.Z[iz])

            err_r = TARGET_R - curr_r
            err_z = TARGET_Z - curr_z

            ctrl_r = float(self.brain_R.step(err_r))
            ctrl_z = float(self.brain_Z.step(err_z))

            conf_r = float(self.brain_R.confidence) if self.brain_R is not None else 1.0
            conf_z = float(self.brain_Z.confidence) if self.brain_Z is not None else 1.0
            self._min_confidence_r = min(self._min_confidence_r, conf_r)
            self._min_confidence_z = min(self._min_confidence_z, conf_z)
            if conf_r < self.safety_confidence_threshold or conf_z < self.safety_confidence_threshold:
                self._enter_safe_shutdown(reason="low_confidence")

            if not math.isfinite(ctrl_r) or not math.isfinite(ctrl_z):
                self._overflow_trap_count += 1
                self._enter_safe_shutdown(reason="overflow_trap")
                ctrl_r = 0.0
                ctrl_z = 0.0

            ctrl_r = self._safe_command(ctrl_r)
            ctrl_z = self._safe_command(ctrl_z)
            self._advance_safe_state()

            coils[2]["current"] = float(coils[2]["current"]) + ctrl_r
            coils[0]["current"] = float(coils[0]["current"]) - ctrl_z
            coils[4]["current"] = float(coils[4]["current"]) + ctrl_z

            solve_kernel(self.kernel)

            self.history["t"].append(float(t))
            self.history["Ip"].append(float(target_ip))
            self.history["R_axis"].append(curr_r)
            self.history["Z_axis"].append(curr_z)
            self.history["Err_R"].append(float(err_r))
            self.history["Err_Z"].append(float(err_z))
            self.history["Control_R"].append(ctrl_r)
            self.history["Control_Z"].append(ctrl_z)
            self.history["Spike_Rates"].append(float(self.brain_R.last_rate_pos - self.brain_R.last_rate_neg))
            self.history["Confidence_R"].append(conf_r)
            self.history["Confidence_Z"].append(conf_z)
            cast(list[str], self.history["Safety_State"]).append(self.safety_state)

            if verbose:
                logger.info(
                    f"T={t}: Pos=({curr_r:.2f}, {curr_z:.2f}) | "
                    f"Err=({err_r:.3f}, {err_z:.3f}) | "
                    f"Brain_Out=({ctrl_r:.3f}, {ctrl_z:.3f})"
                )

        plot_saved = False
        plot_error: str | None = None
        if save_plot:
            try:
                self.visualize(title, output_path=output_path, verbose=verbose)
                plot_saved = True
            except (OSError, ValueError, RuntimeError) as exc:
                plot_error = str(exc)
                if verbose:
                    logger.info(f"Plot export skipped due to error: {exc}")

        err_r_arr = np.asarray(self.history["Err_R"], dtype=np.float64)
        err_z_arr = np.asarray(self.history["Err_Z"], dtype=np.float64)
        ctrl_r_arr = np.asarray(self.history["Control_R"], dtype=np.float64)
        ctrl_z_arr = np.asarray(self.history["Control_Z"], dtype=np.float64)
        summary: Dict[str, Any] = {
            "seed": self.seed,
            "steps": int(self.shot_duration),
            "mode": str(mode),
            "backend_r": self.brain_R.backend,
            "backend_z": self.brain_Z.backend,
            "final_r": float(self.history["R_axis"][-1]),
            "final_z": float(self.history["Z_axis"][-1]),
            "mean_abs_err_r": float(np.mean(np.abs(err_r_arr))),
            "mean_abs_err_z": float(np.mean(np.abs(err_z_arr))),
            "max_abs_control_r": float(np.max(np.abs(ctrl_r_arr))),
            "max_abs_control_z": float(np.max(np.abs(ctrl_z_arr))),
            "mean_spike_imbalance": float(np.mean(self.history["Spike_Rates"])),
            "min_confidence_r": float(np.min(self.history["Confidence_R"])),
            "min_confidence_z": float(np.min(self.history["Confidence_Z"])),
            "safety_state": str(self.safety_state),
            "safety_reason": str(self.safety_reason),
            "safety_trigger_count": int(self._safety_trigger_count),
            "safety_overflow_traps": int(self._overflow_trap_count),
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }
        return summary

    def visualize(
        self,
        title: str,
        *,
        output_path: str | None = None,
        verbose: bool = True,
    ) -> str:
        """Plot the control history and return the saved figure path.

        Parameters
        ----------
        title
            Plot title prefix.
        output_path
            Optional path for the saved figure.
        verbose
            Whether to log the save location.

        Returns
        -------
        str
            The path to the saved figure.

        Raises
        ------
        RuntimeError
            If matplotlib is unavailable.
        """
        if not HAS_MPL or plt is None:
            raise RuntimeError("matplotlib is required to visualize neuro-cybernetic controller history")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.set_title(f"{title} Control")
        ax1.plot(self.history["t"], self.history["R_axis"], "b-", label="R (Radial)")
        ax1.plot(self.history["t"], self.history["Z_axis"], "r-", label="Z (Vertical)")
        ax1.axhline(TARGET_R, color="b", linestyle="--", alpha=0.3)
        ax1.axhline(TARGET_Z, color="r", linestyle="--", alpha=0.3)
        ax1.set_ylabel("Position (m)")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Neural Control Activity")
        ax2.plot(self.history["t"], self.history["Control_R"], "k-", label="Radial Command")
        ax2.set_ylabel("Current Delta (A)")
        ax2.set_xlabel("Time Step")
        ax2.legend()

        filename = output_path if output_path is not None else f"{title.replace(' ', '_')}_Result.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        if verbose:
            logger.info(f"Analysis saved: {filename}")
        return filename


def run_neuro_cybernetic_control(
    *,
    config_file: str,
    shot_duration: int = SHOT_DURATION,
    seed: int = 42,
    quantum: bool = False,
    save_plot: bool = False,
    verbose: bool = False,
    output_path: str | None = None,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
    kernel_factory: Callable[[str], Any] | None = None,
    safety_confidence_threshold: float = 0.35,
    safe_shutdown_ramp_steps: int = 3,
) -> Dict[str, Any]:
    """Run neuro-cybernetic control in deterministic non-interactive mode."""
    controller = NeuroCyberneticController(
        config_file,
        seed=seed,
        shot_duration=shot_duration,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        kernel_factory=kernel_factory,
        safety_confidence_threshold=safety_confidence_threshold,
        safe_shutdown_ramp_steps=safe_shutdown_ramp_steps,
    )
    if quantum:
        return controller.run_quantum_shot(save_plot=save_plot, verbose=verbose, output_path=output_path)
    return controller.run_shot(save_plot=save_plot, verbose=verbose, output_path=output_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "iter_config.json"
    nc = NeuroCyberneticController(str(cfg))
    if len(sys.argv) > 1 and sys.argv[1] == "quantum":
        nc.run_quantum_shot()
    else:
        nc.run_shot()
