# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Director interface.
"""Adapter contracts for exchanging control objectives with external director services."""

from __future__ import annotations

from importlib import import_module
import logging
import re
from pathlib import Path
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
import numpy as np


def _load_fusion_kernel() -> tuple[type[Any], bool]:
    """Load the native FusionKernel when present, otherwise the Python kernel."""

    def load_python_kernel() -> type[Any]:
        try:
            fusion_kernel = import_module("scpn_control.core.fusion_kernel")
        except ImportError as exc:
            raise ImportError(
                "Unable to import FusionKernel. Run with PYTHONPATH=src "
                "or use `python -m scpn_control.control.director_interface`."
            ) from exc
        fusion_kernel_module = cast(Any, fusion_kernel)
        return cast(type[Any], fusion_kernel_module.FusionKernel)

    try:
        rust_compat = import_module("scpn_control.core._rust_compat")
    except ImportError:
        return load_python_kernel(), False
    rust_compat_module = cast(Any, rust_compat)
    if not hasattr(rust_compat_module, "FusionKernel"):
        return load_python_kernel(), False
    return cast(type[Any], rust_compat_module.FusionKernel), bool(getattr(rust_compat_module, "RUST_BACKEND", False))


FusionKernel, RUST_BACKEND = _load_fusion_kernel()

from scpn_control.control.neuro_cybernetic_controller import NeuroCyberneticController
from scpn_control.control import solve_kernel


class _RuleBasedDirector:
    """Deterministic fallback director when external DIRECTOR_AI is unavailable."""

    def __init__(self, entropy_threshold: float = 0.3, history_window: int = 10) -> None:
        entropy_threshold = float(entropy_threshold)
        if not np.isfinite(entropy_threshold) or entropy_threshold <= 0.0:
            raise ValueError("entropy_threshold must be finite and > 0.")
        history_window = int(history_window)
        if history_window < 1:
            raise ValueError("history_window must be >= 1.")
        self.entropy_threshold = entropy_threshold
        self.history_window = history_window
        self._scores: list[float] = []

    def review_action(self, prompt: str, _proposed_action: str) -> tuple[bool, float]:
        m_stability = re.search(r"Stability=([A-Za-z]+)", prompt)
        stability = m_stability.group(1) if m_stability else "Unstable"

        m_entropy = re.search(r"BrainEntropy=([0-9.]+)", prompt)
        entropy = float(m_entropy.group(1)) if m_entropy else self.entropy_threshold * 2.0

        sec_score = float(np.clip(entropy / self.entropy_threshold, 0.0, 10.0))
        self._scores.append(sec_score)
        if len(self._scores) > self.history_window:
            self._scores = self._scores[-self.history_window :]

        rolling = float(np.mean(self._scores))
        approved = stability == "Stable" and rolling <= 1.0
        return bool(approved), float(sec_score)


class DirectorInterface:
    """
    Interfaces the 'Director' (Layer 16: Coherence Oversight) with the Fusion Reactor.

    Role:
    The Director does NOT control the coils (Layer 2 does that).
    The Director controls the *Controller*. It sets the strategy and monitors for "Backfire".

    Mechanism:
    1. Sample System State (Physics + Neural Activity).
    2. Format as a "Prompt" for the Director.
    3. Director calculates Entropy/Risk.
    4. If Safe: Director updates Target Parameters.
    5. If Unsafe: Director triggers corrective action.
    """

    def __init__(
        self,
        config_path: str,
        *,
        allow_fallback: bool = False,
        allow_legacy_fallback: bool = False,
        director: Any | None = None,
        controller_factory: Callable[[str], Any] = NeuroCyberneticController,
        entropy_threshold: float = 0.3,
        history_window: int = 10,
    ) -> None:
        if allow_fallback and not allow_legacy_fallback:
            raise ValueError(
                "allow_fallback=True requires allow_legacy_fallback=True; "
                "rule-based legacy fallback is disabled by default."
            )
        self.nc = controller_factory(config_path)
        # A director is provided one of two honest ways: inject a concrete director
        # object (the integration seam — any external director, e.g. a future
        # director_ai adapter, is wired in by the caller), or opt into the built-in
        # rule-based legacy fallback. There is no implicit `director_module` import:
        # no such package exists and the real sibling (`director_ai`) does not expose
        # a `DirectorModule`, so that branch was dead and has been removed.
        if director is not None:
            self.director = director
            self.director_backend = "injected"
        elif allow_fallback:
            self.director = _RuleBasedDirector(
                entropy_threshold=float(entropy_threshold),
                history_window=int(history_window),
            )
            self.director_backend = "fallback_rule_based"
        else:
            raise ValueError(
                "Cannot initialize DirectorInterface without a director: pass `director=...` "
                "to inject one, or set allow_fallback=True (with allow_legacy_fallback=True) "
                "for the rule-based legacy director."
            )

        self.step_count = 0
        self.log: list[dict[str, float]] = []

    def format_state_for_director(
        self,
        t: int,
        ip: float,
        err_r: float,
        err_z: float,
        brain_activity: list[float],
    ) -> str:
        """
        Translate physical telemetry into a semantic prompt for the Director.
        """
        ip = float(ip)
        err_r = float(err_r)
        err_z = float(err_z)
        brain_activity_arr = np.asarray(brain_activity, dtype=np.float64)

        if not np.isfinite(ip):
            raise ValueError("ip must be finite.")
        if not np.isfinite(err_r):
            raise ValueError("err_r must be finite.")
        if not np.isfinite(err_z):
            raise ValueError("err_z must be finite.")
        if not np.all(np.isfinite(brain_activity_arr)):
            raise ValueError("brain_activity must contain finite values.")

        stability = "Stable"
        if abs(err_r) > 0.1 or abs(err_z) > 0.1:
            stability = "Unstable"
        if abs(err_r) > 0.5:
            stability = "Critical"

        neural_entropy = float(np.std(brain_activity_arr))
        return f"Time={t}, Ip={ip:.1f}, Stability={stability}, BrainEntropy={neural_entropy:.2f}"

    def run_directed_mission(
        self,
        duration: int = 100,
        *,
        use_quantum: bool = True,
        glitch_start_step: int = 50,
        glitch_std: float = 500.0,
        rng_seed: int = 42,
        save_plot: bool = True,
        output_path: str = "Director_Interface_Result.png",
        verbose: bool = True,
        allow_plot_fallback: bool = False,
        allow_legacy_plot_fallback: bool = False,
        allow_runtime_contract_fallback: bool = False,
        allow_legacy_runtime_contract_fallback: bool = False,
    ) -> dict[str, Any]:
        """Run a Director-mediated closed-loop fusion-control mission.

        Parameters
        ----------
        duration
            Number of mission steps; must be at least 1.
        use_quantum
            Whether to engage the quantum-assisted control path.
        glitch_start_step
            Step at which an injected sensor glitch begins.
        glitch_std
            Standard deviation of the injected glitch.
        rng_seed
            Random seed for reproducibility.
        save_plot
            Whether to render the mission figure.
        output_path
            File path for the rendered figure.
        verbose
            Whether to log progress.
        allow_plot_fallback, allow_legacy_plot_fallback
            Explicit opt-ins for degraded plotting paths.
        allow_runtime_contract_fallback, allow_legacy_runtime_contract_fallback
            Explicit opt-ins for degraded runtime-contract paths.

        Returns
        -------
        dict[str, Any]
            The mission summary and recorded telemetry log.

        Raises
        ------
        ValueError
            If ``duration`` is less than 1.
        """
        duration = int(duration)
        if duration < 1:
            raise ValueError("duration must be >= 1.")
        glitch_start_step = int(glitch_start_step)
        if glitch_start_step < 0:
            raise ValueError("glitch_start_step must be >= 0.")
        glitch_std = float(glitch_std)
        if not np.isfinite(glitch_std) or glitch_std < 0.0:
            raise ValueError("glitch_std must be finite and >= 0.")
        if allow_plot_fallback and not allow_legacy_plot_fallback:
            raise ValueError(
                "allow_plot_fallback=True requires allow_legacy_plot_fallback=True; "
                "legacy plot fallback is disabled by default."
            )
        if allow_runtime_contract_fallback and not allow_legacy_runtime_contract_fallback:
            raise ValueError(
                "allow_runtime_contract_fallback=True requires "
                "allow_legacy_runtime_contract_fallback=True; "
                "legacy runtime contract fallback is disabled by default."
            )
        rng = np.random.default_rng(int(rng_seed))

        if verbose:
            logger.info("--- DIRECTOR-GHOSTED FUSION MISSION ---")
            logger.info("Layer 16 (Director) is now overseeing Layer 2 (Neurocore).")
            logger.info("Director backend: %s", self.director_backend)

        solve_kernel(self.nc.kernel)
        self.nc.initialize_brains(use_quantum=bool(use_quantum))

        current_target_ip = 5.0
        self.log = []

        for t in range(duration):
            if not hasattr(self.nc.kernel, "cfg"):
                if not allow_runtime_contract_fallback:
                    raise RuntimeError(
                        "Kernel does not expose required cfg contract and legacy runtime contract fallback is disabled."
                    )
                break
            if not isinstance(self.nc.kernel.cfg, dict):
                if not allow_runtime_contract_fallback:
                    raise RuntimeError("Kernel cfg must be a mapping and legacy runtime contract fallback is disabled.")
                break
            if "physics" not in self.nc.kernel.cfg or "coils" not in self.nc.kernel.cfg:
                if not allow_runtime_contract_fallback:
                    raise RuntimeError(
                        "Kernel cfg must contain physics and coils and legacy runtime contract fallback is disabled."
                    )
                break
            if not isinstance(self.nc.kernel.cfg["coils"], list):
                if not allow_runtime_contract_fallback:
                    raise RuntimeError(
                        "Kernel cfg['coils'] must be a list and legacy runtime contract fallback is disabled."
                    )
                break
            if len(self.nc.kernel.cfg["coils"]) < 5:
                if not allow_runtime_contract_fallback:
                    raise RuntimeError(
                        "Kernel cfg['coils'] requires at least 5 channels and "
                        "legacy runtime contract fallback is disabled."
                    )
                break

            self.nc.kernel.cfg["physics"]["plasma_current_target"] = current_target_ip

            if t >= glitch_start_step and glitch_std > 0.0:
                self.nc.kernel.cfg["coils"][2]["current"] += float(rng.normal(0.0, glitch_std))

            idx_max = int(np.argmax(self.nc.kernel.Psi))
            iz, ir = np.unravel_index(idx_max, self.nc.kernel.Psi.shape)
            curr_r = float(self.nc.kernel.R[ir])
            curr_z = float(self.nc.kernel.Z[iz])

            err_r = 6.2 - curr_r
            err_z = 0.0 - curr_z

            ctrl_r = float(self.nc.brain_R.step(err_r))
            ctrl_z = float(self.nc.brain_Z.step(err_z))

            self.nc.kernel.cfg["coils"][2]["current"] += ctrl_r
            self.nc.kernel.cfg["coils"][0]["current"] -= ctrl_z
            self.nc.kernel.cfg["coils"][4]["current"] += ctrl_z

            approved = True
            if t % 5 == 0:
                brain_activity = [ctrl_r, ctrl_z]
                proposed_action = f"Increase Ip to {current_target_ip + 1.0}"
                prompt = self.format_state_for_director(t, current_target_ip, err_r, err_z, brain_activity)
                approved, sec_score = self.director.review_action(prompt, proposed_action)

                if verbose:
                    status = "APPROVED" if approved else "DENIED"
                    logger.info(
                        "[Director] T=%d | State: %s | Proposal: %s -> %s (SEC=%.2f)",
                        t,
                        prompt,
                        proposed_action,
                        status,
                        sec_score,
                    )

                if approved:
                    current_target_ip += 1.0
                else:
                    if verbose:
                        logger.info("[Director] INTERVENTION: Reducing Power to restore Coherence.")
                    current_target_ip = max(1.0, current_target_ip - 2.0)

            solve_kernel(self.nc.kernel)
            self.log.append(
                {
                    "t": float(t),
                    "Ip": float(current_target_ip),
                    "Err_R": float(err_r),
                    "Director_Intervention": float(0 if approved else 1),
                }
            )

        self.step_count = duration
        plot_error: str | None = None
        plot_saved = False
        if save_plot:
            try:
                self.visualize(output_path=output_path)
                plot_saved = True
            except (OSError, ValueError) as exc:
                if not allow_plot_fallback:
                    raise RuntimeError(
                        "Director plot export failed and legacy plot fallback is disabled. "
                        "Set allow_plot_fallback=True and allow_legacy_plot_fallback=True "
                        "for explicit degraded-mode operation."
                    ) from exc
                plot_error = f"{exc.__class__.__name__}: {exc}"

        err = np.array([x["Err_R"] for x in self.log], dtype=np.float64)
        interventions = np.array([x["Director_Intervention"] for x in self.log], dtype=np.float64)
        return {
            "backend": self.director_backend,
            "steps": int(duration),
            "final_target_ip": float(current_target_ip),
            "mean_abs_err_r": float(np.mean(np.abs(err))) if err.size > 0 else 0.0,
            "intervention_count": int(np.sum(interventions)),
            "plot_saved": bool(plot_saved),
            "plot_error": plot_error,
        }

    def visualize(self, output_path: str = "Director_Interface_Result.png") -> str:
        """Render the Director-mediated control telemetry figure.

        Parameters
        ----------
        output_path
            Destination path for the PNG figure.

        Returns
        -------
        str
            The path to the written figure.
        """
        t = [x["t"] for x in self.log]
        ip = [x["Ip"] for x in self.log]
        err = [x["Err_R"] for x in self.log]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_title("Director-Mediated Fusion Control")
        ax1.plot(t, ip, "b-", label="Plasma Current Target (Director Controlled)")
        ax1.set_ylabel("Current (MA)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(t, err, "r--", label="Radial Error (Instability)")
        ax2.set_ylabel("Error (m)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.axvline(50, color="k", linestyle=":", label="External Disturbance")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
        plt.tight_layout()
        plt.savefig(output_path)
        return str(output_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    cfg = repo_root / "iter_config.json"
    di = DirectorInterface(str(cfg))
    di.run_directed_mission()
