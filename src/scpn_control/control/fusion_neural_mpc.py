# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gradient MPC and pulsed-shot admission adapter.
"""Gradient-based trajectory optimizer over a linearized surrogate.

Not a full MPC in the Rawlings-Mayne sense: no terminal cost, no
explicit state constraints, no convergence guarantee. The surrogate
is a finite-difference Jacobian linearization of the equilibrium
solver, not a neural network despite the class name.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

try:
    from scpn_control.core._rust_compat import FusionKernel
except ImportError:
    from scpn_control.core.fusion_kernel import FusionKernel

import logging

from scpn_control.control import normalize_bounds, solve_kernel
from scpn_control.control.capacitor_bank_state import CapacitorBank, PulseSpec, WaveformName
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    PulsedScenarioScheduler,
    PulsedScenarioState,
)
from scpn_control.core.magnetic_axis import require_magnetic_axis

logger = logging.getLogger(__name__)

# --- Neural-MPC parameters ---
PREDICTION_HORIZON = 10
SHOT_LENGTH = 100
PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION = "scpn-control.pulsed-mpc-decision-evidence.v1"


def _float_array_sha256(values: AnyFloatArray) -> str:
    """Return the SHA-256 of the values flattened as little-endian float64 bytes."""
    arr = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _bool_array_sha256(values: AnyFloatArray) -> str:
    """Return the SHA-256 of the values flattened as one byte (0 or 1) per element."""
    arr = np.asarray(values, dtype=np.bool_).reshape(-1).astype(np.uint8, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _evidence_digest(payload: Mapping[str, float | str | bool]) -> str:
    """Return the SHA-256 of the payload as sorted, compact, ASCII-only JSON."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class NeuralSurrogate:
    """Linearized surrogate model around current operating point."""

    def __init__(self, n_coils: int, n_state: int, verbose: bool = True) -> None:
        """Start with identity state dynamics ``A`` and a zero coil response ``B``."""
        self.verbose = bool(verbose)
        self.A = np.eye(int(n_state), dtype=np.float64)
        self.B = np.zeros((int(n_state), int(n_coils)), dtype=np.float64)

    def _log(self, message: str) -> None:
        """Log ``message`` at INFO level when the surrogate is verbose."""
        if self.verbose:
            logger.info(message)

    def train_on_kernel(self, kernel: Any, perturbation: float = 1.0) -> None:
        """Identify the surrogate response matrix by perturbing each coil.

        Parameters
        ----------
        kernel
            The Grad-Shafranov physics kernel to perturb and re-solve.
        perturbation
            Coil-current perturbation amplitude; must be finite and positive.
        """
        self._log("[neural-mpc] Training Neural Surrogate on Physics Kernel...")
        solve_kernel(kernel)
        base_state = self.get_state(kernel)
        p = float(perturbation)
        if not np.isfinite(p) or p <= 0.0:
            raise ValueError("perturbation must be finite and > 0.")

        for i in range(len(kernel.cfg["coils"])):
            old_i = float(kernel.cfg["coils"][i].get("current", 0.0))
            kernel.cfg["coils"][i]["current"] = old_i + p
            solve_kernel(kernel)
            new_state = self.get_state(kernel)
            self.B[:, i] = (new_state - base_state) / p
            kernel.cfg["coils"][i]["current"] = old_i

        solve_kernel(kernel)
        self._log("[neural-mpc] Surrogate Training Complete.")

    def get_state(self, kernel: Any) -> FloatArray:
        """Extract the shape state ``[R_axis, Z_axis, R_xpoint, Z_xpoint]`` in metres.

        Parameters
        ----------
        kernel
            The solved physics kernel.

        Returns
        -------
        FloatArray
            The magnetic-axis and X-point coordinates, shape ``(4,)``.

        Raises
        ------
        MagneticAxisError
            If the solved flux has no interior O-point.
        """
        axis = require_magnetic_axis(kernel.Psi, kernel.R, kernel.Z)
        xp_pos, _ = kernel.find_x_point(kernel.Psi)
        return np.array([axis.r, axis.z, float(xp_pos[0]), float(xp_pos[1])], dtype=np.float64)

    def predict(self, current_state: AnyFloatArray, action_delta: AnyFloatArray) -> FloatArray:
        """Predict the next shape state under a coil-current change.

        Parameters
        ----------
        current_state
            Current shape state.
        action_delta
            Coil-current increment.

        Returns
        -------
        FloatArray
            The linearly predicted next state ``state + B @ action``.
        """
        return np.asarray(current_state, dtype=np.float64) + (self.B @ np.asarray(action_delta, dtype=np.float64))


def horizon_coupling_spectral_radius(horizon: int) -> float:
    """Return the largest eigenvalue of the horizon coupling matrix ``T_H[i, j] = H - max(i, j)``.

    Under ``x_{k+1} = x_k + B u_k`` the tracking part of the MPC cost has the Hessian
    ``(B^T B) ⊗ T_H``: action ``u_i`` and ``u_j`` both move every state after
    ``max(i, j)``. ``T_H`` is the min-matrix ``min(a, b)`` on ``1..H``, whose largest
    eigenvalue is ``1 / (4 sin^2(pi / (2 (2H + 1))))``.

    Parameters
    ----------
    horizon
        Prediction horizon ``H >= 1``.

    Returns
    -------
    float
        ``lambda_max(T_H)``.
    """
    half_angle_sine = math.sin(math.pi / (2.0 * (2 * horizon + 1)))
    return 1.0 / (4.0 * half_angle_sine * half_angle_sine)


def horizon_cost_and_gradient(
    model: NeuralSurrogate,
    target: FloatArray,
    initial_state: FloatArray,
    actions: FloatArray,
    regularization: float,
) -> tuple[float, FloatArray]:
    """Return the horizon tracking cost and its exact gradient for an action sequence.

    The cost is ``sum_t 1/2 ||x_{t+1} - x*||^2 + regularization/2 ||u_t||^2`` along the
    linear surrogate rollout ``x_{t+1} = x_t + B u_t``. Because ``u_t`` moves every
    later state, the gradient is ``B^T sum_{k >= t} (x_{k+1} - x*) + regularization u_t``;
    it is accumulated backwards in time as an adjoint (costate) sum.

    Parameters
    ----------
    model
        Linear surrogate with input matrix ``B``.
    target
        Target state ``x*``.
    initial_state
        Rollout start ``x_0``.
    actions
        Action sequence, shape ``(H, n_coils)``.
    regularization
        Action penalty ``lambda >= 0``.

    Returns
    -------
    tuple[float, FloatArray]
        The cost and the gradient with the shape of ``actions``.
    """
    horizon = actions.shape[0]
    state = initial_state.copy()
    errors = np.empty((horizon, state.shape[0]), dtype=np.float64)
    for t in range(horizon):
        state = model.predict(state, actions[t])
        errors[t] = state - target
    cost = 0.5 * float(np.sum(errors * errors)) + 0.5 * regularization * float(np.sum(actions * actions))
    gradient = np.empty_like(actions)
    costate = np.zeros(state.shape[0], dtype=np.float64)
    for t in range(horizon - 1, -1, -1):
        costate = costate + errors[t]
        gradient[t] = model.B.T @ costate + regularization * actions[t]
    return cost, gradient


class ModelPredictiveController:
    """Gradient-based MPC planner over surrogate dynamics."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        target_state: AnyFloatArray,
        *,
        prediction_horizon: int = PREDICTION_HORIZON,
        learning_rate: float = 0.5,
        iterations: int = 20,
        action_limit: float = 2.0,
        action_regularization: float = 0.1,
    ) -> None:
        """Validate and store the horizon, step, iteration, clipping and penalty settings.

        Raises
        ------
        ValueError
            If the horizon or iteration count is below 1, or the learning rate,
            action limit or action penalty is not finite or out of range.
        """
        self.model = surrogate
        self.target = np.asarray(target_state, dtype=np.float64).reshape(-1)
        horizon = int(prediction_horizon)
        if horizon < 1:
            raise ValueError("prediction_horizon must be >= 1.")
        learning_rate = float(learning_rate)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0.")
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        action_limit = float(action_limit)
        if not np.isfinite(action_limit) or action_limit <= 0.0:
            raise ValueError("action_limit must be finite and > 0.")
        action_regularization = float(action_regularization)
        if not np.isfinite(action_regularization) or action_regularization < 0.0:
            raise ValueError("action_regularization must be finite and >= 0.")
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.action_limit = action_limit
        self.action_regularization = action_regularization

    def stable_step(self) -> float:
        """Return the gradient step: ``learning_rate`` capped at ``1 / L`` for the horizon cost.

        ``L = ||B||_F^2 · lambda_max(T_H) + action_regularization`` bounds the Lipschitz
        constant of the exact horizon gradient, since ``||B||_F^2 >= lambda_max(B^T B)``.
        A step of at most ``1 / L`` makes every iteration decrease the cost before clipping,
        which the full multi-step gradient needs where a one-step gradient did not.

        Returns
        -------
        float
            The step actually used by :meth:`plan_trajectory`.
        """
        lipschitz = (
            float(np.sum(self.model.B * self.model.B)) * horizon_coupling_spectral_radius(self.horizon)
            + self.action_regularization
        )
        return self.learning_rate if lipschitz <= 0.0 else min(self.learning_rate, 1.0 / lipschitz)

    def plan_trajectory(self, current_state: AnyFloatArray) -> FloatArray:
        """Plan by projected gradient descent and return the first action to apply.

        The whole action sequence over the prediction horizon is optimised against the
        linear surrogate with the exact horizon gradient
        (:func:`horizon_cost_and_gradient`), a step of :meth:`stable_step`, and
        clipping to ``±action_limit`` after each step. Receding-horizon control applies
        only the first planned action.

        Parameters
        ----------
        current_state
            The current shape state.

        Returns
        -------
        FloatArray
            The first planned action, shape ``(n_coils,)``.
        """
        n_coils = int(self.model.B.shape[1])
        planned_actions: FloatArray = np.zeros((self.horizon, n_coils), dtype=np.float64)
        state0 = np.asarray(current_state, dtype=np.float64).reshape(-1)
        step = self.stable_step()

        for _ in range(self.iterations):
            _, grads = horizon_cost_and_gradient(
                self.model, self.target, state0, planned_actions, self.action_regularization
            )
            planned_actions -= step * grads
            np.clip(planned_actions, -self.action_limit, self.action_limit, out=planned_actions)

        return np.asarray(planned_actions[0], dtype=np.float64)


@dataclass(frozen=True)
class PulsedShotMPCDecision:
    """Admitted pulsed-shot MPC action and its control-boundary rationale."""

    action: AnyFloatArray
    mpc_objective: float
    constraint_slack: float
    scheduler_state: str
    bank_feasibility: str
    reason: str
    bank_feasible: bool
    safe_action_applied: bool
    burn_components_masked: bool
    peak_current_A: float
    evidence_schema_version: str
    action_sha256: str
    safe_action_sha256: str
    burn_action_mask_sha256: str
    admission_digest: str


class PulsedShotMPCAdapter:
    """Gate an existing MPC action through pulsed-shot lifecycle and bank guards."""

    def __init__(
        self,
        nmpc: ModelPredictiveController,
        scheduler: PulsedScenarioScheduler,
        bank: CapacitorBank,
        *,
        burn_action_mask: AnyFloatArray | None = None,
        safe_action: AnyFloatArray | None = None,
        pulse_duration_s: float = 0.001,
        pulse_waveform: WaveformName = "half_sine",
        refuse_burn_when_uncharged: bool = True,
    ) -> None:
        """Bind the MPC to the pulsed-shot scheduler and capacitor bank, and validate the admission settings.

        Raises
        ------
        ValueError
            If the burn mask or safe action does not match the MPC action dimension,
            the mask selects no component, the safe action is not finite, the pulse
            duration is not finite and positive, or the waveform is unknown.
        """
        self.nmpc = nmpc
        self.scheduler = scheduler
        self.bank = bank
        n_actions = int(self.nmpc.model.B.shape[1])
        if burn_action_mask is None:
            mask = np.ones(n_actions, dtype=bool)
        else:
            mask = np.asarray(burn_action_mask, dtype=bool).reshape(-1)
        if mask.shape != (n_actions,):
            raise ValueError("burn_action_mask length must match MPC action dimension")
        if not bool(np.any(mask)):
            raise ValueError("burn_action_mask must select at least one burn action component")
        self.burn_action_mask = mask
        if safe_action is None:
            safe = np.zeros(n_actions, dtype=np.float64)
        else:
            safe = np.asarray(safe_action, dtype=np.float64).reshape(-1)
        if safe.shape != (n_actions,) or not np.all(np.isfinite(safe)):
            raise ValueError("safe_action must be finite and match MPC action dimension")
        self.safe_action = safe
        duration = float(pulse_duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("pulse_duration_s must be finite and > 0")
        if pulse_waveform not in ("rect", "half_sine", "exp_decay"):
            raise ValueError("pulse_waveform must be one of: rect, half_sine, exp_decay")
        self.pulse_duration_s = duration
        self.pulse_waveform: WaveformName = pulse_waveform
        self.refuse_burn_when_uncharged = bool(refuse_burn_when_uncharged)
        self._last_decision: PulsedShotMPCDecision | None = None

    def step(
        self,
        state: AnyFloatArray,
        ref: AnyFloatArray | None = None,
        context: object | None = None,
        *,
        pulse: PulseSpec | None = None,
    ) -> FloatArray:
        """Return an MPC action admitted by scheduler and capacitor-bank guards."""
        state_vec = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_vec.shape != self.nmpc.target.shape or not np.all(np.isfinite(state_vec)):
            raise ValueError("state must be finite and match MPC target dimension")
        target = self.nmpc.target
        if ref is not None:
            target = np.asarray(ref, dtype=np.float64).reshape(-1)
            if target.shape != self.nmpc.target.shape or not np.all(np.isfinite(target)):
                raise ValueError("ref must be finite and match MPC target dimension")

        original_target = self.nmpc.target.copy()
        try:
            if ref is not None:
                self.nmpc.target = target
            raw_action = np.asarray(self.nmpc.plan_trajectory(state_vec), dtype=np.float64).reshape(-1)
        finally:
            self.nmpc.target = original_target

        if raw_action.shape != self.safe_action.shape or not np.all(np.isfinite(raw_action)):
            raise ValueError("MPC action must be finite and match safe_action dimension")

        scheduler_state = self._scheduler_state(context)
        pulse_spec = self._pulse_from_action(raw_action, pulse)
        bank_feasible = True
        bank_reason = "not evaluated outside burn"
        constraint_slack = self._constraint_slack(pulse_spec)
        action = raw_action.copy()
        safe_action_applied = False
        burn_components_masked = False
        reason = "burn action admitted"

        if scheduler_state is not PulsedScenarioState.BURN:
            action[self.burn_action_mask] = self.safe_action[self.burn_action_mask]
            burn_components_masked = True
            reason = f"scheduler state {scheduler_state.value} masks burn action"
        elif self.refuse_burn_when_uncharged:
            if pulse_spec is None:
                bank_reason = "no burn demand"
            else:
                bank_feasible, bank_reason = self.bank.feasibility(pulse_spec)
            if not bank_feasible:
                action = self.safe_action.copy()
                safe_action_applied = True
                reason = f"bank feasibility rejected burn action: {bank_reason}"
        else:
            bank_reason = "bank guard disabled by policy"

        mpc_objective = self._objective(state_vec, raw_action, target)
        evidence = self._decision_evidence(
            action=action,
            safe_action=self.safe_action,
            burn_action_mask=self.burn_action_mask,
            mpc_objective=mpc_objective,
            constraint_slack=constraint_slack,
            scheduler_state=scheduler_state.value,
            bank_feasibility=bank_reason,
            reason=reason,
            bank_feasible=bool(bank_feasible),
            safe_action_applied=safe_action_applied,
            burn_components_masked=burn_components_masked,
            peak_current_A=0.0 if pulse_spec is None else float(pulse_spec.peak_current_A),
        )

        self._last_decision = PulsedShotMPCDecision(
            action=action.copy(),
            mpc_objective=mpc_objective,
            constraint_slack=constraint_slack,
            scheduler_state=scheduler_state.value,
            bank_feasibility=bank_reason,
            reason=reason,
            bank_feasible=bool(bank_feasible),
            safe_action_applied=safe_action_applied,
            burn_components_masked=burn_components_masked,
            peak_current_A=0.0 if pulse_spec is None else float(pulse_spec.peak_current_A),
            evidence_schema_version=str(evidence["schema_version"]),
            action_sha256=str(evidence["action_sha256"]),
            safe_action_sha256=str(evidence["safe_action_sha256"]),
            burn_action_mask_sha256=str(evidence["burn_action_mask_sha256"]),
            admission_digest=_evidence_digest(evidence),
        )
        return action

    def explain_last_decision(self) -> dict[str, float | str | bool]:
        """Return the latest scheduler, bank, and MPC admission summary."""
        if self._last_decision is None:
            raise RuntimeError("no pulsed MPC decision has been recorded")
        decision = self._last_decision
        return {
            "mpc_objective": decision.mpc_objective,
            "constraint_slack": decision.constraint_slack,
            "scheduler_state": decision.scheduler_state,
            "bank_feasibility": decision.bank_feasibility,
            "reason": decision.reason,
            "bank_feasible": decision.bank_feasible,
            "safe_action_applied": decision.safe_action_applied,
            "burn_components_masked": decision.burn_components_masked,
            "peak_current_A": decision.peak_current_A,
            "evidence_schema_version": decision.evidence_schema_version,
            "action_sha256": decision.action_sha256,
            "safe_action_sha256": decision.safe_action_sha256,
            "burn_action_mask_sha256": decision.burn_action_mask_sha256,
            "admission_digest": decision.admission_digest,
        }

    def _scheduler_state(self, context: object | None) -> PulsedScenarioState:
        """Resolve the scheduler state from an enum, string, mapping or ``state`` attribute, else the scheduler."""
        if context is None:
            return PulsedScenarioState(self.scheduler.state)
        if isinstance(context, PulsedScenarioState):
            return context
        if isinstance(context, str):
            return PulsedScenarioState(context)
        if isinstance(context, Mapping) and "state" in context:
            return PulsedScenarioState(context["state"])
        state = getattr(context, "state", None)
        if state is not None:
            return PulsedScenarioState(state)
        return PulsedScenarioState(self.scheduler.state)

    def _pulse_from_action(self, action: AnyFloatArray, pulse: PulseSpec | None) -> PulseSpec | None:
        """Return the given pulse, else one sized by the peak burn-component current, else None for zero current."""
        if pulse is not None:
            return pulse
        peak_current = float(np.max(np.abs(action[self.burn_action_mask])))
        if peak_current <= 0.0:
            return None
        return PulseSpec(
            peak_current_A=peak_current,
            duration_s=self.pulse_duration_s,
            waveform=self.pulse_waveform,
        )

    def _constraint_slack(self, pulse: PulseSpec | None) -> float:
        """Return the bank energy left after the pulse's estimated resistive loss, or all of it without a pulse."""
        if pulse is None:
            return float(self.bank.state.energy_J)
        factor = {
            "rect": 1.0,
            "half_sine": 0.5,
            "exp_decay": 0.25 * (1.0 - np.exp(-10.0)),
        }[pulse.waveform]
        estimated_loss = self.bank.spec.series_resistance_ohm * pulse.peak_current_A**2 * factor * pulse.duration_s
        return float(self.bank.state.energy_J - estimated_loss)

    def _objective(self, state: AnyFloatArray, action: AnyFloatArray, target: AnyFloatArray) -> float:
        """Return the one-step squared tracking error plus the action penalty of the proposal."""
        predicted = self.nmpc.model.predict(state, action)
        error = predicted - target
        return float(np.dot(error, error) + self.nmpc.action_regularization * np.dot(action, action))

    def _decision_evidence(
        self,
        *,
        action: AnyFloatArray,
        safe_action: AnyFloatArray,
        burn_action_mask: AnyFloatArray,
        mpc_objective: float,
        constraint_slack: float,
        scheduler_state: str,
        bank_feasibility: str,
        reason: str,
        bank_feasible: bool,
        safe_action_applied: bool,
        burn_components_masked: bool,
        peak_current_A: float,
    ) -> dict[str, float | str | bool]:
        """Validate the admission outcome and return its digest-bound decision-evidence fields.

        Raises
        ------
        ValueError
            If a numeric field is not finite or the peak current is negative.
        """
        if not np.isfinite(mpc_objective):
            raise ValueError("mpc_objective must be finite")
        if not np.isfinite(constraint_slack):
            raise ValueError("constraint_slack must be finite")
        if not np.isfinite(peak_current_A) or peak_current_A < 0.0:
            raise ValueError("peak_current_A must be finite and non-negative")
        return {
            "schema_version": PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
            "scheduler_state": scheduler_state,
            "bank_feasibility": bank_feasibility,
            "reason": reason,
            "bank_feasible": bool(bank_feasible),
            "safe_action_applied": bool(safe_action_applied),
            "burn_components_masked": bool(burn_components_masked),
            "constraint_slack": float(constraint_slack),
            "mpc_objective": float(mpc_objective),
            "peak_current_A": float(peak_current_A),
            "action_sha256": _float_array_sha256(action),
            "safe_action_sha256": _float_array_sha256(safe_action),
            "burn_action_mask_sha256": _bool_array_sha256(burn_action_mask),
        }


def _plot_telemetry(
    h_r: AnyFloatArray,
    h_z: AnyFloatArray,
    h_xr: AnyFloatArray,
    h_xz: AnyFloatArray,
    target_vec: AnyFloatArray,
    output_path: str,
) -> Tuple[bool, str | None]:
    """Plot the axis and X-point tracking histories to ``output_path``; return success and any error text."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title("MPC Axis Tracking")
        ax1.plot(h_r, label="R Axis")
        ax1.plot(h_z, label="Z Axis")
        ax1.axhline(target_vec[0], color="blue", linestyle="--", alpha=0.5, label="Target R")
        ax1.axhline(target_vec[1], color="orange", linestyle="--", alpha=0.5, label="Target Z")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("MPC Divertor (X-Point) Stabilization")
        ax2.plot(h_xr, label="X-Point R")
        ax2.plot(h_xz, label="X-Point Z")
        ax2.axhline(target_vec[2], color="blue", linestyle="--", alpha=0.5, label="Target XR")
        ax2.axhline(target_vec[3], color="orange", linestyle="--", alpha=0.5, label="Target XZ")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True, None
    except (OSError, ValueError) as exc:
        return False, str(exc)


def run_neural_mpc_simulation(
    config_file: str | None = None,
    shot_length: int = SHOT_LENGTH,
    prediction_horizon: int = PREDICTION_HORIZON,
    target_vector: AnyFloatArray | None = None,
    disturbance_start_step: int = 20,
    disturbance_per_step_ma: float = 0.1,
    current_target_bounds: Tuple[float, float] = (5.0, 16.0),
    action_limit: float = 2.0,
    coil_current_limits: Tuple[float, float] = (-40.0, 40.0),
    save_plot: bool = True,
    output_path: str = "neural_mpc_results.png",
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """Run a closed-loop surrogate-MPC shape-control simulation.

    Parameters
    ----------
    config_file
        Optional kernel configuration path; defaults to the repository
        ``iter_config.json``.
    shot_length
        Number of control steps.
    prediction_horizon
        MPC prediction horizon.
    target_vector
        Optional target shape state.
    disturbance_start_step
        Step at which the current disturbance begins.
    disturbance_per_step_ma
        Per-step current disturbance in MA.
    current_target_bounds
        Allowed plasma-current target range in MA.
    action_limit
        Per-step coil-action limit.
    coil_current_limits
        Coil-current bounds in kA-turn.
    save_plot
        Whether to save a results plot.
    output_path
        Output path for the results plot.
    verbose
        Whether to log progress.
    kernel_factory
        Factory building the physics kernel from a config path.

    Returns
    -------
    Dict[str, Any]
        The simulation history and final tracking metrics.
    """
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    lo_ip, hi_ip = normalize_bounds(current_target_bounds, "current_target_bounds")
    lo_i, hi_i = normalize_bounds(coil_current_limits, "coil_current_limits")
    steps = int(shot_length)
    if steps < 1:
        raise ValueError("shot_length must be >= 1.")
    drift_start = int(disturbance_start_step)
    if drift_start < 0:
        raise ValueError("disturbance_start_step must be >= 0.")
    drift_per = float(disturbance_per_step_ma)
    if target_vector is None:
        target_vec = np.array([6.0, 0.0, 5.0, -3.5], dtype=np.float64)
    else:
        target_vec = np.asarray(target_vector, dtype=np.float64).reshape(4)

    if verbose:
        logger.info("\n--- SCPN Fusion: Neural-MPC Hybrid Control ---")

    kernel = kernel_factory(str(config_file))
    surrogate = NeuralSurrogate(
        n_coils=len(kernel.cfg["coils"]),
        n_state=4,
        verbose=verbose,
    )
    surrogate.train_on_kernel(kernel)

    mpc = ModelPredictiveController(
        surrogate,
        target_vec,
        prediction_horizon=prediction_horizon,
        action_limit=action_limit,
    )

    h_r: list[float] = []
    h_z: list[float] = []
    h_xr: list[float] = []
    h_xz: list[float] = []
    h_error: list[float] = []
    h_action: list[float] = []
    h_coil_abs: list[float] = []

    physics_cfg = kernel.cfg.setdefault("physics", {})
    target_ip_ma = float(np.clip(physics_cfg.get("plasma_current_target", lo_ip), lo_ip, hi_ip))
    physics_cfg["plasma_current_target"] = target_ip_ma

    if verbose:
        logger.info(f"Starting {steps} step simulation with MPC Horizon={prediction_horizon}...")
    start_time = time.time()
    for t in range(steps):
        curr_state = surrogate.get_state(kernel)
        best_action = mpc.plan_trajectory(curr_state)
        h_action.append(float(np.max(np.abs(best_action))) if best_action.size else 0.0)

        for i, delta in enumerate(best_action):
            old_i = float(kernel.cfg["coils"][i].get("current", 0.0))
            kernel.cfg["coils"][i]["current"] = float(np.clip(old_i + float(delta), lo_i, hi_i))

        if t >= drift_start:
            target_ip_ma = float(np.clip(target_ip_ma + drift_per, lo_ip, hi_ip))
            physics_cfg["plasma_current_target"] = target_ip_ma

        solve_kernel(kernel)
        h_coil_abs.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            [float(c.get("current", 0.0)) for c in kernel.cfg["coils"]],
                            dtype=np.float64,
                        )
                    )
                )
            )
        )

        h_r.append(float(curr_state[0]))
        h_z.append(float(curr_state[1]))
        h_xr.append(float(curr_state[2]))
        h_xz.append(float(curr_state[3]))
        err = float(np.linalg.norm(curr_state - target_vec))
        h_error.append(err)

        if verbose and t % 10 == 0:
            logger.info(
                f"Step {t}: R={curr_state[0]:.2f}, Z={curr_state[1]:.2f} | "
                f"X-Point=({curr_state[2]:.2f},{curr_state[3]:.2f}) | Err={err:.3f}"
            )

    runtime_s = float(time.time() - start_time)
    if verbose:
        logger.info(f"Simulation finished in {runtime_s:.2f}s")

    plot_saved = False
    plot_error: str | None = None
    if save_plot:
        plot_saved, plot_error = _plot_telemetry(
            np.asarray(h_r, dtype=np.float64),
            np.asarray(h_z, dtype=np.float64),
            np.asarray(h_xr, dtype=np.float64),
            np.asarray(h_xz, dtype=np.float64),
            target_vec,
            output_path,
        )
        if verbose and plot_saved:
            logger.info(f"Neural-MPC analysis saved: {output_path}")

    return {
        "config_path": str(config_file),
        "steps": int(steps),
        "prediction_horizon": int(prediction_horizon),
        "runtime_seconds": runtime_s,
        "final_target_ip_ma": float(target_ip_ma),
        "final_r_axis": float(h_r[-1]) if h_r else 0.0,
        "final_z_axis": float(h_z[-1]) if h_z else 0.0,
        "final_xpoint_r": float(h_xr[-1]) if h_xr else 0.0,
        "final_xpoint_z": float(h_xz[-1]) if h_xz else 0.0,
        "mean_tracking_error": float(np.mean(np.asarray(h_error, dtype=np.float64))) if h_error else 0.0,
        "max_abs_action": float(np.max(np.asarray(h_action, dtype=np.float64))) if h_action else 0.0,
        "max_abs_coil_current": float(np.max(np.asarray(h_coil_abs, dtype=np.float64))) if h_coil_abs else 0.0,
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }


if __name__ == "__main__":
    run_neural_mpc_simulation()
