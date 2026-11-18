// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust gradient MPC and pulsed-shot admission adapter.
//! Model Predictive Control for tokamak shape control.
//!
//! Port of `fusion_neural_mpc.py`.
//! Linear surrogate + gradient descent over prediction horizon.

use control_types::error::{FusionError, FusionResult};
use ndarray::{Array1, Array2};
use sha2::{Digest, Sha256};

/// Prediction horizon. Python: 10.
const HORIZON: usize = 10;

/// Gradient descent iterations. Python: 20.
const GD_ITERATIONS: usize = 20;

/// Learning rate. Python: 0.5.
const LR: f64 = 0.5;

/// Regularization weight. Python: 0.1.
const LAMBDA: f64 = 0.1;

/// Action clipping. Python: 2.0.
const ACTION_CLIP: f64 = 2.0;

/// Schema version for digest-bound pulsed MPC admission evidence.
pub const PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION: &str =
    "scpn-control.pulsed-mpc-decision-evidence.v1";

/// Refuse a vector holding NaN or an infinity, naming it in the error.
fn ensure_finite_vector(name: &str, values: &Array1<f64>) -> FusionResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "mpc {name} must contain only finite values"
        )));
    }
    Ok(())
}

/// Refuse a burn mask whose length differs from the action dimension or that selects nothing.
fn ensure_mask_shape(name: &str, mask: &[bool], expected: usize) -> FusionResult<()> {
    if mask.len() != expected {
        return Err(FusionError::ConfigError(format!(
            "mpc {name} length mismatch: expected {expected}, got {}",
            mask.len()
        )));
    }
    if !mask.iter().any(|value| *value) {
        return Err(FusionError::ConfigError(format!(
            "mpc {name} must select at least one burn action component"
        )));
    }
    Ok(())
}

/// Return the canonical lower-case, underscore scheduler state, refusing unknown states.
fn normalise_scheduler_state(value: &str) -> FusionResult<String> {
    let state = value.trim().to_ascii_lowercase().replace('-', "_");
    match state.as_str() {
        "idle" | "ramp_up" | "flat_top" | "burn" | "expansion" | "dump" | "recharge"
        | "cool_down" => Ok(state),
        _ => Err(FusionError::ConfigError(format!(
            "mpc scheduler_state must be one of idle, ramp_up, flat_top, burn, expansion, dump, recharge, cool_down; got {value:?}"
        ))),
    }
}

/// SHA-256 of the values as consecutive little-endian `f64` bytes.
fn hash_f64_array(values: &Array1<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    crate::to_hex(&hasher.finalize())
}

/// SHA-256 of the mask as one byte (0 or 1) per element.
fn hash_bool_mask(values: &[bool]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update([u8::from(*value)]);
    }
    crate::to_hex(&hasher.finalize())
}

/// Feed `name`, a 0 separator, the UTF-8 `value` and a 255 terminator into the hasher.
fn update_str(hasher: &mut Sha256, name: &str, value: &str) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update(value.as_bytes());
    hasher.update([255]);
}

/// Feed `name`, a 0 separator, the value as one byte and a 255 terminator into the hasher.
fn update_bool(hasher: &mut Sha256, name: &str, value: bool) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update([u8::from(value)]);
    hasher.update([255]);
}

/// Feed `name`, a 0 separator, the little-endian `f64` bytes and a 255 terminator into the hasher.
fn update_f64(hasher: &mut Sha256, name: &str, value: f64) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update(value.to_le_bytes());
    hasher.update([255]);
}

/// SHA-256 binding every admission input and output of `decision`, field by field in a
/// fixed order with name separators. The decision's own `admission_digest` is not an input.
fn decision_evidence_digest(decision: &PulsedMpcDecision) -> String {
    let mut hasher = Sha256::new();
    update_str(
        &mut hasher,
        "schema_version",
        decision.evidence_schema_version,
    );
    update_str(&mut hasher, "scheduler_state", &decision.scheduler_state);
    update_str(&mut hasher, "bank_feasibility", &decision.bank_feasibility);
    update_str(&mut hasher, "reason", &decision.reason);
    update_bool(&mut hasher, "bank_feasible", decision.bank_feasible);
    update_bool(
        &mut hasher,
        "safe_action_applied",
        decision.safe_action_applied,
    );
    update_bool(
        &mut hasher,
        "burn_components_masked",
        decision.burn_components_masked,
    );
    update_f64(&mut hasher, "constraint_slack", decision.constraint_slack);
    update_f64(&mut hasher, "mpc_objective", decision.mpc_objective);
    update_f64(&mut hasher, "peak_current_A", decision.peak_current_a);
    update_str(&mut hasher, "action_sha256", &decision.action_sha256);
    update_str(
        &mut hasher,
        "safe_action_sha256",
        &decision.safe_action_sha256,
    );
    update_str(
        &mut hasher,
        "burn_action_mask_sha256",
        &decision.burn_action_mask_sha256,
    );
    crate::to_hex(&hasher.finalize())
}

/// Largest eigenvalue of the horizon coupling matrix `T_H[i, j] = H - max(i, j)`.
///
/// Under `x_{k+1} = x_k + B u_k` the tracking part of the MPC cost has the Hessian
/// `(B^T B) ⊗ T_H`. `T_H` is the min-matrix `min(a, b)` on `1..H`, whose largest
/// eigenvalue is `1 / (4 sin^2(pi / (2 (2H + 1))))`. Python:
/// `horizon_coupling_spectral_radius`.
pub fn horizon_coupling_spectral_radius(horizon: usize) -> f64 {
    let half_angle_sine = (std::f64::consts::PI / (2.0 * (2 * horizon + 1) as f64)).sin();
    1.0 / (4.0 * half_angle_sine * half_angle_sine)
}

/// Linear surrogate model: x_{t+1} = x_t + B·u_t.
pub struct NeuralSurrogate {
    /// Control impact matrix (n_state × n_coils).
    pub b_matrix: Array2<f64>,
}

impl NeuralSurrogate {
    /// Construct a linear surrogate from its state-by-coil impact matrix.
    pub fn new(b_matrix: Array2<f64>) -> Self {
        NeuralSurrogate { b_matrix }
    }

    /// Predict next state.
    pub fn predict(&self, state: &Array1<f64>, action: &Array1<f64>) -> FusionResult<Array1<f64>> {
        let n_state = self.b_matrix.nrows();
        let n_coils = self.b_matrix.ncols();
        if state.len() != n_state {
            return Err(FusionError::ConfigError(format!(
                "mpc state length mismatch: expected {n_state}, got {}",
                state.len()
            )));
        }
        if action.len() != n_coils {
            return Err(FusionError::ConfigError(format!(
                "mpc action length mismatch: expected {n_coils}, got {}",
                action.len()
            )));
        }
        ensure_finite_vector("state", state)?;
        ensure_finite_vector("action", action)?;
        Ok(state + &self.b_matrix.dot(action))
    }
}

/// Result of the Rust pulsed-shot MPC admission adapter.
pub struct PulsedMpcDecision {
    /// Admitted coil action vector.
    pub action: Array1<f64>,
    /// Objective value of the original MPC proposal.
    pub mpc_objective: f64,
    /// Minimum constraint slack associated with the proposal.
    pub constraint_slack: f64,
    /// Canonical pulsed-shot scheduler state.
    pub scheduler_state: String,
    /// Canonical capacitor-bank feasibility state.
    pub bank_feasibility: String,
    /// Human-readable admission or fallback reason.
    pub reason: String,
    /// Whether the capacitor-bank constraints admit the proposal.
    pub bank_feasible: bool,
    /// Whether a safe fallback action replaced the proposal.
    pub safe_action_applied: bool,
    /// Whether burn-only components were masked outside a burn state.
    pub burn_components_masked: bool,
    /// Maximum absolute coil current in amperes.
    pub peak_current_a: f64,
    /// Version of the deterministic decision-evidence schema.
    pub evidence_schema_version: &'static str,
    /// SHA-256 digest of the proposed action vector.
    pub action_sha256: String,
    /// SHA-256 digest of the configured safe action vector.
    pub safe_action_sha256: String,
    /// SHA-256 digest of the burn-component mask.
    pub burn_action_mask_sha256: String,
    /// SHA-256 digest binding all admission inputs and outputs.
    pub admission_digest: String,
}

/// Model Predictive Controller.
pub struct MPController {
    /// Linear state-transition surrogate used for rollouts.
    pub model: NeuralSurrogate,
    /// Desired state vector.
    pub target: Array1<f64>,
    /// Number of control steps in each optimization rollout.
    pub horizon: usize,
}

impl MPController {
    /// Validate model and target dimensions and construct an MPC controller.
    pub fn new(model: NeuralSurrogate, target: Array1<f64>) -> FusionResult<Self> {
        let n_state = model.b_matrix.nrows();
        let n_coils = model.b_matrix.ncols();
        if n_state == 0 || n_coils == 0 {
            return Err(FusionError::ConfigError(
                "mpc b_matrix must have non-zero state and coil dimensions".to_string(),
            ));
        }
        if target.len() != n_state {
            return Err(FusionError::ConfigError(format!(
                "mpc target length mismatch: expected {n_state}, got {}",
                target.len()
            )));
        }
        if model.b_matrix.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "mpc b_matrix must contain only finite values".to_string(),
            ));
        }
        if target.iter().any(|v| !v.is_finite()) {
            return Err(FusionError::ConfigError(
                "mpc target must contain only finite values".to_string(),
            ));
        }
        Ok(MPController {
            model,
            target,
            horizon: HORIZON,
        })
    }

    /// Return the horizon tracking cost and its exact gradient for an action sequence.
    ///
    /// The cost is `sum_t 1/2 ||x_{t+1} - x*||^2 + LAMBDA/2 ||u_t||^2` along the linear
    /// rollout `x_{t+1} = x_t + B u_t`. Because `u_t` moves every later state, the
    /// gradient is `B^T sum_{k >= t} (x_{k+1} - x*) + LAMBDA u_t`; it is accumulated
    /// backwards in time as an adjoint (costate) sum. Python:
    /// `horizon_cost_and_gradient`.
    pub fn horizon_cost_and_gradient(
        &self,
        initial_state: &Array1<f64>,
        actions: &[Array1<f64>],
    ) -> FusionResult<(f64, Vec<Array1<f64>>)> {
        let mut state = initial_state.clone();
        let mut errors: Vec<Array1<f64>> = Vec::with_capacity(actions.len());
        for action in actions {
            state = self.model.predict(&state, action)?;
            errors.push(&state - &self.target);
        }
        let tracking: f64 = errors.iter().map(|e| e.dot(e)).sum();
        let effort: f64 = actions.iter().map(|u| u.dot(u)).sum();
        let cost = 0.5 * tracking + 0.5 * LAMBDA * effort;
        let mut gradient: Vec<Array1<f64>> = Vec::with_capacity(actions.len());
        let mut costate = Array1::<f64>::zeros(state.len());
        for (error, action) in errors.iter().zip(actions.iter()).rev() {
            costate = &costate + error;
            gradient.push(self.model.b_matrix.t().dot(&costate) + action * LAMBDA);
        }
        gradient.reverse();
        Ok((cost, gradient))
    }

    /// Return the gradient step: `LR` capped at `1 / L` for the horizon cost.
    ///
    /// `L = ||B||_F^2 * lambda_max(T_H) + LAMBDA` bounds the Lipschitz constant of the
    /// exact horizon gradient, since `||B||_F^2 >= lambda_max(B^T B)`. A step of at
    /// most `1 / L` makes every iteration decrease the cost before clipping. `LAMBDA`
    /// is positive, so `L` is too. Python: `ModelPredictiveController.stable_step`.
    pub fn stable_step(&self) -> f64 {
        let frobenius_sq: f64 = self.model.b_matrix.iter().map(|v| v * v).sum();
        let lipschitz = frobenius_sq * horizon_coupling_spectral_radius(self.horizon) + LAMBDA;
        LR.min(1.0 / lipschitz)
    }

    /// Plan by projected gradient descent and return the first action to apply.
    ///
    /// The whole action sequence over the horizon is optimised with the exact horizon
    /// gradient, a step of [`Self::stable_step`], and clipping to `±ACTION_CLIP` after
    /// each step. Receding-horizon control applies only the first planned action.
    pub fn plan(&self, current_state: &Array1<f64>) -> FusionResult<Array1<f64>> {
        ensure_finite_vector("current_state", current_state)?;
        let n_coils = self.model.b_matrix.ncols();
        let mut actions: Vec<Array1<f64>> =
            (0..self.horizon).map(|_| Array1::zeros(n_coils)).collect();
        let step = self.stable_step();

        for _ in 0..GD_ITERATIONS {
            let (_, grads) = self.horizon_cost_and_gradient(current_state, &actions)?;
            for (action, grad) in actions.iter_mut().zip(grads.iter()) {
                *action = &*action - &(grad * step);
                action.mapv_inplace(|v| v.clamp(-ACTION_CLIP, ACTION_CLIP));
            }
        }

        Ok(actions[0].clone())
    }

    /// Plan action accounting for known actuation delay and sensor bias.
    ///
    /// `delayed_actions` are already-issued actions that will still take effect
    /// before the next new control is applied.
    pub fn plan_with_delay_and_bias(
        &self,
        measured_state: &Array1<f64>,
        delayed_actions: &[Array1<f64>],
        sensor_bias: Option<&Array1<f64>>,
    ) -> FusionResult<Array1<f64>> {
        ensure_finite_vector("measured_state", measured_state)?;
        let n_coils = self.model.b_matrix.ncols();
        let mut predicted_state = measured_state.clone();
        if let Some(bias) = sensor_bias {
            if bias.len() != predicted_state.len() {
                return Err(FusionError::ConfigError(format!(
                    "mpc sensor_bias length mismatch: expected {}, got {}",
                    predicted_state.len(),
                    bias.len()
                )));
            }
            ensure_finite_vector("sensor_bias", bias)?;
            predicted_state = &predicted_state + bias;
        }

        for (idx, action) in delayed_actions.iter().enumerate() {
            if action.len() != n_coils {
                return Err(FusionError::ConfigError(format!(
                    "mpc delayed action[{idx}] length mismatch: expected {n_coils}, got {}",
                    action.len()
                )));
            }
            if action.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(format!(
                    "mpc delayed action[{idx}] must contain only finite values"
                )));
            }
            predicted_state = self.model.predict(&predicted_state, action)?;
        }

        self.plan(&predicted_state)
    }

    /// Delay-aware MPC rollout with first-order actuator lag.
    ///
    /// This provides a reduced DDE-like controller path where delayed commands
    /// are filtered before being applied to the surrogate dynamics.
    pub fn plan_with_delay_dynamics(
        &self,
        measured_state: &Array1<f64>,
        delayed_actions: &[Array1<f64>],
        sensor_noise: Option<&Array1<f64>>,
        actuator_lag_alpha: f64,
    ) -> FusionResult<Array1<f64>> {
        if !actuator_lag_alpha.is_finite() || !(0.0..=1.0).contains(&actuator_lag_alpha) {
            return Err(FusionError::ConfigError(
                "mpc actuator_lag_alpha must be finite and within [0, 1]".to_string(),
            ));
        }

        ensure_finite_vector("measured_state", measured_state)?;
        let n_coils = self.model.b_matrix.ncols();
        let mut predicted_state = measured_state.clone();
        if let Some(noise) = sensor_noise {
            if noise.len() != predicted_state.len() {
                return Err(FusionError::ConfigError(format!(
                    "mpc sensor_noise length mismatch: expected {}, got {}",
                    predicted_state.len(),
                    noise.len()
                )));
            }
            ensure_finite_vector("sensor_noise", noise)?;
            predicted_state = &predicted_state + noise;
        }

        let mut delayed_applied = Array1::zeros(n_coils);
        let alpha = actuator_lag_alpha;
        for (idx, action) in delayed_actions.iter().enumerate() {
            if action.len() != n_coils {
                return Err(FusionError::ConfigError(format!(
                    "mpc delayed action[{idx}] length mismatch: expected {n_coils}, got {}",
                    action.len()
                )));
            }
            if action.iter().any(|v| !v.is_finite()) {
                return Err(FusionError::ConfigError(format!(
                    "mpc delayed action[{idx}] must contain only finite values"
                )));
            }
            delayed_applied = &delayed_applied * (1.0 - alpha) + action * alpha;
            predicted_state = self.model.predict(&predicted_state, &delayed_applied)?;
        }

        self.plan(&predicted_state)
    }

    /// Plan an action and admit it through pulsed-shot scheduler and bank guards.
    pub fn plan_pulsed(
        &self,
        current_state: &Array1<f64>,
        scheduler_state: &str,
        bank_feasible: bool,
        burn_action_mask: &[bool],
        safe_action: &Array1<f64>,
        constraint_slack: f64,
    ) -> FusionResult<PulsedMpcDecision> {
        ensure_finite_vector("current_state", current_state)?;
        ensure_finite_vector("safe_action", safe_action)?;
        let n_coils = self.model.b_matrix.ncols();
        if safe_action.len() != n_coils {
            return Err(FusionError::ConfigError(format!(
                "mpc safe_action length mismatch: expected {n_coils}, got {}",
                safe_action.len()
            )));
        }
        if !constraint_slack.is_finite() {
            return Err(FusionError::ConfigError(
                "mpc constraint_slack must be finite".to_string(),
            ));
        }
        ensure_mask_shape("burn_action_mask", burn_action_mask, n_coils)?;

        let raw_action = self.plan(current_state)?;
        let state_label = normalise_scheduler_state(scheduler_state)?;
        let mut action = raw_action.clone();
        let mut reason = "burn action admitted".to_string();
        let mut bank_reason = "ok".to_string();
        let mut safe_action_applied = false;
        let mut burn_components_masked = false;

        if state_label != "burn" {
            for (idx, selected) in burn_action_mask.iter().enumerate() {
                if *selected {
                    action[idx] = safe_action[idx];
                }
            }
            burn_components_masked = true;
            bank_reason = "not evaluated outside burn".to_string();
            reason = format!("scheduler state {state_label} masks burn action");
        } else if !bank_feasible {
            action = safe_action.clone();
            safe_action_applied = true;
            bank_reason = "bank feasibility rejected burn action".to_string();
            reason = bank_reason.clone();
        }

        let predicted = self.model.predict(current_state, &raw_action)?;
        let error = &predicted - &self.target;
        let mpc_objective = error.dot(&error) + raw_action.dot(&raw_action) * LAMBDA;
        let peak_current_a = burn_action_mask
            .iter()
            .enumerate()
            .filter(|(_, selected)| **selected)
            .map(|(idx, _)| raw_action[idx].abs())
            .fold(0.0_f64, f64::max);
        let action_sha256 = hash_f64_array(&action);
        let safe_action_sha256 = hash_f64_array(safe_action);
        let burn_action_mask_sha256 = hash_bool_mask(burn_action_mask);
        let mut decision = PulsedMpcDecision {
            action,
            mpc_objective,
            constraint_slack,
            scheduler_state: state_label,
            bank_feasibility: bank_reason,
            reason,
            bank_feasible,
            safe_action_applied,
            burn_components_masked,
            peak_current_a,
            evidence_schema_version: PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
            action_sha256,
            safe_action_sha256,
            burn_action_mask_sha256,
            admission_digest: String::new(),
        };
        decision.admission_digest = decision_evidence_digest(&decision);
        Ok(decision)
    }
}

/// Unit tests of the surrogate, the planner, its delay paths and the exact horizon gradient.
#[cfg(test)]
mod tests {
    use super::*;

    /// The linear surrogate returns `x + B u`.
    #[test]
    fn test_surrogate_prediction() {
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let state = Array1::from_vec(vec![1.0, 2.0]);
        let action = Array1::from_vec(vec![0.5, -0.3]);
        let next = model
            .predict(&state, &action)
            .expect("matching state/action shape");
        assert!((next[0] - 1.5).abs() < 1e-10);
        assert!((next[1] - 1.7).abs() < 1e-10);
    }

    /// The first planned action pushes each state component toward its target.
    #[test]
    fn test_mpc_moves_toward_target() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target.clone()).expect("matching target length");

        let state = Array1::from_vec(vec![5.0, 1.0]);
        let action = mpc.plan(&state).expect("matching state shape");

        // Action should push state toward target
        assert!(action[0] > 0.0, "Should push R positive: {}", action[0]);
        assert!(action[1] < 0.0, "Should push Z negative: {}", action[1]);
    }

    /// A strong surrogate far from target still yields actions within the clip.
    #[test]
    fn test_mpc_action_clipped() {
        let b = Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 10.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![100.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");

        let state = Array1::from_vec(vec![0.0, 0.0]);
        let action = mpc.plan(&state).expect("matching state shape");
        for &v in action.iter() {
            assert!(
                v.abs() <= ACTION_CLIP + 1e-10,
                "Action should be clipped: {v}"
            );
        }
    }

    /// Delayed actions and sensor bias shift the planning start as documented.
    #[test]
    fn test_mpc_delay_and_bias_path() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");

        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let delayed = vec![Array1::from_vec(vec![0.2, -0.1])];
        let bias = Array1::from_vec(vec![0.05, -0.02]);
        let action = mpc
            .plan_with_delay_and_bias(&measured, &delayed, Some(&bias))
            .expect("valid delayed action and bias sizes");

        assert!(
            action[0] > 0.0,
            "Delayed MPC should push R positive: {}",
            action[0]
        );
        assert!(
            action[1] < 0.0,
            "Delayed MPC should push Z negative: {}",
            action[1]
        );
        for &v in action.iter() {
            assert!(
                v.abs() <= ACTION_CLIP + 1e-10,
                "Action should remain clipped: {v}"
            );
        }
    }

    /// The first-order actuator lag filters delayed actions before planning.
    #[test]
    fn test_mpc_delay_dynamics_path() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");

        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let delayed = vec![
            Array1::from_vec(vec![0.4, -0.2]),
            Array1::from_vec(vec![0.4, -0.2]),
        ];
        let noise = Array1::from_vec(vec![0.03, -0.01]);
        let action = mpc
            .plan_with_delay_dynamics(&measured, &delayed, Some(&noise), 0.5)
            .expect("valid actuator lag alpha");

        assert!(
            action[0] > 0.0,
            "Delay dynamics MPC should push R positive: {}",
            action[0]
        );
        assert!(
            action[1] < 0.0,
            "Delay dynamics MPC should push Z negative: {}",
            action[1]
        );
        for &v in action.iter() {
            assert!(v.abs() <= ACTION_CLIP + 1e-10);
        }
    }

    /// A lag coefficient outside [0, 1] or non-finite is refused.
    #[test]
    fn test_mpc_delay_dynamics_rejects_invalid_actuator_lag_alpha() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");
        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let delayed = vec![Array1::from_vec(vec![0.1, -0.05])];

        assert!(mpc
            .plan_with_delay_dynamics(&measured, &delayed, None, -0.1)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured, &delayed, None, 1.1)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured, &delayed, None, f64::NAN)
            .is_err());
    }

    /// Bias or delayed actions of the wrong length are refused.
    #[test]
    fn test_mpc_delay_and_bias_rejects_mismatched_vector_lengths() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");
        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let bad_bias = Array1::from_vec(vec![0.02, -0.01, 0.0]);
        let bad_delayed = vec![Array1::from_vec(vec![0.1, -0.05, 0.01])];

        assert!(mpc
            .plan_with_delay_and_bias(&measured, &[], Some(&bad_bias))
            .is_err());
        assert!(mpc
            .plan_with_delay_and_bias(&measured, &bad_delayed, None)
            .is_err());
    }

    /// Noise or delayed actions of the wrong length are refused on the lag path.
    #[test]
    fn test_mpc_delay_dynamics_rejects_mismatched_vector_lengths() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");
        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let bad_noise = Array1::from_vec(vec![0.01, -0.01, 0.0]);
        let bad_delayed = vec![Array1::from_vec(vec![0.1, -0.05, 0.01])];

        assert!(mpc
            .plan_with_delay_dynamics(&measured, &[], Some(&bad_noise), 0.5)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured, &bad_delayed, None, 0.5)
            .is_err());
    }

    /// A target whose length differs from the state dimension is refused.
    #[test]
    fn test_mpc_constructor_rejects_target_length_mismatch() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let bad_target = Array1::from_vec(vec![6.0, 0.0, -0.1]);
        assert!(MPController::new(model, bad_target).is_err());
    }

    /// States and actions of the wrong length are refused by the surrogate.
    #[test]
    fn test_surrogate_predict_rejects_state_and_action_shape_mismatch() {
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let bad_state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let bad_action = Array1::from_vec(vec![0.5]);
        assert!(model
            .predict(&bad_state, &Array1::from_vec(vec![0.2, 0.1]))
            .is_err());
        assert!(model
            .predict(&Array1::from_vec(vec![1.0, 2.0]), &bad_action)
            .is_err());
    }

    /// A non-finite `B` or target is refused at construction.
    #[test]
    fn test_mpc_constructor_rejects_non_finite_matrix_or_target_values() {
        let bad_b = Array2::from_shape_vec((2, 2), vec![0.1, f64::NAN, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(bad_b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        assert!(MPController::new(model, target).is_err());

        let good_b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(good_b);
        let bad_target = Array1::from_vec(vec![6.0, f64::INFINITY]);
        assert!(MPController::new(model, bad_target).is_err());
    }

    /// Non-finite states and actions are refused by the surrogate.
    #[test]
    fn test_surrogate_predict_rejects_non_finite_state_or_action_values() {
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let model = NeuralSurrogate::new(b);
        let bad_state = Array1::from_vec(vec![f64::NAN, 2.0]);
        let bad_action = Array1::from_vec(vec![0.5, f64::INFINITY]);
        assert!(model
            .predict(&bad_state, &Array1::from_vec(vec![0.1, 0.2]))
            .is_err());
        assert!(model
            .predict(&Array1::from_vec(vec![1.0, 2.0]), &bad_action)
            .is_err());
    }

    /// Both delay paths refuse non-finite states, bias, noise and delayed actions.
    #[test]
    fn test_mpc_delay_paths_reject_non_finite_vectors() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let target = Array1::from_vec(vec![6.0, 0.0]);
        let mpc = MPController::new(model, target).expect("matching target length");
        let measured_bad = Array1::from_vec(vec![5.0, f64::NAN]);
        let measured = Array1::from_vec(vec![5.0, 1.0]);
        let bad_bias = Array1::from_vec(vec![0.05, f64::INFINITY]);
        let bad_noise = Array1::from_vec(vec![0.03, f64::NAN]);
        let bad_delayed = vec![Array1::from_vec(vec![0.2, f64::NAN])];

        assert!(mpc
            .plan_with_delay_and_bias(&measured_bad, &[], None)
            .is_err());
        assert!(mpc
            .plan_with_delay_and_bias(&measured, &[], Some(&bad_bias))
            .is_err());
        assert!(mpc
            .plan_with_delay_and_bias(&measured, &bad_delayed, None)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured_bad, &[], None, 0.5)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured, &[], Some(&bad_noise), 0.5)
            .is_err());
        assert!(mpc
            .plan_with_delay_dynamics(&measured, &bad_delayed, None, 0.5)
            .is_err());
    }

    /// A three-state, two-coil controller with `B` scaled by `scale`, and its start state.
    fn quadratic_fixture(scale: f64) -> (MPController, Array1<f64>) {
        let b =
            Array2::from_shape_vec((3, 2), vec![0.7, -0.2, 0.1, 0.5, -0.3, 0.4]).unwrap() * scale;
        let target = Array1::from_vec(vec![0.4, -0.6, 0.2]);
        let mpc = MPController::new(NeuralSurrogate::new(b), target).expect("matching target");
        (mpc, Array1::from_vec(vec![1.0, 0.5, -0.8]))
    }

    /// A deterministic, non-trivial action sequence of the given horizon.
    fn sample_actions(horizon: usize) -> Vec<Array1<f64>> {
        (0..horizon)
            .map(|t| Array1::from_vec(vec![0.1 * t as f64 - 0.3, 0.2 - 0.05 * t as f64]))
            .collect()
    }

    /// The closed form matches the largest eigenvalue for horizons 1, 2 and 3.
    #[test]
    fn test_horizon_coupling_spectral_radius_matches_known_values() {
        // T_1 = [1]; T_2 = [[2, 1], [1, 1]] has the largest eigenvalue (3 + sqrt 5) / 2.
        assert!((horizon_coupling_spectral_radius(1) - 1.0).abs() < 1e-12);
        let golden = (3.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((horizon_coupling_spectral_radius(2) - golden).abs() < 1e-12);
        // T_3 = [[3, 2, 1], [2, 2, 1], [1, 1, 1]]: power iteration reaches the dominant eigenvector,
        // whose Rayleigh quotient is the largest eigenvalue.
        let lambda = horizon_coupling_spectral_radius(3);
        let t3: Array2<f64> =
            Array2::from_shape_vec((3, 3), vec![3.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
                .unwrap();
        let mut v: Array1<f64> = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        for _ in 0..200 {
            let w = t3.dot(&v);
            v = &w / w.dot(&w).sqrt();
        }
        assert!((v.dot(&t3.dot(&v)) - lambda).abs() < 1e-10);
    }

    /// The adjoint gradient equals a central finite difference of the horizon cost.
    #[test]
    fn test_horizon_gradient_matches_central_finite_difference() {
        let (mpc, x0) = quadratic_fixture(1.0);
        let actions = sample_actions(mpc.horizon);
        let (_, gradient) = mpc.horizon_cost_and_gradient(&x0, &actions).unwrap();
        let h = 1e-6;
        for t in 0..mpc.horizon {
            for j in 0..2 {
                let mut plus = actions.clone();
                let mut minus = actions.clone();
                plus[t][j] += h;
                minus[t][j] -= h;
                let (cp, _) = mpc.horizon_cost_and_gradient(&x0, &plus).unwrap();
                let (cm, _) = mpc.horizon_cost_and_gradient(&x0, &minus).unwrap();
                let fd = (cp - cm) / (2.0 * h);
                assert!(
                    (gradient[t][j] - fd).abs() < 1e-6,
                    "t={t} j={j}: adjoint {} vs finite difference {fd}",
                    gradient[t][j]
                );
            }
        }
    }

    /// The first gradient entry differs from the one-step term.
    #[test]
    fn test_horizon_gradient_includes_later_stage_errors() {
        // A one-step gradient would make the first entry B^T e_0 + LAMBDA u_0 only.
        let (mpc, x0) = quadratic_fixture(1.0);
        let actions = sample_actions(mpc.horizon);
        let (_, gradient) = mpc.horizon_cost_and_gradient(&x0, &actions).unwrap();
        let first_error = mpc.model.predict(&x0, &actions[0]).unwrap() - &mpc.target;
        let one_step = mpc.model.b_matrix.t().dot(&first_error) + &actions[0] * LAMBDA;
        assert!((&gradient[0] - &one_step).iter().any(|d| d.abs() > 1e-3));
    }

    /// For a strong surrogate the capped step descends where `LR` overshoots.
    #[test]
    fn test_stable_step_decreases_cost_where_the_fixed_rate_diverges() {
        let (mpc, x0) = quadratic_fixture(3.0);
        let step = mpc.stable_step();
        assert!(step < LR);
        let zero: Vec<Array1<f64>> = (0..mpc.horizon).map(|_| Array1::zeros(2)).collect();
        let (c0, g0) = mpc.horizon_cost_and_gradient(&x0, &zero).unwrap();
        let moved = |rate: f64| -> f64 {
            let next: Vec<Array1<f64>> = zero
                .iter()
                .zip(g0.iter())
                .map(|(u, g)| u - &(g * rate))
                .collect();
            mpc.horizon_cost_and_gradient(&x0, &next).unwrap().0
        };
        assert!(moved(step) < c0);
        assert!(moved(LR) > c0);
    }

    /// A weak surrogate leaves `LR` in force.
    #[test]
    fn test_stable_step_keeps_the_rate_for_a_weak_surrogate() {
        let (mpc, _) = quadratic_fixture(0.05);
        assert_eq!(mpc.stable_step(), LR);
    }

    /// Actions of the wrong length are refused by the gradient.
    #[test]
    fn test_horizon_gradient_rejects_mismatched_actions() {
        let (mpc, x0) = quadratic_fixture(1.0);
        let bad = vec![Array1::from_vec(vec![0.1, 0.2, 0.3])];
        assert!(mpc.horizon_cost_and_gradient(&x0, &bad).is_err());
    }
}
