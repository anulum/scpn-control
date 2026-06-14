// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust gradient MPC and pulsed-shot admission adapter.
//! Model Predictive Control for tokamak shape control.
//!
//! Port of `fusion_sota_mpc.py`.
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

fn ensure_finite_vector(name: &str, values: &Array1<f64>) -> FusionResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FusionError::ConfigError(format!(
            "mpc {name} must contain only finite values"
        )));
    }
    Ok(())
}

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

fn hash_f64_array(values: &Array1<f64>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    crate::to_hex(&hasher.finalize())
}

fn hash_bool_mask(values: &[bool]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update([u8::from(*value)]);
    }
    crate::to_hex(&hasher.finalize())
}

fn update_str(hasher: &mut Sha256, name: &str, value: &str) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update(value.as_bytes());
    hasher.update([255]);
}

fn update_bool(hasher: &mut Sha256, name: &str, value: bool) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update([u8::from(value)]);
    hasher.update([255]);
}

fn update_f64(hasher: &mut Sha256, name: &str, value: f64) {
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update(value.to_le_bytes());
    hasher.update([255]);
}

#[allow(clippy::too_many_arguments)]
fn decision_evidence_digest(
    scheduler_state: &str,
    bank_feasibility: &str,
    reason: &str,
    bank_feasible: bool,
    safe_action_applied: bool,
    burn_components_masked: bool,
    constraint_slack: f64,
    mpc_objective: f64,
    peak_current_a: f64,
    action_sha256: &str,
    safe_action_sha256: &str,
    burn_action_mask_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    update_str(
        &mut hasher,
        "schema_version",
        PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
    );
    update_str(&mut hasher, "scheduler_state", scheduler_state);
    update_str(&mut hasher, "bank_feasibility", bank_feasibility);
    update_str(&mut hasher, "reason", reason);
    update_bool(&mut hasher, "bank_feasible", bank_feasible);
    update_bool(&mut hasher, "safe_action_applied", safe_action_applied);
    update_bool(
        &mut hasher,
        "burn_components_masked",
        burn_components_masked,
    );
    update_f64(&mut hasher, "constraint_slack", constraint_slack);
    update_f64(&mut hasher, "mpc_objective", mpc_objective);
    update_f64(&mut hasher, "peak_current_A", peak_current_a);
    update_str(&mut hasher, "action_sha256", action_sha256);
    update_str(&mut hasher, "safe_action_sha256", safe_action_sha256);
    update_str(
        &mut hasher,
        "burn_action_mask_sha256",
        burn_action_mask_sha256,
    );
    crate::to_hex(&hasher.finalize())
}

/// Linear surrogate model: x_{t+1} = x_t + B·u_t.
pub struct NeuralSurrogate {
    /// Control impact matrix (n_state × n_coils).
    pub b_matrix: Array2<f64>,
}

impl NeuralSurrogate {
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
    pub action: Array1<f64>,
    pub mpc_objective: f64,
    pub constraint_slack: f64,
    pub scheduler_state: String,
    pub bank_feasibility: String,
    pub reason: String,
    pub bank_feasible: bool,
    pub safe_action_applied: bool,
    pub burn_components_masked: bool,
    pub peak_current_a: f64,
    pub evidence_schema_version: &'static str,
    pub action_sha256: String,
    pub safe_action_sha256: String,
    pub burn_action_mask_sha256: String,
    pub admission_digest: String,
}

/// Model Predictive Controller.
pub struct MPController {
    pub model: NeuralSurrogate,
    pub target: Array1<f64>,
    pub horizon: usize,
}

impl MPController {
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

    /// Plan optimal action via gradient descent over horizon.
    /// Returns first action to apply.
    pub fn plan(&self, current_state: &Array1<f64>) -> FusionResult<Array1<f64>> {
        ensure_finite_vector("current_state", current_state)?;
        let n_coils = self.model.b_matrix.ncols();
        let mut actions: Vec<Array1<f64>> =
            (0..self.horizon).map(|_| Array1::zeros(n_coils)).collect();

        for _ in 0..GD_ITERATIONS {
            // Forward rollout to compute gradients
            let mut grads: Vec<Array1<f64>> =
                (0..self.horizon).map(|_| Array1::zeros(n_coils)).collect();

            let mut state = current_state.clone();
            for t in 0..self.horizon {
                let next = self.model.predict(&state, &actions[t])?;
                let error = &next - &self.target;
                // Gradient of ||error||² w.r.t. u_t = B^T · error
                let grad = self.model.b_matrix.t().dot(&error) + &actions[t] * LAMBDA;
                grads[t] = grad;
                state = next;
            }

            // Update actions
            for t in 0..self.horizon {
                actions[t] = &actions[t] - &(&grads[t] * LR);
                // Clip
                for v in actions[t].iter_mut() {
                    *v = v.clamp(-ACTION_CLIP, ACTION_CLIP);
                }
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
        let admission_digest = decision_evidence_digest(
            &state_label,
            &bank_reason,
            &reason,
            bank_feasible,
            safe_action_applied,
            burn_components_masked,
            constraint_slack,
            mpc_objective,
            peak_current_a,
            &action_sha256,
            &safe_action_sha256,
            &burn_action_mask_sha256,
        );
        Ok(PulsedMpcDecision {
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
            admission_digest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_mpc_constructor_rejects_target_length_mismatch() {
        let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = NeuralSurrogate::new(b);
        let bad_target = Array1::from_vec(vec![6.0, 0.0, -0.1]);
        assert!(MPController::new(model, bad_target).is_err());
    }

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
}
