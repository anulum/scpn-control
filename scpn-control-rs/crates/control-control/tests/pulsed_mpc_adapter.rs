// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust pulsed-shot MPC admission tests.
use control_control::mpc::{
    MPController, NeuralSurrogate, PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
};
use ndarray::{array, Array1, Array2};

fn controller() -> MPController {
    let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    let model = NeuralSurrogate::new(b);
    MPController::new(model, array![6.0, 0.0]).expect("valid MPC")
}

#[test]
fn non_burn_state_masks_selected_components() {
    let mpc = controller();
    let state = array![5.0, 1.0];
    let safe = array![0.0, 0.0];
    let decision = mpc
        .plan_pulsed(&state, "flat_top", true, &[true, false], &safe, 12.0)
        .expect("valid pulsed decision");

    assert_eq!(decision.action[0], 0.0);
    assert!(decision.action[1] < 0.0);
    assert!(decision.burn_components_masked);
    assert!(!decision.safe_action_applied);
    assert_eq!(decision.scheduler_state, "flat_top");
    assert_eq!(
        decision.evidence_schema_version,
        PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    );
    assert_eq!(decision.admission_digest.len(), 64);
    assert_eq!(decision.action_sha256.len(), 64);
    assert_eq!(decision.safe_action_sha256.len(), 64);
    assert_eq!(decision.burn_action_mask_sha256.len(), 64);
}

#[test]
fn infeasible_bank_replaces_burn_action_with_safe_action() {
    let mpc = controller();
    let state = array![5.0, 1.0];
    let safe = array![0.0, 0.0];
    let decision = mpc
        .plan_pulsed(&state, "burn", false, &[true, true], &safe, -0.5)
        .expect("valid pulsed decision");

    assert_eq!(decision.action, Array1::<f64>::zeros(2));
    assert!(decision.safe_action_applied);
    assert!(!decision.bank_feasible);
    assert_eq!(decision.constraint_slack, -0.5);
    assert_eq!(decision.peak_current_a, decision.peak_current_a.abs());
    assert_eq!(decision.admission_digest.len(), 64);
}

#[test]
fn rejects_bad_mask_and_safe_action_shapes() {
    let mpc = controller();
    let state = array![5.0, 1.0];
    let safe = array![0.0, 0.0];
    assert!(mpc
        .plan_pulsed(&state, "burn", true, &[false, false], &safe, 1.0)
        .is_err());
    assert!(mpc
        .plan_pulsed(&state, "burn", true, &[true], &safe, 1.0)
        .is_err());
    assert!(mpc
        .plan_pulsed(&state, "burn", true, &[true, true], &array![0.0], 1.0)
        .is_err());
    assert!(mpc
        .plan_pulsed(&state, "unknown", true, &[true, true], &safe, 1.0)
        .is_err());
}

#[test]
fn decision_digest_changes_when_admission_state_changes() {
    let mpc = controller();
    let state = array![5.0, 1.0];
    let safe = array![0.0, 0.0];
    let non_burn = mpc
        .plan_pulsed(&state, "flat_top", true, &[true, false], &safe, 12.0)
        .expect("valid non-burn pulsed decision");
    let burn = mpc
        .plan_pulsed(&state, "burn", true, &[true, false], &safe, 12.0)
        .expect("valid burn pulsed decision");

    assert_ne!(non_burn.action_sha256, burn.action_sha256);
    assert_ne!(non_burn.admission_digest, burn.admission_digest);
    assert!(non_burn.burn_components_masked);
    assert!(!burn.burn_components_masked);
}
