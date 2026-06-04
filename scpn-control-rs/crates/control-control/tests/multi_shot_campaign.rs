// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust multi-shot campaign tests.
use control_control::capacitor_bank::CapacitorBankSpec;
use control_control::multi_shot_campaign::{
    CampaignShotPlan, CampaignShotSample, MultiShotCampaignOrchestrator,
};
use control_control::pulsed_scenario::{
    CapacitorBankTelemetry, PulsedPlasmaTelemetry, PulsedScenarioSpec,
};

fn scheduler_spec() -> PulsedScenarioSpec {
    PulsedScenarioSpec::new(
        100.0, 2.0e6, 0.01, 0.002, 1.0e3, 2.0e6, 1.0e3, 40.0, 0.95, 20.0, 1.0e3, 0.0,
    )
    .expect("valid scheduler spec")
}

fn bank_spec() -> CapacitorBankSpec {
    CapacitorBankSpec::new(100e-6, 100e-6, 0.5, 10_000.0, 20.0).expect("valid bank spec")
}

fn plasma(
    coil_current_a: f64,
    temperature_ev: f64,
    phase_lock_error_rad: f64,
    reference_error_m: f64,
    fusion_power_w: f64,
    radial_velocity_m_s: f64,
) -> PulsedPlasmaTelemetry {
    PulsedPlasmaTelemetry::new(
        coil_current_a,
        temperature_ev,
        phase_lock_error_rad,
        reference_error_m,
        fusion_power_w,
        radial_velocity_m_s,
    )
    .expect("valid plasma")
}

fn bank(voltage_v: f64, energy_j: f64) -> CapacitorBankTelemetry {
    CapacitorBankTelemetry::new(voltage_v, 10_000.0, energy_j).expect("valid bank")
}

fn admission_digest(prefix: &str) -> String {
    format!("{prefix:0<64}")[..64].to_string()
}

fn sample(
    t_s: f64,
    plasma: PulsedPlasmaTelemetry,
    bank: CapacitorBankTelemetry,
) -> CampaignShotSample {
    CampaignShotSample::new(t_s, plasma, bank).expect("valid sample")
}

fn complete_shot(shot_id: &str) -> CampaignShotPlan {
    CampaignShotPlan::new(
        shot_id,
        vec![
            sample(
                0.0,
                plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            sample(
                1.0e-3,
                plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            sample(
                2.0e-3,
                plasma(2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            sample(
                3.0e-3,
                plasma(2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0),
                bank(9800.0, 200.0),
            ),
            sample(
                4.0e-3,
                plasma(0.0, 200.0, 0.02, 0.01, 0.0, 1500.0),
                bank(9800.0, 200.0),
            ),
            sample(
                5.0e-3,
                plasma(0.0, 120.0, 0.02, 0.01, 0.0, 0.0),
                bank(2000.0, 20.0),
            ),
            sample(
                6.0e-3,
                plasma(0.0, 40.0, 0.02, 0.01, 0.0, 0.0),
                bank(9700.0, 180.0),
            ),
            sample(
                7.0e-3,
                plasma(100.0, 15.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
        ],
        5000.0,
        0.0,
    )
    .expect("valid shot")
    .with_pulsed_mpc_admission_digest(&admission_digest("a"))
    .expect("valid admission digest")
}

fn orchestrator() -> MultiShotCampaignOrchestrator {
    MultiShotCampaignOrchestrator::new("campaign-a", scheduler_spec(), bank_spec(), true)
        .expect("valid orchestrator")
}

#[test]
fn runs_two_complete_shots() {
    let report = orchestrator()
        .run(&[complete_shot("shot-001"), complete_shot("shot-002")])
        .expect("valid campaign");

    assert_eq!(report.shot_count, 2);
    assert_eq!(report.passed_count, 2);
    assert_eq!(report.failed_count, 0);
    assert_eq!(report.shots[0].terminal_state, "idle");
    assert_eq!(report.shots[0].trigger_timestamp_ns, Some(2_000_000));
    assert_eq!(report.shots[0].energy_recovered_j, 180.0);
    assert_eq!(report.pulsed_mpc_admission_digest_count, 2);
    assert_eq!(
        report.pulsed_mpc_evidence_schema_version,
        "scpn-control.pulsed-mpc-decision-evidence.v1"
    );
    let expected_digest = admission_digest("a");
    assert_eq!(
        report.shots[0].pulsed_mpc_admission_digest.as_deref(),
        Some(expected_digest.as_str())
    );
    assert_eq!(report.payload_sha256.len(), 64);
}

#[test]
fn rejects_duplicate_shot_ids() {
    let shot = complete_shot("shot-001");
    let err = orchestrator()
        .run(&[shot.clone(), shot])
        .expect_err("duplicate rejected");

    assert!(err.to_string().contains("duplicate shot_id"));
}

#[test]
fn rejects_malformed_pulsed_mpc_digest() {
    let err = CampaignShotPlan::new(
        "bad-digest",
        vec![sample(
            0.0,
            plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0),
            bank(9800.0, 200.0),
        )],
        5000.0,
        0.0,
    )
    .expect("valid shot")
    .with_pulsed_mpc_admission_digest(&"A".repeat(64))
    .expect_err("uppercase digest rejected");

    assert!(err.to_string().contains("lowercase SHA-256"));
}

#[test]
fn incomplete_lifecycle_fails_shot_without_aborting_campaign() {
    let incomplete = CampaignShotPlan::new(
        "incomplete",
        vec![
            sample(
                0.0,
                plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
            sample(
                1.0e-3,
                plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0),
                bank(9800.0, 200.0),
            ),
        ],
        5000.0,
        0.0,
    )
    .expect("valid shot");

    let report = orchestrator()
        .run(&[incomplete, complete_shot("shot-002")])
        .expect("valid campaign");

    assert_eq!(report.passed_count, 1);
    assert_eq!(report.failed_count, 1);
    assert!(!report.shots[0].passed);
    assert_eq!(
        report.shots[0].failure_reason.as_deref(),
        Some("shot did not traverse the complete pulsed lifecycle")
    );
}
