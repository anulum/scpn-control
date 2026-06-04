// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust multi-shot campaign benchmark harness.
use control_control::capacitor_bank::CapacitorBankSpec;
use control_control::multi_shot_campaign::{
    CampaignShotPlan, CampaignShotSample, MultiShotCampaignOrchestrator,
};
use control_control::pulsed_scenario::{
    CapacitorBankTelemetry, PulsedPlasmaTelemetry, PulsedScenarioSpec,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    arg_value(args, name)
        .map(|value| value.parse::<usize>().expect("usize argument"))
        .unwrap_or(default)
}

fn parse_path(args: &[String], name: &str) -> Option<PathBuf> {
    arg_value(args, name).map(PathBuf::from)
}

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

fn admission_digest(label: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"scpn-control.multi-shot-campaign.benchmark");
    hasher.update([0]);
    hasher.update(label.as_bytes());
    format!("{:x}", hasher.finalize())
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
    .with_pulsed_mpc_admission_digest(&admission_digest(shot_id))
    .expect("valid admission digest")
}

fn stats(samples_ns: &[u128]) -> Value {
    let mut ordered = samples_ns.to_vec();
    ordered.sort_unstable();
    let n = ordered.len();
    let sum: u128 = ordered.iter().sum();
    json!({
        "samples": n,
        "mean_us": sum as f64 / n as f64 / 1_000.0,
        "median_us": ordered[n / 2] as f64 / 1_000.0,
        "p95_us": ordered[((n as f64 * 0.95) as usize).min(n - 1)] as f64 / 1_000.0,
        "p99_us": ordered[((n as f64 * 0.99) as usize).min(n - 1)] as f64 / 1_000.0,
        "min_us": ordered[0] as f64 / 1_000.0,
        "max_us": ordered[n - 1] as f64 / 1_000.0,
    })
}

fn write_json(path: &PathBuf, payload: &Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    fs::write(path, serde_json::to_string_pretty(payload).unwrap() + "\n").expect("write JSON");
}

fn write_markdown(path: &PathBuf, payload: &Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    let stats = &payload["result"]["stats"];
    let body = format!(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n\
<!-- Commercial license available -->\n\
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- ORCID: 0009-0009-3560-0851 -->\n\
<!-- Contact: www.anulum.li | protoscience@anulum.li -->\n\
<!-- SCPN Control — Rust multi-shot campaign benchmark report. -->\n\n\
# Rust Multi-Shot Campaign Benchmark\n\n\
| Samples | Mean us | Median us | p95 us | p99 us |\n\
|---:|---:|---:|---:|---:|\n\
| {} | {:.6} | {:.6} | {:.6} | {:.6} |\n\n\
Payload SHA-256: `{}`\n",
        stats["samples"].as_u64().unwrap(),
        stats["mean_us"].as_f64().unwrap(),
        stats["median_us"].as_f64().unwrap(),
        stats["p95_us"].as_f64().unwrap(),
        stats["p99_us"].as_f64().unwrap(),
        payload["payload_sha256"].as_str().unwrap()
    );
    fs::write(path, body).expect("write Markdown");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let steps = parse_usize(&args, "--steps", 2_000);
    let warmup = parse_usize(&args, "--warmup", 200);
    assert!(steps > 0, "--steps must be >= 1");
    let orchestrator =
        MultiShotCampaignOrchestrator::new("campaign-a", scheduler_spec(), bank_spec(), true)
            .expect("valid orchestrator");
    let shots = vec![complete_shot("shot-001"), complete_shot("shot-002")];
    for _ in 0..warmup {
        let _ = orchestrator.run(&shots).expect("valid campaign");
    }
    let mut timings = Vec::with_capacity(steps);
    let mut passed_count = 0usize;
    let mut pulsed_mpc_admission_digest_count = 0usize;
    for _ in 0..steps {
        let start = Instant::now();
        let report = orchestrator.run(&shots).expect("valid campaign");
        timings.push(start.elapsed().as_nanos());
        passed_count = report.passed_count;
        pulsed_mpc_admission_digest_count = report.pulsed_mpc_admission_digest_count;
    }
    let mut payload = json!({
        "schema_version": "scpn-control.rust-multi-shot-campaign-benchmark.v1.1",
        "generated_unix_s": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "command": args.join(" "),
        "evidence_class": "local_regression",
        "production_claim_allowed": false,
        "steps": steps,
        "warmup": warmup,
        "result": {
            "stats": stats(&timings),
            "last_passed_count": passed_count,
            "last_pulsed_mpc_admission_digest_count": pulsed_mpc_admission_digest_count,
        },
    });
    let digest = Sha256::digest(serde_json::to_vec(&payload).unwrap());
    payload["payload_sha256"] = Value::String(format!("{digest:x}"));
    if let Some(path) = parse_path(&args, "--json-out") {
        write_json(&path, &payload);
    }
    if let Some(path) = parse_path(&args, "--md-out") {
        write_markdown(&path, &payload);
    }
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}
