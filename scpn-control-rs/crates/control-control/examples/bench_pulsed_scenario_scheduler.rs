// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust pulsed-scenario scheduler benchmark harness.
use control_control::pulsed_scenario::{
    CapacitorBankTelemetry, PulsedPlasmaTelemetry, PulsedScenarioScheduler, PulsedScenarioSpec,
};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::time::Instant;

fn arg_usize(args: &[String], name: &str, default: usize) -> usize {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].parse::<usize>().expect("usize argument"))
        .unwrap_or(default)
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn cpu_affinity() -> String {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .map(str::trim)
                    .map(str::to_string)
                    .filter(|value| !value.is_empty())
            })
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

fn loadavg() -> String {
    read_trimmed("/proc/loadavg").unwrap_or_else(|| "unavailable".to_string())
}

fn spec() -> PulsedScenarioSpec {
    PulsedScenarioSpec::new(
        100.0, 2.0e6, 0.01, 0.002, 1.0e3, 2.0e6, 1.0e3, 40.0, 0.95, 20.0, 1.0e3, 0.0,
    )
    .expect("valid scheduler spec")
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
    .expect("valid plasma telemetry")
}

fn bank(voltage_v: f64, energy_j: f64) -> CapacitorBankTelemetry {
    CapacitorBankTelemetry::new(voltage_v, 10_000.0, energy_j).expect("valid bank telemetry")
}

fn campaign() -> [(f64, PulsedPlasmaTelemetry, CapacitorBankTelemetry); 8] {
    [
        (
            0.0,
            plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0),
            bank(9800.0, 200.0),
        ),
        (
            1.0e-3,
            plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0),
            bank(9800.0, 200.0),
        ),
        (
            2.0e-3,
            plasma(2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0),
            bank(9800.0, 200.0),
        ),
        (
            3.0e-3,
            plasma(2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0),
            bank(9800.0, 200.0),
        ),
        (
            4.0e-3,
            plasma(0.0, 200.0, 0.02, 0.01, 0.0, 1500.0),
            bank(9800.0, 200.0),
        ),
        (
            5.0e-3,
            plasma(0.0, 120.0, 0.02, 0.01, 0.0, 0.0),
            bank(2000.0, 20.0),
        ),
        (
            6.0e-3,
            plasma(0.0, 40.0, 0.02, 0.01, 0.0, 0.0),
            bank(9700.0, 180.0),
        ),
        (
            7.0e-3,
            plasma(100.0, 15.0, 0.02, 0.01, 0.0, 0.0),
            bank(9800.0, 200.0),
        ),
    ]
}

fn run_campaign() -> Vec<&'static str> {
    let mut scheduler = PulsedScenarioScheduler::new(spec());
    let states = campaign()
        .into_iter()
        .map(|(time, plasma, bank)| {
            scheduler
                .step(time, plasma, bank)
                .expect("valid campaign step")
                .state
                .as_str()
        })
        .collect::<Vec<_>>();
    assert_eq!(scheduler.state.as_str(), "idle");
    assert_eq!(scheduler.audit_log().len(), 8);
    states
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

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let iterations = arg_usize(&args, "--iterations", 50_000);
    let warmup = arg_usize(&args, "--warmup", 2_000);
    assert!(iterations > 0, "--iterations must be >= 1");
    for _ in 0..warmup {
        std::hint::black_box(run_campaign());
    }
    let loadavg_start = loadavg();
    let mut timings = Vec::with_capacity(iterations);
    let mut last_states = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        last_states = run_campaign();
        timings.push(start.elapsed().as_nanos());
    }
    let payload = json!({
        "schema_version": "scpn-control.rust-pulsed-scenario-scheduler-benchmark.v1",
        "evidence_class": "local_proxy",
        "production_claim_allowed": false,
        "public_claim_allowed": false,
        "claim_boundary": "Loaded-workstation orientation only; not PREEMPT_RT, HIL, target-hardware, or facility timing evidence.",
        "command": args.join(" "),
        "iterations": iterations,
        "warmup": warmup,
        "campaign_steps": 8,
        "last_states": last_states,
        "stats": stats(&timings),
        "context": {
            "cpu_affinity": cpu_affinity(),
            "loadavg_start": loadavg_start,
            "loadavg_end": loadavg(),
            "kernel_release": read_trimmed("/proc/sys/kernel/osrelease").unwrap_or_else(|| "unavailable".to_string()),
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "rust_crate_version": env!("CARGO_PKG_VERSION"),
        }
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}
