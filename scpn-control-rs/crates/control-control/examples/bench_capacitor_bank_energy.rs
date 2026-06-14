// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust capacitor-bank energy ledger benchmark harness.
use control_control::capacitor_bank::{CapacitorBank, CapacitorBankSpec, PulseSpec, PulseWaveform};
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

fn parse_f64(args: &[String], name: &str, default: f64) -> f64 {
    arg_value(args, name)
        .map(|value| value.parse::<f64>().expect("f64 argument"))
        .unwrap_or(default)
}

fn parse_path(args: &[String], name: &str) -> Option<PathBuf> {
    arg_value(args, name).map(PathBuf::from)
}

fn spec() -> CapacitorBankSpec {
    CapacitorBankSpec::new(100e-6, 100e-6, 0.5, 10_000.0, 20.0).expect("valid bank spec")
}

fn run_once(discharge_steps: usize, dt_s: f64) -> Value {
    let mut bank = CapacitorBank::new(spec(), 5_000.0, 12.0).expect("valid bank");
    let pulse = PulseSpec::new(
        500.0,
        discharge_steps as f64 * dt_s,
        PulseWaveform::HalfSine,
    )
    .expect("valid pulse");
    let report = bank
        .discharge(pulse, dt_s, discharge_steps)
        .expect("discharge evaluates");
    json!({
        "energy_initial_J": report.energy_initial_j,
        "energy_remaining_J": report.energy_remaining_j,
        "energy_delivered_J": report.energy_delivered_j,
        "resistive_loss_J": report.resistive_loss_j,
        "load_energy_J": report.load_energy_j,
        "energy_balance_residual_J": report.energy_balance_residual_j,
        "energy_balance_relative_error": report.energy_balance_relative_error,
        "energy_balance_passed": report.energy_balance_passed,
        "rlc_regime": report.rlc_regime.as_str(),
    })
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

fn proc_loadavg() -> Option<String> {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|value| value.trim().to_string())
}

fn proc_cpu_affinity() -> Option<String> {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .map(|value| value.trim().to_string())
            })
        })
}

fn payload_sha256(payload: &Value) -> String {
    let encoded = serde_json::to_vec(payload).expect("payload encodes");
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    control_control::to_hex(&hasher.finalize())
}

fn write_markdown(path: PathBuf, payload: &Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("report directory");
    }
    let stats = &payload["result"]["stats"];
    let report = &payload["result"]["last_report"];
    let body = format!(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n\
<!-- Commercial license available -->\n\
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- ORCID: 0009-0009-3560-0851 -->\n\
<!-- Contact: www.anulum.li | protoscience@anulum.li -->\n\
<!-- SCPN Control — Rust capacitor-bank energy ledger benchmark report. -->\n\n\
# Rust Capacitor-Bank Energy Ledger Benchmark\n\n\
- Generated unix seconds: `{}`\n\
- Evidence class: `{}`\n\
- Production claim allowed: `{}`\n\
- CPU affinity: `{}`\n\
- Load average start: `{}`\n\
- Load average end: `{}`\n\
- Discharge steps per sample: `{}`\n\
- Step size s: `{}`\n\n\
| Samples | Mean us | Median us | p95 us | p99 us |\n\
|---:|---:|---:|---:|---:|\n\
| {} | {:.6} | {:.6} | {:.6} | {:.6} |\n\n\
## Last energy ledger\n\n\
- Energy balance passed: `{}`\n\
- Relative residual: `{}`\n\
- Residual J: `{}`\n\
- RLC regime: `{}`\n",
        payload["generated_unix_seconds"],
        payload["evidence_class"],
        payload["production_claim_allowed"],
        payload["context"]["cpu_affinity"],
        payload["context"]["loadavg_start"],
        payload["context"]["loadavg_end"],
        payload["settings"]["discharge_steps"],
        payload["settings"]["dt_s"],
        stats["samples"].as_u64().expect("samples"),
        stats["mean_us"].as_f64().expect("mean"),
        stats["median_us"].as_f64().expect("median"),
        stats["p95_us"].as_f64().expect("p95"),
        stats["p99_us"].as_f64().expect("p99"),
        report["energy_balance_passed"],
        report["energy_balance_relative_error"],
        report["energy_balance_residual_J"],
        report["rlc_regime"],
    );
    fs::write(path, body).expect("markdown report");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let steps = parse_usize(&args, "--steps", 500);
    let warmup = parse_usize(&args, "--warmup", 50);
    let discharge_steps = parse_usize(&args, "--discharge-steps", 200);
    let dt_s = parse_f64(&args, "--dt-s", 1.0e-7);
    let evidence_class =
        arg_value(&args, "--evidence-class").unwrap_or_else(|| "local_regression".to_string());
    let loadavg_start = proc_loadavg();

    for _ in 0..warmup {
        let _ = run_once(discharge_steps, dt_s);
    }

    let mut samples_ns = Vec::with_capacity(steps);
    let mut last_report = Value::Null;
    for _ in 0..steps {
        let start = Instant::now();
        last_report = run_once(discharge_steps, dt_s);
        samples_ns.push(start.elapsed().as_nanos());
    }

    let generated_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_secs();
    let mut payload = json!({
        "schema_version": "scpn-control.rust-capacitor-bank-energy-benchmark.v1",
        "generated_unix_seconds": generated_unix_seconds,
        "evidence_class": evidence_class,
        "production_claim_allowed": false,
        "settings": {
            "steps": steps,
            "warmup": warmup,
            "discharge_steps": discharge_steps,
            "dt_s": dt_s,
        },
        "context": {
            "cpu_affinity": proc_cpu_affinity(),
            "loadavg_start": loadavg_start,
            "loadavg_end": proc_loadavg(),
            "isolation_method": "caller-provided affinity only; not a production benchmark",
        },
        "result": {
            "stats": stats(&samples_ns),
            "last_report": last_report,
        },
    });
    let digest = payload_sha256(&payload);
    payload["payload_sha256"] = Value::String(digest);

    if let Some(path) = parse_path(&args, "--json-out") {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("report directory");
        }
        fs::write(
            path,
            serde_json::to_string_pretty(&payload).expect("payload encodes") + "\n",
        )
        .expect("json report");
    }
    if let Some(path) = parse_path(&args, "--markdown-out") {
        write_markdown(path, &payload);
    }
    if parse_path(&args, "--json-out").is_none() && parse_path(&args, "--markdown-out").is_none() {
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).expect("payload encodes")
        );
    }
}
