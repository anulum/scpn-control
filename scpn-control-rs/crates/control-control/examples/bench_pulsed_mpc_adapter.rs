// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Rust pulsed-shot MPC adapter benchmark harness.
use control_control::mpc::{MPController, NeuralSurrogate};
use ndarray::{array, Array1, Array2};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn controller() -> MPController {
    let b = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
    let model = NeuralSurrogate::new(b);
    MPController::new(model, array![6.0, 0.0]).expect("valid MPC")
}

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

fn stats(samples_ns: &[u128]) -> Value {
    let mut ordered = samples_ns.to_vec();
    ordered.sort_unstable();
    let n = ordered.len();
    assert!(n > 0, "at least one sample is required");
    let sum: u128 = ordered.iter().sum();
    let mean_us = sum as f64 / n as f64 / 1_000.0;
    let median_us = ordered[n / 2] as f64 / 1_000.0;
    let p95_us = ordered[((n as f64 * 0.95) as usize).min(n - 1)] as f64 / 1_000.0;
    let p99_us = ordered[((n as f64 * 0.99) as usize).min(n - 1)] as f64 / 1_000.0;
    json!({
        "samples": n,
        "mean_us": mean_us,
        "median_us": median_us,
        "p95_us": p95_us,
        "p99_us": p99_us,
        "min_us": ordered[0] as f64 / 1_000.0,
        "max_us": ordered[n - 1] as f64 / 1_000.0,
    })
}

fn measure<F>(steps: usize, warmup: usize, mut f: F) -> Value
where
    F: FnMut() -> Value,
{
    for _ in 0..warmup {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(steps);
    let mut last_value = Value::Null;
    for _ in 0..steps {
        let start = Instant::now();
        last_value = f();
        samples.push(start.elapsed().as_nanos());
    }
    json!({
        "stats": stats(&samples),
        "last_value": last_value,
    })
}

fn decision_to_json(
    action: &Array1<f64>,
    decision: &control_control::mpc::PulsedMpcDecision,
) -> Value {
    json!({
        "action": action.to_vec(),
        "decision": {
            "mpc_objective": decision.mpc_objective,
            "constraint_slack": decision.constraint_slack,
            "scheduler_state": decision.scheduler_state,
            "bank_feasibility": decision.bank_feasibility,
            "reason": decision.reason,
            "bank_feasible": decision.bank_feasible,
            "safe_action_applied": decision.safe_action_applied,
            "burn_components_masked": decision.burn_components_masked,
        },
    })
}

fn write_markdown(path: &PathBuf, payload: &Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    let results = payload["results"].as_object().expect("results object");
    let mut rows = vec![
        "| Case | Samples | Mean us | Median us | p95 us | p99 us |".to_string(),
        "|---|---:|---:|---:|---:|---:|".to_string(),
    ];
    for (name, result) in results {
        let stats = &result["stats"];
        rows.push(format!(
            "| `{}` | {} | {:.6} | {:.6} | {:.6} | {:.6} |",
            name,
            stats["samples"].as_u64().unwrap(),
            stats["mean_us"].as_f64().unwrap(),
            stats["median_us"].as_f64().unwrap(),
            stats["p95_us"].as_f64().unwrap(),
            stats["p99_us"].as_f64().unwrap()
        ));
    }
    let body = format!(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n\
<!-- Commercial license available -->\n\
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->\n\
<!-- ORCID: 0009-0009-3560-0851 -->\n\
<!-- Contact: www.anulum.li | protoscience@anulum.li -->\n\
<!-- SCPN Control — Rust pulsed-shot MPC adapter benchmark report. -->\n\n\
# Rust Pulsed MPC Adapter Benchmark\n\n\
- Evidence class: `{}`\n\
- Production claim allowed: `{}`\n\
- Steps: `{}`\n\
- Warmup: `{}`\n\n\
## Results\n\n{}\n\n\
## Claim boundary\n\n\
This is local regression evidence for the Rust pulsed MPC admission adapter.\n\
It is not target-hardware timing evidence and does not admit facility PCS claims.\n\n\
Payload SHA-256: `{}`\n",
        payload["evidence_class"].as_str().unwrap(),
        payload["production_claim_allowed"].as_bool().unwrap(),
        payload["steps"].as_u64().unwrap(),
        payload["warmup"].as_u64().unwrap(),
        rows.join("\n"),
        payload["payload_sha256"].as_str().unwrap()
    );
    fs::write(path, body).expect("write Markdown report");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let steps = parse_usize(&args, "--steps", 2_000);
    let warmup = parse_usize(&args, "--warmup", 200);
    assert!(steps > 0, "--steps must be >= 1");

    let mpc = controller();
    let state = array![5.0, 1.0];
    let safe = array![0.0, 0.0];
    let generated_unix_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time after UNIX epoch")
        .as_secs();

    let results = json!({
        "rust_non_burn_mask": measure(steps, warmup, || {
            let decision = mpc
                .plan_pulsed(&state, "flat_top", true, &[true, false], &safe, 12.0)
                .expect("valid non-burn pulsed decision");
            decision_to_json(&decision.action, &decision)
        }),
        "rust_burn_feasible": measure(steps, warmup, || {
            let decision = mpc
                .plan_pulsed(&state, "burn", true, &[true, true], &safe, 12.0)
                .expect("valid burn pulsed decision");
            decision_to_json(&decision.action, &decision)
        }),
        "rust_burn_infeasible_safe": measure(steps, warmup, || {
            let decision = mpc
                .plan_pulsed(&state, "burn", false, &[true, true], &safe, -0.5)
                .expect("valid infeasible-bank pulsed decision");
            decision_to_json(&decision.action, &decision)
        }),
    });

    let mut payload = json!({
        "schema_version": "scpn-control.rust-pulsed-mpc-adapter-benchmark.v1",
        "generated_unix_s": generated_unix_s,
        "command": args.join(" "),
        "evidence_class": "local_regression",
        "production_claim_allowed": false,
        "steps": steps,
        "warmup": warmup,
        "context": {
            "target_os": env::consts::OS,
            "target_arch": env::consts::ARCH,
            "current_dir": env::current_dir().expect("current dir").display().to_string(),
        },
        "results": results,
    });
    let encoded = serde_json::to_vec(&payload).expect("encode payload");
    let digest = Sha256::digest(&encoded);
    payload["payload_sha256"] = Value::String(format!("{digest:x}"));

    if let Some(path) = parse_path(&args, "--json-out") {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create report directory");
        }
        fs::write(
            &path,
            serde_json::to_string_pretty(&payload).expect("pretty JSON") + "\n",
        )
        .expect("write JSON report");
    }
    if let Some(path) = parse_path(&args, "--md-out") {
        write_markdown(&path, &payload);
    }
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}
