// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Control — Native PID step-latency benchmark

//! Measure the native PID over the same sine-error sequence as the Python lane.

use control_control::pid::PIDController;
use serde_json::json;
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

fn argument(args: &[String], name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(default);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{name} needs a value"))?;
    Ok(value.parse::<usize>()?)
}

fn percentile(ordered: &[u128], fraction: f64) -> f64 {
    let index = (fraction * (ordered.len() - 1) as f64).round() as usize;
    ordered[index] as f64 / 1000.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let iterations = argument(&args, "--iterations", 5000)?;
    let warmup = argument(&args, "--warmup", 500)?;
    if iterations == 0 {
        return Err("--iterations must be positive".into());
    }

    let mut pid = PIDController::new(1.0, 0.1, 0.05)?;
    for i in 0..warmup {
        let error = ((i as f64) * 0.01).sin();
        black_box(pid.step(black_box(error))?);
    }
    let mut samples_ns = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let error = (((warmup + i) as f64) * 0.01).sin();
        let start = Instant::now();
        black_box(pid.step(black_box(error))?);
        samples_ns.push(start.elapsed().as_nanos());
    }
    samples_ns.sort_unstable();
    let mean_us = samples_ns.iter().sum::<u128>() as f64 / iterations as f64 / 1000.0;
    let payload = json!({
        "schema": "scpn-control.pid-step-latency.v1",
        "backend": "native-rust",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "platform": { "arch": env::consts::ARCH, "os": env::consts::OS },
        "parameters": { "iterations": iterations, "warmup": warmup, "kp": 1.0, "ki": 0.1, "kd": 0.05 },
        "stats": {
            "n": iterations,
            "p50_us": percentile(&samples_ns, 0.50),
            "p95_us": percentile(&samples_ns, 0.95),
            "p99_us": percentile(&samples_ns, 0.99),
            "mean_us": mean_us,
            "min_us": samples_ns[0] as f64 / 1000.0,
            "max_us": samples_ns[iterations - 1] as f64 / 1000.0
        }
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}
