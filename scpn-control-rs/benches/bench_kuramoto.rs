use control_math::kuramoto;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

// DIII-D tokamak-scale oscillator counts.
// 1k: single flux surface, 16k: full poloidal cross-section,
// 65k: high-res EFIT reconstruction grid.
const DIIID_SIZES: &[usize] = &[64, 256, 1_000, 4_096, 16_384, 65_536];

fn make_population(n: usize) -> (Vec<f64>, Vec<f64>) {
    let pi = std::f64::consts::PI;
    let theta: Vec<f64> = (0..n)
        .map(|i| -pi + (i as f64) * 2.0 * pi / n as f64)
        .collect();
    let omega: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 - n as f64 / 2.0)).collect();
    (theta, omega)
}

// ── 1. Single-layer Kuramoto step (baseline: K only, no ζ) ──────────

fn bench_kuramoto_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_step");
    for &n in DIIID_SIZES {
        let (theta, omega) = make_population(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                kuramoto::kuramoto_sakaguchi_step(
                    black_box(&theta),
                    black_box(&omega),
                    black_box(0.01),
                    black_box(2.0),
                    black_box(0.0),
                    black_box(0.0),
                    black_box(None),
                )
            });
        });
    }
    group.finish();
}

// ── 2. ζ sin(Ψ−θ) global driver vs baseline ─────────────────────────

fn bench_zeta_driver(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeta_driver_vs_baseline");
    for &n in &[1_000, 4_096, 16_384, 65_536] {
        let (theta, omega) = make_population(n);

        // Baseline: ζ=0 (pure Kuramoto coupling)
        group.bench_with_input(BenchmarkId::new("zeta_0", n), &n, |b, _| {
            b.iter(|| {
                kuramoto::kuramoto_sakaguchi_step(
                    black_box(&theta),
                    black_box(&omega),
                    black_box(0.01),
                    black_box(2.0),
                    black_box(0.0),
                    black_box(0.0),
                    black_box(None),
                )
            });
        });

        // With driver: ζ=0.5, Ψ=0.3
        group.bench_with_input(BenchmarkId::new("zeta_0.5", n), &n, |b, _| {
            b.iter(|| {
                kuramoto::kuramoto_sakaguchi_step(
                    black_box(&theta),
                    black_box(&omega),
                    black_box(0.01),
                    black_box(2.0),
                    black_box(0.0),
                    black_box(0.5),
                    black_box(Some(0.3)),
                )
            });
        });
    }
    group.finish();
}

// ── 3. Knm inter-layer PAC bench (16 layers, multi-step) ────────────
//
// Simulates the 16-layer UPDE outer loop: for each target layer m,
// compute order parameters for all source layers n, apply PAC gate
// g = 1 + γ(1 − R_n), accumulate inter-layer coupling.

fn bench_knm_interlayer_pac(c: &mut Criterion) {
    let n_layers: usize = 16;
    let n_per_layer: usize = 256;
    let pac_gamma: f64 = 1.0;
    let dt: f64 = 0.005;

    // Paper 27 Knm: K_base * exp(-alpha * |i-j|)
    let k_base: f64 = 0.45;
    let k_alpha: f64 = 0.3;
    let mut knm: Vec<Vec<f64>> = (0..n_layers)
        .map(|i| {
            (0..n_layers)
                .map(|j| k_base * (-k_alpha * (i as f64 - j as f64).abs()).exp())
                .collect()
        })
        .collect();
    // Calibration anchors
    knm[0][1] = 0.302;
    knm[1][0] = 0.302;
    knm[1][2] = 0.201;
    knm[2][1] = 0.201;
    knm[2][3] = 0.252;
    knm[3][2] = 0.252;
    knm[3][4] = 0.154;
    knm[4][3] = 0.154;
    // Cross-hierarchy
    knm[0][15] = 0.05_f64.max(knm[0][15]);
    knm[15][0] = 0.05_f64.max(knm[15][0]);
    knm[4][6] = 0.15_f64.max(knm[4][6]);
    knm[6][4] = 0.15_f64.max(knm[6][4]);

    // Per-layer populations
    let layers: Vec<(Vec<f64>, Vec<f64>)> = (0..n_layers)
        .map(|_| make_population(n_per_layer))
        .collect();

    let mut group = c.benchmark_group("knm_interlayer_pac");

    // Full 16-layer UPDE step (mirrors upde.py logic)
    group.bench_function("16layer_256osc_pac", |b| {
        b.iter(|| {
            // Pre-compute per-layer order parameters
            let order: Vec<(f64, f64)> = layers
                .iter()
                .map(|(th, _)| kuramoto::order_parameter(th))
                .collect();

            let psi_global = {
                let (re, im) = order
                    .iter()
                    .fold((0.0_f64, 0.0_f64), |(re, im), &(r, psi)| {
                        (re + r * psi.cos(), im + r * psi.sin())
                    });
                im.atan2(re)
            };

            let mut out_layers = Vec::with_capacity(n_layers);

            for m in 0..n_layers {
                let (ref theta, ref omega) = layers[m];
                let (r_m, psi_m) = order[m];
                let kr_intra = black_box(knm[m][m]) * r_m;

                let mut theta_next = vec![0.0_f64; n_per_layer];
                for i in 0..n_per_layer {
                    let th = theta[i];
                    let mut dth = omega[i] + kr_intra * (psi_m - th).sin();

                    // Inter-layer with PAC gate
                    for n in 0..n_layers {
                        if n == m {
                            continue;
                        }
                        let (r_n, psi_n) = order[n];
                        let pac_gate = 1.0 + pac_gamma * (1.0 - r_n);
                        dth += pac_gate * knm[n][m] * r_n * (psi_n - th).sin();
                    }

                    // Global driver ζ=0.5
                    dth += 0.5 * (psi_global - th).sin();

                    theta_next[i] = kuramoto::wrap_phase(th + dt * dth);
                }
                out_layers.push(theta_next);
            }
            black_box(&out_layers);
        });
    });

    // Same without PAC (γ=0) for comparison
    group.bench_function("16layer_256osc_no_pac", |b| {
        b.iter(|| {
            let order: Vec<(f64, f64)> = layers
                .iter()
                .map(|(th, _)| kuramoto::order_parameter(th))
                .collect();

            let psi_global = {
                let (re, im) = order
                    .iter()
                    .fold((0.0_f64, 0.0_f64), |(re, im), &(r, psi)| {
                        (re + r * psi.cos(), im + r * psi.sin())
                    });
                im.atan2(re)
            };

            let mut out_layers = Vec::with_capacity(n_layers);

            for m in 0..n_layers {
                let (ref theta, ref omega) = layers[m];
                let (r_m, psi_m) = order[m];
                let kr_intra = black_box(knm[m][m]) * r_m;

                let mut theta_next = vec![0.0_f64; n_per_layer];
                for i in 0..n_per_layer {
                    let th = theta[i];
                    let mut dth = omega[i] + kr_intra * (psi_m - th).sin();

                    for n in 0..n_layers {
                        if n == m {
                            continue;
                        }
                        let (r_n, psi_n) = order[n];
                        // No PAC gate (γ=0 → gate=1.0)
                        dth += knm[n][m] * r_n * (psi_n - th).sin();
                    }

                    dth += 0.5 * (psi_global - th).sin();
                    theta_next[i] = kuramoto::wrap_phase(th + dt * dth);
                }
                out_layers.push(theta_next);
            }
            black_box(&out_layers);
        });
    });

    group.finish();
}

// ── 4. Order parameter scaling ───────────────────────────────────────

fn bench_kuramoto_order_param(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_order_param");
    for &n in &[256, 4096, 16384, 65536] {
        let theta: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| kuramoto::order_parameter(black_box(&theta)));
        });
    }
    group.finish();
}

// ── 5. Multi-step convergence (DIII-D 100-step trajectory) ───────────

fn bench_kuramoto_run_diiid(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_run_diiid");
    // DIII-D: 4096 modes, 100 steps @ 10 kHz → 10 ms window
    for &(n, steps) in &[(4096_usize, 100_usize), (16384, 100)] {
        let (theta, omega) = make_population(n);
        group.bench_with_input(BenchmarkId::new(format!("N{n}_S{steps}"), n), &n, |b, _| {
            b.iter(|| {
                kuramoto::kuramoto_sakaguchi_run(
                    black_box(&theta),
                    black_box(&omega),
                    black_box(steps),
                    black_box(0.0001), // dt = 100 µs (10 kHz)
                    black_box(2.0),
                    black_box(0.0),
                    black_box(0.5),
                    black_box(Some(0.3)),
                )
            });
        });
    }
    group.finish();
}

// ── 6. DIII-D run with Lyapunov exponent tracking ────────────────────

fn bench_kuramoto_run_lyapunov(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_run_lyapunov");
    for &(n, steps) in &[(4096_usize, 100_usize), (16384, 100)] {
        let (theta, omega) = make_population(n);
        group.bench_with_input(BenchmarkId::new(format!("N{n}_S{steps}"), n), &n, |b, _| {
            b.iter(|| {
                kuramoto::kuramoto_run_lyapunov(
                    black_box(&theta),
                    black_box(&omega),
                    black_box(steps),
                    black_box(0.0001),
                    black_box(2.0),
                    black_box(0.0),
                    black_box(0.5),
                    black_box(Some(0.3)),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kuramoto_step,
    bench_zeta_driver,
    bench_knm_interlayer_pac,
    bench_kuramoto_order_param,
    bench_kuramoto_run_diiid,
    bench_kuramoto_run_lyapunov,
);
criterion_main!(benches);
