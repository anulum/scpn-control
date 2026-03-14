# Benchmarks

This project has two benchmark tracks:

1. Python CLI micro-benchmark (`scpn-control benchmark`)
2. Rust Criterion benches (`cargo bench --workspace`)

## Python CLI benchmark

Run:

```bash
python -m pip install -e .
scpn-control benchmark --n-bench 5000
```

JSON output:

```bash
scpn-control benchmark --n-bench 5000 --json-out
```

Current outputs include:

- `pid_us_per_step`
- `snn_us_per_step`
- `speedup_ratio`

## Kuramoto Phase Sync — Python vs Rust Speedup

Single `kuramoto_sakaguchi_step()` with ζ=0.5, Ψ=0.3.
Python: NumPy vectorised (AMD Ryzen, single-thread).
Rust: Rayon `par_chunks_mut(64)` + criterion harness.

| N | Python (ms) | Rust (ms) | Speedup |
|------:|------------:|----------:|--------:|
| 64 | 0.050 | 0.003 | 17.3× |
| 256 | 0.029 | 0.033 | 0.9× |
| 1 000 | 0.087 | 0.062 | 1.4× |
| 4 096 | 0.328 | 0.180 | 1.8× |
| 16 384 | 1.240 | 0.544 | 2.3× |
| 65 536 | 5.010 | 1.980 | 2.5× |

N=64: Rust wins on per-element throughput (no NumPy dispatch overhead).
N=256: parity — NumPy SIMD matches rayon at this size.
N≥1000: Rust rayon parallelism scales; **sub-ms for N=16k** (0.544 ms).

### Knm 16-Layer UPDE PAC Benchmark

Full 16-layer outer loop (16 × 256 oscillators, Paper 27 Knm, ζ=0.5).
Criterion harness, AMD Ryzen.

| Config | Median (µs) | 95% CI |
|--------|------------:|-------:|
| PAC γ=1.0 | 909 | [860, 921] |
| No PAC γ=0 | 811 | [807, 827] |

PAC gate overhead: ~12% (98 µs per step).
See `docs/bench_pac_vs_nopac.vl.json` for Vega-Lite breakdown.

### Lyapunov Exponent vs ζ Strength

N=1000, 200 steps @ dt=1ms, Ψ=0.3 (exogenous driver).

| ζ | λ (K=0) | λ (K=2) |
|------:|--------:|--------:|
| 0.0 | +0.01 | +0.04 |
| 0.1 | −0.03 | −0.02 |
| 0.5 | −0.23 | −0.24 |
| 1.0 | −0.49 | −0.53 |
| 3.0 | −1.65 | −1.83 |
| 5.0 | −3.01 | −3.35 |

λ < 0 ⟹ stable convergence toward Ψ.
See `docs/bench_lyapunov_vs_zeta.vl.json` for Vega-Lite plot.

Benchmark source: `benches/bench_fusion_snn_hook.py` (Python, pytest-benchmark).

### Interactive Visualization

All three benchmark datasets (speedup, λ-vs-ζ, PAC latency) in a single
interactive Vega-Lite chart with legend-click filtering:

`docs/bench_interactive.vl.json`

Open in the [Vega Editor](https://vega.github.io/editor/) or embed via
`<vega-embed>` / `vegaEmbed()`.  Click legend entries to isolate series.

## Gyrokinetic Linear Benchmark (v0.17.0)

The native linear GK eigenvalue solver is benchmarked via
`validation/benchmark_gk_linear.py`:

| Case | Parameters | gamma_max | Dominant | Runtime |
|------|-----------|-----------|----------|---------|
| Cyclone Base Case | R/a=2.78, q=1.4, s_hat=0.78, R/L_Ti=6.9 | >0 | ITG | ~2s (12 k_y, n_theta=32) |
| SPARC mid-radius | R0=1.85, B0=12.2, q=1.8 | finite | — | ~1s (6 k_y) |
| ITER mid-radius | R0=6.2, B0=5.3, q=1.5 | finite | — | ~1s (6 k_y) |

Multi-code comparison (`benchmark_gk_linear.run_multi_code_comparison()`):

| Model | gamma_max | chi_i | chi_e |
|-------|-----------|-------|-------|
| Native GK eigenvalue | from solver | from quasilinear | from quasilinear |
| Quasilinear dispersion | from analytic | from mixing-length | from mixing-length |

Hybrid accuracy (`validation/benchmark_hybrid_accuracy.py`) measures the
correction layer convergence over 20 transport steps with periodic GK
spot-checks.

## Rust Criterion benchmarks

Run from the Rust workspace root:

```bash
cd scpn-control-rs
cargo bench --workspace
```

Current benchmark targets:

- `benches/bench_boris.rs`
- `benches/bench_lif.rs`
- `benches/bench_transport.rs`
- `benches/bench_kuramoto.rs`

Criterion artifacts are generated under:

- `scpn-control-rs/target/criterion/`

## CI benchmark jobs

### Rust Criterion (Job 8)

- `cargo bench --workspace`
- Uploads `bench-results` from `scpn-control-rs/target/criterion/`

### Python phase-sync benchmark — DIII-D scale (Job 9)

Runs `kuramoto_sakaguchi_step` at N=1000 and N=4096 (DIII-D PCS scale),
plus a `RealtimeMonitor.tick()` (16 layers × 50 oscillators).

Gates:
- Single-step P50 < 5 ms (N=4096)
- RealtimeMonitor tick P50 < 50 ms

## Reproducibility notes

- Run benchmarks on an idle machine.
- Keep `--n-bench` fixed for comparable CLI timing runs.
- Compare same Python/Rust versions and CPU class when evaluating trends.
