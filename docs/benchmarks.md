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

## CI benchmark job

The CI workflow runs:

- `cargo bench --workspace`

and uploads benchmark artifacts as:

- `bench-results`

from:

- `scpn-control-rs/target/criterion/`

## Reproducibility notes

- Run benchmarks on an idle machine.
- Keep `--n-bench` fixed for comparable CLI timing runs.
- Compare same Python/Rust versions and CPU class when evaluating trends.
