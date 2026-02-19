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
