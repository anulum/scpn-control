<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — libFuzzer fuzzing and robustness guide. -->

# Fuzzing the Rust surfaces

The Rust crates under `scpn-control-rs` accept two kinds of input that a unit
test does not fully cover: untrusted text/byte streams (configuration files,
external solver output) and high-volume numeric arrays that cross the PyO3 FFI
boundary. The `scpn-control-rs/fuzz` package drives those surfaces with
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer + nightly
AddressSanitizer) so that malformed, adversarial, or extreme-but-finite input is
exercised for real rather than assumed safe.

Fuzzing is **not** a per-commit blocker. A fast `cargo fuzz build` smoke check
keeps the targets compiling; the time-boxed campaign runs on the nightly
`Fuzz nightly` workflow (and on demand) and fails closed on any crash, timeout,
leak, or out-of-memory artefact.

## Targets and surfaces

| Target | Surface class | Code under test |
|---|---|---|
| `config_json` | parser | `ReactorConfig` JSON deserialisation (`control-types`) |
| `vmec_import` | parser | `import_vmec_like_text` VMEC-like boundary text (`control-core`) |
| `bout_stability` | parser | `parse_bout_stability` BOUT++ output (`control-core`) |
| `capacitor_bank` | safety-critical numeric adapter / FFI | series-RLC discharge ledger (`control-control`) |
| `kuramoto_kernel` | vector kernel / FFI | Kuramoto-Sakaguchi phase step (`control-math`) |

The `capacitor_bank` and `kuramoto_kernel` targets exercise the same Rust code
reached through the PyO3 `PyCapacitorBankModel` and Kuramoto bindings, so the FFI
numeric boundary is covered without driving a Python interpreter inside the
fuzzer.

Each target asserts the invariant appropriate to its surface:

- the parser targets require that malformed input returns an error and that any
  accepted value survives a serialise/export round-trip — never a panic;
- `capacitor_bank` asserts that an *admitted* energy balance
  (`energy_balance_passed`) carries only finite, physically-ordered quantities,
  so a passed admission can never smuggle a NaN/Inf ledger or a negative ohmic
  dissipation;
- `kuramoto_kernel` asserts the order-parameter magnitude stays finite and in
  `[0, 1]` and that `wrap_phase` lands in `(-π, π]` for finite phases.

## Running a campaign

The campaign is driven by `tools/run_fuzz_campaign.py`, which copies the tracked
seed corpus into the working corpus, records provenance, runs each target for a
bounded wall-clock time, collects any reproducer artefacts, and fails closed:

```bash
# Build-only smoke check (fast):
python tools/run_fuzz_campaign.py --build-only

# Full campaign with evidence report (default 300 s per target):
python tools/run_fuzz_campaign.py \
  --max-total-time 300 \
  --json-out artifacts/fuzz/fuzz_campaign_report.json \
  --markdown-out artifacts/fuzz/fuzz_campaign_report.md
```

A single target can also be driven directly:

```bash
cd scpn-control-rs
cargo +nightly fuzz run capacitor_bank -- -max_total_time=60 -timeout=10
```

## Evidence and triage

Every campaign emits a JSON evidence document
(`scpn-control.fuzz-campaign-evidence.v1`) that binds the run to its inputs and
environment: the nightly Rust toolchain, cargo-fuzz version, target triple,
sanitiser configuration, per-target executed-unit counts and duration, the
SHA-256 of every seed (with a per-target aggregate digest), and the list of any
reproducer artefacts. The embedded triage verdict admits a campaign only when
**every** requested target ran and **no** target crashed, timed out, leaked, or
produced an artefact; a missing target is itself a fail-closed failure. The
report carries `production_claim_allowed: false` — fuzzing is a robustness gate,
not a performance or correctness certification.

Reproducer artefacts (`crash-*`, `leak-*`, `timeout-*`, `oom-*`, `slow-unit-*`)
are written under `scpn-control-rs/fuzz/artifacts/<target>/` and uploaded by the
nightly workflow. A confirmed reproducer is copied into
`scpn-control-rs/fuzz/seeds/<target>/` as a tracked regression seed so it runs on
every subsequent campaign.

## Seed corpus

Tracked seeds live in `scpn-control-rs/fuzz/seeds/<target>/`; the working corpus
(`fuzz/corpus/`), build output (`fuzz/target/`), and artefacts
(`fuzz/artifacts/`) are git-ignored. `scpn-control-rs/fuzz/seeds/SHA256SUMS`
records the seed digests so a reviewer can verify the corpus without running the
campaign.

## Recorded findings

- **Capacitor-bank discharge denial-of-service** (`capacitor_bank` target,
  `timeout-be2a810f…`). A denormal capacitance combined with a ~1e103 H/Ω
  inductance and resistance overflowed the assembled Van Loan gramian matrix
  norm to `+inf`. In the Rust `matrix_exp` scaling-and-squaring routine the
  squaring exponent `log2(norm).ceil() as u32` then saturated `+inf` to
  `u32::MAX`, turning the squaring loop into ~4.3×10⁹ matrix multiplies — a
  multi-minute hang on a single input. Fixed by failing closed on a non-finite
  norm (no scaling, prompt non-finite result that callers reject) and clamping
  the squaring exponent; the Python path already failed closed through
  `scipy.linalg.expm`. Covered by Rust regression tests in `h_infinity.rs` and
  `capacitor_bank.rs` and by the tracked regression seed.
