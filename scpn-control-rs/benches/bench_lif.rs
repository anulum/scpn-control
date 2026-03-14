// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Rust Crate
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ──────────────────────────────────────────────────────────────────────

use control_control::snn::LIFNeuron;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_lif_neuron(c: &mut Criterion) {
    c.bench_function("lif_single_step", |b| {
        let mut neuron = LIFNeuron::new();
        b.iter(|| {
            let spiked = neuron
                .step(black_box(10.0), black_box(1e-3))
                .expect("LIF step should succeed for finite inputs");
            black_box(spiked);
        });
    });
}

criterion_group!(benches, bench_lif_neuron);
criterion_main!(benches);
