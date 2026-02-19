use control_core::transport::{transport_step, TransportSolver};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_transport_step(c: &mut Criterion) {
    c.bench_function("transport_single_step", |b| {
        let mut solver = TransportSolver::new();
        b.iter(|| {
            transport_step(&mut solver, black_box(20.0), black_box(0.01))
                .expect("transport step should succeed for finite inputs");
            black_box(solver.profiles.te[0]);
        });
    });
}

criterion_group!(benches, bench_transport_step);
criterion_main!(benches);
