use control_core::particles::{boris_push_step, ChargedParticle};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_boris_pusher(c: &mut Criterion) {
    c.bench_function("boris_single_step", |b| {
        let mut particle = ChargedParticle {
            x_m: 2.0,
            y_m: 0.0,
            z_m: 0.0,
            vx_m_s: 1.2e6,
            vy_m_s: 3.5e5,
            vz_m_s: -2.0e5,
            charge_c: 1.602_176_634e-19,
            mass_kg: 9.109_383_701_5e-31,
            weight: 1.0,
        };
        b.iter(|| {
            boris_push_step(
                &mut particle,
                black_box([0.0, 0.0, 0.0]),
                black_box([0.0, 0.0, 2.5]),
                black_box(1.0e-9),
            )
            .expect("Boris push should succeed for finite inputs");
            black_box((particle.x_m, particle.y_m, particle.z_m));
        });
    });
}

criterion_group!(benches, bench_boris_pusher);
criterion_main!(benches);
