use control_math::kuramoto;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn bench_kuramoto_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_step");
    for &n in &[64, 256, 1000, 4096, 16384] {
        let theta: Vec<f64> = (0..n)
            .map(|i| -std::f64::consts::PI + (i as f64) * 2.0 * std::f64::consts::PI / n as f64)
            .collect();
        let omega: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 - n as f64 / 2.0)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
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

fn bench_kuramoto_order_param(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_order_param");
    for &n in &[256, 4096, 16384] {
        let theta: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| kuramoto::order_parameter(black_box(&theta)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kuramoto_step, bench_kuramoto_order_param);
criterion_main!(benches);
