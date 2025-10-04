use alphaforge::MLStrategy;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

fn prediction_benchmark(c: &mut Criterion) {
    let prices: Array1<f64> = Array1::from_vec(
        (0..1000)
            .map(|i| 100.0 + i as f64 * 0.1 + (i as f64 * 0.1).sin() * 5.0)
            .collect(),
    );

    let mut strategy = MLStrategy::new(0.02, 20);
    strategy.train(&prices).unwrap();

    c.bench_function("generate_signal", |b| {
        b.iter(|| {
            let recent = prices.slice(ndarray::s![-20..]).to_owned();
            black_box(strategy.generate_signal(&recent))
        });
    });
}

criterion_group!(benches, prediction_benchmark);
criterion_main!(benches);
