use criterion::{criterion_group, Criterion};
use std::hint::black_box;

use polars::prelude::*;
use tsfx::extract::{lazy_feature_df, ExtractionSettings, FeatureSetting};

fn criterion_benchmark(c: &mut Criterion) {
    let cdf = df![
        "id" =>    ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "value" => [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    ]
    .unwrap()
    .lazy();

    c.bench_function("extract_minimal", |b| {
        b.iter(|| {
            lazy_feature_df(
                black_box(cdf.clone()),
                ExtractionSettings {
                    grouping_col: "id".to_string(),
                    value_cols: vec!["value".to_string()],
                    feature_setting: FeatureSetting::Minimal,
                    config_path: None,
                    dynamic_settings: None,
                },
            )
            .unwrap()
            .collect()
            .unwrap()
        })
    });
}

fn few_samples() -> Criterion {
    Criterion::default().sample_size(1000)
}

criterion_group! {
    name = benches;
    config = few_samples();
    targets = criterion_benchmark
}
