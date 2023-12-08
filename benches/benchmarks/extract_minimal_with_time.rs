use criterion::{black_box, criterion_group, Criterion};

use polars::prelude::*;
use tsfx::extract::{lazy_feature_df, DynamicGroupBySettings, ExtractionSettings};

fn criterion_benchmark(c: &mut Criterion) {
    let df = LazyCsvReader::new("test_data/all_stocks_5yr.csv")
        .with_dtype_overwrite(Some(&schema))
        .finish()
        .unwrap()
        .drop_nulls();
    c.bench_function("extract_minimal_with_time", |b| {
        b.iter(|| {
            lazy_feature_df(
                black_box(df),
                ExtractionSettings {
                    grouping_col: "Name".to_string(),
                    value_cols: vec![
                        "open".to_string(),
                        "high".to_string(),
                        "low".to_string(),
                        "close".to_string(),
                        "volume".to_string(),
                    ],
                    dynamic_settings: Some(DynamicGroupBySettings {
                        time_col: "date".to_string(),
                        every: "1y".to_string(),
                        period: "1y".to_string(),
                        offset: "0".to_string(),
                        datetime_format: "%Y-%m-%d".to_string(),
                    }),
                },
            )
            .collect()
            .unwrap()
        })
    });
}

fn few_samples() -> Criterion {
    Criterion::default().sample_size(100)
}

criterion_group! {
    name = benches;
    config = few_samples();
    targets = criterion_benchmark
}
