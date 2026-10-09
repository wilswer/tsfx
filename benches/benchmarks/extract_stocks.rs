use criterion::{Criterion, criterion_group};
use std::hint::black_box;
use std::time::Duration;

use polars::prelude::*;
use tsfx::extract::{ExtractionSettings, FeatureSetting, lazy_feature_df};

const VALUE_COLS: [&str; 5] = ["open", "high", "low", "close", "volume"];

/// The stocks dataset (619k rows, 505 groups), loaded once outside the timed loop.
fn load_stocks() -> DataFrame {
    LazyCsvReader::new(PlRefPath::new("test_data/all_stocks_5yr.csv"))
        .finish()
        .unwrap()
        .drop_nulls(None)
        .with_columns(
            VALUE_COLS
                .iter()
                .map(|c| col(*c).cast(DataType::Float64))
                .collect::<Vec<_>>(),
        )
        .collect()
        .unwrap()
}

fn bench_tier(c: &mut Criterion, df: &DataFrame, name: &str, tier: FeatureSetting) {
    c.bench_function(name, |b| {
        b.iter(|| {
            lazy_feature_df(
                black_box(df.clone().lazy()),
                ExtractionSettings {
                    grouping_cols: vec!["Name".to_string()],
                    value_cols: VALUE_COLS.iter().map(|c| c.to_string()).collect(),
                    feature_setting: tier.clone(),
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

fn criterion_benchmark(c: &mut Criterion) {
    let df = load_stocks();
    bench_tier(c, &df, "stocks_minimal", FeatureSetting::Minimal);
    bench_tier(c, &df, "stocks_efficient", FeatureSetting::Efficient);
}

fn stocks_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(20))
}

criterion_group! {
    name = benches;
    config = stocks_config();
    targets = criterion_benchmark
}
