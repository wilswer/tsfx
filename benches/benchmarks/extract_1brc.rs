use criterion::{Criterion, SamplingMode, criterion_group};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use polars::prelude::*;
use tsfx::extract::{ExtractionSettings, FeatureSetting, lazy_feature_df};

/// The 1BRC measurements, generated with `python_benchmarks/1brc/generate_data.py`
/// (not in git: about 16 GB for the full billion rows).
const PATH: &str = "test_data/measurements.txt";
/// 100M rows: about 10,000 stations with about 10,000 values each, so long
/// series where per-value cost dominates (stocks has short ones).
const ROWS: usize = 100_000_000;

/// The first `ROWS` rows, loaded once outside the timed loop.
fn load_1brc() -> DataFrame {
    let schema = Schema::from_iter([
        Field::new("station".into(), DataType::String),
        Field::new("temperature".into(), DataType::Float64),
    ]);
    LazyCsvReader::new(PlRefPath::new(PATH))
        .with_has_header(false)
        .with_separator(b';')
        .with_quote_char(None)
        .with_schema(Some(Arc::new(schema)))
        .with_n_rows(Some(ROWS))
        .finish()
        .unwrap()
        .collect()
        .unwrap()
}

fn bench_tier(c: &mut Criterion, df: &DataFrame, name: &str, tier: FeatureSetting) {
    let mut group = c.benchmark_group("1brc");
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function(name, |b| {
        b.iter(|| {
            lazy_feature_df(
                black_box(df.clone().lazy()),
                ExtractionSettings {
                    grouping_cols: vec!["station".to_string()],
                    value_cols: vec!["temperature".to_string()],
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
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    if !Path::new(PATH).exists() {
        eprintln!("skipping 1brc benches: {PATH} not found (see python_benchmarks/1brc)");
        return;
    }
    let df = load_1brc();
    bench_tier(c, &df, "brc_minimal", FeatureSetting::Minimal);
    bench_tier(c, &df, "brc_efficient", FeatureSetting::Efficient);
}

fn brc_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(60))
}

criterion_group! {
    name = benches;
    config = brc_config();
    targets = criterion_benchmark
}
