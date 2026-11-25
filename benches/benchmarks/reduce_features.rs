//! The feature reduction's fit on extracted features, in polars' pool and on
//! one thread (for the speedup). Extraction runs once, outside the timed loop.

use criterion::{Criterion, criterion_group};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

use polars::prelude::*;
use polars_core::runtime::RAYON;
use tsfx::extract::{
    DynamicGroupBySettings, ExtractionSettings, FeatureSetting, collect_in_memory, lazy_feature_df,
};
use tsfx::reduction::Reduction;

const STOCK_VALUE_COLS: [&str; 5] = ["open", "high", "low", "close", "volume"];

/// The 1BRC measurements (not in git, see `python_benchmarks/1brc`).
const BRC_PATH: &str = "test_dataset/measurements.txt";
/// Enough rows for every station to appear (about 9,000 series, more than
/// `max_samples`, so the sampled path runs).
const BRC_ROWS: usize = 10_000_000;

fn extract(lf: LazyFrame, opts: ExtractionSettings) -> DataFrame {
    collect_in_memory(lazy_feature_df(lf, opts).unwrap()).unwrap()
}

/// Efficient features of the stocks in one-year windows.
fn stock_features(value_cols: &[&str]) -> DataFrame {
    let lf = LazyCsvReader::new(PlRefPath::new("test_dataset/all_stocks_5yr.csv"))
        .finish()
        .unwrap()
        .drop_nulls(None)
        .with_columns(
            STOCK_VALUE_COLS
                .iter()
                .map(|c| col(*c).cast(DataType::Float64))
                .collect::<Vec<_>>(),
        );
    extract(
        lf,
        ExtractionSettings {
            grouping_cols: vec!["Name".to_string()],
            value_cols: value_cols.iter().map(|c| c.to_string()).collect(),
            feature_setting: FeatureSetting::Efficient,
            config_path: None,
            dynamic_settings: Some(DynamicGroupBySettings {
                time_col: "date".to_string(),
                every: "1y".to_string(),
                period: "1y".to_string(),
                offset: "0y".to_string(),
                datetime_format: Some("%Y-%m-%d".to_string()),
            }),
            scaling: None,
            reduction: None,
        },
    )
}

/// Efficient features of the 1BRC stations.
fn brc_features() -> DataFrame {
    let schema = Schema::from_iter([
        Field::new("station".into(), DataType::String),
        Field::new("temperature".into(), DataType::Float64),
    ]);
    let lf = LazyCsvReader::new(PlRefPath::new(BRC_PATH))
        .with_has_header(false)
        .with_separator(b';')
        .with_quote_char(None)
        .with_schema(Some(Arc::new(schema)))
        .with_n_rows(Some(BRC_ROWS))
        .finish()
        .unwrap();
    extract(
        lf,
        ExtractionSettings {
            grouping_cols: vec!["station".to_string()],
            value_cols: vec!["temperature".to_string()],
            feature_setting: FeatureSetting::Efficient,
            config_path: None,
            dynamic_settings: None,
            scaling: None,
            reduction: None,
        },
    )
}

/// The default fit, in polars' pool and with its rayon work in a one-thread
/// pool (polars' own sort of the rows still uses polars' pool).
fn bench_fit(c: &mut Criterion, name: &str, features: &DataFrame, keys: &[String]) {
    let reduction = Reduction::default();
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    c.bench_function(name, |b| {
        b.iter(|| RAYON.install(|| reduction.fit(black_box(features), keys).unwrap()))
    });
    c.bench_function(&format!("{name}_1thread"), |b| {
        b.iter(|| one_thread.install(|| reduction.fit(black_box(features), keys).unwrap()))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let stock_keys = vec!["Name".to_string(), "date".to_string()];
    bench_fit(
        c,
        "reduce_stocks_close",
        &stock_features(&["close"]),
        &stock_keys,
    );
    bench_fit(
        c,
        "reduce_stocks_all",
        &stock_features(&STOCK_VALUE_COLS),
        &stock_keys,
    );
    if Path::new(BRC_PATH).exists() {
        bench_fit(c, "reduce_1brc", &brc_features(), &["station".to_string()]);
    } else {
        eprintln!("skipping reduce_1brc: {BRC_PATH} not found (see python_benchmarks/1brc)");
    }
}

fn reduce_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(20))
}

criterion_group! {
    name = benches;
    config = reduce_config();
    targets = criterion_benchmark
}
