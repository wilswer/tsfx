//! The noise step on real features: copies of real stock features shuffled
//! across rows are noise by construction. About 1 - `noise_sensitivity` of
//! them get through by design, so the test checks rates over several seeds,
//! and that the step drops nearly no real feature. Slow in debug builds: run
//! it with `cargo test --release`.

use polars::prelude::*;
use tsfx::extract::{
    DynamicGroupBySettings, ExtractionSettings, FeatureSetting, collect_in_memory, lazy_feature_df,
};
use tsfx::reduction::Reduction;

/// SplitMix64, as in the crate (whose generator is private), for the shuffles.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Fisher-Yates; the slight modulo bias does not matter here.
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

/// `Efficient` features of the closing prices in one-year windows.
fn stock_features() -> DataFrame {
    let lf = LazyCsvReader::new(PlRefPath::new("test_dataset/all_stocks_5yr.csv"))
        .finish()
        .unwrap()
        .drop_nulls(None)
        .select([
            col("Name"),
            col("date"),
            col("close").cast(DataType::Float64),
        ]);
    let opts = ExtractionSettings {
        grouping_cols: vec!["Name".to_string()],
        value_cols: vec!["close".to_string()],
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
    };
    collect_in_memory(lazy_feature_df(lf, opts).unwrap()).unwrap()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds: run with --release")]
fn test_noise_step_on_stock_features() {
    let keys = vec!["Name".to_string(), "date".to_string()];
    let features = stock_features();
    let base = Reduction::default().fit(&features, &keys).unwrap();
    let sources: Vec<String> = base
        .kept
        .iter()
        .filter(|name| name.as_str() != "length")
        .take(20)
        .cloned()
        .collect();
    assert_eq!(sources.len(), 20);

    let mut copies_kept = 0;
    for seed in 0..5u64 {
        let mut g = SplitMix64(1000 + seed);
        let mut df = features.clone();
        for (i, name) in sources.iter().enumerate() {
            let mut order: Vec<IdxSize> = (0..df.height() as IdxSize).collect();
            g.shuffle(&mut order);
            let copy = features
                .column(name)
                .unwrap()
                .take(&IdxCa::from_vec("order".into(), order))
                .unwrap()
                .with_name(format!("copy_{i}").into());
            df.with_column(copy).unwrap();
        }
        let fit = Reduction {
            seed,
            ..Default::default()
        }
        .fit(&df, &keys)
        .unwrap();
        copies_kept += fit
            .kept
            .iter()
            .filter(|name| name.starts_with("copy_"))
            .count();

        // Real features that reached the noise step, and those it dropped
        let report = &fit.report;
        let feature = report.column("feature").unwrap().str().unwrap();
        let score = report.column("noise_score").unwrap().f64().unwrap();
        let dropped_by = report.column("dropped_by").unwrap().str().unwrap();
        let (mut reached, mut dropped) = (0, 0);
        for i in 0..report.height() {
            if feature.get(i).unwrap().starts_with("copy_") {
                // every copy reached the noise step, so the step decided
                assert!(
                    score.get(i).is_some(),
                    "{} did not reach the noise step",
                    feature.get(i).unwrap()
                );
                continue;
            }
            if score.get(i).is_none() {
                continue;
            }
            reached += 1;
            if dropped_by.get(i) == Some("noise") {
                dropped += 1;
            }
        }
        let copies = fit
            .kept
            .iter()
            .filter(|name| name.starts_with("copy_"))
            .count();
        eprintln!(
            "reduction_stocks margin: seed {seed}: {copies} of 20 copies kept, \
             {dropped} of {reached} real features dropped as noise"
        );
        assert!(
            dropped * 100 <= reached,
            "seed {seed}: {dropped} of {reached} real features dropped as noise"
        );
    }
    assert!(
        copies_kept <= 5,
        "{copies_kept} of 100 shuffled copies kept"
    );
}
