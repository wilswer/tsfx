use anyhow::Result;
use polars::prelude::*;

use polars_core::runtime::RAYON;

use crate::error::ExtractionError;
use crate::feature_extractors::aggregators;
use crate::reduction::{Reduction, ReductionFit};
use crate::scaling::{Scaling, scale_features};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureSetting {
    Minimal,
    Efficient,
    Comprehensive,
}

#[derive(Clone, Debug)]
pub struct DynamicGroupBySettings {
    pub time_col: String,
    pub every: String,
    pub period: String,
    pub offset: String,
    pub datetime_format: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ExtractionSettings {
    pub grouping_cols: Vec<String>,
    pub value_cols: Vec<String>,
    pub feature_setting: FeatureSetting,
    pub config_path: Option<String>,
    pub dynamic_settings: Option<DynamicGroupBySettings>,
    /// Scale the features across groups after extraction; None leaves them
    /// as extracted.
    pub scaling: Option<Scaling>,
    /// Drop redundant and uninformative features after extraction, before
    /// scaling; None keeps them all.
    pub reduction: Option<Reduction>,
}

/// The result of an extraction: the features, and the fitted reduction if
/// one was set.
pub struct Extraction {
    /// The extracted features: reduced and scaled as the settings say.
    pub features: LazyFrame,
    /// The fitted reduction (what was kept and why), if one was set.
    pub reduction: Option<ReductionFit>,
}

/// A step that runs on the extracted features. Each step works on whatever
/// feature columns it is given; none assumes another ran before it.
enum PostStep {
    Reduce(Reduction),
    Scale(Scaling),
}

/// Collect `lf` on the in-memory engine, as `collect()` did before polars 2.0.
/// Since 2.0 a plain `collect()` runs lazy queries on the streaming engine,
/// which costs tsfx more per query and per group.
pub fn collect_in_memory(lf: LazyFrame) -> PolarsResult<DataFrame> {
    Ok(lf.collect_with_engine(Engine::InMemory)?.unwrap_single())
}

/// Parse the `field` duration of the dynamic group-by settings. An invalid
/// string is a settings error (a ValueError in Python) rather than the panic
/// of `Duration::parse`.
fn parse_window_duration(field: &str, value: &str) -> Result<Duration, ExtractionError> {
    Duration::try_parse(value).map_err(|e| {
        ExtractionError::InvalidSettings(format!("invalid {field} duration {value:?}: {e}"))
    })
}

/// Extract the features of `df` as `opts` says, as a lazy frame.
pub fn lazy_feature_df(
    df: LazyFrame,
    opts: ExtractionSettings,
) -> Result<LazyFrame, ExtractionError> {
    Ok(extract(df, opts)?.features)
}

/// Extract the features of `df` as `opts` says: extraction, then the
/// post-extraction steps in order (reduce if `opts.reduction` is set, then
/// scale if `opts.scaling` is set), returning the reduction's fit with the
/// features.
pub fn extract(df: LazyFrame, opts: ExtractionSettings) -> Result<Extraction, ExtractionError> {
    if let Some(scaling) = &opts.scaling {
        scaling.validate()?;
    }
    if let Some(reduction) = &opts.reduction {
        reduction.validate()?;
    }
    let features = aggregators(&opts)?;
    // Columns identifying a group (or window), which scaling leaves alone
    let mut key_cols = opts.grouping_cols.clone();
    if let Some(dynamic_settings) = &opts.dynamic_settings {
        key_cols.push(dynamic_settings.time_col.clone());
    }
    let grouping_cols: Vec<Expr> = opts.grouping_cols.into_iter().map(col).collect();
    let mut selected_cols = grouping_cols.clone();
    for val_col in &opts.value_cols {
        selected_cols.push(col(val_col));
    }
    let gdf = if let Some(dynamic_settings) = opts.dynamic_settings {
        let datetime_format = if let Some(dt_fmt) = dynamic_settings.datetime_format {
            dt_fmt
        } else {
            "%Y-%m-%d %H:%S:%M".to_string()
        };
        let time_options = StrptimeOptions {
            format: Some(datetime_format.into()),
            ..Default::default()
        };
        selected_cols.push(col(&dynamic_settings.time_col));
        df.select(&selected_cols).group_by_dynamic(
            col(&dynamic_settings.time_col).str().to_date(time_options),
            &grouping_cols,
            DynamicGroupOptions {
                index_column: dynamic_settings.time_col.into(),
                every: parse_window_duration("every", &dynamic_settings.every)?,
                period: parse_window_duration("period", &dynamic_settings.period)?,
                offset: parse_window_duration("offset", &dynamic_settings.offset)?,
                ..Default::default()
            },
        )
    } else {
        df.select(&selected_cols).group_by(&grouping_cols)
    };
    let mut fdf = collect_in_memory(gdf.agg(features.aggregations))?;
    for (column, plan) in &features.kernels {
        plan.unpack(&mut fdf, column)?;
    }
    let mut steps = Vec::new();
    if let Some(reduction) = opts.reduction {
        steps.push(PostStep::Reduce(reduction));
    }
    if let Some(scaling) = opts.scaling {
        steps.push(PostStep::Scale(scaling));
    }
    let mut reduction_fit = None;
    for step in steps {
        fdf = match step {
            PostStep::Reduce(reduction) => {
                // In polars' pool, so POLARS_MAX_THREADS caps it too (a rayon
                // pool around the caller is not used: test thread counts on
                // Reduction::fit directly)
                let fit = RAYON.install(|| reduction.fit(&fdf, &key_cols))?;
                let reduced = fit.apply(fdf, &key_cols)?;
                reduction_fit = Some(fit);
                reduced
            }
            // Scale in memory too, whatever engine collects the result: the
            // streaming engine merges a column's partial means and variances
            // in thread order, so its scaled features differ in the last bits
            // from run to run
            PostStep::Scale(scaling) => {
                collect_in_memory(scale_features(fdf, &scaling, &key_cols))?
            }
        };
    }
    Ok(Extraction {
        features: fdf.lazy(),
        reduction: reduction_fit,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DynamicGroupBySettings, ExtractionSettings, FeatureSetting, collect_in_memory, extract,
        lazy_feature_df,
    };
    use crate::error::ExtractionError;
    use crate::reduction::Reduction;
    use crate::scaling::Scaling;
    use polars::prelude::*;

    #[test]
    fn test_extract() {
        let df = df![
            "id" =>    ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "value" => [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        ]
        .unwrap()
        .lazy();
        let opts = ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            config_path: None,
            dynamic_settings: None,
            scaling: None,
            reduction: None,
        };
        let gdf = lazy_feature_df(df, opts).unwrap();

        assert_eq!(
            gdf.clone().select([col("length")]).collect().unwrap(),
            df!["length" => [3, 3, 3]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .select([col("value__sum_values")])
                .collect()
                .unwrap(),
            df!["value__sum_values" => [6.0, 6.0, 6.0]].unwrap()
        );
        assert_eq!(
            gdf.clone().select([col("value__mean")]).collect().unwrap(),
            df!["value__mean" => [2.0, 2.0, 2.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .select([col("value__minimum")])
                .collect()
                .unwrap(),
            df!["value__minimum" => [1.0, 1.0, 1.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .select([col("value__maximum")])
                .collect()
                .unwrap(),
            df!["value__maximum" => [3.0, 3.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.select([col("value__median")]).collect().unwrap(),
            df!["value__median" => [2.0, 2.0, 2.0]].unwrap()
        );
    }

    #[test]
    fn test_extract_short_series() {
        let df = df![
            "id" =>    ["a", "b", "c"],
            "value" => [1.0, 2.0, 3.0],
        ]
        .unwrap()
        .lazy();
        let opts = ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            config_path: None,
            dynamic_settings: None,
            scaling: None,
            reduction: None,
        };
        let gdf = lazy_feature_df(df, opts).unwrap();
        println!("{}", gdf.clone().collect().unwrap());
        assert_eq!(
            gdf.clone().select([col("length")]).collect().unwrap(),
            df!["length" => [1, 1, 1]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    ["id"],
                    SortMultipleOptions {
                        ..Default::default()
                    }
                )
                .select([col("value__sum_values")])
                .collect()
                .unwrap(),
            df!["value__sum_values" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    ["id"],
                    SortMultipleOptions {
                        ..Default::default()
                    }
                )
                .select([col("value__mean")])
                .collect()
                .unwrap(),
            df!["value__mean" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    ["id"],
                    SortMultipleOptions {
                        ..Default::default()
                    }
                )
                .select([col("value__minimum")])
                .collect()
                .unwrap(),
            df!["value__minimum" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    ["id"],
                    SortMultipleOptions {
                        ..Default::default()
                    }
                )
                .select([col("value__median")])
                .collect()
                .unwrap(),
            df!["value__median" => [1.0, 2.0, 3.0]].unwrap()
        );
        // tsfresh: the (population) standard deviation of a single value is 0
        assert_eq!(
            gdf.clone()
                .sort(
                    ["id"],
                    SortMultipleOptions {
                        ..Default::default()
                    }
                )
                .select([col("value__standard_deviation")])
                .collect()
                .unwrap(),
            df!["value__standard_deviation" => [0.0, 0.0, 0.0]].unwrap()
        );
    }

    #[test]
    fn test_invalid_window_duration_is_an_error() {
        // Regression: Duration::parse panicked on a duration without a unit
        for field in ["every", "period", "offset"] {
            let mut dynamic = DynamicGroupBySettings {
                time_col: "time".to_string(),
                every: "1y".to_string(),
                period: "1y".to_string(),
                offset: "0y".to_string(),
                datetime_format: Some("%Y-%m-%d".to_string()),
            };
            let target = match field {
                "every" => &mut dynamic.every,
                "period" => &mut dynamic.period,
                _ => &mut dynamic.offset,
            };
            *target = "0".to_string();
            let df = df![
                "id" => ["a", "a"],
                "time" => ["2001-01-01", "2002-01-01"],
                "value" => [1.0, 2.0],
            ]
            .unwrap()
            .lazy();
            let opts = ExtractionSettings {
                grouping_cols: vec!["id".to_string()],
                value_cols: vec!["value".to_string()],
                feature_setting: FeatureSetting::Minimal,
                config_path: None,
                dynamic_settings: Some(dynamic),
                scaling: None,
                reduction: None,
            };
            let Err(err) = lazy_feature_df(df, opts) else {
                panic!("{field} = \"0\" was accepted");
            };
            assert!(matches!(err, ExtractionError::InvalidSettings(_)), "{err}");
            let expected = format!("invalid {field} duration \"0\"");
            assert!(err.to_string().contains(&expected), "{err}");
        }
    }

    /// The float columns of `df` sorted by `id`, as bit patterns (any NaN as
    /// one value), so that equal means bit-for-bit equal.
    fn float_bits(df: DataFrame) -> Vec<(String, Vec<Option<u64>>)> {
        let df = df.sort(["id"], SortMultipleOptions::default()).unwrap();
        df.columns()
            .iter()
            .filter(|c| c.dtype() == &DataType::Float64)
            .map(|c| {
                let values = c.f64().unwrap().to_vec();
                let bits = values.into_iter();
                let bits = bits.map(|v| v.map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() }));
                (c.name().to_string(), bits.collect())
            })
            .collect()
    }

    /// Scaled features do not depend on the engine that collects the result.
    /// Polars 2's streaming engine reduces a column's mean and standard
    /// deviation in parallel and merges the partial results in whatever order
    /// its threads finish, so a scaling run there differed in the last bits
    /// from run to run.
    #[test]
    fn test_scaling_does_not_depend_on_engine() {
        let df = df![
            "id" => ["a", "b", "c", "d"],
            "value" => [1.0, 2.0, 3.0, 4.0],
        ]
        .unwrap()
        .lazy();
        let opts = || ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            config_path: None,
            dynamic_settings: None,
            scaling: Some(Scaling::Standard),
            reduction: None,
        };
        let expected = collect_in_memory(lazy_feature_df(df.clone(), opts()).unwrap()).unwrap();
        let expected = float_bits(expected);
        for _ in 0..100 {
            let lf = lazy_feature_df(df.clone(), opts()).unwrap();
            let streamed = lf
                .collect_with_engine(Engine::Streaming)
                .unwrap()
                .unwrap_single();
            assert_eq!(float_bits(streamed), expected);
        }
    }

    /// 30 groups of different lengths (5 + g values for group g).
    fn varied() -> LazyFrame {
        let (mut ids, mut values) = (Vec::new(), Vec::new());
        for g in 0..30usize {
            for t in 0..5 + g {
                ids.push(format!("g{g:02}"));
                values.push((0.3 * t as f64 + g as f64).sin() * (1 + g % 4) as f64);
            }
        }
        df!["id" => ids, "value" => values].unwrap().lazy()
    }

    fn minimal(reduction: Option<Reduction>, scaling: Option<Scaling>) -> ExtractionSettings {
        ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            config_path: None,
            dynamic_settings: None,
            scaling,
            reduction,
        }
    }

    fn without_noise() -> Reduction {
        Reduction {
            noise_sensitivity: None,
            ..Default::default()
        }
    }

    fn names(df: &DataFrame) -> Vec<String> {
        df.get_column_names()
            .into_iter()
            .map(|name| name.to_string())
            .collect()
    }

    #[test]
    fn test_extract_with_reduction() {
        let ex = extract(varied(), minimal(Some(without_noise()), None)).unwrap();
        let fit = ex.reduction.unwrap();
        let df = collect_in_memory(ex.features).unwrap();
        let mut expected = vec!["id".to_string()];
        expected.extend(fit.kept.iter().cloned());
        assert_eq!(names(&df), expected);
        // the variance copies the standard deviation (a monotone map)
        assert!(fit.kept.len() < fit.report.height());
        // lengths vary, so `length` is kept, with its dtype
        assert!(fit.kept.contains(&"length".to_string()));
        assert_eq!(df.column("length").unwrap().dtype(), &DataType::UInt32);
    }

    #[test]
    fn test_no_reduction_returns_no_fit() {
        let ex = extract(varied(), minimal(None, None)).unwrap();
        assert!(ex.reduction.is_none());
    }

    #[test]
    fn test_reduction_then_scaling() {
        let reduced = collect_in_memory(
            lazy_feature_df(varied(), minimal(Some(without_noise()), None)).unwrap(),
        )
        .unwrap();
        let both = collect_in_memory(
            lazy_feature_df(
                varied(),
                minimal(Some(without_noise()), Some(Scaling::Standard)),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(names(&reduced), names(&both));
        for column in both.columns().iter().skip(1) {
            let finite: Vec<f64> = column
                .f64()
                .unwrap()
                .iter()
                .flatten()
                .filter(|x| x.is_finite())
                .collect();
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            assert!(mean.abs() < 1e-9, "{}: {mean}", column.name());
        }
    }

    #[test]
    fn test_reduction_with_time_windows() {
        // 15 groups, monthly values over 3 years: 3 yearly windows each
        let (mut ids, mut times, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for g in 0..15usize {
            for y in 0..3usize {
                for m in 1..=12usize {
                    ids.push(format!("g{g:02}"));
                    times.push(format!("{}-{m:02}-01", 2001 + y));
                    values.push(((g * 12 + m) as f64 * 0.7).sin() * (1 + (g + y) % 5) as f64);
                }
            }
        }
        let lf = df!["id" => ids, "time" => times, "value" => values]
            .unwrap()
            .lazy();
        let opts = |reduction: Option<Reduction>| ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            config_path: None,
            dynamic_settings: Some(DynamicGroupBySettings {
                time_col: "time".to_string(),
                every: "1y".to_string(),
                period: "1y".to_string(),
                offset: "0y".to_string(),
                datetime_format: Some("%Y-%m-%d".to_string()),
            }),
            scaling: None,
            reduction,
        };
        let full = collect_in_memory(lazy_feature_df(lf.clone(), opts(None)).unwrap()).unwrap();
        let reduced =
            collect_in_memory(lazy_feature_df(lf, opts(Some(without_noise()))).unwrap()).unwrap();
        assert_eq!(names(&reduced)[..2], ["id", "time"]);
        assert_eq!(reduced.height(), full.height());
        // something was dropped (`length` is 12 in every window)
        assert!(reduced.width() < full.width());
        assert!(!names(&reduced).contains(&"length".to_string()));
        let by_key = |df: DataFrame| {
            df.sort(["id", "time"], SortMultipleOptions::default())
                .unwrap()
        };
        let (full, reduced) = (by_key(full), by_key(reduced));
        for key in ["id", "time"] {
            assert!(
                full.column(key)
                    .unwrap()
                    .as_materialized_series()
                    .equals(reduced.column(key).unwrap().as_materialized_series()),
                "{key}"
            );
        }
    }
}
