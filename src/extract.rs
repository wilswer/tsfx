use anyhow::Result;
use polars::prelude::*;

use crate::error::ExtractionError;
use crate::feature_extractors::aggregators;
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

pub fn lazy_feature_df(
    df: LazyFrame,
    opts: ExtractionSettings,
) -> Result<LazyFrame, ExtractionError> {
    if let Some(scaling) = &opts.scaling {
        scaling.validate()?;
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
    // Scale in memory too, whatever engine collects the result: the streaming
    // engine merges a column's partial means and variances in thread order, so
    // its scaled features differ in the last bits from run to run
    Ok(match &opts.scaling {
        Some(scaling) => collect_in_memory(scale_features(fdf, scaling, &key_cols))?.lazy(),
        None => fdf.lazy(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DynamicGroupBySettings, ExtractionSettings, FeatureSetting, collect_in_memory,
        lazy_feature_df,
    };
    use crate::error::ExtractionError;
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
}
