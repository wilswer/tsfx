use anyhow::Result;
use polars::prelude::*;

use crate::error::ExtractionError;
use crate::feature_extractors::aggregators;

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
}

pub fn lazy_feature_df(
    df: LazyFrame,
    opts: ExtractionSettings,
) -> Result<LazyFrame, ExtractionError> {
    let aggregators = aggregators(&opts);
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
                every: Duration::parse(&dynamic_settings.every),
                period: Duration::parse(&dynamic_settings.period),
                offset: Duration::parse(&dynamic_settings.offset),
                ..Default::default()
            },
        )
    } else {
        df.select(&selected_cols).group_by(&grouping_cols)
    };
    Ok(gdf.agg(aggregators).collect()?.lazy())
}

#[cfg(test)]
mod tests {
    use super::{ExtractionSettings, FeatureSetting, lazy_feature_df};
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
}
