use polars::prelude::*;

use crate::feature_extractors::extras::extra_aggregators;
use crate::feature_extractors::minimal::minimal_aggregators;

#[derive(Clone, Debug)]
pub enum FeatureSetting {
    Minimal,
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
    pub grouping_col: String,
    pub value_cols: Vec<String>,
    pub feature_setting: FeatureSetting,
    pub dynamic_settings: Option<DynamicGroupBySettings>,
}

fn get_aggregators(opts: &ExtractionSettings) -> Vec<Expr> {
    let mut aggregators = minimal_aggregators(&opts.value_cols);
    match opts.feature_setting {
        FeatureSetting::Minimal => aggregators,
        FeatureSetting::Comprehensive => {
            aggregators.append(&mut extra_aggregators(&opts.value_cols));
            aggregators
        }
    }
}

pub fn lazy_feature_df(df: LazyFrame, opts: ExtractionSettings) -> LazyFrame {
    let aggregators = get_aggregators(&opts);
    let mut selected_cols = Vec::new();
    selected_cols.push(col(&opts.grouping_col));
    let df = df.drop_nulls(None);
    for val_col in &opts.value_cols {
        selected_cols.push(col(val_col));
    }
    let gdf = if opts.dynamic_settings.is_some() {
        let dynamic_settings = opts.clone().dynamic_settings.unwrap();
        let datetime_format = if dynamic_settings.datetime_format.is_some() {
            dynamic_settings.clone().datetime_format.unwrap()
        } else {
            "%Y-%m-%d %H:%S:%M".to_string()
        };
        let time_options = StrptimeOptions {
            format: Some(datetime_format),
            ..Default::default()
        };
        selected_cols.push(col(&dynamic_settings.time_col));
        df.select(&selected_cols).group_by_dynamic(
            col(&dynamic_settings.time_col).str().to_date(time_options),
            [col(&opts.grouping_col)],
            DynamicGroupOptions {
                index_column: dynamic_settings.time_col.into(),
                every: Duration::parse(&dynamic_settings.every),
                period: Duration::parse(&dynamic_settings.period),
                offset: Duration::parse(&dynamic_settings.offset),
                ..Default::default()
            },
        )
    } else {
        df.select(&selected_cols)
            .group_by([col(&opts.grouping_col)])
    };
    gdf.agg(aggregators)
}

#[cfg(test)]
mod tests {
    use std::f32::NAN;

    use super::{lazy_feature_df, ExtractionSettings, FeatureSetting};
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
            grouping_col: "id".to_string(),
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            dynamic_settings: None,
        };
        let gdf = lazy_feature_df(df, opts);

        assert_eq!(
            gdf.clone().select([col("length")]).collect().unwrap(),
            df!["length" => [3, 3, 3]].unwrap()
        );
        assert_eq!(
            gdf.clone().select([col("value_sum")]).collect().unwrap(),
            df!["value_sum" => [6.0, 6.0, 6.0]].unwrap()
        );
        assert_eq!(
            gdf.clone().select([col("value_mean")]).collect().unwrap(),
            df!["value_mean" => [2.0, 2.0, 2.0]].unwrap()
        );
        assert_eq!(
            gdf.clone().select([col("value_min")]).collect().unwrap(),
            df!["value_min" => [1.0, 1.0, 1.0]].unwrap()
        );
        assert_eq!(
            gdf.clone().select([col("value_max")]).collect().unwrap(),
            df!["value_max" => [3.0, 3.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.select([col("value_median")]).collect().unwrap(),
            df!["value_median" => [2.0, 2.0, 2.0]].unwrap()
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
            grouping_col: "id".to_string(),
            value_cols: vec!["value".to_string()],
            feature_setting: FeatureSetting::Minimal,
            dynamic_settings: None,
        };
        let gdf = lazy_feature_df(df, opts);
        println!("{}", gdf.clone().collect().unwrap());
        assert_eq!(
            gdf.clone().select([col("length")]).collect().unwrap(),
            df!["length" => [1, 1, 1]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    "id",
                    SortOptions {
                        ..Default::default()
                    }
                )
                .select([col("value_sum")])
                .collect()
                .unwrap(),
            df!["value_sum" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    "id",
                    SortOptions {
                        ..Default::default()
                    }
                )
                .select([col("value_mean")])
                .collect()
                .unwrap(),
            df!["value_mean" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    "id",
                    SortOptions {
                        ..Default::default()
                    }
                )
                .select([col("value_min")])
                .collect()
                .unwrap(),
            df!["value_min" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    "id",
                    SortOptions {
                        ..Default::default()
                    }
                )
                .select([col("value_median")])
                .collect()
                .unwrap(),
            df!["value_median" => [1.0, 2.0, 3.0]].unwrap()
        );
        assert_eq!(
            gdf.clone()
                .sort(
                    "id",
                    SortOptions {
                        ..Default::default()
                    }
                )
                .select([col("value_std")])
                .collect()
                .unwrap(),
            df!["value_std" => [NAN, NAN, NAN]].unwrap()
        );
    }
}
