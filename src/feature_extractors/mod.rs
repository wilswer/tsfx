//! Feature extractors, grouped by what they compute.
//!
//! [`aggregators`] is the single place that decides which features run for a
//! given [`FeatureSetting`] tier and config.

pub mod autocorrelation;
pub mod change;
pub mod counting;
pub mod entropy;
pub mod kernel;
pub mod location;
pub mod reoccurrence;
pub mod statistics;
pub mod trend;

use std::sync::Arc;

use polars::prelude::*;

use crate::extract::{ExtractionSettings, FeatureSetting};
use crate::utils::toml_reader::{ConfigError, load_config};
use autocorrelation::{autocorrelation, c3, time_reversal_asymmetry_statistic};
use change::{absolute_sum_of_changes, cid_ce, mean_absolute_change, mean_change};
use counting::{
    CountAbove, CountAboveMean, CountBelow, CountBelowMean, LongestStrikeAboveMean,
    LongestStrikeBelowMean, NumberCrossingM, NumberPeaks, RangeCount,
};
use entropy::sample_entropy;
use kernel::KernelPlan;
use location::{
    first_location_of_maximum, first_location_of_minimum, index_mass_quantile,
    last_location_of_maximum, last_location_of_minimum,
};
use reoccurrence::{ReoccurrenceFeatures, has_duplicate_max, has_duplicate_min, reoccurrence};
use statistics::{
    AbsoluteEnergy, AbsoluteMaximum, Kurtosis, LargeStandardDeviation, Maximum, Mean,
    MeanNAbsoluteMax, Minimum, RatioBeyondRSigma, RootMeanSquare, Skewness, StandardDeviation,
    SumValues, SymmetryLooking, Variance, VarianceLargerThanStandardDeviation,
    VariationCoefficient, count, expr_median, expr_quantile,
};
use trend::{agg_linear_trend, linear_trend};

/// Feature expressions for the configured features of the selected tier.
///
/// Columns come out tier by tier (Minimal, then Efficient, then
/// Comprehensive), each tier looping over all value columns.
pub fn aggregators(opts: &ExtractionSettings) -> Result<FeatureExprs, ConfigError> {
    let config = load_config(opts.config_path.as_deref())?;
    let tier = &opts.feature_setting;
    let mut aggregators = Vec::new();
    // Features computed by the single `apply` of each value column
    let mut plans: Vec<KernelPlan> = opts
        .value_cols
        .iter()
        .map(|_| KernelPlan::default())
        .collect();

    // Minimal tier
    if config.length.is_some() {
        aggregators.push(count(&opts.value_cols[0]));
    }
    for (col, plan) in opts.value_cols.iter().zip(plans.iter_mut()) {
        if config.sum_values.is_some() {
            plan.push(SumValues);
        }
        if config.mean.is_some() {
            plan.push(Mean);
        }
        if config.median.is_some() {
            aggregators.push(expr_median(col));
        }
        if config.minimum.is_some() {
            plan.push(Minimum);
        }
        if config.maximum.is_some() {
            plan.push(Maximum);
        }
        if config.standard_deviation.is_some() {
            plan.push(StandardDeviation);
        }
        if config.variance.is_some() {
            plan.push(Variance);
        }
        if config.skewness.is_some() {
            plan.push(Skewness);
        }
        if config.root_mean_square.is_some() {
            plan.push(RootMeanSquare);
        }
    }

    // Efficient tier
    if *tier >= FeatureSetting::Efficient {
        for (col, plan) in opts.value_cols.iter().zip(plans.iter_mut()) {
            if config.kurtosis.is_some() {
                plan.push(Kurtosis);
            }
            if config.absolute_energy.is_some() {
                plan.push(AbsoluteEnergy);
            }
            if config.mean_absolute_change.is_some() {
                aggregators.push(mean_absolute_change(col));
            }
            if config.linear_trend.is_some() {
                aggregators.push(linear_trend(col));
            }
            if config.variance_larger_than_standard_deviation.is_some() {
                plan.push(VarianceLargerThanStandardDeviation);
            }
            if let Some(feature) = &config.ratio_beyond_r_sigma {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                plan.push(RatioBeyondRSigma { rs });
            }

            if let Some(feature) = &config.large_standard_deviation {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                plan.push(LargeStandardDeviation { rs });
            }
            if let Some(feature) = &config.symmetry_looking {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                plan.push(SymmetryLooking { rs });
            }
            if config.has_duplicate_max.is_some() {
                aggregators.push(has_duplicate_max(col));
            }
            if config.has_duplicate_min.is_some() {
                aggregators.push(has_duplicate_min(col));
            }
            if let Some(feature) = &config.cid_ce {
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(cid_ce(col, p.normalize));
                }
            }
            if config.absolute_maximum.is_some() {
                plan.push(AbsoluteMaximum);
            }
            if config.absolute_sum_of_changes.is_some() {
                aggregators.push(absolute_sum_of_changes(col));
            }
            if config.count_above_mean.is_some() {
                plan.push(CountAboveMean);
            }
            if config.count_below_mean.is_some() {
                plan.push(CountBelowMean);
            }
            if let Some(feature) = &config.count_above {
                let ts = feature.parameters.iter().map(|p| p.t).collect();
                plan.push(CountAbove { ts });
            }
            if let Some(feature) = &config.count_below {
                let ts = feature.parameters.iter().map(|p| p.t).collect();
                plan.push(CountBelow { ts });
            }
            if config.first_location_of_maximum.is_some() {
                aggregators.push(first_location_of_maximum(col));
            }
            if config.first_location_of_minimum.is_some() {
                aggregators.push(first_location_of_minimum(col));
            }
            if config.last_location_of_maximum.is_some() {
                aggregators.push(last_location_of_maximum(col));
            }
            if config.last_location_of_minimum.is_some() {
                aggregators.push(last_location_of_minimum(col));
            }
            if config.longest_strike_above_mean.is_some() {
                plan.push(LongestStrikeAboveMean);
            }
            if config.longest_strike_below_mean.is_some() {
                plan.push(LongestStrikeBelowMean);
            }
            // The counting-based reoccurrence features share one sort per group
            let reoccurring = ReoccurrenceFeatures {
                has_duplicate: config.has_duplicate.is_some(),
                ratio_value_number_to_time_series_length: config
                    .ratio_value_number_to_time_series_length
                    .is_some(),
                sum_of_reoccurring_values: config.sum_of_reoccurring_values.is_some(),
                sum_of_reoccurring_data_points: config.sum_of_reoccurring_data_points.is_some(),
                percentage_of_reoccurring_values_to_all_values: config
                    .percentage_of_reoccurring_values_to_all_values
                    .is_some(),
                percentage_of_reoccurring_values_to_all_datapoints: config
                    .percentage_of_reoccurring_values_to_all_datapoints
                    .is_some(),
            };
            if reoccurring != ReoccurrenceFeatures::default() {
                aggregators.push(reoccurrence(col, reoccurring));
            }
            if config.variation_coefficient.is_some() {
                plan.push(VariationCoefficient);
            }
            if config.mean_change.is_some() {
                aggregators.push(mean_change(col));
            }
            if let Some(feature) = &config.agg_linear_trend {
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(agg_linear_trend(col, p.chunk_size, p.aggregator.clone()));
                }
            }
            if let Some(feature) = &config.mean_n_absolute_max {
                let params = &feature.parameters;
                let mut ns = Vec::new();
                for p in params {
                    ns.push(p.n);
                }
                plan.push(MeanNAbsoluteMax { ns });
            }
            if let Some(feature) = &config.autocorrelation {
                let params = &feature.parameters;
                let mut lags = Vec::new();
                for p in params {
                    lags.push(p.lag);
                }
                aggregators.push(autocorrelation(col, lags));
            }
            if let Some(feature) = &config.quantile {
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(expr_quantile(col, p.q));
                }
            }
            if let Some(feature) = &config.number_crossing_m {
                let ms = feature.parameters.iter().map(|p| p.m).collect();
                plan.push(NumberCrossingM { ms });
            }
            if let Some(feature) = &config.range_count {
                let ranges = feature.parameters.iter().map(|p| (p.min, p.max)).collect();
                plan.push(RangeCount { ranges });
            }
            if let Some(feature) = &config.index_mass_quantile {
                let params = &feature.parameters;
                let mut qs = Vec::new();
                for p in params {
                    qs.push(p.q);
                }
                aggregators.push(index_mass_quantile(col, qs));
            }
            if let Some(feature) = &config.c3 {
                let parameters = &feature.parameters;
                for p in parameters {
                    aggregators.push(c3(col, p.lag));
                }
            }
            if let Some(feature) = &config.time_reversal_asymmetry_statistic {
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(time_reversal_asymmetry_statistic(col, p.lag));
                }
            }
            if let Some(feature) = &config.number_peaks {
                let ns = feature.parameters.iter().map(|p| p.n).collect();
                plan.push(NumberPeaks { ns });
            }
        }
    }

    // Comprehensive tier: always on, independent of the config
    if *tier >= FeatureSetting::Comprehensive {
        for col in &opts.value_cols {
            aggregators.push(sample_entropy(col));
        }
    }

    let mut kernels = Vec::new();
    for (col, plan) in opts.value_cols.iter().zip(plans) {
        if plan.is_empty() {
            continue;
        }
        let plan = Arc::new(plan);
        aggregators.push(plan.aggregation(col));
        kernels.push((col.clone(), plan));
    }
    Ok(FeatureExprs {
        aggregations: aggregators,
        kernels,
    })
}

/// What [`aggregators`] builds for a group-by.
pub struct FeatureExprs {
    /// Expressions for the group-by aggregation.
    pub aggregations: Vec<Expr>,
    /// Each value column's kernel plan, whose packed feature list must be
    /// unpacked ([`KernelPlan::unpack`]) in the aggregated frame.
    pub kernels: Vec<(String, Arc<KernelPlan>)>,
}

/// The group's values as one contiguous `Float64` chunk, ready for
/// `ArrayView1::from(ca.cont_slice()?)`.
///
/// Unlike `to_ndarray`, this does not copy a `Float64` group that is already
/// a single chunk; other types are cast (one copy, as before). The column
/// must not contain nulls, or `cont_slice` fails.
pub(super) fn f64_values(s: &Column) -> PolarsResult<Float64Chunked> {
    let series = s.as_materialized_series().cast(&DataType::Float64)?;
    Ok(series.f64()?.rechunk().into_owned())
}

pub(super) fn _make_nan_struct_column(
    name: &str,
    parameter_name: &str,
    rs: &[f64],
) -> Result<Column, PolarsError> {
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs.iter() {
        ss.push(Column::new(
            format!("{}__{}_{:2}", name, parameter_name, r).into(),
            &[f64::NAN],
        ))
    }
    let s = DataFrame::new(1, ss)?
        .into_struct(name.into())
        .into_column();
    Ok(s)
}
pub(super) fn _make_nan_struct_column_int(
    name: &str,
    parameter_name: &str,
    ns: &[usize],
) -> Result<Column, PolarsError> {
    let mut ss: Vec<Column> = Vec::with_capacity(ns.len());
    for n in ns.iter() {
        ss.push(Column::new(
            format!("{}__{}_{}", name, parameter_name, n).into(),
            &[f64::NAN],
        ))
    }
    let s = DataFrame::new(1, ss)?
        .into_struct(name.into())
        .into_column();
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{ExtractionSettings, FeatureSetting, lazy_feature_df};
    use polars::datatypes::AnyValue;

    #[test]
    fn test_unit_length_df() {
        // Create a dataframe with a single row
        let df = df![
            "id" => ["a"],
            "val" => [1.0],
        ]
        .unwrap()
        .lazy();

        // Configure extraction settings
        let opts = ExtractionSettings {
            grouping_cols: vec!["id".to_string()],
            feature_setting: FeatureSetting::Efficient,
            value_cols: vec!["val".to_string()],
            config_path: None,
            dynamic_settings: None,
        };

        // Extract features
        let gdf = lazy_feature_df(df, opts).unwrap();

        // Collect the results
        let fdf = gdf.collect().unwrap();

        // Assert that the resulting dataframe has exactly one row
        assert_eq!(fdf.shape().0, 1);

        // Also check that the length column has the expected value
        assert_eq!(
            fdf.column("length").unwrap().get(0).unwrap(),
            AnyValue::UInt32(1)
        );
    }
}
