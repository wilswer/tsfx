//! Feature extractors, grouped by what they compute.
//!
//! [`aggregators`] is the single place that decides which features run for a
//! given [`FeatureSetting`] tier and config.

pub mod autocorrelation;
pub mod change;
pub mod counting;
pub mod entropy;
pub mod location;
pub mod reoccurrence;
pub mod statistics;
pub mod trend;

use polars::prelude::*;

use crate::extract::{ExtractionSettings, FeatureSetting};
use crate::utils::toml_reader::{ConfigError, load_config};
use autocorrelation::{autocorrelation, c3, time_reversal_asymmetry_statistic};
use change::{absolute_sum_of_changes, cid_ce, mean_absolute_change, mean_change};
use counting::{
    count_above, count_above_mean, count_below, count_below_mean, longest_strike_above_mean,
    longest_strike_below_mean, number_crossing_m, number_peaks, range_count,
};
use entropy::sample_entropy;
use location::{
    first_location_of_maximum, first_location_of_minimum, index_mass_quantile,
    last_location_of_maximum, last_location_of_minimum,
};
use reoccurrence::{
    has_duplicate, has_duplicate_max, has_duplicate_min,
    percentage_of_reoccurring_values_to_all_datapoints,
    percentage_of_reoccurring_values_to_all_values, ratio_value_number_to_time_series_length,
    sum_of_reoccurring_data_points, sum_of_reoccurring_values,
};
use statistics::{
    absolute_energy, absolute_maximum, count, expr_median, expr_quantile, kurtosis,
    large_standard_deviation, maximum, mean, mean_n_absolute_max, minimum, ratio_beyond_r_sigma,
    root_mean_square, skewness, standard_deviation, sum_values, symmetry_looking, variance,
    variance_larger_than_standard_deviation, variation_coefficient,
};
use trend::{agg_linear_trend, linear_trend};

/// Feature expressions for the configured features of the selected tier.
///
/// Columns come out tier by tier (Minimal, then Efficient, then
/// Comprehensive), each tier looping over all value columns.
pub fn aggregators(opts: &ExtractionSettings) -> Result<Vec<Expr>, ConfigError> {
    let config = load_config(opts.config_path.as_deref())?;
    let tier = &opts.feature_setting;
    let mut aggregators = Vec::new();

    // Minimal tier
    if config.length.is_some() {
        aggregators.push(count(&opts.value_cols[0]));
    }
    for col in &opts.value_cols {
        if config.sum_values.is_some() {
            aggregators.push(sum_values(col));
        }
        if config.mean.is_some() {
            aggregators.push(mean(col));
        }
        if config.median.is_some() {
            aggregators.push(expr_median(col));
        }
        if config.minimum.is_some() {
            aggregators.push(minimum(col));
        }
        if config.maximum.is_some() {
            aggregators.push(maximum(col));
        }
        if config.standard_deviation.is_some() {
            aggregators.push(standard_deviation(col));
        }
        if config.variance.is_some() {
            aggregators.push(variance(col));
        }
        if config.skewness.is_some() {
            aggregators.push(skewness(col));
        }
        if config.root_mean_square.is_some() {
            aggregators.push(root_mean_square(col));
        }
    }

    // Efficient tier
    if *tier >= FeatureSetting::Efficient {
        for col in &opts.value_cols {
            if config.kurtosis.is_some() {
                aggregators.push(kurtosis(col));
            }
            if config.absolute_energy.is_some() {
                aggregators.push(absolute_energy(col));
            }
            if config.mean_absolute_change.is_some() {
                aggregators.push(mean_absolute_change(col));
            }
            if config.linear_trend.is_some() {
                aggregators.push(linear_trend(col));
            }
            if config.variance_larger_than_standard_deviation.is_some() {
                aggregators.push(variance_larger_than_standard_deviation(col));
            }
            if let Some(feature) = &config.ratio_beyond_r_sigma {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                aggregators.push(ratio_beyond_r_sigma(col, rs));
            }

            if let Some(feature) = &config.large_standard_deviation {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                aggregators.push(large_standard_deviation(col, rs));
            }
            if let Some(feature) = &config.symmetry_looking {
                let params = &feature.parameters;
                let mut rs = Vec::new();
                for p in params {
                    rs.push(p.r);
                }
                aggregators.push(symmetry_looking(col, rs));
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
                aggregators.push(absolute_maximum(col));
            }
            if config.absolute_sum_of_changes.is_some() {
                aggregators.push(absolute_sum_of_changes(col));
            }
            if config.count_above_mean.is_some() {
                aggregators.push(count_above_mean(col));
            }
            if config.count_below_mean.is_some() {
                aggregators.push(count_below_mean(col));
            }
            if let Some(feature) = &config.count_above {
                for p in &feature.parameters {
                    aggregators.push(count_above(col, p.t));
                }
            }
            if let Some(feature) = &config.count_below {
                for p in &feature.parameters {
                    aggregators.push(count_below(col, p.t));
                }
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
                aggregators.push(longest_strike_above_mean(col));
            }
            if config.longest_strike_below_mean.is_some() {
                aggregators.push(longest_strike_below_mean(col));
            }
            if config.has_duplicate.is_some() {
                aggregators.push(has_duplicate(col));
            }
            if config.variation_coefficient.is_some() {
                aggregators.push(variation_coefficient(col));
            }
            if config.mean_change.is_some() {
                aggregators.push(mean_change(col));
            }
            if config.ratio_value_number_to_time_series_length.is_some() {
                aggregators.push(ratio_value_number_to_time_series_length(col));
            }
            if config.sum_of_reoccurring_values.is_some() {
                aggregators.push(sum_of_reoccurring_values(col));
            }
            if config.sum_of_reoccurring_data_points.is_some() {
                aggregators.push(sum_of_reoccurring_data_points(col));
            }
            if config
                .percentage_of_reoccurring_values_to_all_values
                .is_some()
            {
                aggregators.push(percentage_of_reoccurring_values_to_all_values(col));
            }
            if config
                .percentage_of_reoccurring_values_to_all_datapoints
                .is_some()
            {
                aggregators.push(percentage_of_reoccurring_values_to_all_datapoints(col));
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
                aggregators.push(mean_n_absolute_max(col, ns));
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
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(number_crossing_m(col, p.m));
                }
            }
            if let Some(feature) = &config.range_count {
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(range_count(col, p.min, p.max));
                }
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
                let params = &feature.parameters;
                for p in params {
                    aggregators.push(number_peaks(col, p.n));
                }
            }
        }
    }

    // Comprehensive tier: always on, independent of the config
    if *tier >= FeatureSetting::Comprehensive {
        for col in &opts.value_cols {
            aggregators.push(sample_entropy(col));
        }
    }
    Ok(aggregators)
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
