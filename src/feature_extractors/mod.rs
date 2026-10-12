//! Feature extractors, grouped by what they compute.
//!
//! [`aggregators`] is the single place that decides which features run for a
//! given [`FeatureSetting`] tier and config. Every feature except `length`
//! is a [`kernel::Kernel`]; see [`kernel`] for how they run.

pub mod autocorrelation;
pub mod change;
pub mod counting;
pub mod entropy;
pub mod kernel;
pub mod location;
pub mod reoccurrence;
pub mod spectral;
pub mod statistics;
pub mod trend;

use std::sync::Arc;

use polars::prelude::*;

use crate::extract::{ExtractionSettings, FeatureSetting};
use crate::utils::toml_reader::{ConfigError, load_config};
use autocorrelation::{Autocorrelation, C3, TimeReversalAsymmetryStatistic};
use change::{AbsoluteSumOfChanges, CidCe, MeanAbsoluteChange, MeanChange};
use counting::{
    CountAbove, CountAboveMean, CountBelow, CountBelowMean, LongestStrikeAboveMean,
    LongestStrikeBelowMean, NumberCrossingM, NumberPeaks, RangeCount,
};
use entropy::SampleEntropy;
use kernel::KernelPlan;
use location::{
    FirstLocationOfMaximum, FirstLocationOfMinimum, IndexMassQuantile, LastLocationOfMaximum,
    LastLocationOfMinimum,
};
use reoccurrence::{HasDuplicateMax, HasDuplicateMin, ReoccurrenceFeatures};
use spectral::{FftAggregated, FftCoefficient, FourierEntropy, SpktWelchDensity};
use statistics::{
    AbsoluteEnergy, AbsoluteMaximum, Kurtosis, LargeStandardDeviation, Maximum, Mean,
    MeanNAbsoluteMax, Median, Minimum, Quantile, RatioBeyondRSigma, RootMeanSquare, Skewness,
    StandardDeviation, SumValues, SymmetryLooking, Variance, VarianceLargerThanStandardDeviation,
    VariationCoefficient, count,
};
use trend::{AggLinearTrend, LinearTrend};

/// Feature expressions for the configured features of the selected tier.
///
/// Columns come out as `length` first, then, for each value column in turn,
/// all of its features in tier order (Minimal, then Efficient, then
/// Comprehensive).
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
    for plan in plans.iter_mut() {
        if config.sum_values.is_some() {
            plan.push(SumValues);
        }
        if config.mean.is_some() {
            plan.push(Mean);
        }
        if config.median.is_some() {
            plan.push(Median);
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
        for plan in plans.iter_mut() {
            if config.kurtosis.is_some() {
                plan.push(Kurtosis);
            }
            if config.absolute_energy.is_some() {
                plan.push(AbsoluteEnergy);
            }
            if config.mean_absolute_change.is_some() {
                plan.push(MeanAbsoluteChange);
            }
            if config.linear_trend.is_some() {
                plan.push(LinearTrend);
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
                plan.push(HasDuplicateMax);
            }
            if config.has_duplicate_min.is_some() {
                plan.push(HasDuplicateMin);
            }
            if let Some(feature) = &config.cid_ce {
                let normalize = feature.parameters.iter().map(|p| p.normalize).collect();
                plan.push(CidCe { normalize });
            }
            if config.absolute_maximum.is_some() {
                plan.push(AbsoluteMaximum);
            }
            if config.absolute_sum_of_changes.is_some() {
                plan.push(AbsoluteSumOfChanges);
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
                plan.push(FirstLocationOfMaximum);
            }
            if config.first_location_of_minimum.is_some() {
                plan.push(FirstLocationOfMinimum);
            }
            if config.last_location_of_maximum.is_some() {
                plan.push(LastLocationOfMaximum);
            }
            if config.last_location_of_minimum.is_some() {
                plan.push(LastLocationOfMinimum);
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
                plan.push(reoccurring);
            }
            if config.variation_coefficient.is_some() {
                plan.push(VariationCoefficient);
            }
            if config.mean_change.is_some() {
                plan.push(MeanChange);
            }
            if let Some(feature) = &config.agg_linear_trend {
                let params = feature
                    .parameters
                    .iter()
                    .map(|p| (p.chunk_size, p.aggregator.clone()))
                    .collect();
                plan.push(AggLinearTrend { params });
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
                let lags = feature.parameters.iter().map(|p| p.lag).collect();
                plan.push(Autocorrelation { lags });
            }
            if let Some(feature) = &config.quantile {
                let qs = feature.parameters.iter().map(|p| p.q).collect();
                plan.push(Quantile { qs });
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
                let qs = feature.parameters.iter().map(|p| p.q).collect();
                plan.push(IndexMassQuantile { qs });
            }
            if let Some(feature) = &config.c3 {
                let lags = feature.parameters.iter().map(|p| p.lag).collect();
                plan.push(C3 { lags });
            }
            if let Some(feature) = &config.time_reversal_asymmetry_statistic {
                let lags = feature.parameters.iter().map(|p| p.lag).collect();
                plan.push(TimeReversalAsymmetryStatistic { lags });
            }
            if let Some(feature) = &config.number_peaks {
                let ns = feature.parameters.iter().map(|p| p.n).collect();
                plan.push(NumberPeaks { ns });
            }
            if let Some(feature) = &config.fft_coefficient {
                let params = feature
                    .parameters
                    .iter()
                    .map(|p| (p.attr.clone(), p.coeff))
                    .collect();
                plan.push(FftCoefficient { params });
            }
            if let Some(feature) = &config.fft_aggregated {
                let aggtypes = feature
                    .parameters
                    .iter()
                    .map(|p| p.aggtype.clone())
                    .collect();
                plan.push(FftAggregated { aggtypes });
            }
            if let Some(feature) = &config.spkt_welch_density {
                let coeffs = feature.parameters.iter().map(|p| p.coeff).collect();
                plan.push(SpktWelchDensity { coeffs });
            }
            if let Some(feature) = &config.fourier_entropy {
                let bins = feature.parameters.iter().map(|p| p.bins).collect();
                plan.push(FourierEntropy { bins });
            }
        }
    }

    // Comprehensive tier: always on, independent of the config
    if *tier >= FeatureSetting::Comprehensive {
        for plan in plans.iter_mut() {
            plan.push(SampleEntropy);
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
