use serde::Deserialize;
use toml;

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub length: Option<Length>,
    pub sum_values: Option<SumValues>,
    pub mean: Option<Mean>,
    pub median: Option<Median>,
    pub minimum: Option<Minimum>,
    pub maximum: Option<Maximum>,
    pub standard_deviation: Option<StandardDeviation>,
    pub variance: Option<Variance>,
    pub skewness: Option<Skewness>,
    pub root_mean_square: Option<RootMeanSquare>,
    pub kurtosis: Option<Kurtosis>,
    pub absolute_energy: Option<AbsoluteEnergy>,
    pub mean_absolute_change: Option<MeanAbsoluteChange>,
    pub linear_trend: Option<LinearTrend>,
    pub variance_larger_than_standard_deviation: Option<VarianceLargerThanStandardDeviation>,
    pub ratio_beyond_r_sigma: Option<RatioBeyondRSigma>,
    pub large_standard_deviation: Option<LargeStandardDeviation>,
    pub symmetry_looking: Option<SymmetryLooking>,
    pub has_duplicate_max: Option<HasDuplicateMax>,
    pub has_duplicate_min: Option<HasDuplicateMin>,
    pub cid_ce: Option<CidCe>,
    pub absolute_maximum: Option<AbsoluteMaximum>,
    pub absolute_sum_of_changes: Option<AbsoluteSumOfChanges>,
    pub count_above_mean: Option<CountAboveMean>,
    pub count_below_mean: Option<CountBelowMean>,
    pub count_above: Option<CountAbove>,
    pub count_below: Option<CountBelow>,
    pub first_location_of_maximum: Option<FirstLocationOfMaximum>,
    pub first_location_of_minimum: Option<FirstLocationOfMinimum>,
    pub last_location_of_maximum: Option<LastLocationOfMaximum>,
    pub last_location_of_minimum: Option<LastLocationOfMinimum>,
    pub longest_strike_above_mean: Option<LongestStrikeAboveMean>,
    pub longest_strike_below_mean: Option<LongestStrikeBelowMean>,
    pub has_duplicate: Option<HasDuplicate>,
    pub variation_coefficient: Option<VariationCoefficient>,
    pub mean_change: Option<MeanChange>,
    pub ratio_value_number_to_time_series_length: Option<RatioValueNumberToTimeSeriesLength>,
    pub sum_of_reoccurring_values: Option<SumOfReoccurringValues>,
    pub sum_of_reoccurring_data_points: Option<SumOfReoccurringDataPoints>,
    pub percentage_of_reoccurring_values_to_all_values:
        Option<PercentageOfReoccurringValuesToAllValues>,
    pub percentage_of_reoccurring_values_to_all_datapoints:
        Option<PercentageOfReoccurringValuesToAllDataPoints>,
    pub agg_linear_trend: Option<AggLinearTrend>,
    pub mean_n_absolute_max: Option<MeanNAbsoluteMax>,
    pub autocorrelation: Option<Autocorrelation>,
    pub quantile: Option<Quantile>,
    pub number_crossing_m: Option<NumberCrossingM>,
    pub range_count: Option<RangeCount>,
    pub index_mass_quantile: Option<IndexMassQuantile>,
    pub c3: Option<C3>,
    pub time_reversal_asymmetry_statistic: Option<TimeReversalAsymmetryStatistic>,
    pub number_peaks: Option<NumberPeaks>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            length: Some(Length::default()),
            sum_values: Some(SumValues::default()),
            mean: Some(Mean::default()),
            median: Some(Median::default()),
            minimum: Some(Minimum::default()),
            maximum: Some(Maximum::default()),
            standard_deviation: Some(StandardDeviation::default()),
            variance: Some(Variance::default()),
            skewness: Some(Skewness::default()),
            root_mean_square: Some(RootMeanSquare::default()),
            kurtosis: Some(Kurtosis::default()),
            absolute_energy: Some(AbsoluteEnergy::default()),
            mean_absolute_change: Some(MeanAbsoluteChange::default()),
            linear_trend: Some(LinearTrend::default()),
            variance_larger_than_standard_deviation: Some(
                VarianceLargerThanStandardDeviation::default(),
            ),
            ratio_beyond_r_sigma: Some(RatioBeyondRSigma::default()),
            large_standard_deviation: Some(LargeStandardDeviation::default()),
            symmetry_looking: Some(SymmetryLooking::default()),
            has_duplicate_max: Some(HasDuplicateMax::default()),
            has_duplicate_min: Some(HasDuplicateMin::default()),
            cid_ce: Some(CidCe::default()),
            absolute_maximum: Some(AbsoluteMaximum::default()),
            absolute_sum_of_changes: Some(AbsoluteSumOfChanges::default()),
            count_above_mean: Some(CountAboveMean::default()),
            count_below_mean: Some(CountBelowMean::default()),
            count_above: Some(CountAbove::default()),
            count_below: Some(CountBelow::default()),
            first_location_of_maximum: Some(FirstLocationOfMaximum::default()),
            first_location_of_minimum: Some(FirstLocationOfMinimum::default()),
            last_location_of_maximum: Some(LastLocationOfMaximum::default()),
            last_location_of_minimum: Some(LastLocationOfMinimum::default()),
            longest_strike_above_mean: Some(LongestStrikeAboveMean::default()),
            longest_strike_below_mean: Some(LongestStrikeBelowMean::default()),
            has_duplicate: Some(HasDuplicate::default()),
            variation_coefficient: Some(VariationCoefficient::default()),
            mean_change: Some(MeanChange::default()),
            ratio_value_number_to_time_series_length: Some(
                RatioValueNumberToTimeSeriesLength::default(),
            ),
            sum_of_reoccurring_values: Some(SumOfReoccurringValues::default()),
            sum_of_reoccurring_data_points: Some(SumOfReoccurringDataPoints::default()),
            percentage_of_reoccurring_values_to_all_values: Some(
                PercentageOfReoccurringValuesToAllValues::default(),
            ),
            percentage_of_reoccurring_values_to_all_datapoints: Some(
                PercentageOfReoccurringValuesToAllDataPoints::default(),
            ),
            agg_linear_trend: Some(AggLinearTrend::default()),
            mean_n_absolute_max: Some(MeanNAbsoluteMax::default()),
            autocorrelation: Some(Autocorrelation::default()),
            quantile: Some(Quantile::default()),
            number_crossing_m: Some(NumberCrossingM::default()),
            range_count: Some(RangeCount::default()),
            index_mass_quantile: Some(IndexMassQuantile::default()),
            c3: Some(C3::default()),
            time_reversal_asymmetry_statistic: Some(TimeReversalAsymmetryStatistic::default()),
            number_peaks: Some(NumberPeaks::default()),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Length {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SumValues {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Mean {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Median {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Minimum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Maximum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct StandardDeviation {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Variance {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Skewness {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct RootMeanSquare {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Kurtosis {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct AbsoluteEnergy {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct MeanAbsoluteChange {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LinearTrend {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct VarianceLargerThanStandardDeviation {}

#[derive(Deserialize, Clone, Debug)]
pub struct RatioBeyondRSigma {
    pub parameters: Vec<RatioBeyondRSigmaParams>,
}
impl Default for RatioBeyondRSigma {
    fn default() -> Self {
        RatioBeyondRSigma {
            parameters: vec![
                RatioBeyondRSigmaParams { r: 0.5 },
                RatioBeyondRSigmaParams { r: 1.0 },
                RatioBeyondRSigmaParams { r: 1.5 },
                RatioBeyondRSigmaParams { r: 2.0 },
                RatioBeyondRSigmaParams { r: 2.5 },
                RatioBeyondRSigmaParams { r: 3.0 },
                RatioBeyondRSigmaParams { r: 5.0 },
                RatioBeyondRSigmaParams { r: 6.0 },
                RatioBeyondRSigmaParams { r: 7.0 },
                RatioBeyondRSigmaParams { r: 10.0 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RatioBeyondRSigmaParams {
    pub r: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct LargeStandardDeviation {
    pub parameters: Vec<LargeStandardDeviationParams>,
}
impl Default for LargeStandardDeviation {
    fn default() -> Self {
        LargeStandardDeviation {
            parameters: vec![
                LargeStandardDeviationParams { r: 0.05 },
                LargeStandardDeviationParams { r: 0.1 },
                LargeStandardDeviationParams { r: 0.15 },
                LargeStandardDeviationParams { r: 0.2 },
                LargeStandardDeviationParams { r: 0.25 },
                LargeStandardDeviationParams { r: 0.3 },
                LargeStandardDeviationParams { r: 0.35 },
                LargeStandardDeviationParams { r: 0.4 },
                LargeStandardDeviationParams { r: 0.45 },
                LargeStandardDeviationParams { r: 0.5 },
                LargeStandardDeviationParams { r: 0.55 },
                LargeStandardDeviationParams { r: 0.6 },
                LargeStandardDeviationParams { r: 0.65 },
                LargeStandardDeviationParams { r: 0.7 },
                LargeStandardDeviationParams { r: 0.75 },
                LargeStandardDeviationParams { r: 0.8 },
                LargeStandardDeviationParams { r: 0.85 },
                LargeStandardDeviationParams { r: 0.9 },
                LargeStandardDeviationParams { r: 0.95 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct LargeStandardDeviationParams {
    pub r: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SymmetryLooking {
    pub parameters: Vec<SymmetryLookingParams>,
}
impl Default for SymmetryLooking {
    fn default() -> Self {
        SymmetryLooking {
            parameters: vec![
                SymmetryLookingParams { r: 0.05 },
                SymmetryLookingParams { r: 0.1 },
                SymmetryLookingParams { r: 0.15 },
                SymmetryLookingParams { r: 0.2 },
                SymmetryLookingParams { r: 0.25 },
                SymmetryLookingParams { r: 0.3 },
                SymmetryLookingParams { r: 0.35 },
                SymmetryLookingParams { r: 0.4 },
                SymmetryLookingParams { r: 0.45 },
                SymmetryLookingParams { r: 0.5 },
                SymmetryLookingParams { r: 0.55 },
                SymmetryLookingParams { r: 0.6 },
                SymmetryLookingParams { r: 0.65 },
                SymmetryLookingParams { r: 0.7 },
                SymmetryLookingParams { r: 0.75 },
                SymmetryLookingParams { r: 0.8 },
                SymmetryLookingParams { r: 0.85 },
                SymmetryLookingParams { r: 0.9 },
                SymmetryLookingParams { r: 0.95 },
            ],
        }
    }
}
#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SymmetryLookingParams {
    pub r: f64,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HasDuplicateMax {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HasDuplicateMin {}

#[derive(Deserialize, Clone, Debug)]
pub struct CidCe {
    pub parameters: Vec<CidCeParams>,
}
impl Default for CidCe {
    fn default() -> Self {
        CidCe {
            parameters: vec![
                CidCeParams { normalize: true },
                CidCeParams { normalize: false },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CidCeParams {
    pub normalize: bool,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct AbsoluteMaximum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct AbsoluteSumOfChanges {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct CountAboveMean {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct CountBelowMean {}

#[derive(Deserialize, Clone, Debug)]
pub struct CountAbove {
    pub parameters: Vec<CountAboveParams>,
}

impl Default for CountAbove {
    fn default() -> Self {
        CountAbove {
            parameters: vec![CountAboveParams { t: 0.0 }],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CountAboveParams {
    t: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CountBelow {
    pub parameters: Vec<CountBelowParams>,
}

impl Default for CountBelow {
    fn default() -> Self {
        CountBelow {
            parameters: vec![CountBelowParams { t: 0.0 }],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CountBelowParams {
    t: f64,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct FirstLocationOfMaximum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct FirstLocationOfMinimum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LastLocationOfMaximum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LastLocationOfMinimum {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LongestStrikeAboveMean {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LongestStrikeBelowMean {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HasDuplicate {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct VariationCoefficient {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct MeanChange {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct RatioValueNumberToTimeSeriesLength {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SumOfReoccurringValues {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SumOfReoccurringDataPoints {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct PercentageOfReoccurringValuesToAllValues {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct PercentageOfReoccurringValuesToAllDataPoints {}

#[derive(Deserialize, Clone, Debug)]
pub struct AggLinearTrend {
    pub parameters: Vec<AggLinearTrendParams>,
}

impl Default for AggLinearTrend {
    fn default() -> Self {
        AggLinearTrend {
            parameters: vec![
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "var".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "var".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "var".to_string(),
                },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct AggLinearTrendParams {
    pub chunk_size: usize,
    pub aggregator: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct MeanNAbsoluteMax {
    pub parameters: Vec<MeanNAbsoluteMaxParams>,
}

impl Default for MeanNAbsoluteMax {
    fn default() -> Self {
        MeanNAbsoluteMax {
            parameters: vec![MeanNAbsoluteMaxParams { n: 7 }],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct MeanNAbsoluteMaxParams {
    pub n: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Autocorrelation {
    pub parameters: Vec<AutocorrelationParams>,
}

impl Default for Autocorrelation {
    fn default() -> Self {
        Autocorrelation {
            parameters: vec![
                AutocorrelationParams { lag: 0 },
                AutocorrelationParams { lag: 1 },
                AutocorrelationParams { lag: 2 },
                AutocorrelationParams { lag: 3 },
                AutocorrelationParams { lag: 4 },
                AutocorrelationParams { lag: 5 },
                AutocorrelationParams { lag: 6 },
                AutocorrelationParams { lag: 7 },
                AutocorrelationParams { lag: 8 },
                AutocorrelationParams { lag: 9 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct AutocorrelationParams {
    pub lag: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Quantile {
    pub parameters: Vec<QuantileParams>,
}

impl Default for Quantile {
    fn default() -> Self {
        Quantile {
            parameters: vec![
                QuantileParams { q: 0.1 },
                QuantileParams { q: 0.2 },
                QuantileParams { q: 0.3 },
                QuantileParams { q: 0.4 },
                QuantileParams { q: 0.6 },
                QuantileParams { q: 0.7 },
                QuantileParams { q: 0.8 },
                QuantileParams { q: 0.9 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct QuantileParams {
    pub q: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct NumberCrossingM {
    pub parameters: Vec<NumberCrossingMParams>,
}

impl Default for NumberCrossingM {
    fn default() -> Self {
        NumberCrossingM {
            parameters: vec![
                NumberCrossingMParams { m: -1.0 },
                NumberCrossingMParams { m: 0.0 },
                NumberCrossingMParams { m: 1.0 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct NumberCrossingMParams {
    pub m: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RangeCount {
    pub parameters: Vec<RangeCountParams>,
}

impl Default for RangeCount {
    fn default() -> Self {
        RangeCount {
            parameters: vec![
                RangeCountParams {
                    min: -1.0,
                    max: 1.0,
                },
                RangeCountParams {
                    min: -1_000_000_000_000.0,
                    max: 0.0,
                },
                RangeCountParams {
                    min: 0.0,
                    max: 1_000_000_000_000.0,
                },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RangeCountParams {
    pub min: f64,
    pub max: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct IndexMassQuantile {
    pub parameters: Vec<IndexMassQuantileParams>,
}

impl Default for IndexMassQuantile {
    fn default() -> Self {
        IndexMassQuantile {
            parameters: vec![
                IndexMassQuantileParams { q: 0.1 },
                IndexMassQuantileParams { q: 0.2 },
                IndexMassQuantileParams { q: 0.3 },
                IndexMassQuantileParams { q: 0.4 },
                IndexMassQuantileParams { q: 0.5 },
                IndexMassQuantileParams { q: 0.6 },
                IndexMassQuantileParams { q: 0.7 },
                IndexMassQuantileParams { q: 0.8 },
                IndexMassQuantileParams { q: 0.9 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct IndexMassQuantileParams {
    pub q: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct C3 {
    pub parameters: Vec<C3Params>,
}

impl Default for C3 {
    fn default() -> Self {
        C3 {
            parameters: vec![
                C3Params { lag: 1 },
                C3Params { lag: 2 },
                C3Params { lag: 3 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct C3Params {
    pub lag: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct TimeReversalAsymmetryStatistic {
    pub parameters: Vec<TimeReversalAsymmetryStatisticParams>,
}

impl Default for TimeReversalAsymmetryStatistic {
    fn default() -> Self {
        TimeReversalAsymmetryStatistic {
            parameters: vec![
                TimeReversalAsymmetryStatisticParams { lag: 1 },
                TimeReversalAsymmetryStatisticParams { lag: 2 },
                TimeReversalAsymmetryStatisticParams { lag: 3 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TimeReversalAsymmetryStatisticParams {
    pub lag: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct NumberPeaks {
    pub parameters: Vec<NumberPeaksParams>,
}

impl Default for NumberPeaks {
    fn default() -> Self {
        NumberPeaks {
            parameters: vec![
                NumberPeaksParams { n: 1 },
                NumberPeaksParams { n: 3 },
                NumberPeaksParams { n: 5 },
                NumberPeaksParams { n: 10 },
                NumberPeaksParams { n: 50 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct NumberPeaksParams {
    pub n: usize,
}

pub fn load_config(file_path: Option<&str>) -> Config {
    let file_path = file_path.unwrap_or(".tsfx-config.toml");
    if !std::path::Path::new(file_path).exists() {
        println!("tsfx: No config file detected. Using default config.");
        return Config::default();
    }
    let config_str = std::fs::read_to_string(file_path);
    match config_str {
        Ok(config_str) => toml::from_str(&config_str).unwrap(),
        Err(_) => {
            panic!(
                "Error reading config file.
                Please create a .tsfx-config.toml file in the root of your project."
            )
        }
    }
}

// cargo test toml file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_config_from_file() {
        let config = load_config(None);
        assert!(config.length.is_some());
        assert!(config.sum_values.is_some());
        assert!(config.mean.is_some());
        assert!(config.median.is_some());
        assert!(config.minimum.is_some());
        assert!(config.maximum.is_some());
        assert!(config.standard_deviation.is_some());
        assert!(config.variance.is_some());
        assert!(config.skewness.is_some());
        assert!(config.root_mean_square.is_some());
        assert!(config.kurtosis.is_some());
        assert!(config.absolute_energy.is_some());
        assert!(config.mean_absolute_change.is_some());
        assert!(config.linear_trend.is_some());
        assert!(config.variance_larger_than_standard_deviation.is_some());
        assert!(config.ratio_beyond_r_sigma.is_some());
        assert_eq!(
            config
                .clone()
                .ratio_beyond_r_sigma
                .unwrap()
                .parameters
                .len(),
            10
        );
        assert_eq!(
            config.clone().ratio_beyond_r_sigma.unwrap().parameters[..],
            [
                RatioBeyondRSigmaParams { r: 0.5 },
                RatioBeyondRSigmaParams { r: 1.0 },
                RatioBeyondRSigmaParams { r: 1.5 },
                RatioBeyondRSigmaParams { r: 2.0 },
                RatioBeyondRSigmaParams { r: 2.5 },
                RatioBeyondRSigmaParams { r: 3.0 },
                RatioBeyondRSigmaParams { r: 5.0 },
                RatioBeyondRSigmaParams { r: 6.0 },
                RatioBeyondRSigmaParams { r: 7.0 },
                RatioBeyondRSigmaParams { r: 10.0 },
            ]
        );
        assert!(config.large_standard_deviation.is_some());
        assert_eq!(
            config
                .clone()
                .large_standard_deviation
                .unwrap()
                .parameters
                .len(),
            19
        );
        assert_eq!(
            config.clone().large_standard_deviation.unwrap().parameters[..],
            [
                LargeStandardDeviationParams { r: 0.05 },
                LargeStandardDeviationParams { r: 0.1 },
                LargeStandardDeviationParams { r: 0.15 },
                LargeStandardDeviationParams { r: 0.2 },
                LargeStandardDeviationParams { r: 0.25 },
                LargeStandardDeviationParams { r: 0.3 },
                LargeStandardDeviationParams { r: 0.35 },
                LargeStandardDeviationParams { r: 0.4 },
                LargeStandardDeviationParams { r: 0.45 },
                LargeStandardDeviationParams { r: 0.5 },
                LargeStandardDeviationParams { r: 0.55 },
                LargeStandardDeviationParams { r: 0.6 },
                LargeStandardDeviationParams { r: 0.65 },
                LargeStandardDeviationParams { r: 0.7 },
                LargeStandardDeviationParams { r: 0.75 },
                LargeStandardDeviationParams { r: 0.8 },
                LargeStandardDeviationParams { r: 0.85 },
                LargeStandardDeviationParams { r: 0.9 },
                LargeStandardDeviationParams { r: 0.95 },
            ]
        );
        assert!(config.symmetry_looking.is_some());
        assert_eq!(
            config.clone().symmetry_looking.unwrap().parameters.len(),
            19
        );
        assert_eq!(
            config.clone().symmetry_looking.unwrap().parameters[..],
            [
                SymmetryLookingParams { r: 0.05 },
                SymmetryLookingParams { r: 0.1 },
                SymmetryLookingParams { r: 0.15 },
                SymmetryLookingParams { r: 0.2 },
                SymmetryLookingParams { r: 0.25 },
                SymmetryLookingParams { r: 0.3 },
                SymmetryLookingParams { r: 0.35 },
                SymmetryLookingParams { r: 0.4 },
                SymmetryLookingParams { r: 0.45 },
                SymmetryLookingParams { r: 0.5 },
                SymmetryLookingParams { r: 0.55 },
                SymmetryLookingParams { r: 0.6 },
                SymmetryLookingParams { r: 0.65 },
                SymmetryLookingParams { r: 0.7 },
                SymmetryLookingParams { r: 0.75 },
                SymmetryLookingParams { r: 0.8 },
                SymmetryLookingParams { r: 0.85 },
                SymmetryLookingParams { r: 0.9 },
                SymmetryLookingParams { r: 0.95 },
            ]
        );
        assert!(config.has_duplicate_max.is_some());
        assert!(config.has_duplicate_min.is_some());
        assert!(config.cid_ce.is_some());
        assert_eq!(
            config.clone().cid_ce.unwrap().parameters[..],
            [
                CidCeParams { normalize: true },
                CidCeParams { normalize: false }
            ]
        );
        assert!(config.absolute_maximum.is_some());
        assert!(config.absolute_sum_of_changes.is_some());
        assert!(config.count_above_mean.is_some());
        assert!(config.count_below_mean.is_some());
        assert!(config.count_above.is_some());
        assert_eq!(config.clone().count_above.unwrap().parameters.len(), 1);
        assert_eq!(
            config.clone().count_above.unwrap().parameters[..],
            [CountAboveParams { t: 0.0 }]
        );
        assert!(config.count_below.is_some());
        assert_eq!(config.clone().count_below.unwrap().parameters.len(), 1);
        assert_eq!(
            config.clone().count_below.unwrap().parameters[..],
            [CountBelowParams { t: 0.0 }]
        );
        assert!(config.first_location_of_maximum.is_some());
        assert!(config.first_location_of_minimum.is_some());
        assert!(config.last_location_of_maximum.is_some());
        assert!(config.last_location_of_minimum.is_some());
        assert!(config.longest_strike_above_mean.is_some());
        assert!(config.longest_strike_below_mean.is_some());
        assert!(config.has_duplicate.is_some());
        assert!(config.variation_coefficient.is_some());
        assert!(config.mean_change.is_some());
        assert!(config.ratio_value_number_to_time_series_length.is_some());
        assert!(config.sum_of_reoccurring_values.is_some());
        assert!(config.sum_of_reoccurring_data_points.is_some());
        assert!(config
            .percentage_of_reoccurring_values_to_all_values
            .is_some());
        assert!(config
            .percentage_of_reoccurring_values_to_all_datapoints
            .is_some());
        assert!(config.agg_linear_trend.is_some());
        assert_eq!(
            config.clone().agg_linear_trend.unwrap().parameters.len(),
            12
        );
        assert_eq!(
            config.clone().agg_linear_trend.unwrap().parameters[..],
            [
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: "var".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: "var".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "mean".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "min".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "max".to_string(),
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: "var".to_string(),
                },
            ]
        );
        assert!(config.mean_n_absolute_max.is_some());
        assert_eq!(
            config.clone().mean_n_absolute_max.unwrap().parameters.len(),
            1
        );
        assert_eq!(
            config.clone().mean_n_absolute_max.unwrap().parameters[..],
            [MeanNAbsoluteMaxParams { n: 7 }]
        );
        assert!(config.autocorrelation.is_some());
        assert_eq!(config.clone().autocorrelation.unwrap().parameters.len(), 10);
        assert_eq!(
            config.clone().autocorrelation.unwrap().parameters[..],
            [
                AutocorrelationParams { lag: 0 },
                AutocorrelationParams { lag: 1 },
                AutocorrelationParams { lag: 2 },
                AutocorrelationParams { lag: 3 },
                AutocorrelationParams { lag: 4 },
                AutocorrelationParams { lag: 5 },
                AutocorrelationParams { lag: 6 },
                AutocorrelationParams { lag: 7 },
                AutocorrelationParams { lag: 8 },
                AutocorrelationParams { lag: 9 },
            ]
        );
        assert!(config.quantile.is_some());
        assert_eq!(config.clone().quantile.unwrap().parameters.len(), 8);
        assert_eq!(
            config.clone().quantile.unwrap().parameters[..],
            [
                QuantileParams { q: 0.1 },
                QuantileParams { q: 0.2 },
                QuantileParams { q: 0.3 },
                QuantileParams { q: 0.4 },
                QuantileParams { q: 0.6 },
                QuantileParams { q: 0.7 },
                QuantileParams { q: 0.8 },
                QuantileParams { q: 0.9 },
            ]
        );
        assert!(config.number_crossing_m.is_some());
        assert_eq!(
            config.clone().number_crossing_m.unwrap().parameters.len(),
            3
        );
        assert_eq!(
            config.clone().number_crossing_m.unwrap().parameters[..],
            [
                NumberCrossingMParams { m: -1.0 },
                NumberCrossingMParams { m: 0.0 },
                NumberCrossingMParams { m: 1.0 },
            ]
        );
        assert!(config.range_count.is_some());
        assert_eq!(config.clone().range_count.unwrap().parameters.len(), 3);
        assert_eq!(
            config.clone().range_count.unwrap().parameters[..],
            [
                RangeCountParams {
                    min: -1.0,
                    max: 1.0
                },
                RangeCountParams {
                    min: -1_000_000_000_000.0,
                    max: 0.0
                },
                RangeCountParams {
                    min: 0.0,
                    max: 1_000_000_000_000.0
                }
            ]
        );
        assert!(config.index_mass_quantile.is_some());
        assert_eq!(
            config.clone().index_mass_quantile.unwrap().parameters.len(),
            9
        );
        assert_eq!(
            config.clone().index_mass_quantile.unwrap().parameters[..],
            [
                IndexMassQuantileParams { q: 0.1 },
                IndexMassQuantileParams { q: 0.2 },
                IndexMassQuantileParams { q: 0.3 },
                IndexMassQuantileParams { q: 0.4 },
                IndexMassQuantileParams { q: 0.5 },
                IndexMassQuantileParams { q: 0.6 },
                IndexMassQuantileParams { q: 0.7 },
                IndexMassQuantileParams { q: 0.8 },
                IndexMassQuantileParams { q: 0.9 },
            ]
        );
        assert!(config.c3.is_some());
        assert_eq!(config.clone().c3.unwrap().parameters.len(), 3);
        assert_eq!(
            config.clone().c3.unwrap().parameters[..],
            [
                C3Params { lag: 1 },
                C3Params { lag: 2 },
                C3Params { lag: 3 }
            ]
        );
        assert!(config.time_reversal_asymmetry_statistic.is_some());
        assert_eq!(
            config
                .clone()
                .time_reversal_asymmetry_statistic
                .unwrap()
                .parameters
                .len(),
            3
        );
        assert_eq!(
            config
                .clone()
                .time_reversal_asymmetry_statistic
                .unwrap()
                .parameters[..],
            [
                TimeReversalAsymmetryStatisticParams { lag: 1 },
                TimeReversalAsymmetryStatisticParams { lag: 2 },
                TimeReversalAsymmetryStatisticParams { lag: 3 }
            ]
        );
        assert!(config.number_peaks.is_some());
        assert_eq!(config.clone().number_peaks.unwrap().parameters.len(), 5);
        assert_eq!(
            config.clone().number_peaks.unwrap().parameters[..],
            [
                NumberPeaksParams { n: 1 },
                NumberPeaksParams { n: 3 },
                NumberPeaksParams { n: 5 },
                NumberPeaksParams { n: 10 },
                NumberPeaksParams { n: 50 }
            ]
        );
    }
}
