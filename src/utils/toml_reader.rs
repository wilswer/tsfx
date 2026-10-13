use std::fmt::Display;

use serde::Deserialize;
use thiserror::Error;
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
    pub mean_second_derivative_central: Option<MeanSecondDerivativeCentral>,
    pub value_count: Option<ValueCount>,
    pub energy_ratio_by_chunks: Option<EnergyRatioByChunks>,
    pub change_quantiles: Option<ChangeQuantiles>,
    pub benford_correlation: Option<BenfordCorrelation>,
    pub query_similarity_count: Option<QuerySimilarityCount>,
    pub fft_coefficient: Option<FftCoefficient>,
    pub fft_aggregated: Option<FftAggregated>,
    pub spkt_welch_density: Option<SpktWelchDensity>,
    pub fourier_entropy: Option<FourierEntropy>,
    pub cwt_coefficients: Option<CwtCoefficients>,
    pub number_cwt_peaks: Option<NumberCwtPeaks>,
    pub agg_autocorrelation: Option<AggAutocorrelation>,
    pub partial_autocorrelation: Option<PartialAutocorrelation>,
    pub ar_coefficient: Option<ArCoefficient>,
    pub augmented_dickey_fuller: Option<AugmentedDickeyFuller>,
    pub binned_entropy: Option<BinnedEntropy>,
    pub lempel_ziv_complexity: Option<LempelZivComplexity>,
    pub permutation_entropy: Option<PermutationEntropy>,
    pub approximate_entropy: Option<ApproximateEntropy>,
    pub friedrich_coefficients: Option<FriedrichCoefficients>,
    pub max_langevin_fixed_point: Option<MaxLangevinFixedPoint>,
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
            mean_second_derivative_central: Some(MeanSecondDerivativeCentral::default()),
            value_count: Some(ValueCount::default()),
            energy_ratio_by_chunks: Some(EnergyRatioByChunks::default()),
            change_quantiles: Some(ChangeQuantiles::default()),
            benford_correlation: Some(BenfordCorrelation::default()),
            // Needs a query; tsfresh's default (no query) only gives NaN
            query_similarity_count: None,
            fft_coefficient: Some(FftCoefficient::default()),
            fft_aggregated: Some(FftAggregated::default()),
            spkt_welch_density: Some(SpktWelchDensity::default()),
            fourier_entropy: Some(FourierEntropy::default()),
            cwt_coefficients: Some(CwtCoefficients::default()),
            number_cwt_peaks: Some(NumberCwtPeaks::default()),
            agg_autocorrelation: Some(AggAutocorrelation::default()),
            partial_autocorrelation: Some(PartialAutocorrelation::default()),
            ar_coefficient: Some(ArCoefficient::default()),
            augmented_dickey_fuller: Some(AugmentedDickeyFuller::default()),
            binned_entropy: Some(BinnedEntropy::default()),
            lempel_ziv_complexity: Some(LempelZivComplexity::default()),
            permutation_entropy: Some(PermutationEntropy::default()),
            approximate_entropy: Some(ApproximateEntropy::default()),
            friedrich_coefficients: Some(FriedrichCoefficients::default()),
            max_langevin_fixed_point: Some(MaxLangevinFixedPoint::default()),
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
    pub t: f64,
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
    pub t: f64,
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
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Var,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Var,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Var,
                },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct AggLinearTrendParams {
    pub chunk_size: usize,
    pub aggregator: ChunkAggregator,
}

/// How `agg_linear_trend` aggregates each chunk. Unknown names are rejected
/// when the config is loaded.
#[derive(Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChunkAggregator {
    #[default]
    Mean,
    Median,
    Min,
    Max,
    Var,
}

impl Display for ChunkAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ChunkAggregator::Mean => write!(f, "mean"),
            ChunkAggregator::Median => write!(f, "median"),
            ChunkAggregator::Max => write!(f, "max"),
            ChunkAggregator::Min => write!(f, "min"),
            ChunkAggregator::Var => write!(f, "var"),
        }
    }
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

#[derive(Deserialize, Clone, Debug, Default)]
pub struct MeanSecondDerivativeCentral {}

#[derive(Deserialize, Clone, Debug)]
pub struct ValueCount {
    pub parameters: Vec<ValueCountParams>,
}
impl Default for ValueCount {
    fn default() -> Self {
        ValueCount {
            parameters: [0.0, 1.0, -1.0]
                .map(|value| ValueCountParams { value })
                .to_vec(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ValueCountParams {
    pub value: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct EnergyRatioByChunks {
    pub parameters: Vec<EnergyRatioByChunksParams>,
}
impl Default for EnergyRatioByChunks {
    fn default() -> Self {
        EnergyRatioByChunks {
            parameters: (0..10)
                .map(|segment_focus| EnergyRatioByChunksParams {
                    num_segments: 10,
                    segment_focus,
                })
                .collect(),
        }
    }
}

/// Rejected when loaded unless `segment_focus < num_segments`.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(try_from = "RawEnergyRatioByChunksParams")]
pub struct EnergyRatioByChunksParams {
    pub num_segments: usize,
    pub segment_focus: usize,
}

#[derive(Deserialize)]
struct RawEnergyRatioByChunksParams {
    num_segments: usize,
    segment_focus: usize,
}

impl TryFrom<RawEnergyRatioByChunksParams> for EnergyRatioByChunksParams {
    type Error = String;

    fn try_from(raw: RawEnergyRatioByChunksParams) -> Result<Self, Self::Error> {
        if raw.segment_focus >= raw.num_segments {
            return Err(format!(
                "energy_ratio_by_chunks needs segment_focus < num_segments, \
                 got segment_focus = {} and num_segments = {}",
                raw.segment_focus, raw.num_segments
            ));
        }
        Ok(EnergyRatioByChunksParams {
            num_segments: raw.num_segments,
            segment_focus: raw.segment_focus,
        })
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ChangeQuantiles {
    pub parameters: Vec<ChangeQuantilesParams>,
}
impl Default for ChangeQuantiles {
    fn default() -> Self {
        // tsfresh's grid: every ql < qh, both isabs, mean and var
        let mut parameters = Vec::new();
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8] {
            for qh in [0.2, 0.4, 0.6, 0.8, 1.0] {
                for isabs in [false, true] {
                    for f_agg in [ChangeAggregator::Mean, ChangeAggregator::Var] {
                        if ql < qh {
                            parameters.push(ChangeQuantilesParams {
                                ql,
                                qh,
                                isabs,
                                f_agg,
                            });
                        }
                    }
                }
            }
        }
        ChangeQuantiles { parameters }
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct ChangeQuantilesParams {
    pub ql: f64,
    pub qh: f64,
    pub isabs: bool,
    pub f_agg: ChangeAggregator,
}

/// How `change_quantiles` aggregates the changes inside the corridor.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChangeAggregator {
    Mean,
    Var,
    Std,
    Median,
}

impl Display for ChangeAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ChangeAggregator::Mean => "mean",
            ChangeAggregator::Var => "var",
            ChangeAggregator::Std => "std",
            ChangeAggregator::Median => "median",
        };
        write!(f, "{s}")
    }
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct BenfordCorrelation {}

/// Off by default: there is no sensible default query.
#[derive(Deserialize, Clone, Debug)]
pub struct QuerySimilarityCount {
    pub parameters: Vec<QuerySimilarityCountParams>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct QuerySimilarityCountParams {
    pub query: Vec<f64>,
    #[serde(default)]
    pub threshold: f64,
    #[serde(default = "default_true")]
    pub normalize: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, Clone, Debug)]
pub struct FftCoefficient {
    pub parameters: Vec<FftCoefficientParams>,
}
impl Default for FftCoefficient {
    fn default() -> Self {
        let attrs = [FftAttr::Real, FftAttr::Imag, FftAttr::Abs, FftAttr::Angle];
        FftCoefficient {
            parameters: attrs
                .into_iter()
                .flat_map(|attr| {
                    (0..100).map(move |coeff| FftCoefficientParams {
                        attr: attr.clone(),
                        coeff,
                    })
                })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FftCoefficientParams {
    pub attr: FftAttr,
    pub coeff: usize,
}

/// Which part of a Fourier coefficient `fft_coefficient` returns. Unknown
/// names are rejected when the config is loaded.
#[derive(Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum FftAttr {
    #[default]
    Real,
    Imag,
    Abs,
    Angle,
}

impl Display for FftAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FftAttr::Real => write!(f, "real"),
            FftAttr::Imag => write!(f, "imag"),
            FftAttr::Abs => write!(f, "abs"),
            FftAttr::Angle => write!(f, "angle"),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct FftAggregated {
    pub parameters: Vec<FftAggregatedParams>,
}
impl Default for FftAggregated {
    fn default() -> Self {
        FftAggregated {
            parameters: vec![
                FftAggregatedParams {
                    aggtype: FftAggType::Centroid,
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Variance,
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Skew,
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Kurtosis,
                },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FftAggregatedParams {
    pub aggtype: FftAggType,
}

/// Which statistic of the Fourier spectrum `fft_aggregated` returns.
/// Unknown names are rejected when the config is loaded.
#[derive(Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum FftAggType {
    #[default]
    Centroid,
    Variance,
    Skew,
    Kurtosis,
}

impl Display for FftAggType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FftAggType::Centroid => write!(f, "centroid"),
            FftAggType::Variance => write!(f, "variance"),
            FftAggType::Skew => write!(f, "skew"),
            FftAggType::Kurtosis => write!(f, "kurtosis"),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct SpktWelchDensity {
    pub parameters: Vec<SpktWelchDensityParams>,
}
impl Default for SpktWelchDensity {
    fn default() -> Self {
        SpktWelchDensity {
            parameters: vec![
                SpktWelchDensityParams { coeff: 2 },
                SpktWelchDensityParams { coeff: 5 },
                SpktWelchDensityParams { coeff: 8 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SpktWelchDensityParams {
    pub coeff: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct FourierEntropy {
    pub parameters: Vec<FourierEntropyParams>,
}
impl Default for FourierEntropy {
    fn default() -> Self {
        FourierEntropy {
            parameters: vec![
                FourierEntropyParams { bins: 2 },
                FourierEntropyParams { bins: 3 },
                FourierEntropyParams { bins: 5 },
                FourierEntropyParams { bins: 10 },
                FourierEntropyParams { bins: 100 },
            ],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FourierEntropyParams {
    pub bins: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CwtCoefficients {
    pub parameters: Vec<CwtCoefficientsParams>,
}
impl Default for CwtCoefficients {
    fn default() -> Self {
        let widths = vec![2, 5, 10, 20];
        let mut parameters = Vec::new();
        for coeff in 0..15 {
            for &w in &widths {
                parameters.push(CwtCoefficientsParams {
                    widths: widths.clone(),
                    coeff,
                    w,
                });
            }
        }
        CwtCoefficients { parameters }
    }
}

/// One `cwt_coefficients` entry; `w` must be one of `widths`, which must be
/// positive.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(try_from = "RawCwtCoefficientsParams")]
pub struct CwtCoefficientsParams {
    pub widths: Vec<usize>,
    pub coeff: usize,
    pub w: usize,
}

#[derive(Deserialize)]
struct RawCwtCoefficientsParams {
    widths: Vec<usize>,
    coeff: usize,
    w: usize,
}

impl TryFrom<RawCwtCoefficientsParams> for CwtCoefficientsParams {
    type Error = String;

    fn try_from(raw: RawCwtCoefficientsParams) -> Result<Self, Self::Error> {
        if raw.widths.is_empty() || raw.widths.contains(&0) {
            return Err(format!(
                "cwt_coefficients widths must be positive, got {:?}",
                raw.widths
            ));
        }
        if !raw.widths.contains(&raw.w) {
            return Err(format!(
                "cwt_coefficients w = {} is not one of widths {:?}",
                raw.w, raw.widths
            ));
        }
        Ok(CwtCoefficientsParams {
            widths: raw.widths,
            coeff: raw.coeff,
            w: raw.w,
        })
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct NumberCwtPeaks {
    pub parameters: Vec<NumberCwtPeaksParams>,
}
impl Default for NumberCwtPeaks {
    fn default() -> Self {
        NumberCwtPeaks {
            parameters: vec![NumberCwtPeaksParams { n: 1 }, NumberCwtPeaksParams { n: 5 }],
        }
    }
}

/// One `number_cwt_peaks` entry; `n` must be positive.
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(try_from = "RawNumberCwtPeaksParams")]
pub struct NumberCwtPeaksParams {
    pub n: usize,
}

#[derive(Deserialize)]
struct RawNumberCwtPeaksParams {
    n: usize,
}

impl TryFrom<RawNumberCwtPeaksParams> for NumberCwtPeaksParams {
    type Error = String;

    fn try_from(raw: RawNumberCwtPeaksParams) -> Result<Self, Self::Error> {
        if raw.n == 0 {
            return Err("number_cwt_peaks n must be positive".to_string());
        }
        Ok(NumberCwtPeaksParams { n: raw.n })
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct AggAutocorrelation {
    pub parameters: Vec<AggAutocorrelationParams>,
}

impl Default for AggAutocorrelation {
    fn default() -> Self {
        AggAutocorrelation {
            parameters: [
                AutocorrelationAggregator::Mean,
                AutocorrelationAggregator::Median,
                AutocorrelationAggregator::Var,
            ]
            .into_iter()
            .map(|f_agg| AggAutocorrelationParams { f_agg, maxlag: 40 })
            .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct AggAutocorrelationParams {
    pub f_agg: AutocorrelationAggregator,
    pub maxlag: usize,
}

/// How `agg_autocorrelation` aggregates the autocorrelations (the numpy
/// function of that name). Unknown names are rejected when the config is
/// loaded.
#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum AutocorrelationAggregator {
    Mean,
    Median,
    Var,
    Std,
}

impl Display for AutocorrelationAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AutocorrelationAggregator::Mean => write!(f, "mean"),
            AutocorrelationAggregator::Median => write!(f, "median"),
            AutocorrelationAggregator::Var => write!(f, "var"),
            AutocorrelationAggregator::Std => write!(f, "std"),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct PartialAutocorrelation {
    pub parameters: Vec<PartialAutocorrelationParams>,
}

impl Default for PartialAutocorrelation {
    fn default() -> Self {
        PartialAutocorrelation {
            parameters: (0..10)
                .map(|lag| PartialAutocorrelationParams { lag })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct PartialAutocorrelationParams {
    pub lag: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ArCoefficient {
    pub parameters: Vec<ArCoefficientParams>,
}

impl Default for ArCoefficient {
    fn default() -> Self {
        ArCoefficient {
            parameters: (0..=10)
                .map(|coeff| ArCoefficientParams { coeff, k: 10 })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct ArCoefficientParams {
    pub coeff: usize,
    pub k: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct AugmentedDickeyFuller {
    pub parameters: Vec<AugmentedDickeyFullerParams>,
}

impl Default for AugmentedDickeyFuller {
    fn default() -> Self {
        AugmentedDickeyFuller {
            parameters: [AdfAttr::Teststat, AdfAttr::Pvalue, AdfAttr::Usedlag]
                .into_iter()
                .map(|attr| AugmentedDickeyFullerParams {
                    attr,
                    autolag: AdfAutolag::default(),
                })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct AugmentedDickeyFullerParams {
    pub attr: AdfAttr,
    /// Optional in the config; `"AIC"` when left out, as in tsfresh.
    #[serde(default)]
    pub autolag: AdfAutolag,
}

/// Which result of the augmented Dickey-Fuller test to return. Unknown names
/// are rejected when the config is loaded.
#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum AdfAttr {
    Teststat,
    Pvalue,
    Usedlag,
}

impl Display for AdfAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AdfAttr::Teststat => write!(f, "teststat"),
            AdfAttr::Pvalue => write!(f, "pvalue"),
            AdfAttr::Usedlag => write!(f, "usedlag"),
        }
    }
}

/// How the augmented Dickey-Fuller test chooses its number of lags
/// (statsmodels' `autolag`): `"AIC"`, `"BIC"`, `"t-stat"` or `"None"` (always
/// the maximum lag). Unknown names are rejected when the config is loaded.
#[derive(Deserialize, Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum AdfAutolag {
    #[default]
    #[serde(rename = "AIC")]
    Aic,
    #[serde(rename = "BIC")]
    Bic,
    #[serde(rename = "t-stat")]
    TStat,
    #[serde(rename = "None")]
    None,
}

impl Display for AdfAutolag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AdfAutolag::Aic => write!(f, "AIC"),
            AdfAutolag::Bic => write!(f, "BIC"),
            AdfAutolag::TStat => write!(f, "t-stat"),
            AdfAutolag::None => write!(f, "None"),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct BinnedEntropy {
    pub parameters: Vec<BinnedEntropyParams>,
}

impl Default for BinnedEntropy {
    fn default() -> Self {
        BinnedEntropy {
            parameters: vec![BinnedEntropyParams { max_bins: 10 }],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct BinnedEntropyParams {
    pub max_bins: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct LempelZivComplexity {
    pub parameters: Vec<LempelZivComplexityParams>,
}

impl Default for LempelZivComplexity {
    fn default() -> Self {
        LempelZivComplexity {
            parameters: [2, 3, 5, 10, 100]
                .map(|bins| LempelZivComplexityParams { bins })
                .to_vec(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct LempelZivComplexityParams {
    pub bins: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PermutationEntropy {
    pub parameters: Vec<PermutationEntropyParams>,
}

impl Default for PermutationEntropy {
    fn default() -> Self {
        PermutationEntropy {
            parameters: (3..=7)
                .map(|dimension| PermutationEntropyParams { tau: 1, dimension })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct PermutationEntropyParams {
    pub tau: usize,
    pub dimension: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ApproximateEntropy {
    pub parameters: Vec<ApproximateEntropyParams>,
}

impl Default for ApproximateEntropy {
    fn default() -> Self {
        ApproximateEntropy {
            parameters: [0.1, 0.3, 0.5, 0.7, 0.9]
                .map(|r| ApproximateEntropyParams { m: 2, r })
                .to_vec(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ApproximateEntropyParams {
    pub m: usize,
    pub r: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct FriedrichCoefficients {
    pub parameters: Vec<FriedrichCoefficientsParams>,
}

impl Default for FriedrichCoefficients {
    fn default() -> Self {
        FriedrichCoefficients {
            parameters: (0..=3)
                .map(|coeff| FriedrichCoefficientsParams { coeff, m: 3, r: 30 })
                .collect(),
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FriedrichCoefficientsParams {
    pub coeff: usize,
    pub m: usize,
    pub r: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct MaxLangevinFixedPoint {
    pub parameters: Vec<MaxLangevinFixedPointParams>,
}

impl Default for MaxLangevinFixedPoint {
    fn default() -> Self {
        MaxLangevinFixedPoint {
            parameters: vec![MaxLangevinFixedPointParams { m: 3, r: 30 }],
        }
    }
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
pub struct MaxLangevinFixedPointParams {
    pub m: usize,
    pub r: usize,
}

/// A config file that could not be read or parsed.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("could not read config file {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid config file {path}: {source}")]
    Parse {
        path: String,
        source: toml::de::Error,
    },
}

/// The config file looked for in the working directory when no path is
/// given.
pub const DEFAULT_CONFIG_FILE: &str = ".tsfx-config.toml";

/// Load the feature config from `file_path`, or without one from
/// [`DEFAULT_CONFIG_FILE`] in the working directory. Without a `file_path`, a
/// missing [`DEFAULT_CONFIG_FILE`] gives the default config; a given
/// `file_path` must exist ([`ConfigError::Read`] otherwise), so a mistyped
/// path is not silently replaced by the defaults.
pub fn load_config(file_path: Option<&str>) -> Result<Config, ConfigError> {
    let file_path = match file_path {
        Some(path) => path,
        None if std::path::Path::new(DEFAULT_CONFIG_FILE).exists() => DEFAULT_CONFIG_FILE,
        None => return Ok(Config::default()),
    };
    let config_str = std::fs::read_to_string(file_path).map_err(|source| ConfigError::Read {
        path: file_path.to_string(),
        source,
    })?;
    toml::from_str(&config_str).map_err(|source| ConfigError::Parse {
        path: file_path.to_string(),
        source,
    })
}

// cargo test toml file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_wavelet_params_rejected() {
        let ok: Result<Config, _> = toml::from_str(
            "[cwt_coefficients]\nparameters = [{ widths = [1, 3], coeff = 0, w = 3 }]",
        );
        assert!(ok.is_ok());
        for bad in [
            "[cwt_coefficients]\nparameters = [{ widths = [1, 3], coeff = 0, w = 2 }]",
            "[cwt_coefficients]\nparameters = [{ widths = [0, 3], coeff = 0, w = 3 }]",
            "[number_cwt_peaks]\nparameters = [{ n = 0 }]",
        ] {
            assert!(toml::from_str::<Config>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn test_read_config_from_file() {
        let config = load_config(None).unwrap();
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
        assert!(
            config
                .percentage_of_reoccurring_values_to_all_values
                .is_some()
        );
        assert!(
            config
                .percentage_of_reoccurring_values_to_all_datapoints
                .is_some()
        );
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
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 5,
                    aggregator: ChunkAggregator::Var,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 10,
                    aggregator: ChunkAggregator::Var,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Mean,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Min,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Max,
                },
                AggLinearTrendParams {
                    chunk_size: 50,
                    aggregator: ChunkAggregator::Var,
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
        assert!(config.mean_second_derivative_central.is_some());
        assert_eq!(
            config.clone().value_count.unwrap().parameters[..],
            [
                ValueCountParams { value: 0.0 },
                ValueCountParams { value: 1.0 },
                ValueCountParams { value: -1.0 },
            ]
        );
        let energy = config.clone().energy_ratio_by_chunks.unwrap().parameters;
        assert_eq!(energy.len(), 10);
        assert!(
            energy
                .iter()
                .enumerate()
                .all(|(i, p)| p.num_segments == 10 && p.segment_focus == i)
        );
        let change = config.clone().change_quantiles.unwrap().parameters;
        assert_eq!(change.len(), 60);
        assert_eq!(
            change[0],
            ChangeQuantilesParams {
                ql: 0.0,
                qh: 0.2,
                isabs: false,
                f_agg: ChangeAggregator::Mean,
            }
        );
        assert!(change.iter().all(|p| p.ql < p.qh));
        assert!(config.benford_correlation.is_some());
        assert!(config.query_similarity_count.is_none());
        assert!(config.fft_coefficient.is_some());
        let fft_coefficient = config.clone().fft_coefficient.unwrap().parameters;
        assert_eq!(fft_coefficient.len(), 400);
        assert_eq!(
            fft_coefficient[0],
            FftCoefficientParams {
                attr: FftAttr::Real,
                coeff: 0
            }
        );
        assert_eq!(
            fft_coefficient[399],
            FftCoefficientParams {
                attr: FftAttr::Angle,
                coeff: 99
            }
        );
        assert!(config.fft_aggregated.is_some());
        assert_eq!(
            config.clone().fft_aggregated.unwrap().parameters[..],
            [
                FftAggregatedParams {
                    aggtype: FftAggType::Centroid
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Variance
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Skew
                },
                FftAggregatedParams {
                    aggtype: FftAggType::Kurtosis
                },
            ]
        );
        assert!(config.spkt_welch_density.is_some());
        assert_eq!(
            config.clone().spkt_welch_density.unwrap().parameters[..],
            [
                SpktWelchDensityParams { coeff: 2 },
                SpktWelchDensityParams { coeff: 5 },
                SpktWelchDensityParams { coeff: 8 },
            ]
        );
        assert!(config.fourier_entropy.is_some());
        assert_eq!(
            config.clone().fourier_entropy.unwrap().parameters[..],
            [
                FourierEntropyParams { bins: 2 },
                FourierEntropyParams { bins: 3 },
                FourierEntropyParams { bins: 5 },
                FourierEntropyParams { bins: 10 },
                FourierEntropyParams { bins: 100 },
            ]
        );
        assert!(config.cwt_coefficients.is_some());
        let cwt = config.clone().cwt_coefficients.unwrap().parameters;
        assert_eq!(cwt.len(), 60);
        assert_eq!(
            cwt[..2],
            [
                CwtCoefficientsParams {
                    widths: vec![2, 5, 10, 20],
                    coeff: 0,
                    w: 2
                },
                CwtCoefficientsParams {
                    widths: vec![2, 5, 10, 20],
                    coeff: 0,
                    w: 5
                },
            ]
        );
        assert_eq!(
            cwt[59],
            CwtCoefficientsParams {
                widths: vec![2, 5, 10, 20],
                coeff: 14,
                w: 20
            }
        );
        assert!(config.number_cwt_peaks.is_some());
        assert_eq!(
            config.clone().number_cwt_peaks.unwrap().parameters[..],
            [NumberCwtPeaksParams { n: 1 }, NumberCwtPeaksParams { n: 5 }]
        );
        assert!(config.agg_autocorrelation.is_some());
        assert_eq!(
            config.clone().agg_autocorrelation.unwrap().parameters[..],
            [
                AggAutocorrelationParams {
                    f_agg: AutocorrelationAggregator::Mean,
                    maxlag: 40
                },
                AggAutocorrelationParams {
                    f_agg: AutocorrelationAggregator::Median,
                    maxlag: 40
                },
                AggAutocorrelationParams {
                    f_agg: AutocorrelationAggregator::Var,
                    maxlag: 40
                },
            ]
        );
        assert!(config.partial_autocorrelation.is_some());
        assert_eq!(
            config.clone().partial_autocorrelation.unwrap().parameters[..],
            (0..10)
                .map(|lag| PartialAutocorrelationParams { lag })
                .collect::<Vec<_>>()[..]
        );
        assert!(config.ar_coefficient.is_some());
        assert_eq!(
            config.clone().ar_coefficient.unwrap().parameters[..],
            (0..=10)
                .map(|coeff| ArCoefficientParams { coeff, k: 10 })
                .collect::<Vec<_>>()[..]
        );
        assert!(config.augmented_dickey_fuller.is_some());
        assert_eq!(
            config.clone().augmented_dickey_fuller.unwrap().parameters[..],
            [AdfAttr::Teststat, AdfAttr::Pvalue, AdfAttr::Usedlag].map(|attr| {
                AugmentedDickeyFullerParams {
                    attr,
                    autolag: AdfAutolag::Aic,
                }
            })
        );
        assert_eq!(
            config.clone().binned_entropy.unwrap().parameters[..],
            [BinnedEntropyParams { max_bins: 10 }]
        );
        assert_eq!(
            config.clone().lempel_ziv_complexity.unwrap().parameters[..],
            [2, 3, 5, 10, 100].map(|bins| LempelZivComplexityParams { bins })
        );
        assert_eq!(
            config.clone().permutation_entropy.unwrap().parameters[..],
            [3, 4, 5, 6, 7].map(|dimension| PermutationEntropyParams { tau: 1, dimension })
        );
        assert_eq!(
            config.clone().approximate_entropy.unwrap().parameters[..],
            [0.1, 0.3, 0.5, 0.7, 0.9].map(|r| ApproximateEntropyParams { m: 2, r })
        );
        assert_eq!(
            config.clone().friedrich_coefficients.unwrap().parameters[..],
            [0, 1, 2, 3].map(|coeff| FriedrichCoefficientsParams { coeff, m: 3, r: 30 })
        );
        assert_eq!(
            config.clone().max_langevin_fixed_point.unwrap().parameters[..],
            [MaxLangevinFixedPointParams { m: 3, r: 30 }]
        );
    }

    #[test]
    fn test_query_similarity_count_defaults() {
        let config: Config =
            toml::from_str("[query_similarity_count]\nparameters = [{ query = [1.0, 2.0, 3.0] }]")
                .unwrap();
        assert_eq!(
            config.query_similarity_count.unwrap().parameters[..],
            [QuerySimilarityCountParams {
                query: vec![1.0, 2.0, 3.0],
                threshold: 0.0,
                normalize: true,
            }]
        );
    }

    #[test]
    fn test_invalid_energy_ratio_segment_focus() {
        let result = toml::from_str::<Config>(
            "[energy_ratio_by_chunks]\nparameters = [{ num_segments = 3, segment_focus = 3 }]",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_change_quantiles_aggregator() {
        let result = toml::from_str::<Config>(
            "[change_quantiles]\nparameters = [{ ql = 0.0, qh = 1.0, isabs = false, f_agg = \"sum\" }]",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_adf_autolag_optional_and_validated() {
        let config: Config = toml::from_str(
            r#"
            [augmented_dickey_fuller]
            parameters = [{ attr = "pvalue" }, { attr = "usedlag", autolag = "t-stat" }]
            "#,
        )
        .unwrap();
        assert_eq!(
            config.augmented_dickey_fuller.unwrap().parameters[..],
            [
                AugmentedDickeyFullerParams {
                    attr: AdfAttr::Pvalue,
                    autolag: AdfAutolag::Aic
                },
                AugmentedDickeyFullerParams {
                    attr: AdfAttr::Usedlag,
                    autolag: AdfAutolag::TStat
                },
            ]
        );
        assert!(
            toml::from_str::<Config>(
                "[augmented_dickey_fuller]\nparameters = [{ attr = \"pvalue\", autolag = \"aic\" }]"
            )
            .is_err()
        );
        assert!(
            toml::from_str::<Config>(
                "[agg_autocorrelation]\nparameters = [{ f_agg = \"max\", maxlag = 4 }]"
            )
            .is_err()
        );
    }
}
