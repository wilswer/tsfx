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
    pub linear_trend_intercept: Option<LinearTrendIntercept>,
    pub linear_trend_slope: Option<LinearTrendSlope>,
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
            linear_trend_intercept: Some(LinearTrendIntercept::default()),
            linear_trend_slope: Some(LinearTrendSlope::default()),
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
pub struct LinearTrendIntercept {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LinearTrendSlope {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct VarianceLargerThanStandardDeviation {}

#[derive(Deserialize, Clone, Debug)]
pub struct RatioBeyondRSigma {
    parameters: Vec<RatioBeyondRSigmaParams>,
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
    r: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct LargeStandardDeviation {
    parameters: Vec<LargeStandardDeviationParams>,
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
    r: f64,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SymmetryLooking {
    parameters: Vec<SymmetryLookingParams>,
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
    r: f64,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HasDuplicateMax {}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HasDuplicateMin {}

#[derive(Deserialize, Clone, Debug)]
pub struct CidCe {
    parameters: Vec<CidCeParams>,
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
    normalize: bool,
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
    parameters: Vec<CountAboveParams>,
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
    parameters: Vec<CountBelowParams>,
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

pub fn load_config(file_path: Option<&str>) -> Config {
    let file_path = if let Some(file_path) = file_path {
        file_path
    } else {
        ".tsfx-config.toml"
    };
    if !std::path::Path::new(file_path).exists() {
        println!("Config file not found. Using default config.");
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
        assert!(config.linear_trend_intercept.is_some());
        assert!(config.linear_trend_slope.is_some());
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
            &config.clone().ratio_beyond_r_sigma.unwrap().parameters[..],
            &[
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
            &config.clone().large_standard_deviation.unwrap().parameters[..],
            &[
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
            &config.clone().symmetry_looking.unwrap().parameters[..],
            &[
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
    }
}
