use serde::Deserialize;
use toml;

#[derive(Deserialize, Clone, Debug)]
#[allow(dead_code)]
struct Config {
    length: Option<Length>,
    sum_values: Option<SumValues>,
    mean: Option<Mean>,
    median: Option<Median>,
    minimum: Option<Minimum>,
    maximum: Option<Maximum>,
    standard_deviation: Option<StandardDeviation>,
    variance: Option<Variance>,
    skewness: Option<Skewness>,
    root_mean_square: Option<RootMeanSquare>,
    kurtosis: Option<Kurtosis>,
    absolute_energy: Option<AbsoluteEnergy>,
    mean_absolute_change: Option<MeanAbsoluteChange>,
    linear_trend_intercept: Option<LinearTrendIntercept>,
    linear_trend_slope: Option<LinearTrendSlope>,
    variance_larger_than_standard_deviation: Option<VarianceLargerThanStandardDeviation>,
    ratio_beyond_r_sigma: Option<RatioBeyondRSigma>,
}

#[derive(Deserialize, Clone, Debug)]
struct Length {}
#[derive(Deserialize, Clone, Debug)]
struct SumValues {}
#[derive(Deserialize, Clone, Debug)]
struct Mean {}
#[derive(Deserialize, Clone, Debug)]
struct Median {}
#[derive(Deserialize, Clone, Debug)]
struct Minimum {}
#[derive(Deserialize, Clone, Debug)]
struct Maximum {}
#[derive(Deserialize, Clone, Debug)]
struct StandardDeviation {}
#[derive(Deserialize, Clone, Debug)]
struct Variance {}
#[derive(Deserialize, Clone, Debug)]
struct Skewness {}
#[derive(Deserialize, Clone, Debug)]
struct RootMeanSquare {}
#[derive(Deserialize, Clone, Debug)]
struct Kurtosis {}
#[derive(Deserialize, Clone, Debug)]
struct AbsoluteEnergy {}
#[derive(Deserialize, Clone, Debug)]
struct MeanAbsoluteChange {}
#[derive(Deserialize, Clone, Debug)]
struct LinearTrendIntercept {}
#[derive(Deserialize, Clone, Debug)]
struct LinearTrendSlope {}
#[derive(Deserialize, Clone, Debug)]
struct VarianceLargerThanStandardDeviation {}
#[derive(Deserialize, Clone, Debug)]
#[allow(dead_code)]
struct RatioBeyondRSigma {
    parameters: Vec<RatioBeyondRSigmaParams>,
}
#[derive(Deserialize, Clone, Debug, PartialEq)]
struct RatioBeyondRSigmaParams {
    r: f64,
}

#[allow(dead_code)]
fn read_config_from_file(file_path: Option<&str>) -> Config {
    let file_path = if let Some(file_path) = file_path {
        file_path
    } else {
        ".tsfx.toml"
    };
    let config_str = std::fs::read_to_string(file_path).unwrap();
    let config: Config = toml::from_str(&config_str).unwrap();
    config
}

// cargo test toml file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_config_from_file() {
        let config = read_config_from_file(None);
        assert_eq!(config.length.is_some(), true);
        assert_eq!(config.sum_values.is_some(), true);
        assert_eq!(config.mean.is_some(), true);
        assert_eq!(config.median.is_some(), true);
        assert_eq!(config.minimum.is_some(), true);
        assert_eq!(config.maximum.is_some(), true);
        assert_eq!(config.standard_deviation.is_some(), true);
        assert_eq!(config.variance.is_some(), true);
        assert_eq!(config.skewness.is_some(), true);
        assert_eq!(config.root_mean_square.is_some(), true);
        assert_eq!(config.kurtosis.is_some(), true);
        assert_eq!(config.absolute_energy.is_some(), true);
        assert_eq!(config.mean_absolute_change.is_some(), true);
        assert_eq!(config.linear_trend_intercept.is_some(), true);
        assert_eq!(config.linear_trend_slope.is_some(), true);
        assert_eq!(
            config.variance_larger_than_standard_deviation.is_some(),
            true
        );
        assert_eq!(config.ratio_beyond_r_sigma.is_some(), true);
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
            &config.ratio_beyond_r_sigma.unwrap().parameters[..],
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
    }
}
