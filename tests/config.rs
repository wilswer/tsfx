use tsfx::utils::toml_reader::load_config;

#[test]
fn test_read_config_from_empty_file() {
    let config = load_config(Some("./tests/data/.tsfx-config-empty.toml"));
    assert_eq!(config.length.is_some(), false);
    assert_eq!(config.sum_values.is_some(), false);
    assert_eq!(config.mean.is_some(), false);
    assert_eq!(config.median.is_some(), false);
    assert_eq!(config.minimum.is_some(), false);
    assert_eq!(config.maximum.is_some(), false);
    assert_eq!(config.standard_deviation.is_some(), false);
    assert_eq!(config.variance.is_some(), false);
    assert_eq!(config.skewness.is_some(), false);
    assert_eq!(config.root_mean_square.is_some(), false);
    assert_eq!(config.kurtosis.is_some(), false);
    assert_eq!(config.absolute_energy.is_some(), false);
    assert_eq!(config.mean_absolute_change.is_some(), false);
    assert_eq!(config.linear_trend_intercept.is_some(), false);
    assert_eq!(config.linear_trend_slope.is_some(), false);
    assert_eq!(
        config.variance_larger_than_standard_deviation.is_some(),
        false
    );
    assert_eq!(config.ratio_beyond_r_sigma.is_some(), false);
    assert_eq!(config.large_standard_deviation.is_some(), false);
}
