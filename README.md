# TSFX

_TSFX -- Time Series Feature eXtraction_

## About

TSFX is a Python library for extracting features from time series data.
Inspired by the great [TSFresh](https://tsfresh.com/) library, TSFX aims to
provide a similar feature set focused on performance on large datasets.
In order to achieve this, TSFX is built on top of the
[Polars](https://pola.rs/) DataFrame library, and feature extractors are
implemented in Rust.

## Installation

Install from PyPI:

```bash
pip install tsfx
```

## Usage

Below is a simple example of extracting features from a time series dataset:

```python
import polars as pl
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)

df = pl.DataFrame(
    {
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
    },
).lazy()
settings = ExtractionSettings(
    grouping_col="id",
    feature_setting=FeatureSetting.Efficient,
    value_cols=["val", "value"],
)
gdf = extract_features(df, settings)
gdf = gdf.sort(by="id")
print(gdf)
```

which produces the following output:

```bash
shape: (3, 316)
┌─────┬────────┬─────────────┬───────────┬───┬─────────────┬─────────────┬────────────┬────────────┐
│ id  ┆ length ┆ val__sum_va ┆ val__mean ┆ … ┆ value__numb ┆ value__numb ┆ value__num ┆ value__num │
│ --- ┆ ---    ┆ lues        ┆ ---       ┆   ┆ er_peaks__n ┆ er_peaks__n ┆ ber_peaks_ ┆ ber_peaks_ │
│ str ┆ u32    ┆ ---         ┆ f32       ┆   ┆ _3          ┆ _5          ┆ _n_10      ┆ _n_50      │
│     ┆        ┆ f32         ┆           ┆   ┆ ---         ┆ ---         ┆ ---        ┆ ---        │
│     ┆        ┆             ┆           ┆   ┆ f32         ┆ f32         ┆ f32        ┆ f32        │
╞═════╪════════╪═════════════╪═══════════╪═══╪═════════════╪═════════════╪════════════╪════════════╡
│ a   ┆ 3      ┆ 6.0         ┆ 2.0       ┆ … ┆ 0.0         ┆ 0.0         ┆ 0.0        ┆ 0.0        │
│ b   ┆ 3      ┆ 6.0         ┆ 2.0       ┆ … ┆ 0.0         ┆ 0.0         ┆ 0.0        ┆ 0.0        │
│ c   ┆ 3      ┆ 6.0         ┆ 2.0       ┆ … ┆ 0.0         ┆ 0.0         ┆ 0.0        ┆ 0.0        │
└─────┴────────┴─────────────┴───────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

### Extracting over a time window

An additional feature of TSFX is the ability to extract features over a time window.
Below is an example of extracting features over a 3 year window:

```python
import polars as pl
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)
tdf = pl.DataFrame(
    {
        "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "time": [
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
            "2001-01-01",
            "2002-01-01",
            "2003-01-01",
        ],
        "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
    },
).lazy()

dyn_settings = DynamicGroupBySettings(
    time_col="time",
    every="3y",
    period="3y",
    offset="0",
    datetime_format="%Y-%m-%d",
)
settings = ExtractionSettings(
    grouping_col="id",
    value_cols=["val", "value"],
    feature_setting=FeatureSetting.Efficient,
    dynamic_settings=dyn_settings,
)
gdf = extract_features(tdf, settings)
gdf = gdf.sort(by="id")
print(gdf)
```

which produces the following output:

```bash
shape: (3, 317)
┌─────┬────────────┬────────┬─────────────┬───┬─────────────┬────────────┬────────────┬────────────┐
│ id  ┆ time       ┆ length ┆ val__sum_va ┆ … ┆ value__numb ┆ value__num ┆ value__num ┆ value__num │
│ --- ┆ ---        ┆ ---    ┆ lues        ┆   ┆ er_peaks__n ┆ ber_peaks_ ┆ ber_peaks_ ┆ ber_peaks_ │
│ str ┆ date       ┆ u32    ┆ ---         ┆   ┆ _3          ┆ _n_5       ┆ _n_10      ┆ _n_50      │
│     ┆            ┆        ┆ f32         ┆   ┆ ---         ┆ ---        ┆ ---        ┆ ---        │
│     ┆            ┆        ┆             ┆   ┆ f32         ┆ f32        ┆ f32        ┆ f32        │
╞═════╪════════════╪════════╪═════════════╪═══╪═════════════╪════════════╪════════════╪════════════╡
│ a   ┆ 2001-01-01 ┆ 3      ┆ 6.0         ┆ … ┆ 0.0         ┆ 0.0        ┆ 0.0        ┆ 0.0        │
│ b   ┆ 2001-01-01 ┆ 3      ┆ 6.0         ┆ … ┆ 0.0         ┆ 0.0        ┆ 0.0        ┆ 0.0        │
│ c   ┆ 2001-01-01 ┆ 3      ┆ 6.0         ┆ … ┆ 0.0         ┆ 0.0        ┆ 0.0        ┆ 0.0        │
└─────┴────────────┴────────┴─────────────┴───┴─────────────┴────────────┴────────────┴────────────┘
```

For more examples, see the [examples](examples) directory.

## Feature Coverage Compared to TSFresh

| Implemented | Function                                                    | Description                                                                                                                                                                                                                                                                                                        |
| ----------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| &#9745;     | `abs_energy(x)`                                             | Returns the absolute energy of the time series which is the sum over the squared values                                                                                                                                                                                                                            |
| &#9745;     | `absolute_maximum(x)`                                       | Calculates the highest absolute value of the time series x                                                                                                                                                                                                                                                         |
| &#9745;     | `absolute_sum_of_changes(x)`                                | Returns the sum over the absolute value of consecutive changes in the series x                                                                                                                                                                                                                                     |
| &#9744;     | `agg_autocorrelation(x, param)`                             | Descriptive statistics on the autocorrelation of the time series                                                                                                                                                                                                                                                   |
| &#9745;     | `agg_linear_trend(x, param)`                                | Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus the sequence from 0 up to the number of chunks minus one                                                                                                                                        |
| &#9744;     | `approximate_entropy(x, m, r)`                              | Implements a vectorized Approximate entropy algorithm                                                                                                                                                                                                                                                              |
| &#9744;     | `ar_coefficient(x, param)`                                  | This feature calculator fits the unconditional maximum likelihood of an autoregressive AR(k) process                                                                                                                                                                                                               |
| &#9744;     | `augmented_dickey_fuller(x, param)`                         | Does the time series have a unit root?                                                                                                                                                                                                                                                                             |
| &#9745;     | `autocorrelation(x, lag)`                                   | Calculates the autocorrelation of the specified lag, according to the formula [1]                                                                                                                                                                                                                                  |
| &#9744;     | `benford_correlation(x)`                                    | Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when                                                                                                                                                                                                       |
| &#9744;     | `binned_entropy(x, max_bins)`                               | First bins the values of x into max_bins equidistant bins                                                                                                                                                                                                                                                          |
| &#9745;     | `c3(x, lag)`                                                | Uses c3 statistics to measure non linearity in the time series                                                                                                                                                                                                                                                     |
| &#9744;     | `change_quantiles(x, ql, qh, isabs, f_agg)`                 | First fixes a corridor given by the quantiles ql and qh of the distribution of x                                                                                                                                                                                                                                   |
| &#9745;     | `cid_ce(x, normalize)`                                      | This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks, valleys etc.).                                                                                                                                                                                |
| &#9745;     | `count_above(x, t)`                                         | Returns the percentage of values in x that are higher than t                                                                                                                                                                                                                                                       |
| &#9745;     | `count_above_mean(x)`                                       | Returns the number of values in x that are higher than the mean of x                                                                                                                                                                                                                                               |
| &#9745;     | `count_below(x, t)`                                         | Returns the percentage of values in x that are lower than t                                                                                                                                                                                                                                                        |
| &#9745;     | `count_below_mean(x)`                                       | Returns the number of values in x that are lower than the mean of x                                                                                                                                                                                                                                                |
| &#9744;     | `cwt_coefficients(x, param)`                                | Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is defined by                                                                                                                                                                                      |
| &#9744;     | `energy_ratio_by_chunks(x, param)`                          | Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole series.                                                                                                                                                                                       |
| &#9744;     | `fft_aggregated(x, param)`                                  | Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum                                                                                                                                                                                                      |
| &#9744;     | `fft_coefficient(x, param)`                                 | Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast fourier transformation algorithm                                                                                                                                                                      |
| &#9745;     | `first_location_of_maximum(x)`                              | Returns the first location of the maximum value of x                                                                                                                                                                                                                                                               |
| &#9745;     | `first_location_of_minimum(x)`                              | Returns the first location of the minimal value of x                                                                                                                                                                                                                                                               |
| &#9744;     | `fourier_entropy(x, bins)`                                  | Calculate the binned entropy of the power spectral density of the time series (using the welch method)                                                                                                                                                                                                             |
| &#9744;     | `friedrich_coefficients(x, param)`                          | Coefficients of polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model                                                                                                                                                                                                             |
| &#9745;     | `has_duplicate(x)`                                          | Checks if any value in x occurs more than once                                                                                                                                                                                                                                                                     |
| &#9745;     | `has_duplicate_max(x)`                                      | Checks if the maximum value of x is observed more than once                                                                                                                                                                                                                                                        |
| &#9745;     | `has_duplicate_min(x)`                                      | Checks if the minimal value of x is observed more than once                                                                                                                                                                                                                                                        |
| &#9745;     | `index_mass_quantile(x, param)`                             | Calculates the relative index i of time series x where q% of the mass of x lies left of i.                                                                                                                                                                                                                         |
| &#9745;     | `kurtosis(x)`                                               | Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).                                                                                                                                                                                                        |
| &#9745;     | `large_standard_deviation(x, r)`                            | Does time series have large standard deviation?                                                                                                                                                                                                                                                                    |
| &#9745;     | `last_location_of_maximum(x)`                               | Returns the relative last location of the maximum value of x.                                                                                                                                                                                                                                                      |
| &#9745;     | `last_location_of_minimum(x)`                               | Returns the last location of the minimal value of x.                                                                                                                                                                                                                                                               |
| &#9744;     | `lempel_ziv_complexity(x, bins)`                            | Calculate a complexity estimate based on the Lempel-Ziv compression algorithm.                                                                                                                                                                                                                                     |
| &#9745;     | `length(x)`                                                 | Returns the length of x                                                                                                                                                                                                                                                                                            |
| &#9745;     | `linear_trend(x, param)`                                    | Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one.                                                                                                                                                                   |
| &#9744;     | `linear_trend_timewise(x, param)`                           | Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one.                                                                                                                                                                   |
| &#9745;     | `longest_strike_above_mean(x)`                              | Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x                                                                                                                                                                                                                   |
| &#9745;     | `longest_strike_below_mean(x)`                              | Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x                                                                                                                                                                                                                  |
| &#9744;     | `matrix_profile(x, param)`                                  | Calculates the 1-D Matrix Profile[1] and returns Tukey's Five Number Set plus the mean of that Matrix Profile.                                                                                                                                                                                                     |
| &#9744;     | `max_langevin_fixed_point(x, r, m)`                         | Largest fixed point of dynamics `:math:argmax_x {h(x)=0}` estimated from polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model                                                                                                                                                    |
| &#9745;     | `maximum(x)`                                                | Calculates the highest value of the time series x.                                                                                                                                                                                                                                                                 |
| &#9745;     | `mean(x)`                                                   | Returns the mean of x                                                                                                                                                                                                                                                                                              |
| &#9745;     | `mean_abs_change(x)`                                        | Average over first differences.                                                                                                                                                                                                                                                                                    |
| &#9745;     | `mean_change(x)`                                            | Average over time series differences.                                                                                                                                                                                                                                                                              |
| &#9745;     | `mean_n_absolute_max(x, number_of_maxima)`                  | Calculates the arithmetic mean of the n absolute maximum values of the time series.                                                                                                                                                                                                                                |
| &#9744;     | `mean_second_derivative_central(x)`                         | Returns the mean value of a central approximation of the second derivative                                                                                                                                                                                                                                         |
| &#9745;     | `median(x)`                                                 | Returns the median of x                                                                                                                                                                                                                                                                                            |
| &#9745;     | `minimum(x)`                                                | Calculates the lowest value of the time series x.                                                                                                                                                                                                                                                                  |
| &#9745;     | `number_crossing_m(x, m)`                                   | Calculates the number of crossings of x on m.                                                                                                                                                                                                                                                                      |
| &#9744;     | `number_cwt_peaks(x, n)`                                    | Number of different peaks in x.                                                                                                                                                                                                                                                                                    |
| &#9745;     | `number_peaks(x, n)`                                        | Calculates the number of peaks of at least support n in the time series x.                                                                                                                                                                                                                                         |
| &#9744;     | `partial_autocorrelation(x, param)`                         | Calculates the value of the partial autocorrelation function at the given lag.                                                                                                                                                                                                                                     |
| &#9745;     | `percentage_of_reoccurring_datapoints_to_all_datapoints(x)` | Returns the percentage of non-unique data points.                                                                                                                                                                                                                                                                  |
| &#9745;     | `percentage_of_reoccurring_values_to_all_values(x)`         | Returns the percentage of values that are present in the time series more than once.                                                                                                                                                                                                                               |
| &#9744;     | `permutation_entropy(x, tau, dimension)`                    | Calculate the permutation entropy.                                                                                                                                                                                                                                                                                 |
| &#9745;     | `quantile(x, q)`                                            | Calculates the q quantile of x.                                                                                                                                                                                                                                                                                    |
| &#9744;     | `query_similarity_count(x, param)`                          | This feature calculator accepts an input query subsequence parameter, compares the query (under z-normalized Euclidean distance)to all subsequences within the time series, and returns a count of the number of times the query was found in the time series (within some predefined maximum distance threshold). |
| &#9745;     | `range_count(x, min, max)`                                  | Count observed values within the interval [min, max].                                                                                                                                                                                                                                                              |
| &#9745;     | `ratio_beyond_r_sigma(x, r)`                                | Ratio of values that are more than r \* std(x) (so r times sigma) away from the mean of x.                                                                                                                                                                                                                         |
| &#9745;     | `ratio_value_number_to_time_series_length(x)`               | Returns a factor which is 1 if all values in the time series occur only once, and below one if this is not the case.                                                                                                                                                                                               |
| &#9745;     | `root_mean_square(x)`                                       | Returns the root mean square (rms) of the time series.                                                                                                                                                                                                                                                             |
| &#9745;     | `sample_entropy(x)`                                         | Calculate and return sample entropy of x.                                                                                                                                                                                                                                                                          |
| &#9745;     | `skewness(x)`                                               | Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G1).                                                                                                                                                                                                 |
| &#9744;     | `spkt_welch_density(x, param)`                              | This feature calculator estimates the cross power spectral density of the time series x at different frequencies.                                                                                                                                                                                                  |
| &#9745;     | `standard_deviation(x)`                                     | Returns the standard deviation of x                                                                                                                                                                                                                                                                                |
| &#9745;     | `sum_of_reoccurring_data_points(x)`                         | Returns the sum of all data points, that are present in the time series more than once.                                                                                                                                                                                                                            |
| &#9745;     | `sum_of_reoccurring_values(x)`                              | Returns the sum of all values, that are present in the time series more than once.                                                                                                                                                                                                                                 |
| &#9745;     | `sum_values(x)`                                             | Calculates the sum over the time series values                                                                                                                                                                                                                                                                     |
| &#9745;     | `symmetry_looking(x, param)`                                | Boolean variable denoting if the distribution of x looks symmetric.                                                                                                                                                                                                                                                |
| &#9745;     | `time_reversal_asymmetry_statistic(x, lag)`                 | Returns the time reversal asymmetry statistic.                                                                                                                                                                                                                                                                     |
| &#9744;     | `value_count(x, value)`                                     | Count occurrences of value in time series x.                                                                                                                                                                                                                                                                       |
| &#9745;     | `variance(x)`                                               | Returns the variance of x                                                                                                                                                                                                                                                                                          |
| &#9745;     | `variance_larger_than_standard_deviation(x)`                | Is variance higher than the standard deviation?                                                                                                                                                                                                                                                                    |
| &#9745;     | `variation_coefficient(x)`                                  | Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.                                                                                                                                                                                                      |

## Acknowledgement

The `tsfx` package was developed within the [Vinnova](https://www.vinnova.se) projects
[DFusion](https://www.vinnova.se/en/p/dfusion---disturbance-data-fusion/),
[TolkAI](https://www.vinnova.se/en/p/intepretable-ai-from-start-to-finish/), and
[SIFT](https://www.vinnova.se/en/p/similarity-search-of-time-series-data-evaluation-of-search-engine-in-industrial-process-datasift-/).
