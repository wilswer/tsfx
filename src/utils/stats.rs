//! Shared statistical conventions for the feature extractors.

use std::ops::{Add, Div, Mul, Rem, Sub};

use itertools::Itertools;
use ndarray::{Array1, ArrayView1, Axis};
use ndarray_stats::errors::QuantileError;
use num::FromPrimitive;

/// Population variance (`ddof = 0`) of a series.
///
/// Matches `np.var` as used by tsfresh. Returns the raw value; how a zero
/// variance is handled is up to each feature.
pub fn population_var(arr: &ArrayView1<f64>) -> f64 {
    arr.var(0.0)
}

/// Population standard deviation (`ddof = 0`) of a series.
///
/// Matches `np.std` as used by tsfresh. Returns the raw value; how a zero
/// standard deviation is handled is up to each feature.
pub fn population_std(arr: &ArrayView1<f64>) -> f64 {
    arr.std(0.0)
}

/// Treat a sum of `k`-th power deviations as zero when it is within
/// floating-point noise, as pandas 3 does (`_zero_out_fperr` with tolerance
/// `(eps * max|x|)^k * n`).
fn zero_out_fperr(m: f64, k: i32, max_abs: f64, n: f64) -> f64 {
    let tol = (f64::EPSILON * max_abs).powi(k) * n;
    if m.abs() < tol { 0.0 } else { m }
}

/// Mean and largest absolute value of a series, as pandas computes them.
fn mean_and_max_abs(arr: &ArrayView1<f64>) -> (f64, f64) {
    let mean = arr.sum() / arr.len() as f64;
    let max_abs = arr.fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    (mean, max_abs)
}

/// Bias-adjusted sample skewness (Fisher-Pearson G1) of a series.
///
/// Matches `pandas.Series.skew` (pandas 3), which tsfresh uses:
/// $$ G_1 = \frac{n \sqrt{n - 1}}{n - 2} \frac{m_3}{m_2^{3/2}}, $$
/// with $m_k = \sum_{i=1}^{n} (x_i - \mu)^k$. Returns NaN for fewer than 3
/// values and 0 for a (numerically) constant series.
pub fn sample_skewness(arr: &ArrayView1<f64>) -> f64 {
    if arr.len() < 3 {
        return f64::NAN;
    }
    let n = arr.len() as f64;
    let (mean, max_abs) = mean_and_max_abs(arr);
    let (m2, m3) = arr.fold((0.0, 0.0), |(m2, m3), &x| {
        let d2 = (x - mean).powi(2);
        (m2 + d2, m3 + d2 * (x - mean))
    });
    let m2 = zero_out_fperr(m2, 2, max_abs, n);
    let m3 = zero_out_fperr(m3, 3, max_abs, n);
    if m2 == 0.0 {
        return 0.0;
    }
    (n * (n - 1.0).sqrt() / (n - 2.0)) * (m3 / m2.powf(1.5))
}

/// Bias-adjusted sample excess kurtosis of a series.
///
/// Matches `pandas.Series.kurtosis` (pandas 3), which tsfresh uses:
/// $$ G_2 = \frac{n (n + 1) (n - 1) m_4}{(n - 2) (n - 3) m_2^2}
///        - \frac{3 (n - 1)^2}{(n - 2) (n - 3)}, $$
/// with $m_k = \sum_{i=1}^{n} (x_i - \mu)^k$. A normal distribution scores 0.
/// Returns NaN for fewer than 4 values and 0 for a (numerically) constant
/// series.
pub fn sample_excess_kurtosis(arr: &ArrayView1<f64>) -> f64 {
    if arr.len() < 4 {
        return f64::NAN;
    }
    let n = arr.len() as f64;
    let (mean, max_abs) = mean_and_max_abs(arr);
    let (m2, m4) = arr.fold((0.0, 0.0), |(m2, m4), &x| {
        let d2 = (x - mean).powi(2);
        (m2 + d2, m4 + d2 * d2)
    });
    let m2 = zero_out_fperr(m2, 2, max_abs, n);
    let m4 = zero_out_fperr(m4, 4, max_abs, n);
    let denominator = (n - 2.0) * (n - 3.0) * m2 * m2;
    if denominator == 0.0 {
        return 0.0;
    }
    let adj = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
    n * (n + 1.0) * (n - 1.0) * m4 / denominator - adj
}

/// Calculates the Ordinary Least Squares (OLS) slope and intercept
/// for a sequence of y-values where x-values are assumed to be a sequential index (0, 1, 2, ..., N-1).
/// Returns a tuple of (intercept, slope).
pub(crate) fn calculate_sequential_ols(
    y_values: impl IntoIterator<Item = f64>,
    n: usize,
) -> (f64, f64) {
    // Cannot fit a line with fewer than 2 points
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let n_f64 = n as f64;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;

    // Calculate sums in a single pass
    for (i, val) in y_values.into_iter().enumerate() {
        sum_y += val;
        sum_xy += (i as f64) * val;
    }

    let mean_x = (n_f64 - 1.0) / 2.0;
    let mean_y = sum_y / n_f64;

    // Sum of squares of x (SS_xx) for a sequence 0..n-1 has a known closed-form formula
    let ss_xx = n_f64 * (n_f64 * n_f64 - 1.0) / 12.0;

    // Sum of products (SS_xy)
    let ss_xy = sum_xy - n_f64 * mean_x * mean_y;

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    (intercept, slope)
}

/// Number of distinct values, counting all NaNs as one value like `np.unique`.
pub(crate) fn count_unique(arr: &ArrayView1<f64>) -> usize {
    let sorted = arr
        .iter()
        .filter(|x| !x.is_nan())
        .sorted_by(|a, b| a.total_cmp(b))
        .collect::<Vec<_>>();
    let distinct = if sorted.is_empty() {
        0
    } else {
        1 + sorted.windows(2).filter(|win| win[0] != win[1]).count()
    };
    let has_nan = arr.iter().any(|x| x.is_nan());
    distinct + has_nan as usize
}

/// Reduce the non-NaN values with `f`, like pandas' `max`/`min` (skipna).
/// NaN if every value is NaN.
pub(crate) fn skip_nan_reduce(x: &Array1<f64>, f: fn(f64, f64) -> f64) -> f64 {
    x.iter()
        .copied()
        .filter(|v| !v.is_nan())
        .reduce(f)
        .unwrap_or(f64::NAN)
}

pub(crate) fn aggregate_on_chunks(
    x: Array1<f64>,
    chunk_size: usize,
    aggregator: impl Fn(Array1<f64>) -> f64,
) -> Array1<f64> {
    let mut agg_arr = Vec::with_capacity(x.len().div_ceil(chunk_size));
    for chunk in x.axis_chunks_iter(Axis(0), chunk_size) {
        agg_arr.push(aggregator(chunk.to_owned()));
    }
    Array1::from_vec(agg_arr)
}

pub(crate) fn get_length_sequences_where(x: &ndarray::Array1<bool>) -> Vec<usize> {
    let mut group_lengths = Vec::new();
    for (key, group) in &x.into_iter().chunk_by(|elt| *elt) {
        if *key {
            group_lengths.push(group.count());
        }
    }
    group_lengths
}

/// Return the median. Sorts its argument in place.
pub(crate) fn median_mut<T>(xs: &mut Array1<T>) -> Result<T, QuantileError>
where
    T: Clone + Copy + Ord + FromPrimitive,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>,
{
    if xs.is_empty() {
        return Err(QuantileError::EmptyInput);
    }
    xs.as_slice_mut().unwrap().sort_unstable();
    Ok(if xs.len().is_multiple_of(2) {
        (xs[xs.len() / 2] + xs[xs.len() / 2 - 1]) / (T::from_u64(2).unwrap())
    } else {
        xs[xs.len() / 2]
    })
}

pub(crate) fn roll(x: &mut [f64], shift: isize) -> &[f64] {
    if shift > 0 {
        x.rotate_right(shift as usize);
    } else {
        x.rotate_left(shift.unsigned_abs());
    }
    x
}

/// Mean of the non-NaN values, like pandas' `mean` (skipna). NaN if every
/// value is NaN.
pub(crate) fn skip_nan_mean(x: &Array1<f64>) -> f64 {
    let (sum, n) = x
        .iter()
        .filter(|v| !v.is_nan())
        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// Sample variance (`ddof = 1`) of the non-NaN values, like pandas' `var`
/// (skipna). NaN with fewer than 2 non-NaN values.
pub(crate) fn skip_nan_sample_var(x: &Array1<f64>) -> f64 {
    let values = Array1::from_iter(x.iter().copied().filter(|v| !v.is_nan()));
    if values.len() < 2 {
        f64::NAN
    } else {
        values.var(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_population_var_and_std() {
        let arr = array![1.0, 2.0, 3.0, 4.0];
        assert_eq!(population_var(&arr.view()), 1.25);
        assert_eq!(population_std(&arr.view()), 1.25_f64.sqrt());
    }

    #[test]
    fn test_single_value_is_zero() {
        let arr = array![5.0];
        assert_eq!(population_var(&arr.view()), 0.0);
        assert_eq!(population_std(&arr.view()), 0.0);
    }

    // Expected values from tsfresh v0.21.2 / pandas 3.0.2.
    #[test]
    fn test_sample_skewness() {
        let arr = array![1.0, 1.0, 1.0, 2.0, 2.0];
        assert!((sample_skewness(&arr.view()) - 0.6085806194501855).abs() < 1e-12);
        assert!(sample_skewness(&array![1.0, 2.0].view()).is_nan());
        assert_eq!(sample_skewness(&array![1.0, 1.0, 1.0].view()), 0.0);
        assert_eq!(sample_skewness(&array![0.1, 0.1, 0.1].view()), 0.0);
    }

    #[test]
    fn test_sample_excess_kurtosis() {
        let arr = array![1.0, 1.0, 1.0, 2.0, 2.0];
        assert!((sample_excess_kurtosis(&arr.view()) + 3.333333333333333).abs() < 1e-12);
        assert!(sample_excess_kurtosis(&array![1.0, 1.0, 2.0].view()).is_nan());
        assert_eq!(
            sample_excess_kurtosis(&array![1.0, 1.0, 1.0, 1.0].view()),
            0.0
        );
        assert_eq!(
            sample_excess_kurtosis(&array![0.1, 0.1, 0.1, 0.1].view()),
            0.0
        );
    }
}
