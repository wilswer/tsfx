//! Shared statistical conventions for the feature extractors.

use std::ops::{Add, Div, Mul, Rem, Sub};

use itertools::Itertools;
use ndarray::{Array1, ArrayView1};
use ndarray_stats::errors::QuantileError;
use num::FromPrimitive;

/// Sum of `value(0), ..., value(n - 1)` exactly as numpy's `add.reduce` sums
/// a contiguous `float64` array (numpy 2.x): pairwise summation with eight
/// accumulators for blocks of up to 128 values, split in halves above that,
/// added to an initial `0.0`. Results equal numpy's bit for bit, which a
/// straight or Welford sum does not.
pub fn numpy_sum(n: usize, value: impl Fn(usize) -> f64) -> f64 {
    0.0 + pairwise_sum(0, n, &value)
}

fn pairwise_sum(start: usize, n: usize, value: &impl Fn(usize) -> f64) -> f64 {
    if n < 8 {
        (start..start + n).fold(0.0, |res, i| res + value(i))
    } else if n <= 128 {
        let mut r: [f64; 8] = std::array::from_fn(|j| value(start + j));
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += value(start + i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for k in i..n {
            res += value(start + k);
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(start, n2, value) + pairwise_sum(start + n2, n - n2, value)
    }
}

/// Arithmetic mean, exactly as `np.mean` (see [`numpy_sum`]). NaN for an
/// empty series or if any value is NaN.
pub fn numpy_mean(x: &[f64]) -> f64 {
    numpy_sum(x.len(), |i| x[i]) / x.len() as f64
}

/// Population variance (`ddof = 0`) of a series.
///
/// Exactly `np.var` as used by tsfresh: the mean, then the mean of the
/// squared deviations, both summed as numpy does (see [`numpy_sum`]).
/// Returns the raw value; how a zero variance is handled is up to each
/// feature.
pub fn population_var(arr: &ArrayView1<f64>) -> f64 {
    let n = arr.len();
    let mean = numpy_sum(n, |i| arr[i]) / n as f64;
    numpy_sum(n, |i| {
        let d = arr[i] - mean;
        d * d
    }) / n as f64
}

/// Population standard deviation (`ddof = 0`) of a series: the square root
/// of [`population_var`], exactly `np.std` as used by tsfresh. Returns the
/// raw value; how a zero standard deviation is handled is up to each
/// feature.
pub fn population_std(arr: &ArrayView1<f64>) -> f64 {
    population_var(arr).sqrt()
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

/// The `q`-quantile of `sorted` (ascending, not empty), exactly as
/// `np.quantile` computes it (method "linear"): index $h = (n - 1)\,q$,
/// neighbours $a = x_{(\lfloor h \rfloor)}$, $b = x_{(\lfloor h \rfloor + 1)}$
/// and weight $t = h - \lfloor h \rfloor$, then numpy's `_lerp`:
/// $a + (b - a)\,t$ for $t < 0.5$, $b - (b - a)(1 - t)$ otherwise. At or
/// beyond the last index numpy uses the last value for both neighbours with
/// $t = h + 1$ (it computes the weight from the replacement index $-1$).
/// So an infinite neighbour can give NaN, exactly as in numpy.
pub fn numpy_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let h = (n as f64 - 1.0) * q;
    let (i, j, t) = if h >= (n - 1) as f64 {
        (n - 1, n - 1, h + 1.0)
    } else {
        let previous = h.floor();
        (previous as usize, previous as usize + 1, h - previous)
    };
    let (a, b) = (sorted[i], sorted[j]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// The median of `values` (not empty, reordered in place), exactly as
/// `np.median` computes it: the `np.mean` of the middle value, or of the two
/// middle values for an even count. Found with a selection, $O(n)$.
pub fn numpy_median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (left, &mut upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return numpy_mean(&[upper]);
    }
    let lower = left.iter().copied().max_by(f64::total_cmp).unwrap_or(upper);
    numpy_mean(&[lower, upper])
}

/// Index of the first maximum, like `np.argmax`: if the series contains NaN,
/// the index of the first NaN. `None` for an empty series.
pub(crate) fn np_argmax(x: &ArrayView1<f64>) -> Option<usize> {
    np_arg_extreme(x, |a, b| a > b)
}

/// Index of the first minimum, like `np.argmin`: if the series contains NaN,
/// the index of the first NaN. `None` for an empty series.
pub(crate) fn np_argmin(x: &ArrayView1<f64>) -> Option<usize> {
    np_arg_extreme(x, |a, b| a < b)
}

fn np_arg_extreme(x: &ArrayView1<f64>, better: fn(f64, f64) -> bool) -> Option<usize> {
    if let Some(i) = x.iter().position(|v| v.is_nan()) {
        return Some(i);
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in x.iter().enumerate() {
        match best {
            Some((_, b)) if !better(v, b) => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
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
