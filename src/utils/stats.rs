//! Shared statistical conventions for the feature extractors.

use ndarray::ArrayView1;

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
