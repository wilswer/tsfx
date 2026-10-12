//! Lag-based relationships within the series.

use super::kernel::{GroupContext, Kernel};
use crate::utils::stats::numpy_sum;

/// Autocorrelation feature.
///
/// The autocorrelation of the series with itself shifted by `lag`, using the
/// mean $\mu$ and population variance $\sigma^2$ of the whole series:
/// $$ R(l) = \frac{1}{(n - l)\,\sigma^2} \sum_{t=1}^{n-l} (x_t - \mu)(x_{t+l} - \mu). $$
/// See [`crate::utils::stats::population_var`].
///
/// # Parameters
/// - `lag`: shift $l$. Config:
///   `[autocorrelation] parameters = [{ lag = 0 }, ...]`; one column per entry.
///
/// # Output column
/// `{name}__autocorrelation__lag_{lag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - `lag` $\geq n$ gives NaN.
/// - A series with (near-)zero variance, $|\sigma^2| \leq 10^{-8}$ (tsfresh's
///   `np.isclose(v, 0)`), gives NaN for every lag.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.autocorrelation` (v0.21.2).
pub struct Autocorrelation {
    pub lags: Vec<usize>,
}

impl Kernel for Autocorrelation {
    fn names(&self) -> Vec<String> {
        self.lags
            .iter()
            .map(|lag| format!("autocorrelation__lag_{lag}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let (x, mean, v) = (ctx.x(), ctx.mean(), ctx.var());
        // tsfresh: autocorrelation is undefined without variance (np.isclose(v, 0))
        if v.abs() <= 1e-8 {
            out.extend(self.lags.iter().map(|_| f64::NAN));
            return;
        }
        for &lag in &self.lags {
            out.push(if x.len() < lag {
                f64::NAN
            } else {
                // np.sum((y1 - x_mean) * (y2 - x_mean)), y1 = x[:n-lag], y2 = x[lag:]
                let m = x.len() - lag;
                let sum_product = numpy_sum(m, |i| (x[i] - mean) * (x[i + lag] - mean));
                sum_product / (m as f64 * v)
            });
        }
    }
}

/// `np.mean` of `term(i)` for `i` in `0..n - 2 * lag`, or 0 if the series is
/// too short (`n <= 2 * lag`), as tsfresh.
fn lagged_mean(n: usize, lag: usize, term: impl Fn(usize) -> f64) -> f64 {
    if n <= 2 * lag {
        return 0.0;
    }
    numpy_sum(n - 2 * lag, term) / (n - 2 * lag) as f64
}

/// C3 non-linearity feature.
///
/// The mean of the product of the series at three points spaced `lag`
/// apart, a measure of non-linearity (Schreiber & Schmitz, 1997):
/// $$ \frac{1}{n - 2l} \sum_{i=1}^{n-2l} x_{i+2l}\, x_{i+l}\, x_i. $$
///
/// # Parameters
/// - `lag`: spacing $l$. Config: `[c3] parameters = [{ lag = 1 }, ...]`; one
///   column per entry.
///
/// # Output column
/// `{name}__c3__lag_{lag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - $2l \geq n$ (no complete triple) gives 0, as in tsfresh.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.c3` (v0.21.2).
pub struct C3 {
    pub lags: Vec<usize>,
}

impl Kernel for C3 {
    fn names(&self) -> Vec<String> {
        self.lags
            .iter()
            .map(|lag| format!("c3__lag_{lag:.0}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        for &lag in &self.lags {
            out.push(lagged_mean(x.len(), lag, |i| {
                x[i + 2 * lag] * x[i + lag] * x[i]
            }));
        }
    }
}

/// Time reversal asymmetry statistic feature.
///
/// A measure of how differently the series behaves forwards and backwards in
/// time (Fulcher & Jones, 2014):
/// $$ \frac{1}{n - 2l} \sum_{i=1}^{n-2l} \left( x_{i+2l}^2\, x_{i+l} - x_{i+l}\, x_i^2 \right). $$
/// A time-reversible series scores close to 0.
///
/// # Parameters
/// - `lag`: spacing $l$. Config:
///   `[time_reversal_asymmetry_statistic] parameters = [{ lag = 1 }, ...]`; one
///   column per entry.
///
/// # Output column
/// `{name}__time_reversal_asymmetry_statistic__lag_{lag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - $2l \geq n$ (no complete triple) gives 0, as in tsfresh.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.time_reversal_asymmetry_statistic` (v0.21.2).
pub struct TimeReversalAsymmetryStatistic {
    pub lags: Vec<usize>,
}

impl Kernel for TimeReversalAsymmetryStatistic {
    fn names(&self) -> Vec<String> {
        self.lags
            .iter()
            .map(|lag| format!("time_reversal_asymmetry_statistic__lag_{lag:.0}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        for &lag in &self.lags {
            out.push(lagged_mean(x.len(), lag, |i| {
                let (two, one) = (x[i + 2 * lag], x[i + lag]);
                two * two * one - one * x[i] * x[i]
            }));
        }
    }
}
