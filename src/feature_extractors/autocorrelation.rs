//! Lag-based relationships within the series.

use ndarray::ArrayView1;

use super::kernel::{GroupContext, Kernel};
use crate::utils::stats::{numpy_mean, numpy_median, numpy_sum, population_var};
use crate::utils::toml_reader::AutocorrelationAggregator;

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

/// The adjusted autocovariances $\gamma(0), \dots, \gamma(l_{max})$ of the
/// demeaned series, $\gamma(l) = \frac{1}{n - l} \sum_{t=1}^{n-l}
/// (x_t - \mu)(x_{t+l} - \mu)$, as statsmodels' `acovf(x, adjusted=True)`
/// (direct sums; statsmodels switches to an FFT for long series, which
/// differs only by rounding). `max_lag` must be below `x.len()`.
fn adjusted_acovf(x: &[f64], mean: f64, max_lag: usize) -> Vec<f64> {
    let n = x.len();
    (0..=max_lag)
        .map(|lag| numpy_sum(n - lag, |i| (x[i + lag] - mean) * (x[i] - mean)) / (n - lag) as f64)
        .collect()
}

/// Aggregated autocorrelation feature.
///
/// An aggregate (mean, median, variance or standard deviation) of the
/// autocorrelations $R(1), \dots, R(m)$ of the series, with
/// $m = \min(\text{maxlag}, n - 1)$ and
/// $$ R(l) = \frac{1}{(n - l)\,\sigma^2} \sum_{t=1}^{n-l} (x_t - \mu)(x_{t+l} - \mu), $$
/// where $\sigma^2$ here is the lag-0 autocovariance (the population
/// variance), as statsmodels' `acf(x, adjusted=True)`.
///
/// # Parameters
/// - `f_agg`: `mean`, `median`, `var` or `std` (numpy semantics, `ddof = 0`).
///   Any other name is rejected when the config is loaded.
/// - `maxlag`: the largest lag $m$.
///
/// Config: `[agg_autocorrelation] parameters = [{ f_agg = "mean", maxlag = 40 }, ...]`;
/// one column per entry.
///
/// # Output column
/// `{name}__agg_autocorrelation__f_agg_{f_agg}__maxlag_{maxlag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A single value, or a population variance below $10^{-10}$, makes every
///   autocorrelation 0 (as in tsfresh), so the aggregate is 0.
/// - `maxlag = 0` gives NaN (the aggregate of no values).
/// - NaN or $\pm\infty$ values give NaN.
///
/// # tsfresh
/// `feature_calculators.agg_autocorrelation` (v0.21.2).
pub struct AggAutocorrelation {
    /// `(f_agg, maxlag)` per config entry.
    pub params: Vec<(AutocorrelationAggregator, usize)>,
}

impl Kernel for AggAutocorrelation {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(f_agg, maxlag)| format!("agg_autocorrelation__f_agg_{f_agg}__maxlag_{maxlag}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let n = x.len();
        let max_maxlag = self.params.iter().map(|p| p.1).max().unwrap_or(0);
        // tsfresh: np.abs(var) < 10**-10 or n == 1 -> all autocorrelations 0
        let acf: Vec<f64> = if ctx.var().abs() < 1e-10 || n == 1 {
            vec![0.0; n]
        } else {
            // acf(...)[1:]: statsmodels computes all lags and keeps nlags + 1
            let acov = adjusted_acovf(x, ctx.mean(), max_maxlag.min(n - 1));
            acov[1..].iter().map(|c| c / acov[0]).collect()
        };
        for &(f_agg, maxlag) in &self.params {
            let a = &acf[..maxlag.min(acf.len())];
            out.push(if a.is_empty() {
                f64::NAN
            } else {
                match f_agg {
                    AutocorrelationAggregator::Mean => numpy_mean(a),
                    AutocorrelationAggregator::Median => {
                        if a.iter().any(|v| v.is_nan()) {
                            f64::NAN
                        } else {
                            numpy_median(&mut a.to_vec())
                        }
                    }
                    AutocorrelationAggregator::Var => population_var(&ArrayView1::from(a)),
                    AutocorrelationAggregator::Std => population_var(&ArrayView1::from(a)).sqrt(),
                }
            });
        }
    }
}

/// Partial autocorrelation feature.
///
/// The partial autocorrelation $\alpha_l$ of the series at lag $l$: the
/// correlation of $x_t$ and $x_{t-l}$ after removing the linear influence of
/// $x_{t-1}, \dots, x_{t-l+1}$. Computed with the Levinson-Durbin recursion
/// on the adjusted autocovariances, as statsmodels' `pacf(x, method="ld")`;
/// $\alpha_0 = 1$.
///
/// # Parameters
/// - `lag`: the lag $l$. Config:
///   `[partial_autocorrelation] parameters = [{ lag = 1 }, ...]`; one column
///   per entry.
///
/// # Output column
/// `{name}__partial_autocorrelation__lag_{lag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - Lags are only computed up to $\lfloor n/2 \rfloor - 1$ (statsmodels'
///   limit; tsfresh caps the requested lags there); larger lags give NaN.
/// - Lag 0 is 1 only if some lag $\geq 1$ can be computed, i.e. at least 4
///   values and a configured lag $\geq 1$; otherwise every lag, including 0,
///   gives NaN (as tsfresh, whose result for lag 0 depends on the other
///   lags requested).
/// - A constant series gives 1 for lag 0 and NaN for the others.
/// - NaN or $\pm\infty$ values give NaN for every lag but 0.
///
/// # tsfresh
/// `feature_calculators.partial_autocorrelation` (v0.21.2).
pub struct PartialAutocorrelation {
    pub lags: Vec<usize>,
}

impl Kernel for PartialAutocorrelation {
    fn names(&self) -> Vec<String> {
        self.lags
            .iter()
            .map(|lag| format!("partial_autocorrelation__lag_{lag}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let n = x.len();
        let max_demanded = self.lags.iter().copied().max().unwrap_or(0);
        // tsfresh caps the lags at statsmodels' limit of 50% of the sample
        let max_lag = if n <= 1 {
            0
        } else if max_demanded >= n / 2 {
            n / 2 - 1
        } else {
            max_demanded
        };
        let pacf = if max_lag > 0 {
            levinson_durbin_pacf(&adjusted_acovf(x, ctx.mean(), max_lag))
        } else {
            Vec::new()
        };
        out.extend(
            self.lags
                .iter()
                .map(|&lag| pacf.get(lag).copied().unwrap_or(f64::NAN)),
        );
    }
}

/// Partial autocorrelations $\alpha_0, \dots, \alpha_m$ from the
/// autocovariances $\gamma(0), \dots, \gamma(m)$ ($m \geq 1$) by the
/// Levinson-Durbin recursion, exactly as statsmodels' `levinson_durbin`.
fn levinson_durbin_pacf(acov: &[f64]) -> Vec<f64> {
    let order = acov.len() - 1;
    // phi[j][k]: coefficient j of the AR(k) fit
    let mut phi = vec![vec![0.0; order + 1]; order + 1];
    let mut sig = vec![0.0; order + 1];
    phi[1][1] = acov[1] / acov[0];
    sig[1] = acov[0] - phi[1][1] * acov[1];
    for k in 2..=order {
        let dot: f64 = (1..k).map(|j| phi[j][k - 1] * acov[k - j]).sum();
        phi[k][k] = (acov[k] - dot) / sig[k - 1];
        for j in 1..k {
            phi[j][k] = phi[j][k - 1] - phi[k][k] * phi[k - j][k - 1];
        }
        sig[k] = sig[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
    }
    let mut pacf: Vec<f64> = (0..=order).map(|k| phi[k][k]).collect();
    pacf[0] = 1.0;
    pacf
}
