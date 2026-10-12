//! Summaries of the value distribution, independent of order.

use itertools::Itertools;
use ndarray::Array1;
use ndarray_stats::QuantileExt;
use noisy_float::types::n64;
use ordered_float::OrderedFloat;
use polars::lazy::dsl::*;
use polars::prelude::*;

use super::kernel::{GroupContext, Kernel, scalar_kernel};
use crate::utils::stats::{median_mut, sample_excess_kurtosis, sample_skewness};
/// Length feature.
///
/// The number of non-null values in the time series. Only the first value
/// column is counted, since all value columns share the same rows.
///
/// # Output column
/// `length` (no column-name prefix)
///
/// # Edge cases
/// - Nulls are not counted; NaN values are. A group with no non-null values
///   gives 0.
///
/// # tsfresh
/// `feature_calculators.length` (v0.21.2).
pub fn count(name: &str) -> Expr {
    col(name).count().alias("length")
}

scalar_kernel! {
/// Sum of values feature.
///
/// The sum of all values in the time series:
/// $$ \sum_{i=1}^{n} x_i. $$
///
/// # Output column
/// `{name}__sum_values`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.sum_values` (v0.21.2).
    SumValues, "sum_values", |ctx| ctx.x().sum()
}

scalar_kernel! {
/// Mean feature.
///
/// The arithmetic mean of all values in the time series:
/// $$ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i, $$
/// where $n$ is the number of values in the time series.
///
/// # Output column
/// `{name}__mean`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.mean` (v0.21.2).
    Mean, "mean", |ctx| ctx.mean()
}

scalar_kernel! {
/// Minimum feature.
///
/// The smallest value in the time series, $\min_i x_i$.
///
/// # Output column
/// `{name}__minimum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.minimum` (v0.21.2).
    Minimum, "minimum", |ctx| ctx.min()
}

scalar_kernel! {
/// Maximum feature.
///
/// The largest value in the time series, $\max_i x_i$.
///
/// # Output column
/// `{name}__maximum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.maximum` (v0.21.2).
    Maximum, "maximum", |ctx| ctx.max()
}

scalar_kernel! {
/// Standard deviation feature.
///
/// The population standard deviation (`ddof = 0`) of all values in the time
/// series:
/// $$ \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}, $$
/// where $n$ is the number of values in the time series and $\mu$ is its mean.
/// See [`crate::utils::stats::population_std`].
///
/// # Output column
/// `{name}__standard_deviation`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
/// - A single value gives 0.
///
/// # tsfresh
/// `feature_calculators.standard_deviation` (v0.21.2), i.e. `np.std`.
    StandardDeviation, "standard_deviation", |ctx| ctx.std()
}

scalar_kernel! {
/// Variance feature.
///
/// The population variance (`ddof = 0`) of all values in the time series:
/// $$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2, $$
/// where $n$ is the number of values in the time series and $\mu$ is its mean.
/// See [`crate::utils::stats::population_var`].
///
/// # Output column
/// `{name}__variance`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
/// - A single value gives 0.
///
/// # tsfresh
/// `feature_calculators.variance` (v0.21.2), i.e. `np.var`.
    Variance, "variance", |ctx| ctx.var()
}

scalar_kernel! {
/// Root mean square feature.
///
/// The square root of the mean of the squared values in the time series:
/// $$ \text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}, $$
/// where $n$ is the number of values in the time series.
///
/// # Output column
/// `{name}__root_mean_square`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.root_mean_square` (v0.21.2).
    RootMeanSquare, "root_mean_square", |ctx| ctx.x().mapv(|x| x.powi(2)).mean().map(f64::sqrt).unwrap_or(f64::NAN)
}

scalar_kernel! {
/// Skewness feature.
///
/// The bias-adjusted sample skewness (Fisher-Pearson $G_1$) of all values in
/// the time series:
/// $$ G_1 = \frac{n \sqrt{n - 1}}{n - 2} \frac{m_3}{m_2^{3/2}}, \quad m_k = \sum_{i=1}^{n} (x_i - \mu)^k, $$
/// where $n$ is the number of values in the time series and $\mu$ is its mean.
/// See [`crate::utils::stats::sample_skewness`].
///
/// # Output column
/// `{name}__skewness`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
/// - Fewer than 3 values give NaN.
/// - A constant series gives 0. Near-constant series are treated as constant
///   using pandas 3's floating-point tolerance.
///
/// # tsfresh
/// `feature_calculators.skewness` (v0.21.2), i.e. `pandas.Series.skew`.
    Skewness, "skewness", |ctx| sample_skewness(ctx.x())
}

scalar_kernel! {
/// Absolute energy feature.
///
/// The sum of the squared values of the time series:
/// $$ E = \sum_{i=1}^{n} x_i^2, $$
/// where $n$ is the number of values in the time series.
///
/// # Output column
/// `{name}__absolute_energy`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.abs_energy` (v0.21.2).
    AbsoluteEnergy, "absolute_energy", |ctx| ctx.x().mapv(|x| x.powi(2)).sum()
}

scalar_kernel! {
/// Kurtosis feature.
///
/// The bias-adjusted sample excess kurtosis $G_2$ of all values in the time
/// series:
/// $$ G_2 = \frac{n (n + 1) (n - 1) m_4}{(n - 2) (n - 3) m_2^2} - \frac{3 (n - 1)^2}{(n - 2) (n - 3)}, \quad m_k = \sum_{i=1}^{n} (x_i - \mu)^k, $$
/// where $n$ is the number of non-NaN values and $\mu$ is their mean. A
/// normal distribution scores 0. See
/// [`crate::utils::stats::sample_excess_kurtosis`].
///
/// # Output column
/// `{name}__kurtosis`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are skipped, like pandas' `skipna=True`.
/// - Fewer than 4 (non-NaN) values give NaN.
/// - A constant series gives 0. Near-constant series are treated as constant
///   using pandas 3's floating-point tolerance.
///
/// # tsfresh
/// `feature_calculators.kurtosis` (v0.21.2), i.e. `pandas.Series.kurtosis`.
    Kurtosis, "kurtosis", |ctx| {
        // tsfresh calls pandas' kurtosis with skipna=True
        let x = Array1::from_iter(ctx.x().iter().copied().filter(|x| !x.is_nan()));
        sample_excess_kurtosis(&x.view())
    }
}

scalar_kernel! {
/// Variance larger than standard deviation feature.
///
/// Whether the population variance exceeds the population standard
/// deviation, $\sigma^2 > \sigma$, which holds exactly when $\sigma^2 > 1$.
/// Returns 1.0 for true and 0.0 for false.
///
/// # Output column
/// `{name}__variance_larger_than_standard_deviation`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 (comparisons with NaN are false).
///
/// # tsfresh
/// `feature_calculators.variance_larger_than_standard_deviation` (v0.21.2).
    VarianceLargerThanStandardDeviation, "variance_larger_than_standard_deviation", |ctx| if ctx.var() > ctx.std() { 1.0 } else { 0.0 }
}

scalar_kernel! {
/// Absolute maximum feature.
///
/// The largest absolute value in the time series, $\max_i |x_i|$.
///
/// # Output column
/// `{name}__absolute_maximum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.absolute_maximum` (v0.21.2).
    AbsoluteMaximum, "absolute_maximum", |ctx| *ctx.x().mapv(f64::abs).max().unwrap_or(&f64::NAN)
}

scalar_kernel! {
/// Variation coefficient feature.
///
/// The population standard deviation relative to the mean, $\sigma / \mu$.
///
/// # Output column
/// `{name}__variation_coefficient`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are kept and make the result NaN.
/// - A mean of exactly 0 gives NaN.
///
/// # tsfresh
/// `feature_calculators.variation_coefficient` (v0.21.2).
    VariationCoefficient, "variation_coefficient", |ctx| if ctx.mean() == 0.0 { f64::NAN } else { ctx.std() / ctx.mean() }
}

/// Median feature.
///
/// The median of all values in the time series: the middle value after
/// sorting, or the mean of the two middle values for an even count. Computed
/// with the native Polars API.
///
/// # Output column
/// `{name}__median`
///
/// # Edge cases
/// - Nulls are ignored; a group with no non-null values gives null (not NaN).
/// - A series containing NaN gives NaN.
///
/// # tsfresh
/// `feature_calculators.median` (v0.21.2).
pub fn expr_median(name: &str) -> Expr {
    // NaN propagation, as np.median in tsfresh
    when(col(name).is_nan().any(true))
        .then(lit(f64::NAN))
        .otherwise(col(name).median().cast(DataType::Float64))
        .alias(format!("{}__median", name))
}

/// Ratio beyond r sigma feature.
///
/// The fraction of values more than $r$ population standard deviations away
/// from the mean:
/// $$ \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\left[\,|x_i - \mu| > r \sigma\,\right], $$
/// where $n$ is the number of values, $\mu$ the mean and $\sigma$ the
/// population standard deviation.
///
/// # Parameters
/// - `r`: number of standard deviations. Config:
///   `[ratio_beyond_r_sigma] parameters = [{ r = 1.0 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__ratio_beyond_r_sigma__r_{r:.2}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 (comparisons with NaN are false).
///
/// # tsfresh
/// `feature_calculators.ratio_beyond_r_sigma` (v0.21.2).
pub struct RatioBeyondRSigma {
    pub rs: Vec<f64>,
}

impl Kernel for RatioBeyondRSigma {
    fn names(&self) -> Vec<String> {
        self.rs
            .iter()
            .map(|r| format!("ratio_beyond_r_sigma__r_{r:.2}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let (mean, std) = (ctx.mean(), ctx.std());
        for r in &self.rs {
            let count = ctx
                .x()
                .mapv(|x| if (x - mean).abs() > r * std { 1.0 } else { 0.0 })
                .sum();
            out.push(count / ctx.len() as f64);
        }
    }
}

/// Large standard deviation feature.
///
/// Whether the population standard deviation is larger than $r$ times the
/// range of the series, $\sigma > r\,(\max_i x_i - \min_i x_i)$. Returns
/// 1.0 for true and 0.0 for false.
///
/// # Parameters
/// - `r`: fraction of the range. Config:
///   `[large_standard_deviation] parameters = [{ r = 0.05 }, ...]`; one column
///   per entry.
///
/// # Output column
/// `{name}__large_standard_deviation__r_{r:.2}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 (comparisons with NaN are false).
///
/// # tsfresh
/// `feature_calculators.large_standard_deviation` (v0.21.2).
pub struct LargeStandardDeviation {
    pub rs: Vec<f64>,
}

impl Kernel for LargeStandardDeviation {
    fn names(&self) -> Vec<String> {
        self.rs
            .iter()
            .map(|r| format!("large_standard_deviation__r_{r:.2}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // With a NaN, std is NaN and every comparison is false
        let (std, range) = (ctx.std(), ctx.max() - ctx.min());
        out.extend(self.rs.iter().map(|r| (std > r * range) as u8 as f64));
    }
}

/// Symmetry looking feature.
///
/// Whether the distribution looks symmetric: the distance between mean and
/// median is less than $r$ times the range,
/// $|\mu - \tilde{x}| < r\,(\max_i x_i - \min_i x_i)$, where $\mu$ is the
/// mean and $\tilde{x}$ the median. Returns 1.0 for true and 0.0 for false.
///
/// # Parameters
/// - `r`: fraction of the range. Config:
///   `[symmetry_looking] parameters = [{ r = 0.05 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__symmetry_looking__r_{r:.2}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 for every `r` (comparisons with NaN are
///   false).
/// - A constant series gives 0 (the range is 0, and `<` is strict).
///
/// # tsfresh
/// `feature_calculators.symmetry_looking` (v0.21.2).
pub struct SymmetryLooking {
    pub rs: Vec<f64>,
}

impl Kernel for SymmetryLooking {
    fn names(&self) -> Vec<String> {
        self.rs
            .iter()
            .map(|r| format!("symmetry_looking__r_{r:.2}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // tsfresh: every comparison with a NaN mean/median is False
        if ctx.x().iter().any(|x| x.is_nan()) {
            out.extend(self.rs.iter().map(|_| 0.0));
            return;
        }
        let mut arr = ctx.x().mapv(n64);
        // Not empty, so median, mean, max and min all exist
        let median = median_mut(&mut arr).map_or(f64::NAN, f64::from);
        let mean = arr.mean().map_or(f64::NAN, f64::from);
        let max = arr.max().map_or(f64::NAN, |m| f64::from(*m));
        let min = arr.min().map_or(f64::NAN, |m| f64::from(*m));
        let (mean_median_diff, max_min_diff) = ((mean - median).abs(), max - min);
        out.extend(
            self.rs
                .iter()
                .map(|r| (mean_median_diff < r * max_min_diff) as u8 as f64),
        );
    }
}

/// Mean of the n absolute maxima feature.
///
/// The arithmetic mean of the $n$ largest absolute values in the time series.
///
/// # Parameters
/// - `n`: number of maxima. Config:
///   `[mean_n_absolute_max] parameters = [{ n = 7 }]`; one column per entry.
///
/// # Output column
/// `{name}__mean_n_absolute_max__n_{n}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series with fewer than $n$ values gives NaN.
///
/// # tsfresh
/// `feature_calculators.mean_n_absolute_max` (v0.21.2).
pub struct MeanNAbsoluteMax {
    pub ns: Vec<usize>,
}

impl Kernel for MeanNAbsoluteMax {
    fn names(&self) -> Vec<String> {
        self.ns
            .iter()
            .map(|n| format!("mean_n_absolute_max__n_{n}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let largest = ctx
            .x()
            .iter()
            .map(|x| -OrderedFloat::from(x.abs()))
            .k_smallest(self.ns.iter().copied().max().unwrap_or(0))
            .map(|x| -f64::from(x))
            .collect::<Vec<f64>>();
        for &n in &self.ns {
            out.push(if ctx.len() < n {
                f64::NAN
            } else {
                largest.iter().take(n).sum::<f64>() / n as f64
            });
        }
    }
}

/// Quantile feature.
///
/// The $q$-quantile of the values, with linear interpolation between the two
/// nearest order statistics (`np.quantile`'s default): for sorted values
/// $x_{(0)} \le \dots \le x_{(n-1)}$ and $h = (n - 1)\,q$,
/// $$ Q(q) = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)\left(x_{(\lceil h \rceil)} - x_{(\lfloor h \rfloor)}\right). $$
/// Computed with the native Polars API.
///
/// # Parameters
/// - `q`: quantile in $[0, 1]$. Config: `[quantile] parameters = [{ q = 0.1 }, ...]`;
///   one column per entry.
///
/// # Output column
/// `{name}__quantile__q_{q:.1}`
///
/// # Edge cases
/// - Nulls are ignored; a group with no non-null values gives null (not NaN).
/// - A series containing NaN gives NaN.
/// - The column name rounds `q` to one decimal, so e.g. 0.25 and 0.2 would
///   share a name.
///
/// # tsfresh
/// `feature_calculators.quantile` (v0.21.2), i.e. `np.quantile`.
pub fn expr_quantile(name: &str, q: f64) -> Expr {
    // Linear interpolation and NaN propagation, as np.quantile in tsfresh
    when(col(name).is_nan().any(true))
        .then(lit(f64::NAN))
        .otherwise(quantile(name, lit(q), QuantileMethod::Linear).cast(DataType::Float64))
        .alias(format!("{}__quantile__q_{:.1}", name, q))
}
