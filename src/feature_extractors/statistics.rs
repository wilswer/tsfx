//! Summaries of the value distribution, independent of order.

use ndarray::Array1;
use polars::lazy::dsl::*;

use super::kernel::{GroupContext, Kernel, extremes, scalar_kernel};
use crate::utils::stats::{
    numpy_mean, numpy_median, numpy_quantile, numpy_sum, sample_excess_kurtosis, sample_skewness,
};
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
    SumValues, "sum_values", |ctx| numpy_sum(ctx.len(), |i| ctx.values()[i])
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
    RootMeanSquare, "root_mean_square", |ctx| {
        // np.sqrt(np.mean(np.square(x)))
        let x = ctx.values();
        (numpy_sum(x.len(), |i| x[i] * x[i]) / x.len() as f64).sqrt()
    }
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
    AbsoluteMaximum, "absolute_maximum", |ctx| extremes(ctx.values().iter().map(|x| x.abs())).1
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

scalar_kernel! {
/// Median feature.
///
/// The median of all values in the time series: the middle value after
/// sorting, or the mean of the two middle values for an even count, exactly
/// as `np.median` (see [`crate::utils::stats::numpy_median`]).
///
/// # Output column
/// `{name}__median`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives NaN.
///
/// # tsfresh
/// `feature_calculators.median` (v0.21.2).
    Median, "median", |ctx| {
        // NaN propagation, as np.median in tsfresh
        if ctx.x().iter().any(|x| x.is_nan()) {
            f64::NAN
        } else {
            // A selection is O(n); only the quantiles need the full sort
            numpy_median(&mut ctx.values().to_vec())
        }
    }
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
        // np.abs(np.mean(x) - np.median(x)) and np.max(x) - np.min(x), exactly
        let median = numpy_median(&mut ctx.values().to_vec());
        let mean_median_diff = (ctx.mean() - median).abs();
        let max_min_diff = ctx.max() - ctx.min();
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
/// - A series with $n$ or fewer values gives NaN (tsfresh requires more than
///   $n$ values).
/// - A NaN counts as the largest absolute value (it sorts last, as in
///   `np.sort`), so with a NaN the result is NaN.
///
/// # tsfresh
/// `feature_calculators.mean_n_absolute_max` (v0.21.2): the `np.mean` of the
/// last $n$ values of `np.sort(np.abs(x))`, summed as numpy does.
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
        // np.sort(np.abs(x)): ascending, NaN last
        let mut abs = ctx.values().iter().map(|x| x.abs()).collect::<Vec<f64>>();
        abs.sort_unstable_by(f64::total_cmp);
        for &n in &self.ns {
            out.push(if ctx.len() > n {
                numpy_mean(&abs[abs.len() - n..])
            } else {
                f64::NAN
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
/// Computed exactly as `np.quantile` (see
/// [`crate::utils::stats::numpy_quantile`], including numpy's evaluation of
/// the interpolation for $h - \lfloor h \rfloor \ge 0.5$).
///
/// # Parameters
/// - `q`: quantile in $[0, 1]$. Config: `[quantile] parameters = [{ q = 0.1 }, ...]`;
///   one column per entry.
///
/// # Output column
/// `{name}__quantile__q_{q:.1}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives NaN.
/// - The column name rounds `q` to one decimal, so e.g. 0.25 and 0.2 would
///   share a name.
///
/// # tsfresh
/// `feature_calculators.quantile` (v0.21.2), i.e. `np.quantile`.
pub struct Quantile {
    pub qs: Vec<f64>,
}

impl Kernel for Quantile {
    fn names(&self) -> Vec<String> {
        self.qs
            .iter()
            .map(|q| format!("quantile__q_{q:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // Linear interpolation and NaN propagation, as np.quantile in tsfresh
        if ctx.x().iter().any(|x| x.is_nan()) {
            out.extend(self.qs.iter().map(|_| f64::NAN));
            return;
        }
        let sorted = ctx.sorted();
        out.extend(self.qs.iter().map(|&q| numpy_quantile(sorted, q)));
    }
}

/// Energy ratio by chunks feature.
///
/// Split the series into `num_segments` consecutive segments and return the
/// sum of squares of segment `segment_focus` (0-based) as a share of the
/// sum of squares of the whole series:
/// $$ \frac{\sum_{i \in S_k} x_i^2}{\sum_{i=1}^{n} x_i^2}. $$
/// Segments are formed as `np.array_split` does: when $n$ is not a multiple
/// of the number of segments, the first $n \bmod N$ segments get one extra
/// value. The shares of all segments add up to 1.
///
/// # Parameters
/// - `num_segments`: number of segments $N \geq 1$.
/// - `segment_focus`: the segment $k < N$ to return.
///
/// Config:
/// `[energy_ratio_by_chunks] parameters = [{ num_segments = 10, segment_focus = 0 }, ...]`;
/// one column per entry. `num_segments = 0` or `segment_focus >= num_segments`
/// is rejected when the config is loaded.
///
/// # Output column
/// `{name}__energy_ratio_by_chunks__num_segments_{num_segments}__segment_focus_{segment_focus}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series whose sum of squares is 0 gives NaN.
/// - With fewer values than segments, the trailing segments are empty and
///   give 0.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.energy_ratio_by_chunks` (v0.21.2), summed as numpy
/// does.
pub struct EnergyRatioByChunks {
    /// `(num_segments, segment_focus)` per config entry.
    pub params: Vec<(usize, usize)>,
}

impl Kernel for EnergyRatioByChunks {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(num_segments, segment_focus)| {
                format!(
                    "energy_ratio_by_chunks__num_segments_{num_segments}__segment_focus_{segment_focus}"
                )
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let n = x.len();
        // np.sum(x**2)
        let full = numpy_sum(n, |i| x[i] * x[i]);
        for &(num_segments, segment_focus) in &self.params {
            if full == 0.0 {
                out.push(f64::NAN);
                continue;
            }
            // np.array_split: the first n % N segments have one more value
            let (size, extra) = (n / num_segments, n % num_segments);
            let start = segment_focus * size + segment_focus.min(extra);
            let len = size + usize::from(segment_focus < extra);
            let segment = &x[start..start + len];
            out.push(numpy_sum(len, |i| segment[i] * segment[i]) / full);
        }
    }
}

/// Newcomb-Benford's law: the share of numbers with leading digit $d$,
/// $\log_{10}(1 + 1/d)$ for $d = 1, \dots, 9$, as numpy computes it.
const BENFORD_DISTRIBUTION: [f64; 9] = [
    std::f64::consts::LOG10_2,
    0.17609125905568124,
    0.12493873660829993,
    0.09691001300805642,
    0.07918124604762482,
    0.06694678963061322,
    0.05799194697768673,
    0.05115252244738129,
    0.04575749056067514,
];

/// The leading digit of `|x|` as `np.format_float_scientific` writes it
/// (shortest round-trip digits), after `np.nan_to_num`: NaN counts as 0 and
/// $\pm\infty$ as the largest finite float.
fn leading_digit(x: f64) -> u8 {
    let x = if x.is_nan() {
        0.0
    } else if x.is_infinite() {
        f64::MAX
    } else {
        x.abs()
    };
    if x == 0.0 {
        return 0;
    }
    format!("{x:e}").as_bytes()[0] - b'0'
}

scalar_kernel! {
/// Benford correlation feature.
///
/// The Pearson correlation between the distribution of the leading digits
/// of the values and Newcomb-Benford's law, $P(d) = \log_{10}(1 + 1/d)$ for
/// $d = 1, \dots, 9$. Useful for anomaly detection: many natural data sets
/// follow Benford's law closely (correlation near 1).
///
/// # Output column
/// `{name}__benford_correlation`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN and 0 have leading digit 0 (they count towards no digit but still
///   towards $n$); $\pm\infty$ has leading digit 1 (tsfresh's
///   `np.nan_to_num`).
/// - If all nine digit shares are equal (e.g. all values 0, or each digit
///   equally often), the correlation is undefined and the result is NaN.
///
/// # tsfresh
/// `feature_calculators.benford_correlation` (v0.21.2). tsfresh's
/// `np.corrcoef` uses a BLAS dot product, so results agree to rounding.
    BenfordCorrelation, "benford_correlation", |ctx| {
        let x = ctx.values();
        let mut counts = [0usize; 10];
        for &v in x {
            counts[leading_digit(v) as usize] += 1;
        }
        let data: [f64; 9] = std::array::from_fn(|d| counts[d + 1] as f64 / x.len() as f64);
        pearson_correlation(&BENFORD_DISTRIBUTION, &data)
    }
}

/// `np.corrcoef(a, b)[0, 1]`: covariance over the product of standard
/// deviations (`ddof = 1`, which cancels), clipped to $[-1, 1]$. NaN if
/// either series is constant.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mean_a, mean_b) = (numpy_mean(a), numpy_mean(b));
    let fact = 1.0 / (n - 1) as f64;
    let (mut cab, mut caa, mut cbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x - mean_a, y - mean_b);
        cab += dx * dy;
        caa += dx * dx;
        cbb += dy * dy;
    }
    let r = cab * fact / (caa * fact).sqrt() / (cbb * fact).sqrt();
    if r.is_nan() { r } else { r.clamp(-1.0, 1.0) }
}
