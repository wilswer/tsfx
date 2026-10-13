//! Features built from consecutive differences.

use ndarray::{ArrayView1, CowArray, s};

use super::kernel::{GroupContext, Kernel, scalar_kernel};
use crate::utils::stats::{numpy_mean, numpy_median, numpy_quantile, numpy_sum, population_var};
use crate::utils::toml_reader::ChangeAggregator;

scalar_kernel! {
/// Mean absolute change feature.
///
/// The mean of the absolute differences between consecutive values:
/// $$ \frac{1}{n-1} \sum_{i=1}^{n-1} |x_{i+1} - x_i|, $$
/// where $n$ is the number of values in the time series.
///
/// # Output column
/// `{name}__mean_absolute_change`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A single value has no differences and gives NaN.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.mean_abs_change` (v0.21.2).
    MeanAbsoluteChange, "mean_absolute_change", |ctx| {
        // np.mean(np.abs(np.diff(x))); NaN for a single value
        let x = ctx.values();
        numpy_sum(x.len() - 1, |i| (x[i + 1] - x[i]).abs()) / (x.len() - 1) as f64
    }
}

scalar_kernel! {
/// Absolute sum of changes feature.
///
/// The sum of the absolute differences between consecutive values:
/// $$ \sum_{i=1}^{n-1} |x_{i+1} - x_i|. $$
///
/// # Output column
/// `{name}__absolute_sum_of_changes`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A single value has no differences and gives 0.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.absolute_sum_of_changes` (v0.21.2).
    AbsoluteSumOfChanges, "absolute_sum_of_changes", |ctx| {
        // np.sum(np.abs(np.diff(x)))
        let x = ctx.values();
        numpy_sum(x.len() - 1, |i| (x[i + 1] - x[i]).abs())
    }
}

scalar_kernel! {
/// Mean change feature.
///
/// The mean of the differences between consecutive values. The sum
/// telescopes, so only the first and last values matter:
/// $$ \frac{1}{n-1} \sum_{i=1}^{n-1} (x_{i+1} - x_i) = \frac{x_n - x_1}{n-1}. $$
///
/// # Output column
/// `{name}__mean_change`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A single value gives NaN.
/// - Only the endpoints are used, so NaN elsewhere does not affect the
///   result; NaN at either end gives NaN.
///
/// # tsfresh
/// `feature_calculators.mean_change` (v0.21.2).
    MeanChange, "mean_change", |ctx| {
        let x = ctx.values();
        if x.len() < 2 {
            f64::NAN
        } else {
            (x[x.len() - 1] - x[0]) / ((x.len() - 1) as f64)
        }
    }
}

/// Complexity-invariant distance (CID CE) feature.
///
/// An estimate of the time series' complexity: the length of the line
/// connecting consecutive values,
/// $$ \sqrt{\sum_{i=1}^{n-1} (z_{i+1} - z_i)^2}, $$
/// where $z = x$, or with normalisation $z_i = (x_i - \mu)/\sigma$ using the
/// mean $\mu$ and population standard deviation $\sigma$. A more complex
/// series (more peaks and valleys) scores higher.
///
/// # Parameters
/// - `normalize`: z-normalise the series first. Config:
///   `[cid_ce] parameters = [{ normalize = true }, { normalize = false }]`; one
///   column per entry.
///
/// # Output column
/// `{name}__cid_ce__normalize_t` or `{name}__cid_ce__normalize_f`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A single value gives 0.
/// - With `normalize`, a constant series has $\sigma = 0$ and gives 0.
/// - NaN values are kept and make the result NaN.
///
/// # tsfresh
/// `feature_calculators.cid_ce` (v0.21.2).
pub struct CidCe {
    pub normalize: Vec<bool>,
}

impl Kernel for CidCe {
    fn names(&self) -> Vec<String> {
        // `{:.1}` truncates the bool to "t" / "f"; kept for stable names
        self.normalize
            .iter()
            .map(|normalize| format!("cid_ce__normalize_{normalize:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        for &normalize in &self.normalize {
            let x = if normalize {
                // tsfresh: a constant series has no complexity
                if ctx.std() == 0.0 {
                    out.push(0.0);
                    continue;
                }
                CowArray::from((ctx.x() - ctx.mean()) / ctx.std())
            } else {
                CowArray::from(ctx.x().view())
            };
            let diffs = &x.slice(s![1..]) - &x.slice(s![..-1]);
            out.push(diffs.mapv(|d| d.powi(2)).sum().sqrt());
        }
    }
}

scalar_kernel! {
/// Mean second derivative central feature.
///
/// The mean of the central approximation of the second derivative,
/// $\frac{1}{2}(x_{i+2} - 2\,x_{i+1} + x_i)$. The sum telescopes, so only
/// the first two and last two values matter:
/// $$ \frac{1}{n-2} \sum_{i=1}^{n-2} \frac{1}{2}(x_{i+2} - 2\,x_{i+1} + x_i)
///    = \frac{x_n - x_{n-1} - x_2 + x_1}{2\,(n-2)}. $$
///
/// # Output column
/// `{name}__mean_second_derivative_central`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - Fewer than 3 values give NaN.
/// - Only the first two and last two values are used, so NaN elsewhere does
///   not affect the result.
///
/// # tsfresh
/// `feature_calculators.mean_second_derivative_central` (v0.21.2).
    MeanSecondDerivativeCentral, "mean_second_derivative_central", |ctx| {
        let x = ctx.values();
        let n = x.len();
        if n > 2 {
            // (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)), in this order
            (x[n - 1] - x[n - 2] - x[1] + x[0]) / (2 * (n - 2)) as f64
        } else {
            f64::NAN
        }
    }
}

/// Change quantiles feature.
///
/// Fix a corridor between the `ql`- and `qh`-quantiles of the values, then
/// aggregate the consecutive changes $x_{i+1} - x_i$ (or their absolute
/// values) that start and end inside the corridor:
/// $$ f_{agg}\left(\{\, x_{i+1} - x_i : c_i \wedge c_{i+1} \,\}\right), $$
/// where $c_i$ says whether $x_i$ lies in the corridor
/// $(Q(q_l), Q(q_h)]$, the lower edge included.
///
/// The corridor is `pd.qcut(x, [ql, qh])`'s first bin: the edges are the
/// quantiles of the non-NaN values with linear interpolation (as pandas
/// computes them, through `np.percentile` with `100 q`), and values are
/// binned with numpy's `searchsorted` exactly as pandas does.
///
/// # Parameters
/// - `ql`, `qh`: lower and upper quantile of the corridor.
/// - `isabs`: aggregate absolute changes.
/// - `f_agg`: `mean`, `var`, `std` or `median` of the changes (population
///   variance and standard deviation, `ddof = 0`; NaN propagates as in
///   numpy). Any other name is rejected when the config is loaded.
///
/// Config:
/// `[change_quantiles] parameters = [{ ql = 0.0, qh = 0.2, isabs = false, f_agg = "mean" }, ...]`;
/// one column per entry.
///
/// # Output column
/// `{name}__change_quantiles__ql_{ql}__qh_{qh}__isabs_{isabs}__f_agg_{f_agg}`,
/// with `ql` and `qh` written as Python's `repr` would (e.g. `0.2`, `1.0`)
/// and `isabs` as `true` / `false`.
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - $q_l \geq q_h$ gives 0.
/// - No change inside the corridor (e.g. a single value) gives 0.
/// - NaN values never lie in the corridor; a change involving $\pm\infty$
///   can be NaN and then makes the aggregate NaN, as in numpy.
/// - If both edges coincide, pandas (3.x) keeps the duplicate edge for two
///   quantiles, so the corridor holds exactly the values equal to it.
///
/// # tsfresh
/// `feature_calculators.change_quantiles` (v0.21.2).
pub struct ChangeQuantiles {
    /// `(ql, qh, isabs, f_agg)` per config entry.
    pub params: Vec<(f64, f64, bool, ChangeAggregator)>,
}

impl Kernel for ChangeQuantiles {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(ql, qh, isabs, f_agg)| {
                format!("change_quantiles__ql_{ql:?}__qh_{qh:?}__isabs_{isabs}__f_agg_{f_agg}")
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        // The quantiles of the non-NaN values (pandas drops NaN first)
        let finite_sorted = ctx
            .sorted()
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect::<Vec<f64>>();
        // The corridor's changes, shared by the aggregations of the same
        // (ql, qh, isabs); the default grid has 3 per corridor and side
        let mut corridors: Vec<((f64, f64, bool), Vec<f64>)> = Vec::new();
        let mut scratch = Vec::new();
        for &(ql, qh, isabs, f_agg) in &self.params {
            if ql >= qh {
                out.push(0.0);
                continue;
            }
            let key = (ql, qh, isabs);
            let changes = match corridors.iter().position(|(k, _)| *k == key) {
                Some(i) => &corridors[i].1,
                None => {
                    let bins = [ql, qh].map(|q| pandas_quantile(&finite_sorted, q));
                    let inside = qcut_first_bin(x, bins);
                    // Only changes that start and end inside the corridor
                    let changes = (0..x.len().saturating_sub(1))
                        .filter(|&i| inside[i] && inside[i + 1])
                        .map(|i| {
                            let d = x[i + 1] - x[i];
                            if isabs { d.abs() } else { d }
                        })
                        .collect();
                    corridors.push((key, changes));
                    &corridors.last().unwrap().1
                }
            };
            out.push(if changes.is_empty() {
                0.0
            } else {
                // A fresh copy: the median reorders it
                scratch.clear();
                scratch.extend_from_slice(changes);
                f_agg.aggregate(&mut scratch)
            });
        }
    }
}

impl ChangeAggregator {
    /// The numpy function of the same name on `values` (not empty; may be
    /// reordered).
    fn aggregate(&self, values: &mut [f64]) -> f64 {
        match self {
            ChangeAggregator::Mean => numpy_mean(values),
            ChangeAggregator::Var => population_var(&ArrayView1::from(&*values)),
            ChangeAggregator::Std => population_var(&ArrayView1::from(&*values)).sqrt(),
            ChangeAggregator::Median => {
                if values.iter().any(|v| v.is_nan()) {
                    f64::NAN
                } else {
                    numpy_median(values)
                }
            }
        }
    }
}

/// `Series.quantile(q)` of non-NaN `sorted` values as pandas computes it:
/// `np.percentile(x, q * 100)`, which divides by 100 again (so `q = 0.2`
/// becomes `0.20000000000000004`). NaN for no values.
fn pandas_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    numpy_quantile(sorted, (q * 100.0) / 100.0)
}

/// Which values fall in the first bin of `pd.qcut(x, bins, labels=False)`
/// with two edges: `ids = np.searchsorted(bins, x, side="left")`, values
/// equal to the lower edge moved into the bin, NaN never in it.
fn qcut_first_bin(x: &[f64], bins: [f64; 2]) -> Vec<bool> {
    let mut ids = numpy_searchsorted_left(&bins, x);
    for (id, &v) in ids.iter_mut().zip(x) {
        if v == bins[0] {
            *id = 1;
        }
    }
    ids.iter()
        .zip(x)
        .map(|(&id, v)| id == 1 && !v.is_nan())
        .collect()
}

/// `np.searchsorted(arr, keys, side="left")` for float64, including numpy's
/// use of the previous key to narrow the search (which matters only when
/// `arr` is not sorted, e.g. a NaN edge before a number) and its NaN-last
/// ordering.
fn numpy_searchsorted_left(arr: &[f64], keys: &[f64]) -> Vec<usize> {
    let less = |a: f64, b: f64| a < b || (b.is_nan() && !a.is_nan());
    let n = arr.len();
    let (mut min_idx, mut max_idx) = (0, n);
    let Some(&first) = keys.first() else {
        return Vec::new();
    };
    let mut last_key = first;
    keys.iter()
        .map(|&key| {
            if less(last_key, key) {
                max_idx = n;
            } else {
                min_idx = 0;
                max_idx = if max_idx < n { max_idx + 1 } else { n };
            }
            last_key = key;
            while min_idx < max_idx {
                let mid = min_idx + ((max_idx - min_idx) >> 1);
                if less(arr[mid], key) {
                    min_idx = mid + 1;
                } else {
                    max_idx = mid;
                }
            }
            min_idx
        })
        .collect()
}
