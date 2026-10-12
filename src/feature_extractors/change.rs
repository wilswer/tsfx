//! Features built from consecutive differences.

use ndarray::{CowArray, s};

use super::kernel::{GroupContext, Kernel, scalar_kernel};

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
    MeanAbsoluteChange, "mean_absolute_change", |ctx| ctx.diffs().mapv(f64::abs).mean().unwrap_or(f64::NAN)
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
    AbsoluteSumOfChanges, "absolute_sum_of_changes", |ctx| ctx.diffs().mapv(f64::abs).sum()
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
