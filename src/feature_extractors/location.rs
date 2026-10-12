//! Relative positions in the series where something happens.

use ndarray::{Axis, s};

use super::kernel::{GroupContext, Kernel, scalar_kernel};
use crate::utils::stats::{np_argmax, np_argmin};

scalar_kernel! {
/// First location of maximum feature.
///
/// The relative position of the first occurrence of the maximum value, as a
/// fraction of the series length $n$ (0-based index $i$):
/// $$ \frac{i}{n}, \quad i = \min\{\,j : x_j = \max_k x_k\,\}. $$
///
/// # Output column
/// `{name}__first_location_of_maximum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - If the series contains NaN, NaN counts as the maximum (numpy's `np.argmax`
///   rule), so the result is the position of the first NaN.
/// - Ties resolve to the first occurrence.
///
/// # tsfresh
/// `feature_calculators.first_location_of_maximum` (v0.21.2).
    FirstLocationOfMaximum, "first_location_of_maximum", |ctx| np_argmax(ctx.x()).map_or(f64::NAN, |i| i as f64 / ctx.len() as f64)
}

scalar_kernel! {
/// First location of minimum feature.
///
/// The relative position of the first occurrence of the minimum value, as a
/// fraction of the series length $n$ (0-based index $i$):
/// $$ \frac{i}{n}, \quad i = \min\{\,j : x_j = \min_k x_k\,\}. $$
///
/// # Output column
/// `{name}__first_location_of_minimum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - If the series contains NaN, NaN counts as the minimum (numpy's `np.argmin`
///   rule), so the result is the position of the first NaN.
/// - Ties resolve to the first occurrence.
///
/// # tsfresh
/// `feature_calculators.first_location_of_minimum` (v0.21.2).
    FirstLocationOfMinimum, "first_location_of_minimum", |ctx| np_argmin(ctx.x()).map_or(f64::NAN, |i| i as f64 / ctx.len() as f64)
}

scalar_kernel! {
/// Last location of maximum feature.
///
/// The relative position of the last occurrence of the maximum value, as a
/// fraction of the series length $n$ (0-based index $i$):
/// $$ \frac{i + 1}{n}, \quad i = \max\{\,j : x_j = \max_k x_k\,\}. $$
///
/// # Output column
/// `{name}__last_location_of_maximum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - If the series contains NaN, NaN counts as the maximum (numpy's `np.argmax`
///   rule), so the result is the position of the last NaN.
/// - Ties resolve to the last occurrence.
///
/// # tsfresh
/// `feature_calculators.last_location_of_maximum` (v0.21.2).
    LastLocationOfMaximum, "last_location_of_maximum", |ctx| {
        // argmax of the reversed series gives the last occurrence
        np_argmax(&ctx.x().slice(s![..;-1])).map_or(f64::NAN, |i| 1.0 - i as f64 / ctx.len() as f64)
    }
}

scalar_kernel! {
/// Last location of minimum feature.
///
/// The relative position of the last occurrence of the minimum value, as a
/// fraction of the series length $n$ (0-based index $i$):
/// $$ \frac{i + 1}{n}, \quad i = \max\{\,j : x_j = \min_k x_k\,\}. $$
///
/// # Output column
/// `{name}__last_location_of_minimum`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - If the series contains NaN, NaN counts as the minimum (numpy's `np.argmin`
///   rule), so the result is the position of the last NaN.
/// - Ties resolve to the last occurrence.
///
/// # tsfresh
/// `feature_calculators.last_location_of_minimum` (v0.21.2).
    LastLocationOfMinimum, "last_location_of_minimum", |ctx| {
        // argmin of the reversed series gives the last occurrence
        np_argmin(&ctx.x().slice(s![..;-1])).map_or(f64::NAN, |i| 1.0 - i as f64 / ctx.len() as f64)
    }
}

/// Index mass quantile feature.
///
/// The relative position at which the cumulative absolute "mass" of the
/// series first reaches the fraction $q$ of its total:
/// $$ \frac{i + 1}{n}, \quad i = \min\left\{\, j : \frac{\sum_{k=0}^{j} |x_k|}{\sum_{k=0}^{n-1} |x_k|} \geq q \,\right\}, $$
/// with 0-based indices and $n$ the number of values.
///
/// # Parameters
/// - `q`: mass fraction in $[0, 1]$. Config:
///   `[index_mass_quantile] parameters = [{ q = 0.1 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__index_mass_quantile__q_{q}` (`q` in Rust's default formatting,
/// e.g. `q_0.3`, `q_0.99`)
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series whose values are all 0 has no mass and gives NaN.
/// - A series containing NaN gives $1/n$, as in tsfresh.
///
/// # tsfresh
/// `feature_calculators.index_mass_quantile` (v0.21.2).
pub struct IndexMassQuantile {
    pub qs: Vec<f64>,
}

impl Kernel for IndexMassQuantile {
    fn names(&self) -> Vec<String> {
        self.qs
            .iter()
            .map(|q| format!("index_mass_quantile__q_{q:2}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let mut mass = ctx.x().mapv(f64::abs);
        let total = mass.sum();
        if total == 0.0 {
            out.extend(self.qs.iter().map(|_| f64::NAN));
            return;
        }
        mass.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
        let mass = mass.mapv(|x| x / total);
        for q in &self.qs {
            let first = mass.iter().position(|x| x >= q).unwrap_or(0);
            out.push((first + 1) as f64 / ctx.len() as f64);
        }
    }
}
