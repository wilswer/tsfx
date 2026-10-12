//! Counts of values or runs relative to a threshold.

use ndarray::Array1;

use super::kernel::{GroupContext, Kernel, scalar_kernel};
use crate::utils::stats::get_length_sequences_where;

scalar_kernel! {
/// Count above mean feature.
///
/// The number of values strictly greater than the mean of the series:
/// $$ \sum_{i=1}^{n} \mathbb{1}[x_i > \mu]. $$
///
/// # Output column
/// `{name}__count_above_mean`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN has a NaN mean and gives 0.
/// - A constant series gives 0.
///
/// # tsfresh
/// `feature_calculators.count_above_mean` (v0.21.2).
    CountAboveMean, "count_above_mean", |ctx| {
        let mean = ctx.mean();
        ctx.x().mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum()
    }
}

scalar_kernel! {
/// Count below mean feature.
///
/// The number of values strictly less than the mean of the series:
/// $$ \sum_{i=1}^{n} \mathbb{1}[x_i < \mu]. $$
///
/// # Output column
/// `{name}__count_below_mean`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN has a NaN mean and gives 0.
/// - A constant series gives 0.
///
/// # tsfresh
/// `feature_calculators.count_below_mean` (v0.21.2).
    CountBelowMean, "count_below_mean", |ctx| {
        let mean = ctx.mean();
        ctx.x().mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum()
    }
}

scalar_kernel! {
/// Longest strike below mean feature.
///
/// The length of the longest run of consecutive values strictly less than
/// the mean of the series.
///
/// # Output column
/// `{name}__longest_strike_below_mean`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN has a NaN mean and gives 0.
/// - A constant series gives 0.
///
/// # tsfresh
/// `feature_calculators.longest_strike_below_mean` (v0.21.2).
    LongestStrikeBelowMean, "longest_strike_below_mean", |ctx| {
        let mean = ctx.mean();
        let flags: Array1<bool> = ctx.x().mapv(|x| x < mean);
        get_length_sequences_where(&flags).into_iter().max().unwrap_or(0) as f64
    }
}

scalar_kernel! {
/// Longest strike above mean feature.
///
/// The length of the longest run of consecutive values strictly greater than
/// the mean of the series.
///
/// # Output column
/// `{name}__longest_strike_above_mean`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN has a NaN mean and gives 0.
/// - A constant series gives 0.
///
/// # tsfresh
/// `feature_calculators.longest_strike_above_mean` (v0.21.2).
    LongestStrikeAboveMean, "longest_strike_above_mean", |ctx| {
        let mean = ctx.mean();
        let flags: Array1<bool> = ctx.x().mapv(|x| x > mean);
        get_length_sequences_where(&flags).into_iter().max().unwrap_or(0) as f64
    }
}

/// Count above feature.
///
/// The fraction of values greater than or equal to the threshold $t$:
/// $$ \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[x_i \geq t], $$
/// where $n$ is the number of values in the time series.
///
/// # Parameters
/// - `t`: threshold. Config: `[count_above] parameters = [{ t = 0.0 }]`; one
///   column per entry.
///
/// # Output column
/// `{name}__count_above__t_{t:.1}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values count in $n$ but never satisfy $x_i \geq t$.
/// - The column name rounds `t` to one decimal, so e.g. 0.25 and 0.2 would
///   share a name.
///
/// # tsfresh
/// `feature_calculators.count_above` (v0.21.2).
pub struct CountAbove {
    pub ts: Vec<f64>,
}

impl Kernel for CountAbove {
    fn names(&self) -> Vec<String> {
        self.ts
            .iter()
            .map(|t| format!("count_above__t_{t:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // fraction of values >= t, as in tsfresh
        for &t in &self.ts {
            let count = ctx.x().mapv(|x| if x >= t { 1.0 } else { 0.0 }).sum();
            out.push(count / ctx.len() as f64);
        }
    }
}

/// Count below feature.
///
/// The fraction of values less than or equal to the threshold $t$:
/// $$ \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[x_i \leq t], $$
/// where $n$ is the number of values in the time series.
///
/// # Parameters
/// - `t`: threshold. Config: `[count_below] parameters = [{ t = 0.0 }]`; one
///   column per entry.
///
/// # Output column
/// `{name}__count_below__t_{t:.1}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values count in $n$ but never satisfy $x_i \leq t$.
/// - The column name rounds `t` to one decimal, so e.g. 0.25 and 0.2 would
///   share a name.
///
/// # tsfresh
/// `feature_calculators.count_below` (v0.21.2).
pub struct CountBelow {
    pub ts: Vec<f64>,
}

impl Kernel for CountBelow {
    fn names(&self) -> Vec<String> {
        self.ts
            .iter()
            .map(|t| format!("count_below__t_{t:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // fraction of values <= t, as in tsfresh
        for &t in &self.ts {
            let count = ctx.x().mapv(|x| if x <= t { 1.0 } else { 0.0 }).sum();
            out.push(count / ctx.len() as f64);
        }
    }
}

/// Number of crossings of m feature.
///
/// The number of times the series crosses the level $m$. Each value is
/// classified as above ($x_i > m$) or not, and every change of class between
/// neighbouring values counts as one crossing:
/// $$ \sum_{i=1}^{n-1} \mathbb{1}\big(\mathbb{1}(x_i > m) \neq \mathbb{1}(x_{i+1} > m)\big). $$
///
/// # Parameters
/// - `m`: level. Config: `[number_crossing_m] parameters = [{ m = 0.0 }, ...]`;
///   one column per entry.
///
/// # Output column
/// `{name}__number_crossing_m__m_{m:.1}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A value equal to $m$, and NaN, count as "not above".
///
/// # tsfresh
/// `feature_calculators.number_crossing_m` (v0.21.2).
pub struct NumberCrossingM {
    pub ms: Vec<f64>,
}

impl Kernel for NumberCrossingM {
    fn names(&self) -> Vec<String> {
        self.ms
            .iter()
            .map(|m| format!("number_crossing_m__m_{m:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // tsfresh: binarise as x > m (NaN and x == m are "not above") and
        // count every change between neighbours
        for &m in &self.ms {
            let changes = ctx
                .values()
                .windows(2)
                .filter(|w| (w[0] > m) != (w[1] > m))
                .count();
            out.push(changes as f64);
        }
    }
}

/// Range count feature.
///
/// The number of values in the half-open interval $[\text{min}, \text{max})$:
/// $$ \sum_{i=1}^{n} \mathbb{1}[\text{min} \leq x_i < \text{max}]. $$
///
/// # Parameters
/// - `min`, `max`: interval bounds. Config:
///   `[range_count] parameters = [{ min = -1.0, max = 1.0 }, ...]`; one column
///   per entry.
///
/// # Output column
/// `{name}__range_count__min_{min:.1}__max_{max:.1}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - NaN values are never counted.
/// - `max <= min` is an empty interval and gives 0.
///
/// # tsfresh
/// `feature_calculators.range_count` (v0.21.2).
pub struct RangeCount {
    pub ranges: Vec<(f64, f64)>,
}

impl Kernel for RangeCount {
    fn names(&self) -> Vec<String> {
        self.ranges
            .iter()
            .map(|(lower, upper)| format!("range_count__min_{lower:.1}__max_{upper:.1}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        // Half-open interval [lower, upper), as tsfresh; empty if upper <= lower
        for &(lower, upper) in &self.ranges {
            let count = ctx
                .values()
                .iter()
                .filter(|&&x| x >= lower && x < upper)
                .count();
            out.push(count as f64);
        }
    }
}

/// Number of peaks feature.
///
/// The number of peaks of support $n$: values strictly greater than the $n$
/// neighbours on each side, $x_i > x_{i \pm k}$ for all $k = 1, \dots, n$.
/// Values within $n$ of either end cannot be peaks.
///
/// # Parameters
/// - `n`: support. Config: `[number_peaks] parameters = [{ n = 1 }, ...]`; one
///   column per entry.
///
/// # Output column
/// `{name}__number_peaks__n_{n}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series shorter than $2n + 1$ gives 0.
/// - Comparisons with NaN are false, so NaN values are never peaks and
///   neighbouring a NaN prevents a peak.
///
/// # tsfresh
/// `feature_calculators.number_peaks` (v0.21.2).
pub struct NumberPeaks {
    pub ns: Vec<usize>,
}

impl Kernel for NumberPeaks {
    fn names(&self) -> Vec<String> {
        self.ns
            .iter()
            .map(|n| format!("number_peaks__n_{n}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        out.extend(
            peak_counts(ctx.values(), &self.ns)
                .into_iter()
                .map(|c| c as f64),
        );
    }
}

/// For each `n` in `ns`, the number of positions `j` with `n <= j < len - n`
/// whose value is strictly greater than its `n` neighbours on each side.
///
/// One pass computes each position's support (how many neighbours it beats on
/// both sides, capped at the largest `n`); the scan stops at the first
/// neighbour that is not smaller. Comparisons with NaN are false, as in
/// tsfresh.
fn peak_counts(x: &[f64], ns: &[usize]) -> Vec<usize> {
    let len = x.len();
    let cap = ns.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0; ns.len()];
    for j in 0..len {
        let mut left = 0;
        while left < cap && left < j && x[j] > x[j - 1 - left] {
            left += 1;
        }
        if left == 0 {
            continue;
        }
        let mut right = 0;
        while right < left && j + 1 + right < len && x[j] > x[j + 1 + right] {
            right += 1;
        }
        let support = left.min(right);
        for (count, &n) in counts.iter_mut().zip(ns) {
            if n <= support && j >= n && j + n < len {
                *count += 1;
            }
        }
    }
    counts
}
