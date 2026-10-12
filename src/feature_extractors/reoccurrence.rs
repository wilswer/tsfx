//! Features about repeated values.

use super::kernel::{GroupContext, Kernel, scalar_kernel};

scalar_kernel! {
/// Has duplicate max feature.
///
/// Whether the maximum value occurs more than once. Returns 1.0 for true and
/// 0.0 for false.
///
/// # Output column
/// `{name}__has_duplicate_max`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 (tsfresh's maximum is NaN, and nothing
///   compares equal to NaN).
///
/// # tsfresh
/// `feature_calculators.has_duplicate_max` (v0.21.2).
    HasDuplicateMax, "has_duplicate_max", |ctx| {
        // tsfresh: np.max is NaN and x == NaN is never true, so the answer is 0
        if ctx.x().iter().any(|x| x.is_nan()) {
            0.0
        } else {
            let max = ctx.max();
            (ctx.x().iter().filter(|&&x| x == max).count() > 1) as u8 as f64
        }
    }
}

scalar_kernel! {
/// Has duplicate min feature.
///
/// Whether the minimum value occurs more than once. Returns 1.0 for true and
/// 0.0 for false.
///
/// # Output column
/// `{name}__has_duplicate_min`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series containing NaN gives 0 (tsfresh's minimum is NaN, and nothing
///   compares equal to NaN).
///
/// # tsfresh
/// `feature_calculators.has_duplicate_min` (v0.21.2).
    HasDuplicateMin, "has_duplicate_min", |ctx| {
        // tsfresh: np.min is NaN and x == NaN is never true, so the answer is 0
        if ctx.x().iter().any(|x| x.is_nan()) {
            0.0
        } else {
            let min = ctx.min();
            (ctx.x().iter().filter(|&&x| x == min).count() > 1) as u8 as f64
        }
    }
}

/// Which of the counting-based reoccurrence features to compute. They share
/// one sort of the series, so computing several together costs about the same
/// as computing one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReoccurrenceFeatures {
    /// Has duplicate feature.
    ///
    /// Whether any value occurs more than once. Returns 1.0 for true and 0.0 for
    /// false.
    ///
    /// # Output column
    /// `{name}__has_duplicate`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - All NaN values count as one value (like `np.unique`), so two NaNs are a
    ///   duplicate.
    ///
    /// # tsfresh
    /// `feature_calculators.has_duplicate` (v0.21.2).
    pub has_duplicate: bool,
    /// Ratio of unique values to length feature.
    ///
    /// The number of distinct values divided by the number of values:
    /// $$ \frac{|\{x_1, \dots, x_n\}|}{n}. $$
    /// 1 means every value is unique.
    ///
    /// # Output column
    /// `{name}__ratio_value_number_to_time_series_length`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - All NaN values count as one distinct value (like `np.unique`).
    ///
    /// # tsfresh
    /// `feature_calculators.ratio_value_number_to_time_series_length` (v0.21.2).
    pub ratio_value_number_to_time_series_length: bool,
    /// Sum of reoccurring values feature.
    ///
    /// The sum of the distinct values that occur more than once, each counted
    /// once: for `[1, 1, 2, 3, 3]` it is 1 + 3 = 4.
    ///
    /// # Output column
    /// `{name}__sum_of_reoccurring_values`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - A NaN or ±∞ that occurs more than once makes the result NaN or ±∞.
    /// - **Deliberate deviation from tsfresh:** a NaN or ±∞ that occurs only once
    ///   is left out. tsfresh excludes such values by multiplying them by 0, and
    ///   since 0 · NaN = 0 · ∞ = NaN it returns NaN instead.
    ///
    /// # tsfresh
    /// `feature_calculators.sum_of_reoccurring_values` (v0.21.2).
    pub sum_of_reoccurring_values: bool,
    /// Sum of reoccurring data points feature.
    ///
    /// The sum of all values whose value occurs more than once, counting every
    /// occurrence: for `[1, 1, 2, 3, 3]` it is 1 + 1 + 3 + 3 = 8.
    ///
    /// # Output column
    /// `{name}__sum_of_reoccurring_data_points`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - A NaN or ±∞ that occurs more than once makes the result NaN or ±∞.
    /// - **Deliberate deviation from tsfresh:** a NaN or ±∞ that occurs only once
    ///   is left out. tsfresh excludes such values by multiplying them by 0, and
    ///   since 0 · NaN = 0 · ∞ = NaN it returns NaN instead.
    ///
    /// # tsfresh
    /// `feature_calculators.sum_of_reoccurring_data_points` (v0.21.2).
    pub sum_of_reoccurring_data_points: bool,
    /// Percentage of reoccurring values to all values feature.
    ///
    /// The fraction of distinct values that occur more than once:
    /// $$ \frac{\text{number of distinct values occurring more than once}}{\text{number of distinct values}}. $$
    ///
    /// # Output column
    /// `{name}__percentage_of_reoccurring_values_to_all_values`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - All NaN values count as one distinct value (like `np.unique`), which can
    ///   itself be reoccurring.
    ///
    /// # tsfresh
    /// `feature_calculators.percentage_of_reoccurring_values_to_all_values` (v0.21.2).
    pub percentage_of_reoccurring_values_to_all_values: bool,
    /// Percentage of reoccurring data points to all data points feature.
    ///
    /// The fraction of values (data points) whose value occurs more than once:
    /// $$ \frac{\text{number of data points with a reoccurring value}}{n}. $$
    ///
    /// # Output column
    /// `{name}__percentage_of_reoccurring_values_to_all_datapoints`
    ///
    /// # Edge cases
    /// - Nulls are dropped first; a group with no non-null values gives NaN.
    /// - NaN never counts as reoccurring (tsfresh uses pandas `value_counts`,
    ///   which drops NaN), but NaN values still count in $n$. Note that this
    ///   differs from the `..._to_all_values` feature, which counts NaN as a value.
    ///
    /// # tsfresh
    /// `feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints` (v0.21.2).
    pub percentage_of_reoccurring_values_to_all_datapoints: bool,
}

impl ReoccurrenceFeatures {
    /// Names of the enabled features, in output order.
    pub fn enabled(&self) -> Vec<&'static str> {
        [
            (self.has_duplicate, "has_duplicate"),
            (
                self.ratio_value_number_to_time_series_length,
                "ratio_value_number_to_time_series_length",
            ),
            (self.sum_of_reoccurring_values, "sum_of_reoccurring_values"),
            (
                self.sum_of_reoccurring_data_points,
                "sum_of_reoccurring_data_points",
            ),
            (
                self.percentage_of_reoccurring_values_to_all_values,
                "percentage_of_reoccurring_values_to_all_values",
            ),
            (
                self.percentage_of_reoccurring_values_to_all_datapoints,
                "percentage_of_reoccurring_values_to_all_datapoints",
            ),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| name)
        .collect()
    }
}

/// Runs of equal values in a series (sorted once), plus how many NaNs it has.
struct ValueCounts {
    /// (value, count) for each distinct non-NaN value; `-0.0 == 0.0`.
    runs: Vec<(f64, usize)>,
    nan_count: usize,
    n: usize,
}

impl ValueCounts {
    fn new(values: Vec<f64>) -> Self {
        let n = values.len();
        let mut non_nan: Vec<f64> = values.into_iter().filter(|x| !x.is_nan()).collect();
        let nan_count = n - non_nan.len();
        non_nan.sort_unstable_by(|a, b| a.total_cmp(b));
        let mut runs: Vec<(f64, usize)> = Vec::new();
        for x in non_nan {
            match runs.last_mut() {
                Some((v, c)) if *v == x => *c += 1,
                _ => runs.push((x, 1)),
            }
        }
        ValueCounts { runs, nan_count, n }
    }

    /// Distinct values, counting all NaNs as one value (like `np.unique`).
    fn distinct(&self) -> usize {
        self.runs.len() + (self.nan_count > 0) as usize
    }

    fn feature(&self, name: &str) -> f64 {
        let n = self.n as f64;
        let recurring = || self.runs.iter().filter(|(_, c)| *c > 1);
        match name {
            "has_duplicate" => ((self.distinct() < self.n) as u8).into(),
            "ratio_value_number_to_time_series_length" => self.distinct() as f64 / n,
            // A NaN that occurs more than once makes the sums NaN; a single NaN
            // or ±inf is left out (deliberate deviation from tsfresh).
            "sum_of_reoccurring_values" => {
                let nan = if self.nan_count > 1 { f64::NAN } else { 0.0 };
                recurring().map(|(v, _)| *v).sum::<f64>() + nan
            }
            "sum_of_reoccurring_data_points" => {
                let nan = if self.nan_count > 1 { f64::NAN } else { 0.0 };
                recurring().map(|(v, c)| *v * *c as f64).sum::<f64>() + nan
            }
            // NaN is one distinct value here (np.unique) and may itself recur.
            "percentage_of_reoccurring_values_to_all_values" => {
                let recurring_values = recurring().count() + (self.nan_count > 1) as usize;
                recurring_values as f64 / self.distinct() as f64
            }
            // NaN never recurs here (pandas value_counts) but counts in n.
            "percentage_of_reoccurring_values_to_all_datapoints" => {
                recurring().map(|(_, c)| *c).sum::<usize>() as f64 / n
            }
            _ => unreachable!("unknown reoccurrence feature {name}"),
        }
    }
}

impl Kernel for ReoccurrenceFeatures {
    fn names(&self) -> Vec<String> {
        self.enabled().into_iter().map(String::from).collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let counts = ValueCounts::new(ctx.values().to_vec());
        out.extend(self.enabled().into_iter().map(|name| counts.feature(name)));
    }
}
