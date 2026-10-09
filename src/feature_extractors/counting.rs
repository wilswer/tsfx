//! Counts of values or runs relative to a threshold.

use anyhow::Result;
use ndarray::{Axis, Ix1};
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::get_length_sequences_where;

fn _count_above_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = arr.mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn count_above_mean(name: &str) -> Expr {
    col(name)
        .apply(_count_above_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__count_above_mean", name))
}

fn _count_below_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = arr.mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn count_below_mean(name: &str) -> Expr {
    col(name)
        .apply(_count_below_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__count_below_mean", name))
}

fn _count_above(s: Column, t: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // fraction of values >= t, as in tsfresh
    let out = arr.mapv(|x| if x >= t { 1.0 } else { 0.0 }).sum() / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
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
pub fn count_above(name: &str, t: f64) -> Expr {
    col(name)
        .apply(
            move |s| _count_above(s, t),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__count_above__t_{:.1}", name, t))
}

fn _count_below(s: Column, t: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // fraction of values <= t, as in tsfresh
    let out = arr.mapv(|x| if x <= t { 1.0 } else { 0.0 }).sum() / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
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
pub fn count_below(name: &str, t: f64) -> Expr {
    col(name)
        .apply(
            move |s| _count_below(s, t),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__count_below__t_{:.1}", name, t))
}

fn _longest_strike_below_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let bool_arr = arr.mapv(|x| x < mean);
    let out = get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(s)
}

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
pub fn longest_strike_below_mean(name: &str) -> Expr {
    col(name)
        .apply(_longest_strike_below_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__longest_strike_below_mean", name))
}

fn _longest_strike_above_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let bool_arr = arr.mapv(|x| x > mean);
    let out = get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(s)
}

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
pub fn longest_strike_above_mean(name: &str) -> Expr {
    col(name)
        .apply(_longest_strike_above_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__longest_strike_above_mean", name))
}

fn _number_crossing_m(s: Column, m: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // tsfresh: binarise as x > m (NaN and x == m are "not above") and count
    // every change between neighbours
    let above = arr.iter().map(|x| *x > m).collect::<Vec<_>>();
    let count = above.windows(2).filter(|w| w[0] != w[1]).count();
    let s = Column::new("".into(), &[count as f64]);
    Ok(s)
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
pub fn number_crossing_m(name: &str, m: f64) -> Expr {
    col(name)
        .apply(
            move |s| _number_crossing_m(s, m),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__number_crossing_m__m_{:.1}", name, m))
}

fn _range_count(s: Column, lower: f64, upper: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // Half-open interval [lower, upper), as tsfresh; empty if upper <= lower
    let count = arr
        .into_iter()
        .filter(|x| x >= &lower && x < &upper)
        .count();
    let s = Column::new("".into(), &[count as f64]);
    Ok(s)
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
pub fn range_count(name: &str, lower: f64, upper: f64) -> Expr {
    col(name)
        .apply(
            move |s| _range_count(s, lower, upper),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!(
            "{}__range_count__min_{:.1}__max_{:.1}",
            name, lower, upper,
        ))
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
pub fn number_peaks(name: &str, n: usize) -> Expr {
    number_peaks_multi(name, vec![n])
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

fn _number_peaks_all(s: Column, ns: &[usize]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls().cast(&DataType::Float64)?;
    let x: Vec<f64> = s
        .as_materialized_series()
        .f64()?
        .into_no_null_iter()
        .collect();
    let counts: Vec<f64> = if x.is_empty() {
        vec![f64::NAN; ns.len()]
    } else {
        peak_counts(&x, ns).into_iter().map(|c| c as f64).collect()
    };
    let columns = ns
        .iter()
        .zip(counts)
        .map(|(n, c)| Column::new(format!("number_peaks__n_{n}").into(), &[c]))
        .collect::<Vec<_>>();
    Ok(DataFrame::new(1, columns)?
        .into_struct("number_peaks".into())
        .into_column())
}

/// Number of peaks for several supports `ns` at once, sharing one pass over
/// each group. See [`number_peaks`].
pub fn number_peaks_multi(name: &str, ns: Vec<usize>) -> Expr {
    let fields = ns
        .iter()
        .map(|n| Field::new(format!("number_peaks__n_{n}").into(), DataType::Float64))
        .collect::<Vec<_>>();
    let new_names = ns
        .iter()
        .map(|n| format!("{name}__number_peaks__n_{n}"))
        .collect::<Vec<_>>();
    col(name)
        .apply(
            move |s| _number_peaks_all(s, &ns),
            move |_, _| Ok(Field::new("".into(), DataType::Struct(fields.clone()))),
        )
        .struct_()
        .rename_fields(new_names)
        .get(0, true)
        .alias(format!("{name}__number_peaks"))
}
