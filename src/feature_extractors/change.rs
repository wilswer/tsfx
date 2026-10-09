//! Features built from consecutive differences.

use anyhow::Result;
use ndarray::{Axis, Ix1, s};
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::population_std;

fn _mean_absolute_change(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let mean_abs_change = diffs.mapv(|x| x.abs()).mean().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[mean_abs_change]);
    Ok(s)
}

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
pub fn mean_absolute_change(name: &str) -> Expr {
    col(name)
        .apply(_mean_absolute_change, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__mean_absolute_change", name))
}

fn _cid_ce(s: Column, normalize: bool) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let arr = if normalize {
        let mean = arr.mean().unwrap_or(f64::NAN);
        let std = population_std(&arr.view());
        // tsfresh: a constant series has no complexity
        if std == 0.0 {
            return Ok(Column::new("".into(), &[0.0]));
        }
        (arr - mean) / std
    } else {
        arr
    };
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.powi(2)).sum().sqrt();
    let s = Column::new("".into(), &[out]);
    Ok(s)
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
pub fn cid_ce(name: &str, normalize: bool) -> Expr {
    col(name)
        .apply(
            move |s| _cid_ce(s, normalize),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__cid_ce__normalize_{:.1}", name, normalize))
}

fn _absolute_sum_of_changes(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.abs()).sum();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn absolute_sum_of_changes(name: &str) -> Expr {
    col(name)
        .apply(_absolute_sum_of_changes, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__absolute_sum_of_changes", name))
}

fn _mean_change(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let out = if arr.len() < 2 {
        f64::NAN
    } else {
        (arr[arr.len() - 1] - arr[0]) / ((arr.len() - 1) as f64)
    };
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn mean_change(name: &str) -> Expr {
    col(name)
        .apply(_mean_change, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__mean_change", name))
}
