//! Features about repeated values.

use anyhow::Result;
use itertools::Itertools;
use ndarray::{Axis, Ix1};
use ndarray_stats::QuantileExt;
use ordered_float::OrderedFloat;
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::count_unique;

fn _has_duplicate_max(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // tsfresh: np.max is NaN and x == NaN is never true, so the answer is 0
    if arr.iter().any(|x| x.is_nan()) {
        return Ok(Column::new("".into(), &[0.0]));
    }
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let count = arr.mapv(|x| if x == *max { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(s)
}

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
pub fn has_duplicate_max(name: &str) -> Expr {
    col(name)
        .apply(_has_duplicate_max, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__has_duplicate_max", name))
}

fn _has_duplicate_min(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // tsfresh: np.min is NaN and x == NaN is never true, so the answer is 0
    if arr.iter().any(|x| x.is_nan()) {
        return Ok(Column::new("".into(), &[0.0]));
    }
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let count = arr.mapv(|x| if x == *min { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(s)
}

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
pub fn has_duplicate_min(name: &str) -> Expr {
    col(name)
        .apply(_has_duplicate_min, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__has_duplicate_min", name))
}

fn _has_duplicate(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let out = count_unique(&arr.view()) < arr.len();
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(s)
}

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
pub fn has_duplicate(name: &str) -> Expr {
    col(name)
        .apply(_has_duplicate, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__has_duplicate", name))
}

fn _ratio_value_number_to_time_series_length(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let out = count_unique(&arr.view()) as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn ratio_value_number_to_time_series_length(name: &str) -> Expr {
    col(name)
        .apply(_ratio_value_number_to_time_series_length, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!(
            "{}__ratio_value_number_to_time_series_length",
            name
        ))
}

fn _sum_of_reoccurring_values(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut sum: f64 = 0.0;
    for (k, v) in counts {
        if v > 1 {
            let k: f64 = k.into();
            sum += k;
        }
    }
    let s = Column::new("".into(), &[sum]);
    Ok(s)
}

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
pub fn sum_of_reoccurring_values(name: &str) -> Expr {
    col(name)
        .apply(_sum_of_reoccurring_values, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__sum_of_reoccurring_values", name))
}

fn _sum_of_reoccurring_data_points(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut sum: f64 = 0.0;
    for (k, v) in counts {
        if v > 1 {
            let k: f64 = k.into();
            sum += (v as f64) * k;
        }
    }
    let s = Column::new("".into(), &[sum]);
    Ok(s)
}

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
pub fn sum_of_reoccurring_data_points(name: &str) -> Expr {
    col(name)
        .apply(_sum_of_reoccurring_data_points, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__sum_of_reoccurring_data_points", name))
}

fn _percentage_of_reoccurring_values_to_all_values(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut more_than_once = 0;
    for v in counts.values() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / counts.keys().len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn percentage_of_reoccurring_values_to_all_values(name: &str) -> Expr {
    col(name)
        .apply(_percentage_of_reoccurring_values_to_all_values, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!(
            "{}__percentage_of_reoccurring_values_to_all_values",
            name
        ))
}

fn _percentage_of_reoccurring_values_to_all_datapoints(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // tsfresh: datapoints whose value occurs more than once, over all
    // datapoints; NaN never counts as reoccurring (pandas value_counts)
    let counts = arr
        .iter()
        .filter(|x| !x.is_nan())
        .map(|x| OrderedFloat(*x))
        .counts();
    let reoccurring: usize = counts.values().filter(|&&c| c > 1).sum();
    let out = reoccurring as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn percentage_of_reoccurring_values_to_all_datapoints(name: &str) -> Expr {
    col(name)
        .apply(
            _percentage_of_reoccurring_values_to_all_datapoints,
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!(
            "{}__percentage_of_reoccurring_values_to_all_datapoints",
            name
        ))
}
