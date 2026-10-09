//! Summaries of the value distribution, independent of order.

use anyhow::Result;
use itertools::Itertools;
use ndarray::{Array1, Axis, Ix1};
use ndarray_stats::QuantileExt;
use noisy_float::types::n64;
use ordered_float::OrderedFloat;
use polars::lazy::dsl::*;
use polars::prelude::*;

use super::{_make_nan_struct_column, _make_nan_struct_column_int};
use crate::utils::stats::{
    median_mut, population_std, population_var, sample_excess_kurtosis, sample_skewness,
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

fn _sum_values(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let sum = arr.sum();
    let s = Column::new("".into(), &[sum]);
    Ok(s)
}

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
pub fn sum_values(name: &str) -> Expr {
    col(name)
        .apply(_sum_values, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__sum_values", name))
}

fn _mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean = arr.mean().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[mean]);
    Ok(s)
}

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
pub fn mean(name: &str) -> Expr {
    col(name)
        .apply(_mean, |_, _| Ok(Field::new("".into(), DataType::Float64)))
        .get(0, true)
        .alias(format!("{}__mean", name))
}

fn _min(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let min = arr.min().unwrap_or(&f64::NAN);
    let s = Column::new("".into(), &[*min]);
    Ok(s)
}

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
pub fn minimum(name: &str) -> Expr {
    col(name)
        .apply(_min, |_, _| Ok(Field::new("".into(), DataType::Float64)))
        .get(0, true)
        .alias(format!("{}__minimum", name))
}

fn _max(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let max = arr.max().unwrap_or(&f64::NAN);
    let s = Column::new("".into(), &[*max]);
    Ok(s)
}

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
pub fn maximum(name: &str) -> Expr {
    col(name)
        .apply(_max, |_, _| Ok(Field::new("".into(), DataType::Float64)))
        .get(0, true)
        .alias(format!("{}__maximum", name))
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

fn _standard_deviation(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let standard_deviation = population_std(&arr.column(0));
    let s = Column::new("".into(), &[standard_deviation]);
    Ok(s)
}

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
pub fn standard_deviation(name: &str) -> Expr {
    col(name)
        .apply(_standard_deviation, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__standard_deviation", name))
}

fn _variance(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let variance = population_var(&arr.column(0));
    let s = Column::new("".into(), &[variance]);
    Ok(s)
}

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
pub fn variance(name: &str) -> Expr {
    col(name)
        .apply(_variance, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__variance", name))
}

fn _rms(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let rms = arr
        .mapv(|x| x.powi(2))
        .mean()
        .map(f64::sqrt)
        .unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[rms]);
    Ok(s)
}

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
pub fn root_mean_square(name: &str) -> Expr {
    col(name)
        .apply(_rms, |_, _| Ok(Field::new("".into(), DataType::Float64)))
        .get(0, true)
        .alias(format!("{}__root_mean_square", name))
}

fn _skewness(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let skewness = sample_skewness(&arr.column(0));
    let s = Column::new("".into(), &[skewness]);
    Ok(s)
}

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
pub fn skewness(name: &str) -> Expr {
    col(name)
        .apply(_skewness, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__skewness", name))
}

fn _absolute_energy(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let abs_energy = arr.mapv(|x| x.powi(2)).sum();
    let s = Column::new("".into(), &[abs_energy]);
    Ok(s)
}

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
pub fn absolute_energy(name: &str) -> Expr {
    col(name)
        .apply(_absolute_energy, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{name}__absolute_energy"))
}

fn _kurtosis(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // tsfresh calls pandas' kurtosis with skipna=True
    let arr = Array1::from_iter(arr.iter().copied().filter(|x| !x.is_nan()));
    let kurtosis = sample_excess_kurtosis(&arr.view());
    let s = Column::new("".into(), &[kurtosis]);
    Ok(s)
}

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
pub fn kurtosis(name: &str) -> Expr {
    col(name)
        .apply(_kurtosis, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__kurtosis", name))
}

fn _variance_larger_than_standard_deviation(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let var = population_var(&arr.column(0));
    let out = if var > var.sqrt() { 1.0 } else { 0.0 };
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn variance_larger_than_standard_deviation(name: &str) -> Expr {
    col(name)
        .apply(_variance_larger_than_standard_deviation, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__variance_larger_than_standard_deviation", name))
}

fn _ratio_beyond_r_sigma(s: Column, rs: &[f64]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("ratio_beyond_r_sigma", "r", rs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return _make_nan_struct_column("ratio_beyond_r_sigma", "r", rs),
    };
    let std = population_std(&arr.column(0));
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs {
        let count = arr
            .mapv(|x| if (x - mean).abs() > r * std { 1.0 } else { 0.0 })
            .sum();
        let ratio = count / arr.len() as f64;
        ss.push(Column::new(
            format!("ratio_beyond_r_sigma__r_{:2}", r).into(),
            &[ratio],
        ));
    }
    let s = DataFrame::new(1, ss)?
        .into_struct("ratio_beyond_r_sigma".into())
        .into_column();
    Ok(s)
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
pub fn ratio_beyond_r_sigma(name: &str, rs: Vec<f64>) -> Expr {
    let name = name.to_string();
    let mut new_field_names = Vec::with_capacity(rs.len());
    let mut struct_names = Vec::with_capacity(rs.len());
    for r in rs.iter() {
        new_field_names.push(format!("{}__ratio_beyond_r_sigma__r_{:.2}", name, r));
        struct_names.push(Field::new(
            format!("ratio_beyond_r_sigma__r_{:2}", r).into(),
            DataType::Float64,
        ));
    }
    col(&name)
        .apply(
            move |s| _ratio_beyond_r_sigma(s, &rs),
            move |_, _| {
                Ok(Field::new(
                    "".into(),
                    DataType::Struct(struct_names.clone()),
                ))
            },
        )
        .struct_()
        .rename_fields(new_field_names)
        .get(0, true)
        .alias(format!("{}__ratio_beyond_r_sigma", name))
}

fn _large_standard_deviation(s: Column, rs: &[f64]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("large_standard_deviation", "r", rs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let min = arr.min().unwrap_or(&0.0);
    let max = arr.max().unwrap_or(&0.0);
    let std = population_std(&arr.column(0));
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs {
        let out = std > r * (max - min);
        ss.push(Column::new(
            format!("large_standard_deviation__r_{:2}", r).into(),
            &[out as u8 as f64],
        ));
    }
    let s = DataFrame::new(1, ss)?
        .into_struct("large_standard_deviation".into())
        .into_column();
    Ok(s)
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
pub fn large_standard_deviation(name: &str, rs: Vec<f64>) -> Expr {
    let mut new_field_names = Vec::with_capacity(rs.len());
    let mut struct_names = Vec::with_capacity(rs.len());
    for r in rs.iter() {
        new_field_names.push(format!("{}__large_standard_deviation__r_{:.2}", name, r));
        struct_names.push(Field::new(
            format!("large_standard_deviation__r_{:2}", r).into(),
            DataType::Float64,
        ));
    }
    col(name)
        .apply(
            move |s| _large_standard_deviation(s, &rs),
            move |_, _| {
                Ok(Field::new(
                    "".into(),
                    DataType::Struct(struct_names.clone()),
                ))
            },
        )
        .struct_()
        .rename_fields(new_field_names)
        .get(0, true)
        .alias(format!("{}__large_standard_deviation", name))
}

fn _symmetry_looking(s: Column, rs: &[f64]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("symmetry_looking", "r", rs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    // tsfresh: every comparison with a NaN mean/median is False
    if arr.iter().any(|x| x.is_nan()) {
        let ss = rs
            .iter()
            .map(|r| Column::new(format!("symmetry_looking__r_{:2}", r).into(), &[0.0]))
            .collect::<Vec<_>>();
        return Ok(DataFrame::new(1, ss)?
            .into_struct("symmetry_looking".into())
            .into_column());
    }
    let mut arr = arr.mapv(n64);
    let median_res = median_mut(&mut arr);
    let median = match median_res {
        Ok(m) => f64::from(m),
        Err(_) => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => f64::from(m),
        None => return _make_nan_struct_column("symmetry_looking", "r", rs),
    };
    let mean_median_diff = (mean - median).abs();
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => f64::from(*m),
        Err(_) => return _make_nan_struct_column("symmetry_looking", "r", rs),
    };
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => f64::from(*m),
        Err(_) => return _make_nan_struct_column("symmetry_looking", "r", rs),
    };
    let max_min_diff = max - min;
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs {
        let out = mean_median_diff < r * max_min_diff;
        ss.push(Column::new(
            format!("symmetry_looking__r_{:2}", r).into(),
            &[out as u8 as f64],
        ));
    }
    let s = DataFrame::new(1, ss)?
        .into_struct("symmetry_looking".into())
        .into_column();
    Ok(s)
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
pub fn symmetry_looking(name: &str, rs: Vec<f64>) -> Expr {
    let mut new_field_names = Vec::with_capacity(rs.len());
    let mut struct_names = Vec::with_capacity(rs.len());
    for r in rs.iter() {
        new_field_names.push(format!("{}__symmetry_looking__r_{:.2}", name, r));
        struct_names.push(Field::new(
            format!("symmetry_looking__r_{:2}", r).into(),
            DataType::Float64,
        ));
    }
    col(name)
        .apply(
            move |s| _symmetry_looking(s, &rs),
            move |_, _| {
                Ok(Field::new(
                    "".into(),
                    DataType::Struct(struct_names.clone()),
                ))
            },
        )
        .struct_()
        .rename_fields(new_field_names)
        .get(0, true)
        .alias(format!("{}__symmetry_looking__r_", name))
}

fn _absolute_maximum(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let abs_arr = arr.mapv(|x| x.abs());
    let max_res = abs_arr.max();
    let max = match max_res {
        Ok(m) => *m,
        Err(_) => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let s = Column::new("".into(), &[max]);
    Ok(s)
}

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
pub fn absolute_maximum(name: &str) -> Expr {
    col(name)
        .apply(_absolute_maximum, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__absolute_maximum", name))
}

fn _variation_coefficient(s: Column) -> Result<Column, PolarsError> {
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
    let std = population_std(&arr.column(0));
    let out = if mean == 0.0 { f64::NAN } else { std / mean };
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
pub fn variation_coefficient(name: &str) -> Expr {
    col(name)
        .apply(_variation_coefficient, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__variation_coefficient", name))
}

fn _mean_n_absolute_max(s: Column, ns: &[usize]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column_int("mean_n_absolute_max", "n", ns);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(|x| -OrderedFloat::from(x.abs()));

    let mut ss: Vec<Column> = Vec::with_capacity(ns.len());
    let sarr = arr
        .iter()
        .k_smallest(
            *ns.iter()
                .max()
                .expect("mean_n_absolute_max parameters didn't have a maximum value..."),
        )
        .map(|x| -f64::from(*x))
        .collect::<Vec<f64>>();
    for n in ns {
        let out = if arr.len() < *n {
            f64::NAN
        } else {
            let _sarr = sarr.iter().take(*n);
            let sum_sarr = _sarr.sum::<f64>();
            sum_sarr / *n as f64
        };
        ss.push(Column::new(
            format!("mean_n_absolute_max__n_{}", n).into(),
            &[out],
        ));
    }
    let s = DataFrame::new(1, ss)?
        .into_struct("mean_n_absolute_max".into())
        .into_column();
    Ok(s)
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
pub fn mean_n_absolute_max(name: &str, ns: Vec<usize>) -> Expr {
    let mut new_field_names = Vec::with_capacity(ns.len());
    let mut struct_names = Vec::with_capacity(ns.len());
    for n in ns.iter() {
        new_field_names.push(format!("{}__mean_n_absolute_max__n_{}", name, n));
        struct_names.push(Field::new(
            format!("mean_n_absolute_max__n_{}", n).into(),
            DataType::Float64,
        ));
    }
    col(name)
        .apply(
            move |s| _mean_n_absolute_max(s, &ns),
            move |_, _| {
                Ok(Field::new(
                    "".into(),
                    DataType::Struct(struct_names.clone()),
                ))
            },
        )
        .struct_()
        .rename_fields(new_field_names)
        .get(0, true)
        .alias(format!("{}__mean_n_absolute_max", name))
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
