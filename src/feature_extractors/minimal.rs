use ndarray_stats::QuantileExt;
use polars::prelude::*;

use crate::utils::stats::{population_std, population_var, sample_skewness};
use crate::{extract::ExtractionSettings, utils::toml_reader::load_config};

pub fn minimal_aggregators(opts: &ExtractionSettings) -> Vec<Expr> {
    let config = match &opts.config_path {
        Some(file) => load_config(Some(file.as_str())),
        None => load_config(None),
    };
    let mut aggregators = Vec::new();
    if config.length.is_some() {
        aggregators.push(count(&opts.value_cols[0]));
    }
    for col in &opts.value_cols {
        if config.sum_values.is_some() {
            aggregators.push(sum_values(col));
        }
        if config.mean.is_some() {
            aggregators.push(mean(col));
        }
        if config.median.is_some() {
            aggregators.push(expr_median(col));
        }
        if config.minimum.is_some() {
            aggregators.push(minimum(col));
        }
        if config.maximum.is_some() {
            aggregators.push(maximum(col));
        }
        if config.standard_deviation.is_some() {
            aggregators.push(standard_deviation(col));
        }
        if config.variance.is_some() {
            aggregators.push(variance(col));
        }
        if config.skewness.is_some() {
            aggregators.push(skewness(col));
        }
        if config.root_mean_square.is_some() {
            aggregators.push(root_mean_square(col));
        }
    }
    aggregators
}

/// Length feature.
///
/// The length of the time series
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

fn _out(_: &Schema, _: &Field) -> Result<Field, PolarsError> {
    Ok(Field::new("".into(), DataType::Float64))
}

/// Sum of values feature.
///
/// The sum of all values in the time series
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
/// The mean of all values in the time series, where mean $\mu$ is
/// $$ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i, $$
/// where $n$ is the number of values in the time series
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
/// The minimum value in the time series
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
/// The maximum value in the time series
pub fn maximum(name: &str) -> Expr {
    col(name)
        .apply(_max, |_, _| Ok(Field::new("".into(), DataType::Float64)))
        .get(0, true)
        .alias(format!("{}__maximum", name))
}

/// Median feature.
///
/// The median of all values in the time series, using the native Polars API
pub fn expr_median(name: &str) -> Expr {
    col(name)
        .median()
        .cast(DataType::Float64)
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
/// The standard deviation of all values in the time series, where the standard deviation $\sigma$ is
/// $$ \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}, $$
/// where $n$ is the number of values in the time series and $\mu$ is the mean of the time series
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
/// The variance of all values in the time series, where the variance $\sigma^2$ is
/// $$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2, $$
/// where $n$ is the number of values in the time Column and $\mu$ is the mean of the time Column
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
/// The root mean square of all values in the time series, where the root mean square (RMS) is
/// $$ \text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}, $$
/// where $n$ is the number of values in the time Column
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
/// The bias-adjusted sample skewness (Fisher-Pearson $G_1$) of all values in the time series,
/// matching tsfresh (`pandas.Series.skew`):
/// $$ G_1 = \frac{n \sqrt{n - 1}}{n - 2} \frac{m_3}{m_2^{3/2}}, \quad m_k = \sum_{i=1}^{n} (x_i - \mu)^k, $$
/// where $n$ is the number of values in the time series and $\mu$ is its mean.
/// NaN for fewer than 3 values, 0 for a constant series. See [`crate::utils::stats::sample_skewness`].
pub fn skewness(name: &str) -> Expr {
    col(name)
        .apply(_skewness, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__skewness", name))
}
