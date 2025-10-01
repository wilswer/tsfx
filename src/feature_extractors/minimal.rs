use ndarray_stats::{QuantileExt, SummaryStatisticsExt};
use polars::prelude::*;

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

fn _sum_values(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let sum = arr.sum();
    let s = Column::new("".into(), &[sum]);
    Ok(Some(s))
}

fn _out(_: &Schema, _: &Field) -> Result<Field, PolarsError> {
    Ok(Field::new("".into(), DataType::Float64))
}

/// Sum of values feature.
///
/// The sum of all values in the time series
pub fn sum_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_values, o)
        .get(0)
        .alias(format!("{}__sum_values", name))
}

/// The sum of all values of the time series, using the native Polars API
pub fn expr_sum(name: &str) -> Expr {
    col(name).sum().alias(format!("{}__sum", name))
}

fn _mean(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean = arr.mean().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[mean]);
    Ok(Some(s))
}

/// Mean feature.
///
/// The mean of all values in the time series, where mean $\mu$ is
/// $$ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i, $$
/// where $n$ is the number of values in the time series
pub fn mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean, o)
        .get(0)
        .alias(format!("{}__mean", name))
}

/// The mean of all values of the time series, using the native Polars API
pub fn expr_mean(name: &str) -> Expr {
    col(name).mean().alias(format!("{}__mean", name))
}

fn _min(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let min = arr.min().unwrap_or(&f64::NAN);
    let s = Column::new("".into(), &[*min]);
    Ok(Some(s))
}

/// Minimum feature.
///
/// The minimum value in the time series
pub fn minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_min, o)
        .get(0)
        .alias(format!("{}__minimum", name))
}

/// The minimum value of the time series, using the native Polars API
pub fn expr_minimum(name: &str) -> Expr {
    col(name).min().alias(format!("{}__minimum", name))
}

fn _max(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let max = arr.max().unwrap_or(&f64::NAN);
    let s = Column::new("".into(), &[*max]);
    Ok(Some(s))
}

/// Maximum feature.
///
/// The maximum value in the time series
pub fn maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_max, o)
        .get(0)
        .alias(format!("{}__maximum", name))
}

/// The maximum value of the time series, using the native Polars API
pub fn expr_maximum(name: &str) -> Expr {
    col(name).max().alias(format!("{}__maximum", name))
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

fn _standard_deviation(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let standard_deviation = arr.std(1.0);
    let s = Column::new("".into(), &[standard_deviation]);
    Ok(Some(s))
}

/// Standard deviation feature.
///
/// The standard deviation of all values in the time series, where the standard deviation $\sigma$ is
/// $$ \sigma = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \mu)^2}, $$
/// where $n$ is the number of values in the time series and $\mu$ is the mean of the time series
pub fn standard_deviation(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_standard_deviation, o)
        .get(0)
        .alias(format!("{}__standard_deviation", name))
}

/// The standard deviation of all values of the time series, using the native Polars API
pub fn expr_standard_deviation(name: &str) -> Expr {
    col(name)
        .std(1)
        .alias(format!("{}__standard_deviation", name))
}

fn _variance(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let variance = arr.var(1.0);
    let s = Column::new("".into(), &[variance]);
    Ok(Some(s))
}

/// Variance feature.
///
/// The variance of all values in the time series, where the variance $\sigma^2$ is
/// $$ \sigma^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \mu)^2, $$
/// where $n$ is the number of values in the time Column and $\mu$ is the mean of the time Column
pub fn variance(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_variance, o)
        .get(0)
        .alias(format!("{}__variance", name))
}

/// The variance of all values of the time series, using the native Polars API
pub fn expr_variance(name: &str) -> Expr {
    col(name).var(1).alias(format!("{}__variance", name))
}

fn _rms(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let rms = arr
        .mapv(|x| x.powi(2))
        .mean()
        .map(f64::sqrt)
        .unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[rms]);
    Ok(Some(s))
}

/// Root mean square feature.
///
/// The root mean square of all values in the time series, where the root mean square (RMS) is
/// $$ \text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}, $$
/// where $n$ is the number of values in the time Column
pub fn root_mean_square(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_rms, o)
        .get(0)
        .alias(format!("{}__root_mean_square", name))
}

/// The root mean square of all values of the time series, using the native Polars API
pub fn expr_root_mean_square(name: &str) -> Expr {
    col(name)
        .pow(2.0)
        .mean()
        .pow(0.5)
        .alias(format!("{}__rms", name))
}

/// The skewness of all values in the time series, using the native Polars API
pub fn expr_skewness(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let standard_deviation = col(name).std(1);
    let skewness = ((col(name) - mean).pow(3)).sum() / ((n - lit(1.0)) * standard_deviation.pow(3));
    skewness.alias(format!("{}__expr_skewness", name))
}

fn _skewness(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let skewness = arr.skewness().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[skewness]);
    Ok(Some(s))
}

/// Skewness feature.
///
/// The skewness of all values in the time series, where the skewness is the third standardized moment:
/// $$ \text{skewness} = \frac{1}{(n-1) \sigma^3} \sum_{i=1}^{n} (x_i - \mu)^3, $$
/// where $n$ is the number of values in the time series, $\mu$ is the mean of the time series,
/// and $\sigma$ is the standard deviation of the time Column
pub fn skewness(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_skewness, o)
        .get(0)
        .alias(format!("{}__skewness", name))
}
