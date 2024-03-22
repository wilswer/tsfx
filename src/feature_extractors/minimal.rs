use ndarray_stats::{QuantileExt, SummaryStatisticsExt};
use polars::prelude::*;

pub fn minimal_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    aggregators.push(count(&value_cols[0]));
    for col in value_cols {
        aggregators.push(sum_values(col));
        aggregators.push(mean(col));
        aggregators.push(expr_median(col));
        aggregators.push(minimum(col));
        aggregators.push(maximum(col));
        aggregators.push(standard_deviation(col));
        aggregators.push(variance(col));
        aggregators.push(skewness(col));
        aggregators.push(root_mean_square(col));
    }
    aggregators
}

pub fn count(name: &str) -> Expr {
    col(name).count().alias("length")
}

fn _sum_values(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let sum = arr.sum();
    let s = Series::new("", &[sum]);
    Ok(Some(s))
}

pub fn sum_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_values, o)
        .get(0)
        .alias(&format!("{}__sum_values", name))
}

pub fn expr_sum(name: &str) -> Expr {
    col(name).sum().alias(&format!("{}__sum", name))
}

fn _mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean = arr.mean().unwrap_or(f64::NAN);
    let s = Series::new("", &[mean]);
    Ok(Some(s))
}

pub fn mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean, o)
        .get(0)
        .alias(&format!("{}__mean", name))
}

pub fn expr_mean(name: &str) -> Expr {
    col(name).mean().alias(&format!("{}__mean", name))
}

fn _min(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let min = arr.min().unwrap_or(&f64::NAN);
    let s = Series::new("", &[*min]);
    Ok(Some(s))
}

pub fn minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_min, o)
        .get(0)
        .alias(&format!("{}__minimum", name))
}

pub fn expr_minimum(name: &str) -> Expr {
    col(name).min().alias(&format!("{}__minimum", name))
}

fn _max(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let max = arr.max().unwrap_or(&f64::NAN);
    let s = Series::new("", &[*max]);
    Ok(Some(s))
}

pub fn maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_max, o)
        .get(0)
        .alias(&format!("{}__maximum", name))
}

pub fn expr_maximum(name: &str) -> Expr {
    col(name).max().alias(&format!("{}__maximum", name))
}

pub fn expr_median(name: &str) -> Expr {
    col(name)
        .median()
        .cast(DataType::Float64)
        .alias(&format!("{}__median", name))
}

fn _standard_deviation(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let standard_deviation = arr.std(1.0);
    let s = Series::new("", &[standard_deviation]);
    Ok(Some(s))
}

pub fn standard_deviation(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_standard_deviation, o)
        .get(0)
        .alias(&format!("{}__standard_deviation", name))
}

pub fn expr_standard_deviation(name: &str) -> Expr {
    col(name)
        .std(1)
        .alias(&format!("{}__standard_deviation", name))
}

fn _variance(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let variance = arr.var(1.0);
    let s = Series::new("", &[variance]);
    Ok(Some(s))
}

pub fn variance(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_variance, o)
        .get(0)
        .alias(&format!("{}__variance", name))
}

pub fn expr_variance(name: &str) -> Expr {
    col(name).var(1).alias(&format!("{}__variance", name))
}

fn _rms(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let rms = arr
        .mapv(|x| x.powi(2))
        .mean()
        .map(f64::sqrt)
        .unwrap_or(f64::NAN);
    let s = Series::new("", &[rms]);
    Ok(Some(s))
}

pub fn root_mean_square(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_rms, o)
        .get(0)
        .alias(&format!("{}__root_mean_sqaure", name))
}

pub fn expr_root_mean_square(name: &str) -> Expr {
    col(name)
        .pow(2.0)
        .mean()
        .pow(0.5)
        .alias(&format!("{}__rms", name))
}

pub fn expr_skewness(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let standard_deviation = col(name).std(1);
    let skewness = ((col(name) - mean).pow(3)).sum() / ((n - lit(1.0)) * standard_deviation.pow(3));
    skewness.alias(&format!("{}__expr_skewness", name))
}

fn _skewness(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let skewness = arr.skewness().unwrap_or(f64::NAN);
    let s = Series::new("", &[skewness]);
    Ok(Some(s))
}

pub fn skewness(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_skewness, o)
        .get(0)
        .alias(&format!("{}__skewness", name))
}
