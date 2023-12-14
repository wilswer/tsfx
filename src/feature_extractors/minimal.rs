use ndarray::Array1;
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};
use polars::prelude::*;

pub fn minimal_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    aggregators.push(count(&value_cols[0]));
    for col in value_cols {
        aggregators.push(sum(col));
        aggregators.push(mean(col));
        aggregators.push(expr_median(col));
        aggregators.push(minimum(col));
        aggregators.push(maximum(col));
        aggregators.push(std(col));
        aggregators.push(var(col));
        aggregators.push(skewness(col));
        aggregators.push(root_mean_square(col));
        aggregators.push(absolute_maximum(col));
    }
    aggregators
}

pub fn count(name: &str) -> Expr {
    col(name).count().alias("length")
}

fn _sum(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let sum = arr.sum();
    let s = Series::new("", &[sum]);
    Ok(Some(s))
}

pub fn sum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_sum, o)
        .get(0)
        .alias(&format!("{}_sum", name))
}

pub fn expr_sum(name: &str) -> Expr {
    col(name).sum().alias(&format!("{}_sum", name))
}

fn _mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let mean = arr.mean();
    let s = Series::new("", &[mean]);
    Ok(Some(s))
}

pub fn mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_mean, o)
        .get(0)
        .alias(&format!("{}_mean", name))
}

pub fn expr_mean(name: &str) -> Expr {
    col(name).mean().alias(&format!("{}_mean", name))
}

fn _min(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let min = arr.min().unwrap();
    let s = Series::new("", &[*min]);
    Ok(Some(s))
}

pub fn minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_min, o)
        .get(0)
        .alias(&format!("{}_min", name))
}

pub fn expr_minimum(name: &str) -> Expr {
    col(name).min().alias(&format!("{}_min", name))
}

fn _max(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let max = arr.max().unwrap();
    let s = Series::new("", &[*max]);
    Ok(Some(s))
}

pub fn maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_max, o)
        .get(0)
        .alias(&format!("{}_max", name))
}

pub fn expr_maximum(name: &str) -> Expr {
    col(name).max().alias(&format!("{}_max", name))
}

fn _abs_max(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let abs_arr: Array1<f32> = arr.iter().map(|x| x.abs()).collect::<Vec<f32>>().into();
    let abs_max = *abs_arr.max().unwrap();
    let s = Series::new("", &[abs_max]);
    Ok(Some(s))
}

pub fn absolute_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_abs_max, o)
        .get(0)
        .alias(&format!("{}_abs_max", name))
}

pub fn expr_median(name: &str) -> Expr {
    col(name).median().alias(&format!("{}_median", name))
}

fn _std(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let std = arr.std(1.0);
    let s = Series::new("", &[std]);
    Ok(Some(s))
}

pub fn std(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_std, o)
        .get(0)
        .alias(&format!("{}_std", name))
}

pub fn expr_std(name: &str) -> Expr {
    col(name).std(1).alias(&format!("{}_std", name))
}

fn _var(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let var = arr.var(1.0);
    let s = Series::new("", &[var]);
    Ok(Some(s))
}

pub fn var(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_var, o)
        .get(0)
        .alias(&format!("{}_var", name))
}

pub fn expr_var(name: &str) -> Expr {
    col(name).var(1).alias(&format!("{}_var", name))
}

fn _rms(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let rms = arr.mapv(|x| x.powi(2)).mean().map(f32::sqrt).unwrap();
    let s = Series::new("", &[rms]);
    Ok(Some(s))
}

pub fn root_mean_square(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_rms, o)
        .get(0)
        .alias(&format!("{}_root_mean_sqaure", name))
}

pub fn expr_root_mean_square(name: &str) -> Expr {
    col(name)
        .pow(2.0)
        .mean()
        .pow(0.5)
        .alias(&format!("{}_rms", name))
}

pub fn expr_skewness(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let std = col(name).std(1);
    let skewness = ((col(name) - mean).pow(3)).sum() / ((n - lit(1.0)) * std.pow(3));
    skewness.alias(&format!("{}_expr_skewness", name))
}

fn _skewness(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let skewness = arr.skewness().unwrap();
    let s = Series::new("", &[skewness]);
    Ok(Some(s))
}

pub fn skewness(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_skewness, o)
        .get(0)
        .alias(&format!("{}_skewness", name))
}
