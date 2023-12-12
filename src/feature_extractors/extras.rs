use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{s, Axis, Ix1};
use ndarray_stats::SummaryStatisticsExt;
use polars::{prelude::*, series::ops::NullBehavior};

pub fn extra_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    for col in value_cols {
        aggregators.push(kurtosis(col));
        aggregators.push(abs_energy(col));
        aggregators.push(mean_change(col));
        aggregators.push(linear_fit_intercept(col));
        aggregators.push(linear_fit_slope(col));
    }
    aggregators
}

fn _abs_energy(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let abs_energy = arr.mapv(|x| x.powi(2)).sum();
    let s = Series::new("", &[abs_energy]).into_series();
    Ok(Some(s))
}

pub fn abs_energy(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_abs_energy, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_abs_energy", name))
}

pub fn expr_abs_energy(name: &str) -> Expr {
    col(name)
        .pow(2)
        .sum()
        .alias(&format!("{}_abs_energy", name))
}

pub fn test_sum(name: &str) -> Expr {
    col(name).sum().alias(&format!("{}_test_sum", name))
}

pub fn test_mean(name: &str) -> Expr {
    let n = col(name).count();
    let s = col(name).sum();
    (s / n).alias(&format!("{}_test_mean", name))
}

fn _mean_change(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let mean_change = diffs.mean().unwrap();
    let s = Series::new("", &[mean_change]).into_series();
    Ok(Some(s))
}

pub fn mean_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_mean_change, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_mean_change", name))
}

pub fn expr_mean_change(name: &str) -> Expr {
    let diffs = col(name).diff(1, NullBehavior::Drop);
    let n = col(name).count() - lit(1);
    (diffs.sum() / n).alias(&format!("{}_mean_change", name))
}

fn _ndarray_sum(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let sum: f32 = arr.sum();
    let s = Series::new("", &[sum]).into_series();
    Ok(Some(s))
}

pub fn ndarray_sum(name: &str, out_type: DataType) -> Expr {
    let o = GetOutput::from_type(out_type);
    col(name)
        .apply(_ndarray_sum, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_ndarray_sum", name))
}

pub fn expr_kurtosis(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let std = col(name).std(1);
    let skewness = ((col(name) - mean).pow(4)).sum() / ((n - lit(1.0)) * std.pow(4));
    skewness.alias(&format!("{}_expr_kurtosis", name))
}

fn _kurtosis(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let kurtosis = arr.kurtosis().unwrap();
    let s = Series::new("", &[kurtosis]).into_series();
    Ok(Some(s))
}

pub fn kurtosis(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_kurtosis, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_kurtosis", name))
}

fn _linear_fit_intercept(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let x = ndarray::Array::range(0., arr.len() as f32, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, arr);
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset).unwrap();
    let s_i = Series::new("intercept", &[model.intercept()]).into_series();
    //let s_p = Series::new("param", &[model.params()[0]]).into_series();
    Ok(Some(s_i))
}

pub fn linear_fit_intercept(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_linear_fit_intercept, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_linear_fit_intercept", name))
}

fn _linear_fit_slope(s: Series) -> Result<Option<Series>, PolarsError> {
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let x = ndarray::Array::range(0., arr.len() as f32, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, arr);
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset).unwrap();
    let s_p = Series::new("", &[model.params()[0]]).into_series();
    Ok(Some(s_p))
}

pub fn linear_fit_slope(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_linear_fit_slope, o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}_linear_fit_slope", name))
}
