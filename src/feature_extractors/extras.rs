use ndarray_stats::SummaryStatisticsExt;
use polars::{prelude::*, series::ops::NullBehavior};

pub fn extra_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    for col in value_cols {
        aggregators.push(expr_kurtosis(col));
        aggregators.push(kurtosis(col));
        aggregators.push(abs_energy(col));
        aggregators.push(mean_change(col));
        aggregators.push(test_sum(col));
        aggregators.push(test_mean(col));
        aggregators.push(ndarray_sum(col, DataType::Float32));
    }
    aggregators
}

pub fn abs_energy(name: &str) -> Expr {
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

pub fn mean_change(name: &str) -> Expr {
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
