use core::f64;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::{fmt::Display, str::FromStr};

use itertools::izip;
use itertools::Itertools;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::ArrayView1;
use ndarray::{s, Array, Array1, Axis, Ix1};
use ndarray_stats::errors::QuantileError;
use ndarray_stats::Quantile1dExt;
use ndarray_stats::{interpolate::Midpoint, QuantileExt, SummaryStatisticsExt};
use noisy_float::types::n64;
use num::FromPrimitive;
use ordered_float::OrderedFloat;
use polars::lazy::dsl::quantile;
use polars::{prelude::*, series::ops::NullBehavior};

pub fn extra_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    for col in value_cols {
        aggregators.push(kurtosis(col));
        aggregators.push(absolute_energy(col));
        aggregators.push(mean_absolute_change(col));
        aggregators.push(linear_trend_intercept(col));
        aggregators.push(linear_trend_slope(col));
        aggregators.push(variance_larger_than_standard_deviation(col));
        aggregators.push(ratio_beyond_r_sigma(col, 0.5));
        aggregators.push(ratio_beyond_r_sigma(col, 1.0));
        aggregators.push(ratio_beyond_r_sigma(col, 1.5));
        aggregators.push(ratio_beyond_r_sigma(col, 2.0));
        aggregators.push(ratio_beyond_r_sigma(col, 2.5));
        aggregators.push(ratio_beyond_r_sigma(col, 3.0));
        aggregators.push(ratio_beyond_r_sigma(col, 5.0));
        aggregators.push(ratio_beyond_r_sigma(col, 6.0));
        aggregators.push(ratio_beyond_r_sigma(col, 7.0));
        aggregators.push(ratio_beyond_r_sigma(col, 10.0));
        for r in ndarray::Array::range(0.05, 1.0, 0.05).iter() {
            aggregators.push(large_standard_deviation(col, *r));
        }
        for r in ndarray::Array::range(0.05, 1.0, 0.05).iter() {
            aggregators.push(symmetry_looking(col, *r));
        }
        aggregators.push(has_duplicate_max(col));
        aggregators.push(has_duplicate_min(col));
        aggregators.push(cid_ce(col, true));
        aggregators.push(cid_ce(col, false));
        aggregators.push(absolute_maximum(col));
        aggregators.push(absolute_sum_of_changes(col));
        aggregators.push(count_above_mean(col));
        aggregators.push(count_below_mean(col));
        aggregators.push(count_above(col, 0.0));
        aggregators.push(count_below(col, 0.0));
        aggregators.push(first_location_of_maximum(col));
        aggregators.push(first_location_of_minimum(col));
        aggregators.push(last_location_of_maximum(col));
        aggregators.push(last_location_of_minimum(col));
        aggregators.push(longest_strike_below_mean(col));
        aggregators.push(longest_strike_above_mean(col));
        aggregators.push(has_duplicate(col));
        aggregators.push(variation_coefficient(col));
        aggregators.push(mean_change(col));
        aggregators.push(ratio_value_number_to_time_series_length(col));
        aggregators.push(sum_of_reoccurring_values(col));
        aggregators.push(sum_of_reoccurring_data_points(col));
        aggregators.push(percentage_of_reoccurring_values_to_all_values(col));
        aggregators.push(percentage_of_reoccurring_values_to_all_datapoints(col));
        let chunk_sizes: [usize; 3] = [5, 10, 50];
        let chunk_aggs: [&str; 4] = ["mean", "min", "max", "var"];
        for ca in chunk_aggs.into_iter() {
            for chunk_size in chunk_sizes.into_iter() {
                aggregators.push(agg_linear_trend_intercept(
                    col,
                    chunk_size,
                    ChunkAggregator::from_str(ca).unwrap(),
                ));
            }
        }
        for ca in chunk_aggs.into_iter() {
            for chunk_size in chunk_sizes.into_iter() {
                aggregators.push(agg_linear_trend_slope(
                    col,
                    chunk_size,
                    ChunkAggregator::from_str(ca).unwrap(),
                ));
            }
        }
        aggregators.push(mean_n_absolute_max(col, 7));
        for r in 0..10 {
            aggregators.push(autocorrelation(col, r));
        }
        for q in ndarray::Array::range(0.1, 1.0, 0.1).into_iter() {
            if q == 0.5 {
                continue;
            }
            aggregators.push(expr_quantile(col, q));
        }
        aggregators.push(number_crossing_m(col, -1.0));
        aggregators.push(number_crossing_m(col, 0.0));
        aggregators.push(number_crossing_m(col, 1.0));
        aggregators.push(range_count(col, -1.0, 1.0));
        aggregators.push(range_count(col, -1_000_000_000_000.0, 0.0));
        aggregators.push(range_count(col, 0.0, 1_000_000_000_000.0));
        for q in ndarray::Array::range(0.1, 1.0, 0.1).into_iter() {
            aggregators.push(index_mass_quantile(col, q));
        }
        aggregators.push(c3(col, 1));
        aggregators.push(c3(col, 2));
        aggregators.push(c3(col, 3));
        aggregators.push(time_reversal_asymmetry_statistic(col, 1));
        aggregators.push(time_reversal_asymmetry_statistic(col, 2));
        aggregators.push(time_reversal_asymmetry_statistic(col, 3));
        aggregators.push(number_peaks(col, 1));
        aggregators.push(number_peaks(col, 3));
        aggregators.push(number_peaks(col, 5));
        aggregators.push(number_peaks(col, 10));
        aggregators.push(number_peaks(col, 50));
    }
    aggregators
}

#[derive(Debug, PartialEq, Clone)]
pub enum ChunkAggregator {
    Mean,
    Min,
    Max,
    Var,
}

impl FromStr for ChunkAggregator {
    type Err = ();

    fn from_str(input: &str) -> Result<ChunkAggregator, Self::Err> {
        match input {
            "mean" => Ok(ChunkAggregator::Mean),
            "min" => Ok(ChunkAggregator::Min),
            "max" => Ok(ChunkAggregator::Max),
            "var" => Ok(ChunkAggregator::Var),
            _ => Err(()),
        }
    }
}

impl Display for ChunkAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ChunkAggregator::Mean => write!(f, "mean"),
            ChunkAggregator::Max => write!(f, "max"),
            ChunkAggregator::Min => write!(f, "min"),
            ChunkAggregator::Var => write!(f, "var"),
        }
    }
}

/// Return the median. Sorts its argument in place.
fn _median_mut<T>(xs: &mut Array1<T>) -> Result<T, QuantileError>
where
    T: Clone + Copy + Ord + FromPrimitive,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T>,
{
    if false {
        // quantile_mut may fail with the error: fatal runtime error: stack overflow
        // See https://github.com/rust-ndarray/ndarray-stats/issues/86
        xs.quantile_mut(n64(0.5), &Midpoint)
    } else {
        if xs.is_empty() {
            return Err(QuantileError::EmptyInput);
        }
        xs.as_slice_mut().unwrap().sort_unstable();
        Ok(if xs.len() % 2 == 0 {
            (xs[xs.len() / 2] + xs[xs.len() / 2 - 1]) / (T::from_u64(2).unwrap())
        } else {
            xs[xs.len() / 2]
        })
    }
}

fn _get_length_sequences_where(x: &ndarray::Array1<bool>) -> Vec<usize> {
    let mut group_lengths = Vec::new();
    for (key, group) in &x.into_iter().group_by(|elt| *elt) {
        if *key {
            group_lengths.push(group.count());
        }
    }
    group_lengths
}

fn _aggregate_on_chunks(
    x: Array1<f64>,
    chunk_size: usize,
    aggregator: impl Fn(Array1<f64>) -> f64,
) -> Array1<f64> {
    let mut agg_arr = Vec::with_capacity(x.len().div_ceil(chunk_size));
    for chunk in x.axis_chunks_iter(Axis(0), chunk_size) {
        agg_arr.push(aggregator(chunk.to_owned()));
    }
    Array1::from_vec(agg_arr)
}

fn _roll(x: &mut [f64], shift: isize) -> &[f64] {
    if shift > 0 {
        x.rotate_right(shift as usize);
    } else {
        x.rotate_left(shift.unsigned_abs());
    }
    x
}

fn _absolute_energy(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let abs_energy = arr.mapv(|x| x.powi(2)).sum();
    let s = Series::new("", &[abs_energy]);
    Ok(Some(s))
}

pub fn absolute_energy(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_energy, o)
        .get(0)
        .alias(&format!("{}__absolute_energy", name))
}

pub fn expr_abs_energy(name: &str) -> Expr {
    col(name)
        .pow(2)
        .sum()
        .alias(&format!("{}__absolute_energy", name))
}

pub fn test_sum(name: &str) -> Expr {
    col(name).sum().alias(&format!("{}__test_sum", name))
}

pub fn test_mean(name: &str) -> Expr {
    let n = col(name).count();
    let s = col(name).sum();
    (s / n).alias(&format!("{}__test_mean", name))
}

fn _mean_absolute_change(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let mean_abs_change = diffs.mapv(|x| x.abs()).mean().unwrap_or(f64::NAN);
    let s = Series::new("", &[mean_abs_change]);
    Ok(Some(s))
}

pub fn mean_absolute_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean_absolute_change, o)
        .get(0)
        .alias(&format!("{}__mean_absolute_change", name))
}

pub fn expr_mean_change(name: &str) -> Expr {
    let diffs = col(name).diff(1, NullBehavior::Drop);
    let n = col(name).count() - lit(1);
    (diffs.sum() / n).alias(&format!("{}__mean_change", name))
}

fn _ndarray_sum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let sum: f64 = arr.sum();
    let s = Series::new("", &[sum]);
    Ok(Some(s))
}

pub fn ndarray_sum(name: &str, out_type: DataType) -> Expr {
    let o = GetOutput::from_type(out_type);
    col(name)
        .apply(_ndarray_sum, o)
        .get(0)
        .alias(&format!("{}__ndarray_sum", name))
}

pub fn expr_kurtosis(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let std = col(name).std(1);
    let skewness = ((col(name) - mean).pow(4)).sum() / ((n - lit(1.0)) * std.pow(4));
    skewness.alias(&format!("{}__expr_kurtosis", name))
}

fn _kurtosis(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let kurtosis = arr.kurtosis().unwrap_or(f64::NAN);
    let s = Series::new("", &[kurtosis]);
    Ok(Some(s))
}

pub fn kurtosis(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_kurtosis, o)
        .get(0)
        .alias(&format!("{}__kurtosis", name))
}

fn _linear_trend_intercept(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let x = ndarray::Array::range(0., arr.len() as f64, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, arr);
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_i = Series::new("", &[model.intercept()]);
            Ok(Some(s_i))
        }
        Err(_) => Ok(Some(Series::new_empty("", &DataType::Float64))),
    }
}

pub fn linear_trend_intercept(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_linear_trend_intercept, o)
        .get(0)
        .alias(&format!("{}__linear_trend_intercept", name))
}

fn _linear_trend_slope(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let x = ndarray::Array::range(0., arr.len() as f64, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, arr);
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_p = Series::new("", &[model.params()[0]]);
            Ok(Some(s_p))
        }
        Err(_) => Ok(Some(Series::new_empty("", &DataType::Float64))),
    }
}

pub fn linear_trend_slope(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_linear_trend_slope, o)
        .get(0)
        .alias(&format!("{}__linear_trend_slope", name))
}

// fn _linear_trend(s: Series) -> Result<DataFrame, PolarsError> {
//     let arr = s
//         .into_frame()
//         .to_ndarray::<Float64Type>(IndexOrder::C)
//         .unwrap();
//     let arr = arr
//         .remove_axis(Axis(1))
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     let x = ndarray::Array::range(0., arr.len() as f64, 1.);
//     let x = x.insert_axis(Axis(1));
//     let dataset = Dataset::new(x, arr);
//     let lin_reg = LinearRegression::new();
//     let model = lin_reg.fit(&dataset).unwrap();
//     let s_i = Series::new("intercept", &[model.intercept()]);
//     let s_p = Series::new("slope", &[model.params()[0]]);
//     DataFrame::new(vec![s_i, s_p])
// }
//
// pub fn linear_trend(name: &str) -> Expr {
//     col(name)
//         .apply(_linear_trend, None)
//         .cast(DataType::Float64)
//         .get(0)
//         .alias(&format!("{}__linear_trend", name))
// }

pub fn variance_larger_than_standard_deviation(name: &str) -> Expr {
    let std = col(name).std(1);
    let var = col(name).var(1);
    (var.gt(std)).cast(DataType::Float64).alias(&format!(
        "{}__variance_larger_than_standard_deviation",
        name
    ))
}

fn _ratio_beyond_r_sigma(s: Series, r: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let std = arr.std(1.0);
    let count = arr
        .mapv(|x| if (x - mean).abs() > r * std { 1.0 } else { 0.0 })
        .sum();
    let ratio = count / arr.len() as f64;
    let s = Series::new("", &[ratio]);
    Ok(Some(s))
}

pub fn ratio_beyond_r_sigma(name: &str, r: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _ratio_beyond_r_sigma(s, r), o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__ratio_beyond_r_sigma__r_{:.1}", name, r))
}

fn _large_standard_deviation(s: Series, r: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let min = arr.min().unwrap_or(&0.0);
    let max = arr.max().unwrap_or(&0.0);
    let std = arr.std(1.0);
    let out = std > r * (max - min);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn large_standard_deviation(name: &str, r: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(move |s| _large_standard_deviation(s, r), o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__large_standard_deviation__r_{:.2}", name, r))
}

fn _symmetry_looking(s: Series, r: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mut arr = arr.mapv(n64);
    let median_res = _median_mut(&mut arr);
    let median = match median_res {
        Ok(m) => f64::from(m),
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => f64::from(m),
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let mean_median_diff = (mean - median).abs();
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => f64::from(*m),
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => f64::from(*m),
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let max_min_diff = max - min;
    let out = mean_median_diff < r * max_min_diff;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn symmetry_looking(name: &str, r: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(move |s| _symmetry_looking(s, r), o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__symmetry_looking__r_{:.2}", name, r))
}

fn _has_duplicate_max(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let count = arr.mapv(|x| if x == *max { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn has_duplicate_max(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(_has_duplicate_max, o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__has_duplicate_max", name))
}

fn _has_duplicate_min(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let count = arr.mapv(|x| if x == *min { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn has_duplicate_min(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(_has_duplicate_min, o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__has_duplicate_min", name))
}

fn _cid_ce(s: Series, normalize: bool) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let arr = if normalize {
        let mean = arr.mean().unwrap_or(f64::NAN);
        let std = arr.std(1.0);
        (arr - mean) / std
    } else {
        arr
    };
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.powi(2)).sum().sqrt();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn cid_ce(name: &str, normalize: bool) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _cid_ce(s, normalize), o)
        .get(0)
        .alias(&format!("{}__cid_ce__normalize_{:.1}", name, normalize))
}

fn _absolute_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let abs_arr = arr.mapv(|x| x.abs());
    let max_res = abs_arr.max();
    let max = match max_res {
        Ok(m) => *m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let s = Series::new("", &[max]);
    Ok(Some(s))
}

pub fn absolute_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_maximum, o)
        .get(0)
        .alias(&format!("{}__absolute_maximum", name))
}

fn _absolute_sum_of_changes(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.abs()).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn absolute_sum_of_changes(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_sum_of_changes, o)
        .get(0)
        .alias(&format!("{}__absolute_sum_of_changes", name))
}

fn _count_above_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = arr.mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_count_above_mean, o)
        .get(0)
        .alias(&format!("{}__count_above_mean", name))
}

fn _count_below_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = arr.mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_count_below_mean, o)
        .get(0)
        .alias(&format!("{}__count_below_mean", name))
}

fn _count_above(s: Series, t: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_above(name: &str, t: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _count_above(s, t), o)
        .get(0)
        .alias(&format!("{}__count_above__t_{:.1}", name, t))
}

fn _count_below(s: Series, t: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_below(name: &str, t: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _count_below(s, t), o)
        .get(0)
        .alias(&format!("{}__count_below__t_{:.1}", name, t))
}

fn _first_location_of_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = max as f64 / arr.len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn first_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_first_location_of_maximum, o)
        .get(0)
        .alias(&format!("{}__first_location_of_maximum", name))
}

fn _first_location_of_minimum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = min as f64 / arr.len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn first_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_first_location_of_minimum, o)
        .get(0)
        .alias(&format!("{}__first_location_of_minimum", name))
}

fn _last_location_of_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = 1.0 - (max as f64 / arr.len() as f64);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn last_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_last_location_of_maximum, o)
        .get(0)
        .alias(&format!("{}__last_location_of_maximum", name))
}

fn _last_location_of_minimum(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let out = 1.0 - (min as f64 / arr.len() as f64);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn last_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_last_location_of_minimum, o)
        .get(0)
        .alias(&format!("{}__last_location_of_minimum", name))
}

fn _longest_strike_below_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let bool_arr = arr.mapv(|x| x < mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Series::new("", &[out as f64]);
    Ok(Some(s))
}

pub fn longest_strike_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_longest_strike_below_mean, o)
        .get(0)
        .alias(&format!("{}__longest_strike_below_mean", name))
}

fn _longest_strike_above_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let bool_arr = arr.mapv(|x| x > mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Series::new("", &[out as f64]);
    Ok(Some(s))
}

pub fn longest_strike_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_longest_strike_above_mean, o)
        .get(0)
        .alias(&format!("{}__longest_strike_above_mean", name))
}

fn _has_duplicate(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let sarr = arr
        .as_slice()
        .unwrap()
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .collect::<Vec<_>>();
    let len = if sarr.is_empty() {
        0
    } else {
        1 + sarr.windows(2).filter(|win| win[0] != win[1]).count()
    };
    let out = len < arr.len();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn has_duplicate(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(_has_duplicate, o)
        .cast(DataType::Float64)
        .get(0)
        .alias(&format!("{}__has_duplicate", name))
}

fn _variation_coefficient(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let std = arr.std(1.0);
    let out = if mean == 0.0 { f64::NAN } else { std / mean };
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn variation_coefficient(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_variation_coefficient, o)
        .get(0)
        .alias(&format!("{}__variation_coefficient", name))
}

fn _mean_change(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let out = if arr.len() < 2 {
        f64::NAN
    } else {
        (arr[arr.len() - 1] - arr[0]) / ((arr.len() - 1) as f64)
    };
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn mean_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean_change, o)
        .get(0)
        .alias(&format!("{}__mean_change", name))
}

fn _ratio_value_number_to_time_series_length(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let sarr = arr
        .as_slice()
        .unwrap()
        .iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .collect::<Vec<_>>();
    let len_unique = if sarr.is_empty() {
        0
    } else {
        1 + sarr.windows(2).filter(|win| win[0] != win[1]).count()
    };
    let out = len_unique as f64 / arr.len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn ratio_value_number_to_time_series_length(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_ratio_value_number_to_time_series_length, o)
        .get(0)
        .alias(&format!(
            "{}__ratio_value_number_to_time_series_length",
            name
        ))
}

fn _sum_of_reoccurring_values(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut sum: f64 = 0.0;
    for (k, v) in counts {
        if v > 1 {
            let k: f64 = k.into();
            sum += k;
        }
    }
    let s = Series::new("", &[sum]);
    Ok(Some(s))
}

pub fn sum_of_reoccurring_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_of_reoccurring_values, o)
        .get(0)
        .alias(&format!("{}__sum_of_reoccurring_values", name))
}

fn _sum_of_reoccurring_data_points(s: Series) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut sum: f64 = 0.0;
    for (k, v) in counts {
        if v > 1 {
            let k: f64 = k.into();
            sum += (v as f64) * k;
        }
    }
    let s = Series::new("", &[sum]);
    Ok(Some(s))
}

pub fn sum_of_reoccurring_data_points(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_of_reoccurring_data_points, o)
        .get(0)
        .alias(&format!("{}__sum_of_reoccurring_data_points", name))
}

fn _percentage_of_reoccurring_values_to_all_values(
    s: Series,
) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut more_than_once = 0;
    for (_, v) in counts.iter() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / counts.keys().len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn percentage_of_reoccurring_values_to_all_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_percentage_of_reoccurring_values_to_all_values, o)
        .get(0)
        .alias(&format!(
            "{}__percentage_of_reoccurring_values_to_all_values",
            name
        ))
}

fn _percentage_of_reoccurring_values_to_all_datapoints(
    s: Series,
) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.iter().counts();
    let mut more_than_once = 0;
    for (_, v) in counts.iter() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / arr.len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn percentage_of_reoccurring_values_to_all_datapoints(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_percentage_of_reoccurring_values_to_all_datapoints, o)
        .get(0)
        .alias(&format!(
            "{}__percentage_of_reoccurring_values_to_all_datapoints",
            name
        ))
}

fn _agg_linear_trend_intercept(
    s: Series,
    chunk_size: usize,
    aggregator: ChunkAggregator,
) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if s.len() < chunk_size {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let agg_arr = match aggregator {
        ChunkAggregator::Mean => _aggregate_on_chunks(arr, chunk_size, |x| x.mean().unwrap()),
        ChunkAggregator::Max => _aggregate_on_chunks(arr, chunk_size, |x| *x.max().unwrap()),
        ChunkAggregator::Min => _aggregate_on_chunks(arr, chunk_size, |x| *x.min().unwrap()),
        ChunkAggregator::Var => _aggregate_on_chunks(arr, chunk_size, |x| x.var(1.0)),
    };
    let x = Array::range(0., agg_arr.len() as f64, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, agg_arr);
    let linreg = LinearRegression::new();
    let model = linreg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_i = Series::new("", &[model.intercept()]);
            Ok(Some(s_i))
        }
        Err(_) => Ok(Some(Series::new("", &[f64::NAN]))),
    }
}

fn agg_linear_trend_intercept(name: &str, chunk_size: usize, aggregator: ChunkAggregator) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    let agg_clone = aggregator.clone();
    col(name)
        .apply(
            move |s| _agg_linear_trend_intercept(s, chunk_size, aggregator.clone()),
            o,
        )
        .get(0)
        .alias(&format!(
            "{}__agg_linear_trend_intercept__chunk_size_{:.1}__agg_{:.1}",
            name, chunk_size, agg_clone
        ))
}

fn _agg_linear_trend_slope(
    s: Series,
    chunk_size: usize,
    aggregator: ChunkAggregator,
) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if s.len() < chunk_size {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let agg_arr = match aggregator {
        ChunkAggregator::Mean => _aggregate_on_chunks(arr, chunk_size, |x| x.mean().unwrap()),
        ChunkAggregator::Max => _aggregate_on_chunks(arr, chunk_size, |x| *x.max().unwrap()),
        ChunkAggregator::Min => _aggregate_on_chunks(arr, chunk_size, |x| *x.min().unwrap()),
        ChunkAggregator::Var => _aggregate_on_chunks(arr, chunk_size, |x| x.var(1.0)),
    };
    let x = Array::range(0., agg_arr.len() as f64, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, agg_arr);
    let linreg = LinearRegression::new();
    let model = linreg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_i = Series::new("", &[model.params()[0]]);
            Ok(Some(s_i))
        }
        Err(_) => Ok(Some(Series::new("", &[f64::NAN]))),
    }
}

fn agg_linear_trend_slope(name: &str, chunk_size: usize, aggregator: ChunkAggregator) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    let agg_clone = aggregator.clone();
    col(name)
        .apply(
            move |s| _agg_linear_trend_slope(s, chunk_size, aggregator.clone()),
            o,
        )
        .get(0)
        .alias(&format!(
            "{}__agg_linear_trend_slope__chunk_size_{:.1}__agg_{:.1}",
            name, chunk_size, agg_clone
        ))
}

fn _mean_n_absolute_max(s: Series, n: usize) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if s.len() < n {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr.mapv(|x| -OrderedFloat::from(x.abs()));
    let sarr = arr.into_iter().k_smallest(n).map(|x| -f64::from(x));
    let out = sarr.sum::<f64>() / n as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn mean_n_absolute_max(name: &str, n: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _mean_n_absolute_max(s, n), o)
        .get(0)
        .alias(&format!("{}__mean_n_absolute_max__n_{:.1}", name, n))
}

fn _autocorrelation(s: Series, lag: usize) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if s.len() < lag {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let y1 = arr.slice(s![..(arr.len() - lag)]);
    let y2 = arr.slice(s![lag..]);
    let sum_product = (y1.to_owned() - mean).dot(&(y2.to_owned() - mean));
    let v = arr.var(1.0);
    let out = sum_product / ((arr.len() - lag) as f64 * v);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn autocorrelation(name: &str, lag: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _autocorrelation(s, lag), o)
        .get(0)
        .alias(&format!("{}__autocorrelation__lag_{:.1}", name, lag))
}

fn _quantile(s: Series, q: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mut arr = arr.mapv(n64);
    let q_res = arr.quantile_axis_mut(Axis(0), n64(q), &Midpoint);
    let q_val = match q_res {
        Ok(m) => m[0],
        Err(_) => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let s = Series::new("", &[f64::from(q_val)]);
    Ok(Some(s))
}

// pub fn quantile(name: &str, q: f64) -> Expr {
//     let o = GetOutput::from_type(DataType::Float64);
//     col(name)
//         .apply(move |s| _quantile(s, q), o)
//         .get(0)
//         .alias(&format!("{}__quantile__q_{:.1}", name, q))
// }

pub fn expr_quantile(name: &str, q: f64) -> Expr {
    quantile(name, lit(q), QuantileInterpolOptions::Midpoint)
        .alias(&format!("{}__quantile__q_{:.1}", name, q))
}

fn _number_crossing_m(s: Series, m: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let iarr = arr.into_iter().filter(|x| x != &m).collect::<Vec<_>>();
    let mut count = 0;
    for (x1, x2) in izip!(iarr.iter(), iarr.iter().skip(1)) {
        if x1.is_nan() {
            return Ok(Some(Series::new("", &[f64::NAN])));
        }
        if (x1 < &m && x2 > &m) || (x1 > &m && x2 < &m) {
            count += 1;
        }
    }
    let s = Series::new("", &[count as f64]);
    Ok(Some(s))
}

pub fn number_crossing_m(name: &str, m: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _number_crossing_m(s, m), o)
        .get(0)
        .alias(&format!("{}__number_crossing_m__m_{:.1}", name, m))
}

fn _range_count(s: Series, lower: f64, upper: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if upper < lower {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let count = arr
        .into_iter()
        .filter(|x| x >= &lower && x <= &upper)
        .count();
    let s = Series::new("", &[count as f64]);
    Ok(Some(s))
}

pub fn range_count(name: &str, lower: f64, upper: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _range_count(s, lower, upper), o)
        .get(0)
        .alias(&format!(
            "{}__range_count__min_{:.1}__max_{:.1}",
            name, lower, upper,
        ))
}

fn _index_mass_quantile(s: Series, q: f64) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mut abs_arr = arr.mapv(|x| x.abs());
    let abs_sum = abs_arr.sum();
    if abs_sum == 0.0 {
        return Ok(Some(Series::new("", &[f64::NAN])));
    }
    abs_arr.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    let mass_centralized = abs_arr.mapv(|x| x / abs_sum);
    let idx_res = mass_centralized
        .into_iter()
        .enumerate()
        .filter(|(_, x)| x >= &q)
        .map(|(i, _)| i);
    let out = (idx_res.min().unwrap_or(0) + 1) as f64 / arr.len() as f64;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn index_mass_quantile(name: &str, q: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _index_mass_quantile(s, q), o)
        .get(0)
        .alias(&format!("{}__index_mass_quantile__q_{:.1}", name, q))
}

fn _c3(s: Series, lag: usize) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Some(Series::new("", &[0 as f64])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let slice = &mut arr.to_vec()[..];
    let neg_lag = -(lag as isize);
    let y1_slice = _roll(slice, 2 * neg_lag);
    let y1 = ArrayView1::from(y1_slice);

    let slice = &mut arr.to_vec()[..];
    let y2_slice = _roll(slice, neg_lag);
    let y2 = ArrayView1::from(y2_slice);
    let y_prod = &y1 * &y2;
    let full_prod = y_prod * arr;
    let prod = full_prod.slice(s![..(n - 2 * lag)]);
    let mean_opt = prod.mean();
    let out = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let s = Series::new("", &[out as f64]);
    Ok(Some(s))
}

pub fn c3(name: &str, lag: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _c3(s, lag), o)
        .get(0)
        .alias(&format!("{}__c3__lag_{:.0}", name, lag))
}

fn _time_reversal_asymmetry_statistic(
    s: Series,
    lag: usize,
) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Some(Series::new("", &[0 as f64])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let slice = &mut arr.to_vec()[..];
    let neg_lag = -(lag as isize);
    let one_lag = _roll(slice, neg_lag);
    let one_lag = ArrayView1::from(one_lag);

    let slice = &mut arr.to_vec()[..];
    let two_lag = _roll(slice, 2 * neg_lag);
    let two_lag = ArrayView1::from(two_lag);
    let full_prod = &two_lag * &two_lag * one_lag - &one_lag * &arr * &arr;
    let prod = full_prod.slice(s![..(n - 2 * lag)]);
    let mean_opt = prod.mean();
    let out = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Series::new_empty("", &DataType::Float64))),
    };
    let s = Series::new("", &[out as f64]);
    Ok(Some(s))
}

pub fn time_reversal_asymmetry_statistic(name: &str, lag: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _time_reversal_asymmetry_statistic(s, lag), o)
        .get(0)
        .alias(&format!(
            "{}__time_reversal_asymmetry_statistic__lag_{:.0}",
            name, lag
        ))
}

fn _number_peaks(s: Series, n: usize) -> Result<Option<Series>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Series::new_empty("", &DataType::Float64)));
    }
    if s.len() < n {
        return Ok(Some(Series::new("", &[0 as f64])));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let arr_reduced = arr.slice(s![n..arr.len() - n]);
    let mut res: Option<Array1<bool>> = None;
    for i in 1..n + 1 {
        let slice = &mut arr.to_vec()[..];
        let rolled = _roll(slice, i as isize);
        let rolled = ArrayView1::from(rolled);
        let rolled = rolled.slice(s![n..arr.len() - n]);
        let result_first = (&arr_reduced - &rolled).mapv(|x| x > 0.0);
        if res.is_none() {
            res = Some(result_first);
        } else {
            res = Some(res.unwrap() & result_first);
        }
        let slice = &mut arr.to_vec()[..];
        let rolled = _roll(slice, -(i as isize));
        let rolled = ArrayView1::from(rolled);
        let rolled = rolled.slice(s![n..arr.len() - n]);
        let result_second = (&arr_reduced - &rolled).mapv(|x| x > 0.0);
        res = Some(res.unwrap() & result_second);
    }
    let count = res.unwrap().into_iter().filter(|x| *x).count();
    let s = Series::new("", &[count as f64]);
    Ok(Some(s))
}

pub fn number_peaks(name: &str, n: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _number_peaks(s, n), o)
        .get(0)
        .alias(&format!("{}__number_peaks__n_{:.0}", name, n))
}
