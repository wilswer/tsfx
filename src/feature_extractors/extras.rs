use core::f64;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use itertools::izip;
use itertools::Itertools;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::ArrayView1;
use ndarray::{s, Array, Array1, Axis, Ix1};
use ndarray_stats::errors::QuantileError;
use ndarray_stats::{interpolate::Midpoint, QuantileExt, SummaryStatisticsExt};
use noisy_float::types::n64;
use num::FromPrimitive;
use ordered_float::OrderedFloat;
use polars::lazy::dsl::*;
use polars::prelude::*;
use polars::series::ops::NullBehavior;

use crate::extract::ExtractionSettings;
use crate::utils::toml_reader::load_config;

pub fn extra_aggregators(opts: &ExtractionSettings) -> Vec<Expr> {
    let config = match &opts.config_path {
        Some(file) => load_config(Some(file.as_str())),
        None => load_config(None),
    };
    let mut aggregators = Vec::new();
    for col in &opts.value_cols {
        if config.kurtosis.is_some() {
            aggregators.push(kurtosis(col));
        }
        if config.absolute_energy.is_some() {
            aggregators.push(absolute_energy(col));
        }
        if config.mean_absolute_change.is_some() {
            aggregators.push(mean_absolute_change(col));
        }
        if config.linear_trend.is_some() {
            aggregators.push(linear_trend(col));
        }
        if config.variance_larger_than_standard_deviation.is_some() {
            aggregators.push(variance_larger_than_standard_deviation(col));
        }
        if let Some(feature) = &config.ratio_beyond_r_sigma {
            let params = &feature.parameters;
            let mut rs = Vec::new();
            for p in params {
                rs.push(p.r);
            }
            aggregators.push(ratio_beyond_r_sigma(col, rs));
        }

        if let Some(feature) = &config.large_standard_deviation {
            let params = &feature.parameters;
            let mut rs = Vec::new();
            for p in params {
                rs.push(p.r);
            }
            aggregators.push(large_standard_deviation(col, rs));
        }
        if let Some(feature) = &config.symmetry_looking {
            let params = &feature.parameters;
            let mut rs = Vec::new();
            for p in params {
                rs.push(p.r);
            }
            aggregators.push(symmetry_looking(col, rs));
        }
        if config.has_duplicate_max.is_some() {
            aggregators.push(has_duplicate_max(col));
        }
        if config.has_duplicate_min.is_some() {
            aggregators.push(has_duplicate_min(col));
        }
        if let Some(feature) = &config.cid_ce {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(cid_ce(col, p.normalize));
            }
        }
        if config.absolute_maximum.is_some() {
            aggregators.push(absolute_maximum(col));
        }
        if config.absolute_sum_of_changes.is_some() {
            aggregators.push(absolute_sum_of_changes(col));
        }
        if config.count_above_mean.is_some() {
            aggregators.push(count_above_mean(col));
        }
        if config.count_below_mean.is_some() {
            aggregators.push(count_below_mean(col));
        }
        if config.count_above.is_some() {
            aggregators.push(count_above(col, 0.0));
        }
        if config.count_below.is_some() {
            aggregators.push(count_below(col, 0.0));
        }
        if config.first_location_of_maximum.is_some() {
            aggregators.push(first_location_of_maximum(col));
        }
        if config.first_location_of_minimum.is_some() {
            aggregators.push(first_location_of_minimum(col));
        }
        if config.last_location_of_maximum.is_some() {
            aggregators.push(last_location_of_maximum(col));
        }
        if config.last_location_of_minimum.is_some() {
            aggregators.push(last_location_of_minimum(col));
        }
        if config.longest_strike_above_mean.is_some() {
            aggregators.push(longest_strike_above_mean(col));
        }
        if config.longest_strike_below_mean.is_some() {
            aggregators.push(longest_strike_below_mean(col));
        }
        if config.has_duplicate.is_some() {
            aggregators.push(has_duplicate(col));
        }
        if config.variation_coefficient.is_some() {
            aggregators.push(variation_coefficient(col));
        }
        if config.mean_change.is_some() {
            aggregators.push(mean_change(col));
        }
        if config.ratio_value_number_to_time_series_length.is_some() {
            aggregators.push(ratio_value_number_to_time_series_length(col));
        }
        if config.sum_of_reoccurring_values.is_some() {
            aggregators.push(sum_of_reoccurring_values(col));
        }
        if config.sum_of_reoccurring_data_points.is_some() {
            aggregators.push(sum_of_reoccurring_data_points(col));
        }
        if config
            .percentage_of_reoccurring_values_to_all_values
            .is_some()
        {
            aggregators.push(percentage_of_reoccurring_values_to_all_values(col));
        }
        if config
            .percentage_of_reoccurring_values_to_all_datapoints
            .is_some()
        {
            aggregators.push(percentage_of_reoccurring_values_to_all_datapoints(col));
        }
        if let Some(feature) = &config.agg_linear_trend {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(agg_linear_trend(col, p.chunk_size, &p.aggregator));
            }
        }
        if let Some(feature) = &config.mean_n_absolute_max {
            let params = &feature.parameters;
            let mut ns = Vec::new();
            for p in params {
                ns.push(p.n);
            }
            aggregators.push(mean_n_absolute_max(col, ns));
        }
        if let Some(feature) = &config.autocorrelation {
            let params = &feature.parameters;
            let mut lags = Vec::new();
            for p in params {
                lags.push(p.lag);
            }
            aggregators.push(autocorrelation(col, lags));
        }
        if let Some(feature) = &config.quantile {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(expr_quantile(col, p.q));
            }
        }
        if let Some(feature) = &config.number_crossing_m {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(number_crossing_m(col, p.m));
            }
        }
        if let Some(feature) = &config.range_count {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(range_count(col, p.min, p.max));
            }
        }
        if let Some(feature) = &config.index_mass_quantile {
            let params = &feature.parameters;
            let mut qs = Vec::new();
            for p in params {
                qs.push(p.q);
            }
            aggregators.push(index_mass_quantile(col, qs));
        }
        if let Some(feature) = &config.c3 {
            let parameters = &feature.parameters;
            for p in parameters {
                aggregators.push(c3(col, p.lag));
            }
        }
        if let Some(feature) = &config.time_reversal_asymmetry_statistic {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(time_reversal_asymmetry_statistic(col, p.lag));
            }
        }
        if let Some(feature) = &config.number_peaks {
            let params = &feature.parameters;
            for p in params {
                aggregators.push(number_peaks(col, p.n));
            }
        }
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

fn _make_nan_struct_column(
    name: &str,
    parameter_name: &str,
    rs: &[f64],
) -> Result<Option<Column>, PolarsError> {
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs.iter() {
        ss.push(Column::new(
            format!("{}__{}_{:2}", name, parameter_name, r).into(),
            &[f64::NAN],
        ))
    }
    let s = DataFrame::new(ss)?.into_struct(name.into()).into_column();
    Ok(Some(s))
}

fn _make_nan_struct_column_int(
    name: &str,
    parameter_name: &str,
    ns: &[usize],
) -> Result<Option<Column>, PolarsError> {
    let mut ss: Vec<Column> = Vec::with_capacity(ns.len());
    for n in ns.iter() {
        ss.push(Column::new(
            format!("{}__{}_{}", name, parameter_name, n).into(),
            &[f64::NAN],
        ))
    }
    let s = DataFrame::new(ss)?.into_struct(name.into()).into_column();
    Ok(Some(s))
}

fn _get_length_sequences_where(x: &ndarray::Array1<bool>) -> Vec<usize> {
    let mut group_lengths = Vec::new();
    for (key, group) in &x.into_iter().chunk_by(|elt| *elt) {
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

fn _absolute_energy(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let abs_energy = arr.mapv(|x| x.powi(2)).sum();
    let s = Column::new("".into(), &[abs_energy]);
    Ok(Some(s))
}

fn _out(_: &Schema, _: &Field) -> Result<Field, PolarsError> {
    Ok(Field::new("".into(), DataType::Float64))
}

/// Abolute energy feature.
///
/// The absolute energy of the time series,
/// defined as the sum of the squared values of the time series:
/// $$ \text{absolute energy} = \sum_{i=1}^{n}x_i^2,$$
/// where $n$ is the number of values in the time series.
pub fn absolute_energy(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_energy, o)
        .get(0)
        .alias(format!("{name}__absolute_energy"))
}

/// The absolute energy of the time series cf. [`absolute_energy`],
/// calculated using the native Polars API.
pub fn expr_abs_energy(name: &str) -> Expr {
    col(name)
        .pow(2)
        .sum()
        .alias(format!("{name}__absolute_energy"))
}

fn _mean_absolute_change(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let mean_abs_change = diffs.mapv(|x| x.abs()).mean().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[mean_abs_change]);
    Ok(Some(s))
}

/// Mean absolute change feature.
///
/// The mean absolute change of a time series is defined as:
/// $$ \text{mean abs. change} = \frac{1}{n-1}\sum_{i=1}^{n-1} \|x_{i + 1} - x_{i}\|.$$
/// It is the average of the absolute value of differences in the time series.
pub fn mean_absolute_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean_absolute_change, o)
        .get(0)
        .alias(format!("{}__mean_absolute_change", name))
}

/// Mean change implemented using the native Polars API.
/// See [`mean_change`].
pub fn expr_mean_change(name: &str) -> Expr {
    let diffs = col(name).diff(1.into(), NullBehavior::Drop);
    let n = col(name).count() - lit(1);
    (diffs.sum() / n).alias(format!("{}__mean_change", name))
}

fn _kurtosis(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let kurtosis = arr.kurtosis().unwrap_or(f64::NAN);
    let s = Column::new("".into(), &[kurtosis]);
    Ok(Some(s))
}

/// Kurtosis feature.
///
/// The kurtosis of all values in the time series, where the kurtosis is the fourth standardized moment:
/// $$ \text{kurtosis} = \frac{1}{(n-1) \sigma^4} \sum_{i=1}^{n} (x_i - \mu)^4, $$
/// where $n$ is the number of values in the time series, $\mu$ is the mean of the time series,
/// and $\sigma$ is the standard deviation of the time series
pub fn kurtosis(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_kurtosis, o)
        .get(0)
        .alias(format!("{}__kurtosis", name))
}

/// Kurtosis implemented using the native Polars API.
/// See [`kurtosis`].
pub fn expr_kurtosis(name: &str) -> Expr {
    let n = col(name).count();
    let mean = col(name).mean();
    let std = col(name).std(1);
    let skewness = ((col(name) - mean).pow(4)).sum() / ((n - lit(1.0)) * std.pow(4));
    skewness.alias(format!("{}__expr_kurtosis", name))
}

fn _linear_trend(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let x = ndarray::Array::range(0., arr.len() as f64, 1.);
    let x = x.insert_axis(Axis(1));
    let dataset = Dataset::new(x, arr);
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_i = model.intercept();
            let s_s = model.params()[0];
            let s = DataFrame::new(vec![
                Column::new("intercept".into(), &[s_i]),
                Column::new("slope".into(), &[s_s]),
            ])?
            .into_struct("linear_trend".into())
            .into_column();
            Ok(Some(s))
        }
        Err(_) => {
            let s = DataFrame::new(vec![
                Column::new("intercept".into(), &[f64::NAN]),
                Column::new("slope".into(), &[f64::NAN]),
            ])?
            .into_struct("linear_trend".into())
            .into_column();
            Ok(Some(s))
        }
    }
}

pub fn linear_trend(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Struct(vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new("slope".into(), DataType::Float64),
    ]));
    let name = name.to_string();
    col(&name)
        .apply(_linear_trend, o)
        .struct_()
        .rename_fields(
            [
                format!("{}__linear_trend_intercept", name),
                format!("{}__linear_trend_slope", name),
            ]
            .to_vec(),
        )
        .get(0)
        .alias(format!("{}__linear_trend", name))
}

pub fn variance_larger_than_standard_deviation(name: &str) -> Expr {
    let std = col(name).std(1);
    let var = col(name).var(1);
    (var.gt(std))
        .cast(DataType::Float64)
        .alias(format!("{}__variance_larger_than_standard_deviation", name))
}

fn _ratio_beyond_r_sigma(s: Column, rs: &[f64]) -> Result<Option<Column>, PolarsError> {
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
    let std = arr.std(1.0);
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
    let s = DataFrame::new(ss)?
        .into_struct("ratio_beyond_r_sigma".into())
        .into_column();
    Ok(Some(s))
}

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
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(&name)
        .apply(move |s| _ratio_beyond_r_sigma(s, &rs), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__ratio_beyond_r_sigma", name))
}

fn _large_standard_deviation(s: Column, rs: &[f64]) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("large_standard_deviation", "r", rs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let min = arr.min().unwrap_or(&0.0);
    let max = arr.max().unwrap_or(&0.0);
    let std = arr.std(1.0);
    let mut ss: Vec<Column> = Vec::with_capacity(rs.len());
    for r in rs {
        let out = std > r * (max - min);
        ss.push(Column::new(
            format!("large_standard_deviation__r_{:2}", r).into(),
            &[out as u8 as f64],
        ));
    }
    let s = DataFrame::new(ss)?
        .into_struct("large_standard_deviation".into())
        .into_column();
    Ok(Some(s))
}

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
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(name)
        .apply(move |s| _large_standard_deviation(s, &rs), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__large_standard_deviation", name))
}

fn _symmetry_looking(s: Column, rs: &[f64]) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("symmetry_looking", "r", rs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mut arr = arr.mapv(n64);
    let median_res = _median_mut(&mut arr);
    let median = match median_res {
        Ok(m) => f64::from(m),
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
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
    let s = DataFrame::new(ss)?
        .into_struct("symmetry_looking".into())
        .into_column();
    Ok(Some(s))
}

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
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(name)
        .apply(move |s| _symmetry_looking(s, &rs), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__symmetry_looking__r_", name))
}

fn _has_duplicate_max(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let count = arr.mapv(|x| if x == *max { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(Some(s))
}

pub fn has_duplicate_max(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_has_duplicate_max, o)
        .get(0)
        .alias(format!("{}__has_duplicate_max", name))
}

fn _has_duplicate_min(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let count = arr.mapv(|x| if x == *min { 1.0 } else { 0.0 }).sum();
    let out = count > 1.0;
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(Some(s))
}

pub fn has_duplicate_min(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_has_duplicate_min, o)
        .get(0)
        .alias(format!("{}__has_duplicate_min", name))
}

fn _cid_ce(s: Column, normalize: bool) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let arr = if normalize {
        let mean = arr.mean().unwrap_or(f64::NAN);
        let std = arr.std(1.0);
        (arr - mean) / std
    } else {
        arr
    };
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.powi(2)).sum().sqrt();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn cid_ce(name: &str, normalize: bool) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _cid_ce(s, normalize), o)
        .get(0)
        .alias(format!("{}__cid_ce__normalize_{:.1}", name, normalize))
}

fn _absolute_maximum(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let abs_arr = arr.mapv(|x| x.abs());
    let max_res = abs_arr.max();
    let max = match max_res {
        Ok(m) => *m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let s = Column::new("".into(), &[max]);
    Ok(Some(s))
}

pub fn absolute_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_maximum, o)
        .get(0)
        .alias(format!("{}__absolute_maximum", name))
}

fn _absolute_sum_of_changes(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let out = diffs.mapv(|x| x.abs()).sum();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn absolute_sum_of_changes(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_absolute_sum_of_changes, o)
        .get(0)
        .alias(format!("{}__absolute_sum_of_changes", name))
}

fn _count_above_mean(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = arr.mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn count_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_count_above_mean, o)
        .get(0)
        .alias(format!("{}__count_above_mean", name))
}

fn _count_below_mean(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = arr.mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn count_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_count_below_mean, o)
        .get(0)
        .alias(format!("{}__count_below_mean", name))
}

fn _count_above(s: Column, t: f64) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn count_above(name: &str, t: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _count_above(s, t), o)
        .get(0)
        .alias(format!("{}__count_above__t_{:.1}", name, t))
}

fn _count_below(s: Column, t: f64) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn count_below(name: &str, t: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _count_below(s, t), o)
        .get(0)
        .alias(format!("{}__count_below__t_{:.1}", name, t))
}

fn _first_location_of_maximum(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = max as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn first_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_first_location_of_maximum, o)
        .get(0)
        .alias(format!("{}__first_location_of_maximum", name))
}

fn _first_location_of_minimum(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = min as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn first_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_first_location_of_minimum, o)
        .get(0)
        .alias(format!("{}__first_location_of_minimum", name))
}

fn _last_location_of_maximum(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = 1.0 - (max as f64 / arr.len() as f64);
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn last_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_last_location_of_maximum, o)
        .get(0)
        .alias(format!("{}__last_location_of_maximum", name))
}

fn _last_location_of_minimum(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let out = 1.0 - (min as f64 / arr.len() as f64);
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn last_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_last_location_of_minimum, o)
        .get(0)
        .alias(format!("{}__last_location_of_minimum", name))
}

fn _longest_strike_below_mean(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let bool_arr = arr.mapv(|x| x < mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(Some(s))
}

pub fn longest_strike_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_longest_strike_below_mean, o)
        .get(0)
        .alias(format!("{}__longest_strike_below_mean", name))
}

fn _longest_strike_above_mean(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let bool_arr = arr.mapv(|x| x > mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(Some(s))
}

pub fn longest_strike_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_longest_strike_above_mean, o)
        .get(0)
        .alias(format!("{}__longest_strike_above_mean", name))
}

fn _has_duplicate(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
    let s = Column::new("".into(), &[out as u8 as f64]);
    Ok(Some(s))
}

pub fn has_duplicate(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_has_duplicate, o)
        .get(0)
        .alias(format!("{}__has_duplicate", name))
}

fn _variation_coefficient(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let std = arr.std(1.0);
    let out = if mean == 0.0 { f64::NAN } else { std / mean };
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn variation_coefficient(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_variation_coefficient, o)
        .get(0)
        .alias(format!("{}__variation_coefficient", name))
}

fn _mean_change(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
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
    Ok(Some(s))
}

/// The mean change of a time series is defined as
/// $$ \text{mean change} = \frac{1}{n-1} \sum_{i=1}^{n-1} x_{i + 1} - x_{i} = \frac{x_{n} - x_1}{n-1} $$.
pub fn mean_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_mean_change, o)
        .get(0)
        .alias(format!("{}__mean_change", name))
}

fn _ratio_value_number_to_time_series_length(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn ratio_value_number_to_time_series_length(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_ratio_value_number_to_time_series_length, o)
        .get(0)
        .alias(format!(
            "{}__ratio_value_number_to_time_series_length",
            name
        ))
}

fn _sum_of_reoccurring_values(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
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
    Ok(Some(s))
}

pub fn sum_of_reoccurring_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_of_reoccurring_values, o)
        .get(0)
        .alias(format!("{}__sum_of_reoccurring_values", name))
}

fn _sum_of_reoccurring_data_points(s: Column) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
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
    Ok(Some(s))
}

pub fn sum_of_reoccurring_data_points(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_sum_of_reoccurring_data_points, o)
        .get(0)
        .alias(format!("{}__sum_of_reoccurring_data_points", name))
}

fn _percentage_of_reoccurring_values_to_all_values(
    s: Column,
) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.into_iter().counts();
    let mut more_than_once = 0;
    for (_, v) in counts.iter() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / counts.keys().len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn percentage_of_reoccurring_values_to_all_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_percentage_of_reoccurring_values_to_all_values, o)
        .get(0)
        .alias(format!(
            "{}__percentage_of_reoccurring_values_to_all_values",
            name
        ))
}

fn _percentage_of_reoccurring_values_to_all_datapoints(
    s: Column,
) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.iter().counts();
    let mut more_than_once = 0;
    for (_, v) in counts.iter() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(Some(s))
}

pub fn percentage_of_reoccurring_values_to_all_datapoints(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(_percentage_of_reoccurring_values_to_all_datapoints, o)
        .get(0)
        .alias(format!(
            "{}__percentage_of_reoccurring_values_to_all_datapoints",
            name
        ))
}

fn _agg_linear_trend(
    s: Column,
    chunk_size: usize,
    aggregator: ChunkAggregator,
) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() || s.len() < chunk_size {
        let s_i = f64::NAN;
        let s_s = f64::NAN;
        let s = DataFrame::new(vec![
            Column::new("agg_intercept".into(), &[s_i]),
            Column::new("agg_slope".into(), &[s_s]),
        ])?
        .into_struct("agg_linear_trend".into())
        .into_column();
        return Ok(Some(s));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
            let s_i = model.intercept();
            let s_s = model.params()[0];
            let s = DataFrame::new(vec![
                Column::new("agg_intercept".into(), &[s_i]),
                Column::new("agg_slope".into(), &[s_s]),
            ])?
            .into_struct("agg_linear_trend".into())
            .into_column();
            Ok(Some(s))
        }
        Err(_) => {
            let s_i = f64::NAN;
            let s_s = f64::NAN;
            let s = DataFrame::new(vec![
                Column::new("agg_intercept".into(), &[s_i]),
                Column::new("agg_slope".into(), &[s_s]),
            ])?
            .into_struct("agg_linear_trend".into())
            .into_column();
            Ok(Some(s))
        }
    }
}

fn agg_linear_trend(name: &str, chunk_size: usize, aggregator: impl Into<String>) -> Expr {
    let o = GetOutput::from_type(DataType::Struct(vec![
        Field::new("agg_intercept".into(), DataType::Float64),
        Field::new("agg_slope".into(), DataType::Float64),
    ]));
    let agg_str = aggregator.into();
    let agg_enum = ChunkAggregator::from_str(&agg_str).unwrap();
    let name = name.to_string();
    col(&name)
        .apply(
            move |s| _agg_linear_trend(s, chunk_size, agg_enum.clone()),
            o,
        )
        .struct_()
        .rename_fields(
            [
                format!(
                    "{}__agg_linear_trend_intercept__chunk_size_{:.1}__agg_{}",
                    name, chunk_size, agg_str
                ),
                format!(
                    "{}__agg_linear_trend_slope__chunk_size_{:.1}__agg_{}",
                    name, chunk_size, agg_str
                ),
            ]
            .to_vec(),
        )
        .get(0)
        .alias(format!(
            "{}__agg_linear_trend__chunk_size_{:.1}__agg_{}",
            name, chunk_size, agg_str
        ))
}

fn _mean_n_absolute_max(s: Column, ns: &[usize]) -> Result<Option<Column>, PolarsError> {
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
    let s = DataFrame::new(ss)?
        .into_struct("mean_n_absolute_max".into())
        .into_column();
    Ok(Some(s))
}

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
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(name)
        .apply(move |s| _mean_n_absolute_max(s, &ns), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__mean_n_absolute_max", name))
}

fn _autocorrelation(s: Column, lags: &[usize]) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column_int("autocorrelation", "lag", lags);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let v = arr.var(1.0);
    let mut ss: Vec<Column> = Vec::with_capacity(lags.len());
    for lag in lags {
        let out = if arr.len() < *lag {
            f64::NAN
        } else {
            let y1 = arr.slice(s![..(arr.len() - lag)]);
            let y2 = arr.slice(s![*lag..]);
            let sum_product = (y1.to_owned() - mean).dot(&(y2.to_owned() - mean));
            sum_product / ((arr.len() - lag) as f64 * v)
        };
        ss.push(Column::new(
            format!("autocorrelation__lag_{}", lag).into(),
            &[out],
        ));
    }
    let s = DataFrame::new(ss)?
        .into_struct("autocorrelation".into())
        .into_column();
    Ok(Some(s))
}

pub fn autocorrelation(name: &str, lags: Vec<usize>) -> Expr {
    let mut new_field_names = Vec::with_capacity(lags.len());
    let mut struct_names = Vec::with_capacity(lags.len());
    for lag in lags.iter() {
        new_field_names.push(format!("{}__autocorrelation__lag_{}", name, lag));
        struct_names.push(Field::new(
            format!("autocorrelation__lag_{}", lag).into(),
            DataType::Float64,
        ));
    }
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(name)
        .apply(move |s| _autocorrelation(s, &lags), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__autocorrelation", name))
}

fn _quantile(s: Column, q: f64) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mut arr = arr.mapv(n64);
    let q_res = arr.quantile_axis_mut(Axis(0), n64(q), &Midpoint);
    let q_val = match q_res {
        Ok(m) => m[0],
        Err(_) => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let s = Column::new("".into(), &[f64::from(q_val)]);
    Ok(Some(s))
}

pub fn expr_quantile(name: &str, q: f64) -> Expr {
    quantile(name, lit(q), QuantileMethod::Midpoint)
        .cast(DataType::Float64)
        .alias(format!("{}__quantile__q_{:.1}", name, q))
}

fn _number_crossing_m(s: Column, m: f64) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let iarr = arr.into_iter().filter(|x| x != &m).collect::<Vec<_>>();
    let mut count = 0;
    for (x1, x2) in izip!(iarr.iter(), iarr.iter().skip(1)) {
        if x1.is_nan() {
            return Ok(Some(Column::new("".into(), &[f64::NAN])));
        }
        if (x1 < &m && x2 > &m) || (x1 > &m && x2 < &m) {
            count += 1;
        }
    }
    let s = Column::new("".into(), &[count as f64]);
    Ok(Some(s))
}

pub fn number_crossing_m(name: &str, m: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _number_crossing_m(s, m), o)
        .get(0)
        .alias(format!("{}__number_crossing_m__m_{:.1}", name, m))
}

fn _range_count(s: Column, lower: f64, upper: f64) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    if upper < lower {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let count = arr
        .into_iter()
        .filter(|x| x >= &lower && x <= &upper)
        .count();
    let s = Column::new("".into(), &[count as f64]);
    Ok(Some(s))
}

pub fn range_count(name: &str, lower: f64, upper: f64) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _range_count(s, lower, upper), o)
        .get(0)
        .alias(format!(
            "{}__range_count__min_{:.1}__max_{:.1}",
            name, lower, upper,
        ))
}

fn _index_mass_quantile(s: Column, qs: &[f64]) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("index_mass_quantile", "q", qs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mut abs_arr = arr.mapv(|x| x.abs());
    let abs_sum = abs_arr.sum();
    if abs_sum == 0.0 {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    abs_arr.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    let mass_centralized = abs_arr.mapv(|x| x / abs_sum);
    let mut ss: Vec<Column> = Vec::with_capacity(qs.len());
    for q in qs {
        let idx_res = mass_centralized
            .iter()
            .enumerate()
            .filter(|(_, x)| x >= &q)
            .map(|(i, _)| i);
        let out = (idx_res.min().unwrap_or(0) + 1) as f64 / arr.len() as f64;
        ss.push(Column::new(
            format!("index_mass_quantile__q_{:2}", q).into(),
            &[out],
        ));
    }
    let s = DataFrame::new(ss)?
        .into_struct("index_mass_quantile".into())
        .into_column();
    Ok(Some(s))
}

pub fn index_mass_quantile(name: &str, qs: Vec<f64>) -> Expr {
    let mut new_field_names = Vec::with_capacity(qs.len());
    let mut struct_names = Vec::with_capacity(qs.len());
    for q in qs.iter() {
        new_field_names.push(format!("{}__index_mass_quantile__q_{:2}", name, q));
        struct_names.push(Field::new(
            format!("index_mass_quantile__q_{:2}", q).into(),
            DataType::Float64,
        ));
    }
    let o = GetOutput::from_type(DataType::Struct(struct_names));
    col(name)
        .apply(move |s| _index_mass_quantile(s, &qs), o)
        .struct_()
        .rename_fields(new_field_names)
        .get(0)
        .alias(format!("{}__index_mass_quantile", name))
}

fn _c3(s: Column, lag: usize) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Some(Column::new("".into(), &[0 as f64])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let s = Column::new("".into(), &[out as f64]);
    Ok(Some(s))
}

pub fn c3(name: &str, lag: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _c3(s, lag), o)
        .get(0)
        .alias(format!("{}__c3__lag_{:.0}", name, lag))
}

fn _time_reversal_asymmetry_statistic(
    s: Column,
    lag: usize,
) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Some(Column::new("".into(), &[0 as f64])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
        None => return Ok(Some(Column::new("".into(), &[f64::NAN]))),
    };
    let s = Column::new("".into(), &[out as f64]);
    Ok(Some(s))
}

pub fn time_reversal_asymmetry_statistic(name: &str, lag: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _time_reversal_asymmetry_statistic(s, lag), o)
        .get(0)
        .alias(format!(
            "{}__time_reversal_asymmetry_statistic__lag_{:.0}",
            name, lag
        ))
}

fn _number_peaks(s: Column, n: usize) -> Result<Option<Column>, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Some(Column::new("".into(), &[f64::NAN])));
    }
    if s.len() < n {
        return Ok(Some(Column::new("".into(), &[0 as f64])));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
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
    let s = Column::new("".into(), &[count as f64]);
    Ok(Some(s))
}

pub fn number_peaks(name: &str, n: usize) -> Expr {
    let o = GetOutput::from_type(DataType::Float64);
    col(name)
        .apply(move |s| _number_peaks(s, n), o)
        .get(0)
        .alias(format!("{}__number_peaks__n_{:.0}", name, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{lazy_feature_df, ExtractionSettings, FeatureSetting};
    use polars::datatypes::AnyValue;

    #[test]
    fn test_unit_length_df() {
        // Create a dataframe with a single row
        let df = df![
            "id" => ["a"],
            "val" => [1.0],
        ]
        .unwrap()
        .lazy();

        // Configure extraction settings
        let opts = ExtractionSettings {
            grouping_col: "id".to_string(),
            feature_setting: FeatureSetting::Efficient,
            value_cols: vec!["val".to_string()],
            config_path: None,
            dynamic_settings: None,
        };

        // Extract features
        let gdf = lazy_feature_df(df, opts).unwrap();

        // Collect the results
        let fdf = gdf.collect().unwrap();

        // Assert that the resulting dataframe has exactly one row
        assert_eq!(fdf.shape().0, 1);

        // Also check that the length column has the expected value
        assert_eq!(
            fdf.column("length").unwrap().get(0).unwrap(),
            AnyValue::UInt32(1)
        );
    }
}
