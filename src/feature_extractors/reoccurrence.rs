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
    let arr = arr.mapv(OrderedFloat);
    let counts = arr.iter().counts();
    let mut more_than_once = 0;
    for v in counts.values() {
        if *v > 1 {
            more_than_once += 1;
        }
    }
    let out = (more_than_once as f64) / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

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
