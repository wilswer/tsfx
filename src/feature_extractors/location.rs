//! Relative positions in the series where something happens.

use anyhow::Result;
use ndarray::{Axis, Ix1, s};
use polars::lazy::dsl::*;
use polars::prelude::*;

use super::_make_nan_struct_column;
use crate::utils::stats::{np_argmax, np_argmin};

fn _first_location_of_maximum(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let max_res = np_argmax(&arr.view());
    let max = match max_res {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = max as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn first_location_of_maximum(name: &str) -> Expr {
    col(name)
        .apply(_first_location_of_maximum, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__first_location_of_maximum", name))
}

fn _first_location_of_minimum(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let min_res = np_argmin(&arr.view());
    let min = match min_res {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = min as f64 / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn first_location_of_minimum(name: &str) -> Expr {
    col(name)
        .apply(_first_location_of_minimum, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__first_location_of_minimum", name))
}

fn _last_location_of_maximum(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    // argmax of the reversed series gives the last occurrence
    let max_res = np_argmax(&arr.slice(s![..;-1]));
    let max = match max_res {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = 1.0 - (max as f64 / arr.len() as f64);
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn last_location_of_maximum(name: &str) -> Expr {
    col(name)
        .apply(_last_location_of_maximum, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__last_location_of_maximum", name))
}

fn _last_location_of_minimum(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    // argmin of the reversed series gives the last occurrence
    let min_res = np_argmin(&arr.slice(s![..;-1]));
    let min = match min_res {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = 1.0 - (min as f64 / arr.len() as f64);
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn last_location_of_minimum(name: &str) -> Expr {
    col(name)
        .apply(_last_location_of_minimum, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__last_location_of_minimum", name))
}

fn _index_mass_quantile(s: Column, qs: &[f64]) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return _make_nan_struct_column("index_mass_quantile", "q", qs);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mut abs_arr = arr.mapv(|x| x.abs());
    let abs_sum = abs_arr.sum();
    if abs_sum == 0.0 {
        return _make_nan_struct_column("index_mass_quantile", "q", qs);
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
    let s = DataFrame::new(1, ss)?
        .into_struct("index_mass_quantile".into())
        .into_column();
    Ok(s)
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
    col(name)
        .apply(
            move |s| _index_mass_quantile(s, &qs),
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
        .alias(format!("{}__index_mass_quantile", name))
}
