//! Lag-based relationships within the series.

use anyhow::Result;
use ndarray::{ArrayView1, Axis, Ix1, s};
use polars::lazy::dsl::*;
use polars::prelude::*;

use super::_make_nan_struct_column_int;
use crate::utils::stats::{population_var, roll};

fn _autocorrelation(s: Column, lags: &[usize]) -> Result<Column, PolarsError> {
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
        None => return _make_nan_struct_column_int("autocorrelation", "lag", lags),
    };
    let v = population_var(&arr.view());
    // tsfresh: autocorrelation is undefined without variance (np.isclose(v, 0))
    if v.abs() <= 1e-8 {
        return _make_nan_struct_column_int("autocorrelation", "lag", lags);
    }
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
    let s = DataFrame::new(1, ss)?
        .into_struct("autocorrelation".into())
        .into_column();
    Ok(s)
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
    col(name)
        .apply(
            move |s| _autocorrelation(s, &lags),
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
        .alias(format!("{}__autocorrelation", name))
}

fn _c3(s: Column, lag: usize) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Column::new("".into(), &[0 as f64]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let slice = &mut arr.to_vec()[..];
    let neg_lag = -(lag as isize);
    let y1_slice = roll(slice, 2 * neg_lag);
    let y1 = ArrayView1::from(y1_slice);

    let slice = &mut arr.to_vec()[..];
    let y2_slice = roll(slice, neg_lag);
    let y2 = ArrayView1::from(y2_slice);
    let y_prod = &y1 * &y2;
    let full_prod = y_prod * arr;
    let prod = full_prod.slice(s![..(n - 2 * lag)]);
    let mean_opt = prod.mean();
    let out = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn c3(name: &str, lag: usize) -> Expr {
    col(name)
        .apply(
            move |s| _c3(s, lag),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__c3__lag_{:.0}", name, lag))
}

fn _time_reversal_asymmetry_statistic(s: Column, lag: usize) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let n = s.len();
    if n <= 2 * lag {
        return Ok(Column::new("".into(), &[0 as f64]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let slice = &mut arr.to_vec()[..];
    let neg_lag = -(lag as isize);
    let one_lag = roll(slice, neg_lag);
    let one_lag = ArrayView1::from(one_lag);

    let slice = &mut arr.to_vec()[..];
    let two_lag = roll(slice, 2 * neg_lag);
    let two_lag = ArrayView1::from(two_lag);
    let full_prod = &two_lag * &two_lag * one_lag - &one_lag * &arr * &arr;
    let prod = full_prod.slice(s![..(n - 2 * lag)]);
    let mean_opt = prod.mean();
    let out = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn time_reversal_asymmetry_statistic(name: &str, lag: usize) -> Expr {
    col(name)
        .apply(
            move |s| _time_reversal_asymmetry_statistic(s, lag),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!(
            "{}__time_reversal_asymmetry_statistic__lag_{:.0}",
            name, lag
        ))
}
