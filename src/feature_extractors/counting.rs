//! Counts of values or runs relative to a threshold.

use anyhow::Result;
use itertools::izip;
use ndarray::{Array1, ArrayView1, Axis, Ix1, s};
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::{get_length_sequences_where, roll};

fn _count_above_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = arr.mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn count_above_mean(name: &str) -> Expr {
    col(name)
        .apply(_count_above_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__count_above_mean", name))
}

fn _count_below_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let out = arr.mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn count_below_mean(name: &str) -> Expr {
    col(name)
        .apply(_count_below_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__count_below_mean", name))
}

fn _count_above(s: Column, t: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // fraction of values >= t, as in tsfresh
    let out = arr.mapv(|x| if x >= t { 1.0 } else { 0.0 }).sum() / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn count_above(name: &str, t: f64) -> Expr {
    col(name)
        .apply(
            move |s| _count_above(s, t),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__count_above__t_{:.1}", name, t))
}

fn _count_below(s: Column, t: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    // fraction of values <= t, as in tsfresh
    let out = arr.mapv(|x| if x <= t { 1.0 } else { 0.0 }).sum() / arr.len() as f64;
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

pub fn count_below(name: &str, t: f64) -> Expr {
    col(name)
        .apply(
            move |s| _count_below(s, t),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__count_below__t_{:.1}", name, t))
}

fn _longest_strike_below_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let bool_arr = arr.mapv(|x| x < mean);
    let out = get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(s)
}

pub fn longest_strike_below_mean(name: &str) -> Expr {
    col(name)
        .apply(_longest_strike_below_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__longest_strike_below_mean", name))
}

fn _longest_strike_above_mean(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(Column::new("".into(), &[f64::NAN])),
    };
    let bool_arr = arr.mapv(|x| x > mean);
    let out = get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Column::new("".into(), &[out as f64]);
    Ok(s)
}

pub fn longest_strike_above_mean(name: &str) -> Expr {
    col(name)
        .apply(_longest_strike_above_mean, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__longest_strike_above_mean", name))
}

fn _number_crossing_m(s: Column, m: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let iarr = arr.into_iter().filter(|x| x != &m).collect::<Vec<_>>();
    let mut count = 0;
    for (x1, x2) in izip!(iarr.iter(), iarr.iter().skip(1)) {
        if x1.is_nan() {
            return Ok(Column::new("".into(), &[f64::NAN]));
        }
        if (x1 < &m && x2 > &m) || (x1 > &m && x2 < &m) {
            count += 1;
        }
    }
    let s = Column::new("".into(), &[count as f64]);
    Ok(s)
}

pub fn number_crossing_m(name: &str, m: f64) -> Expr {
    col(name)
        .apply(
            move |s| _number_crossing_m(s, m),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__number_crossing_m__m_{:.1}", name, m))
}

fn _range_count(s: Column, lower: f64, upper: f64) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    if upper < lower {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let count = arr
        .into_iter()
        .filter(|x| x >= &lower && x <= &upper)
        .count();
    let s = Column::new("".into(), &[count as f64]);
    Ok(s)
}

pub fn range_count(name: &str, lower: f64, upper: f64) -> Expr {
    col(name)
        .apply(
            move |s| _range_count(s, lower, upper),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!(
            "{}__range_count__min_{:.1}__max_{:.1}",
            name, lower, upper,
        ))
}

fn _number_peaks(s: Column, n: usize) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    if s.len() < n {
        return Ok(Column::new("".into(), &[0 as f64]));
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
        let rolled = roll(slice, i as isize);
        let rolled = ArrayView1::from(rolled);
        let rolled = rolled.slice(s![n..arr.len() - n]);
        let result_first = (&arr_reduced - &rolled).mapv(|x| x > 0.0);
        if res.is_none() {
            res = Some(result_first);
        } else {
            res = Some(res.unwrap() & result_first);
        }
        let slice = &mut arr.to_vec()[..];
        let rolled = roll(slice, -(i as isize));
        let rolled = ArrayView1::from(rolled);
        let rolled = rolled.slice(s![n..arr.len() - n]);
        let result_second = (&arr_reduced - &rolled).mapv(|x| x > 0.0);
        res = Some(res.unwrap() & result_second);
    }
    let count = res.unwrap().into_iter().filter(|x| *x).count();
    let s = Column::new("".into(), &[count as f64]);
    Ok(s)
}

pub fn number_peaks(name: &str, n: usize) -> Expr {
    col(name)
        .apply(
            move |s| _number_peaks(s, n),
            |_, _| Ok(Field::new("".into(), DataType::Float64)),
        )
        .get(0, true)
        .alias(format!("{}__number_peaks__n_{:.0}", name, n))
}
