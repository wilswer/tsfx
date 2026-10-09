//! Linear fits over time.

use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use ndarray::{Axis, Ix1};
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::{
    aggregate_on_chunks, calculate_sequential_ols, skip_nan_mean, skip_nan_reduce,
    skip_nan_sample_var,
};

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

fn _linear_trend(s: Column) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();

    // Cast the column to Float64
    let series = s.as_materialized_series().cast(&DataType::Float64)?;
    let ca = series.f64()?;

    // Call our decoupled OLS function
    let (intercept, slope) = calculate_sequential_ols(ca.into_no_null_iter(), ca.len());

    // Build the Struct column result
    let result = DataFrame::new(
        1,
        vec![
            Column::new("intercept".into(), &[intercept]),
            Column::new("slope".into(), &[slope]),
        ],
    )?
    .into_struct("linear_trend".into())
    .into_column();

    Ok(result)
}

pub fn linear_trend(name: &str) -> Expr {
    let name = name.to_string();
    col(&name)
        .apply(_linear_trend, |_, _| {
            Ok(Field::new(
                "".into(),
                DataType::Struct(vec![
                    Field::new("intercept".into(), DataType::Float64),
                    Field::new("slope".into(), DataType::Float64),
                ]),
            ))
        })
        .struct_()
        .rename_fields(
            [
                format!("{}__linear_trend_intercept", name),
                format!("{}__linear_trend_slope", name),
            ]
            .to_vec(),
        )
        .get(0, true)
        .alias(format!("{}__linear_trend", name))
}

fn _agg_linear_trend(
    s: Column,
    chunk_size: usize,
    aggregator: ChunkAggregator,
) -> Result<Column, PolarsError> {
    let s = s.drop_nulls();
    if s.is_empty() || s.len() < chunk_size {
        let s_i = f64::NAN;
        let s_s = f64::NAN;
        let s = DataFrame::new(
            1,
            vec![
                Column::new("agg_intercept".into(), &[s_i]),
                Column::new("agg_slope".into(), &[s_s]),
            ],
        )?
        .into_struct("agg_linear_trend".into())
        .into_column();
        return Ok(s);
    }
    let arr = s.into_frame().to_ndarray::<Float64Type>(IndexOrder::C)?;
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let agg_arr = match aggregator {
        ChunkAggregator::Mean => aggregate_on_chunks(arr, chunk_size, |x| skip_nan_mean(&x)),
        ChunkAggregator::Max => {
            aggregate_on_chunks(arr, chunk_size, |x| skip_nan_reduce(&x, f64::max))
        }
        ChunkAggregator::Min => {
            aggregate_on_chunks(arr, chunk_size, |x| skip_nan_reduce(&x, f64::min))
        }
        // ddof=1 on purpose: tsfresh aggregates chunks with pandas' Series.var
        ChunkAggregator::Var => aggregate_on_chunks(arr, chunk_size, |x| skip_nan_sample_var(&x)),
    };
    let agg_len = agg_arr.len();
    let (s_i, s_s) = calculate_sequential_ols(agg_arr, agg_len);

    let s = DataFrame::new(
        1,
        vec![
            Column::new("agg_intercept".into(), &[s_i]),
            Column::new("agg_slope".into(), &[s_s]),
        ],
    )?
    .into_struct("agg_linear_trend".into())
    .into_column();

    Ok(s)
}

pub fn agg_linear_trend(name: &str, chunk_size: usize, aggregator: impl Into<String>) -> Expr {
    let agg_str = aggregator.into();
    let agg_enum = ChunkAggregator::from_str(&agg_str).unwrap();
    let name = name.to_string();
    col(&name)
        .apply(
            move |s| _agg_linear_trend(s, chunk_size, agg_enum.clone()),
            |_, _| {
                Ok(Field::new(
                    "".into(),
                    DataType::Struct(vec![
                        Field::new("agg_intercept".into(), DataType::Float64),
                        Field::new("agg_slope".into(), DataType::Float64),
                    ]),
                ))
            },
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
        .get(0, true)
        .alias(format!(
            "{}__agg_linear_trend__chunk_size_{:.1}__agg_{}",
            name, chunk_size, agg_str
        ))
}
