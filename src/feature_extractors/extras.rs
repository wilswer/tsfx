use itertools::Itertools;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{s, Axis, Ix1};
use ndarray_stats::{interpolate::Midpoint, QuantileExt, SummaryStatisticsExt};
use noisy_float::types::n64;
use ordered_float::OrderedFloat;
use polars::{prelude::*, series::ops::NullBehavior};

pub fn extra_aggregators(value_cols: &[String]) -> Vec<Expr> {
    let mut aggregators = Vec::new();
    for col in value_cols {
        aggregators.push(kurtosis(col));
        aggregators.push(absolute_energy(col));
        aggregators.push(mean_absolute_change(col));
        aggregators.push(linear_fit_intercept(col));
        aggregators.push(linear_fit_slope(col));
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
    }
    aggregators
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

fn _absolute_energy(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let abs_energy = arr.mapv(|x| x.powi(2)).sum();
    let s = Series::new("", &[abs_energy]);
    Ok(Some(s))
}

pub fn absolute_energy(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_absolute_energy, o)
        .get(0)
        .alias(&format!("{}__abs_energy", name))
}

pub fn expr_abs_energy(name: &str) -> Expr {
    col(name)
        .pow(2)
        .sum()
        .alias(&format!("{}__abs_energy", name))
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
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let diffs = &arr.slice(s![1..]) - &arr.slice(s![..-1]);
    let mean_abs_change = diffs.mapv(|x| x.abs()).mean().unwrap_or(f32::NAN);
    let s = Series::new("", &[mean_abs_change]);
    Ok(Some(s))
}

pub fn mean_absolute_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
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
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let sum: f32 = arr.sum();
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
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let kurtosis = arr.kurtosis().unwrap_or(f32::NAN);
    let s = Series::new("", &[kurtosis]);
    Ok(Some(s))
}

pub fn kurtosis(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_kurtosis, o)
        .get(0)
        .alias(&format!("{}__kurtosis", name))
}

fn _linear_fit_intercept(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
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
    let model = lin_reg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_i = Series::new("", &[model.intercept()]);
            Ok(Some(s_i))
        }
        Err(_) => Ok(None),
    }
}

pub fn linear_fit_intercept(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_linear_fit_intercept, o)
        .get(0)
        .alias(&format!("{}__linear_fit_intercept", name))
}

fn _linear_fit_slope(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
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
    let model = lin_reg.fit(&dataset);
    match model {
        Ok(model) => {
            let s_p = Series::new("", &[model.params()[0]]);
            Ok(Some(s_p))
        }
        Err(_) => Ok(None),
    }
}

pub fn linear_fit_slope(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_linear_fit_slope, o)
        .get(0)
        .alias(&format!("{}__linear_fit_slope", name))
}

// fn _linear_fit(s: Series) -> Result<DataFrame, PolarsError> {
//     let arr = s
//         .into_frame()
//         .to_ndarray::<Float32Type>(IndexOrder::C)
//         .unwrap();
//     let arr = arr
//         .remove_axis(Axis(1))
//         .into_dimensionality::<Ix1>()
//         .unwrap();
//     let x = ndarray::Array::range(0., arr.len() as f32, 1.);
//     let x = x.insert_axis(Axis(1));
//     let dataset = Dataset::new(x, arr);
//     let lin_reg = LinearRegression::new();
//     let model = lin_reg.fit(&dataset).unwrap();
//     let s_i = Series::new("intercept", &[model.intercept()]);
//     let s_p = Series::new("slope", &[model.params()[0]]);
//     DataFrame::new(vec![s_i, s_p])
// }
//
// pub fn linear_fit(name: &str) -> Expr {
//     col(name)
//         .apply(_linear_fit, None)
//         .cast(DataType::Float32)
//         .get(0)
//         .alias(&format!("{}__linear_fit", name))
// }

pub fn variance_larger_than_standard_deviation(name: &str) -> Expr {
    let std = col(name).std(1);
    let var = col(name).var(1);
    (var.gt(std)).cast(DataType::Float32).alias(&format!(
        "{}__variance_larger_than_standard_deviation",
        name
    ))
}

fn _ratio_beyond_r_sigma(s: Series, r: f32) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let std = arr.std(1.0);
    let count = arr
        .mapv(|x| if (x - mean).abs() > r * std { 1.0 } else { 0.0 })
        .sum();
    let ratio = count as f32 / arr.len() as f32;
    let s = Series::new("", &[ratio]);
    Ok(Some(s))
}

pub fn ratio_beyond_r_sigma(name: &str, r: f32) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(move |s| _ratio_beyond_r_sigma(s, r), o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__ratio_beyond_r_sigma__r_{}", name, r))
}

fn _large_standard_deviation(s: Series, r: f32) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let min = arr.min().unwrap_or(&0.0);
    let max = arr.max().unwrap_or(&0.0);
    let std = arr.std(1.0);
    let out = std > r * (max - min);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn large_standard_deviation(name: &str, r: f32) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(move |s| _large_standard_deviation(s, r), o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__large_standard_deviation__r_{}", name, r))
}

fn _symmetry_looking(s: Series, r: f32) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let mut arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let median_res = arr.quantile_axis_skipnan_mut(Axis(0), n64(0.5), &Midpoint);
    let median = match median_res {
        Ok(m) => m[0],
        Err(_) => return Ok(None),
    };
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let mean_median_diff = (mean - median).abs();
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let max_min_diff = max - min;
    let out = mean_median_diff < r * max_min_diff;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn symmetry_looking(name: &str, r: f32) -> Expr {
    let o = GetOutput::from_type(DataType::Boolean);
    col(name)
        .apply(move |s| _symmetry_looking(s, r), o)
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__symmetry_looking__r_{}", name, r))
}

fn _has_duplicate_max(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let max_res = arr.max();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
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
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__has_duplicate_max", name))
}

fn _has_duplicate_min(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let min_res = arr.min();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
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
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__has_duplicate_min", name))
}

fn _cid_ce(s: Series, normalize: bool) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let arr = if normalize {
        let mean = arr.mean().unwrap_or(f32::NAN);
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
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(move |s| _cid_ce(s, normalize), o)
        .get(0)
        .alias(&format!("{}__cid_ce__normalize_{}", name, normalize))
}

fn _absolute_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let abs_arr = arr.mapv(|x| x.abs());
    let max_res = abs_arr.max();
    let max = match max_res {
        Ok(m) => *m,
        Err(_) => return Ok(None),
    };
    let s = Series::new("", &[max]);
    Ok(Some(s))
}

pub fn absolute_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_absolute_maximum, o)
        .get(0)
        .alias(&format!("{}__absolute_maximum", name))
}

fn _absolute_sum_of_changes(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
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
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_absolute_sum_of_changes, o)
        .get(0)
        .alias(&format!("{}__absolute_sum_of_changes", name))
}

fn _count_above_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let out = arr.mapv(|x| if x > mean { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_count_above_mean, o)
        .get(0)
        .alias(&format!("{}__count_above_mean", name))
}

fn _count_below_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let out = arr.mapv(|x| if x < mean { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_count_below_mean, o)
        .get(0)
        .alias(&format!("{}__count_below_mean", name))
}

fn _count_above(s: Series, t: f32) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_above(name: &str, t: f32) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(move |s| _count_above(s, t), o)
        .get(0)
        .alias(&format!("{}__count_above__t_{}", name, t))
}

fn _count_below(s: Series, t: f32) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let out = arr.mapv(|x| if x > t { 1.0 } else { 0.0 }).sum();
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn count_below(name: &str, t: f32) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(move |s| _count_below(s, t), o)
        .get(0)
        .alias(&format!("{}__count_below__t_{}", name, t))
}

fn _first_location_of_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let out = max as f32 / arr.len() as f32;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn first_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_first_location_of_maximum, o)
        .get(0)
        .alias(&format!("{}__first_location_of_maximum", name))
}

fn _first_location_of_minimum(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let out = min as f32 / arr.len() as f32;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn first_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_first_location_of_minimum, o)
        .get(0)
        .alias(&format!("{}__first_location_of_minimum", name))
}

fn _last_location_of_maximum(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let max_res = arr.argmax();
    let max = match max_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let out = 1.0 - (max as f32 / arr.len() as f32);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn last_location_of_maximum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_last_location_of_maximum, o)
        .get(0)
        .alias(&format!("{}__last_location_of_maximum", name))
}

fn _last_location_of_minimum(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let min_res = arr.argmin();
    let min = match min_res {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let out = 1.0 - (min as f32 / arr.len() as f32);
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn last_location_of_minimum(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_last_location_of_minimum, o)
        .get(0)
        .alias(&format!("{}__last_location_of_minimum", name))
}

fn _longest_strike_below_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let bool_arr = arr.mapv(|x| x < mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Series::new("", &[out as f32]);
    Ok(Some(s))
}

pub fn longest_strike_below_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_longest_strike_below_mean, o)
        .get(0)
        .alias(&format!("{}__longest_strike_below_mean", name))
}

fn _longest_strike_above_mean(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let bool_arr = arr.mapv(|x| x > mean);
    let out = _get_length_sequences_where(&bool_arr)
        .into_iter()
        .max()
        .unwrap_or(0);
    let s = Series::new("", &[out as f32]);
    Ok(Some(s))
}

pub fn longest_strike_above_mean(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_longest_strike_above_mean, o)
        .get(0)
        .alias(&format!("{}__longest_strike_above_mean", name))
}

fn _has_duplicate(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
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
        .cast(DataType::Float32)
        .get(0)
        .alias(&format!("{}__has_duplicate", name))
}

fn _variation_coefficient(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let mean_opt = arr.mean();
    let mean = match mean_opt {
        Some(m) => m,
        None => return Ok(None),
    };
    let std = arr.std(1.0);
    let out = if mean == 0.0 { f64::NAN } else { std / mean };
    let s = Series::new("", &[out as f32]);
    Ok(Some(s))
}

pub fn variation_coefficient(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_variation_coefficient, o)
        .get(0)
        .alias(&format!("{}__variation_coefficient", name))
}

fn _mean_change(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
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
    let s = Series::new("", &[out as f32]);
    Ok(Some(s))
}

pub fn mean_change(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_mean_change, o)
        .get(0)
        .alias(&format!("{}__mean_change", name))
}

fn _ratio_value_number_to_time_series_length(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
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
    let out = len_unique as f32 / arr.len() as f32;
    let s = Series::new("", &[out]);
    Ok(Some(s))
}

pub fn ratio_value_number_to_time_series_length(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_ratio_value_number_to_time_series_length, o)
        .get(0)
        .alias(&format!(
            "{}__ratio_value_number_to_time_series_length",
            name
        ))
}

fn _sum_of_reoccurring_values(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
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
    let s = Series::new("", &[sum as f32]);
    Ok(Some(s))
}

pub fn sum_of_reoccurring_values(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_sum_of_reoccurring_values, o)
        .get(0)
        .alias(&format!("{}__sum_of_reoccurring_values", name))
}

fn _sum_of_reoccurring_data_points(s: Series) -> Result<Option<Series>, PolarsError> {
    if s.is_empty() {
        return Ok(None);
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
    let s = Series::new("", &[sum as f32]);
    Ok(Some(s))
}

pub fn sum_of_reoccurring_data_points(name: &str) -> Expr {
    let o = GetOutput::from_type(DataType::Float32);
    col(name)
        .apply(_sum_of_reoccurring_data_points, o)
        .get(0)
        .alias(&format!("{}__sum_of_reoccurring_data_points", name))
}
