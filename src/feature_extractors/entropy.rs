//! Complexity and entropy measures.

use anyhow::Result;
use itertools::Itertools;
use ndarray::{Array1, Axis, Ix1};
use polars::lazy::dsl::*;
use polars::prelude::*;

use crate::utils::stats::population_std;

/// All length-`chunk_size` sliding windows of `x`, one step apart.
fn _into_subchunks(x: &Array1<f64>, chunk_size: usize) -> Vec<Array1<f64>> {
    let mut subchunks = Vec::with_capacity(x.len());
    for chunk in x.axis_windows(Axis(0), chunk_size) {
        subchunks.push(chunk.to_owned());
    }
    subchunks
}

/// Number of unordered pairs of templates within Chebyshev distance `r`.
fn _get_matches(templates: Vec<Array1<f64>>, r: f64) -> usize {
    let mut matches = 0;
    for combo in templates.into_iter().combinations(2) {
        let a = combo[0].to_owned();
        let b = combo[1].to_owned();
        let diff = a - b;
        // tsfresh counts a match at distance <= tolerance
        let dist_check = diff.mapv(|x| if x.abs() <= r { 1 } else { 0 }).sum();
        if dist_check == diff.len() {
            matches += 1;
        }
    }
    matches
}

fn _sample_entropy(s: Column) -> Result<Column, PolarsError> {
    if s.is_empty() {
        return Ok(Column::new("".into(), &[f64::NAN]));
    }
    let arr = s
        .into_frame()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let arr = arr
        .remove_axis(Axis(1))
        .into_dimensionality::<Ix1>()
        .unwrap();
    let m = 2;
    let r = 0.2 * population_std(&arr.view());
    let templates_m = _into_subchunks(&arr, m);
    let matches_m = _get_matches(templates_m, r);
    let templates_m_plus_1 = _into_subchunks(&arr, m + 1);
    let matches_m_plus_1 = _get_matches(templates_m_plus_1, r);
    let out = ((matches_m as f64) / (matches_m_plus_1 as f64)).ln();
    let s = Column::new("".into(), &[out]);
    Ok(s)
}

/// Sample entropy feature.
///
/// The sample entropy (SampEn) of the time series, a measure of its
/// complexity, with embedding dimension $m = 2$ and tolerance
/// $r = 0.2\,\sigma$, where $\sigma$ is the population standard deviation:
/// $$ \text{SampEn} = -\ln \frac{A}{B}, $$
/// where $B$ is the number of pairs of length-$m$ windows and $A$ the number
/// of pairs of length-$(m+1)$ windows whose Chebyshev (maximum) distance is
/// at most $r$. A regular series scores low, an irregular one high.
///
/// # Output column
/// `{name}__sample_entropy`
///
/// # Edge cases
/// - Unlike the other features, nulls are **not** dropped; a null or NaN
///   anywhere in the series gives NaN.
/// - Fewer than 4 values give NaN (no matching pairs of windows).
/// - When no length-$(m+1)$ windows match, the result is $+\infty$.
/// - **Deliberate deviation from tsfresh:** a series containing $\pm\infty$
///   gives NaN. tsfresh's tolerance becomes NaN, every match count turns
///   negative after subtracting self-matches, and the ratio of two negative
///   counts returns a finite but meaningless value.
///
/// # Cost
/// $O(n^2)$ in the series length; only computed for
/// `FeatureSetting::Comprehensive`, where it is always on (the config file
/// does not switch it off).
///
/// # tsfresh
/// `feature_calculators.sample_entropy` (v0.21.2).
pub fn sample_entropy(name: &str) -> Expr {
    col(name)
        .apply(_sample_entropy, |_, _| {
            Ok(Field::new("".into(), DataType::Float64))
        })
        .get(0, true)
        .alias(format!("{}__sample_entropy", name))
}
