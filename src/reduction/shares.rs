//! The per-column numbers of the first two steps: the share of missing values
//! and the share of the most frequent value.

use polars::prelude::*;

/// The values of column `name` of `df` as f64, with null, NaN and ±inf
/// (missing values) as NaN.
pub(crate) fn column_values(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let column = df.column(name)?.cast(&DataType::Float64)?;
    Ok(column
        .f64()?
        .iter()
        .map(|v| match v {
            Some(x) if x.is_finite() => x,
            _ => f64::NAN,
        })
        .collect())
}

/// The share of missing values (NaN) in `xs`.
pub(crate) fn missing_share(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().filter(|x| x.is_nan()).count() as f64 / xs.len() as f64
}

/// Among the non-missing values of `xs`, the share of the most frequent exact
/// value, with -0.0 counted as 0.0; 1.0 when no value is non-missing.
pub(crate) fn top_share(xs: &[f64]) -> f64 {
    // x + 0.0 turns -0.0 into 0.0
    let mut bits: Vec<u64> = xs
        .iter()
        .filter(|x| !x.is_nan())
        .map(|&x| (x + 0.0).to_bits())
        .collect();
    if bits.is_empty() {
        return 1.0;
    }
    bits.sort_unstable();
    let (mut best, mut run) = (1, 1);
    for pair in bits.windows(2) {
        if pair[0] == pair[1] {
            run += 1;
            best = best.max(run);
        } else {
            run = 1;
        }
    }
    best as f64 / bits.len() as f64
}

#[cfg(test)]
mod tests {
    use super::{column_values, missing_share, top_share};
    use polars::prelude::*;

    #[test]
    fn test_missing_share() {
        assert_eq!(missing_share(&[1.0, f64::NAN, f64::NAN, 4.0]), 0.5);
        assert_eq!(missing_share(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_top_share() {
        assert_eq!(top_share(&[0.0, -0.0, 1.0, f64::NAN]), 2.0 / 3.0);
        assert_eq!(top_share(&[f64::NAN, f64::NAN]), 1.0);
        assert_eq!(top_share(&[1.0, 2.0, 3.0, 4.0]), 0.25);
        assert_eq!(top_share(&[5.0, 5.0, 5.0]), 1.0);
    }

    #[test]
    fn test_feature_values_missing_as_nan() {
        let df = df![
            "x" => [Some(1.0), None, Some(f64::INFINITY), Some(f64::NAN), Some(f64::NEG_INFINITY)],
            "n" => [Some(1u32), Some(2), None, Some(4), Some(5)],
        ]
        .unwrap();
        let v = [
            column_values(&df, "x").unwrap(),
            column_values(&df, "n").unwrap(),
        ];
        assert_eq!(v[0][0], 1.0);
        assert!(v[0][1..].iter().all(|x| x.is_nan()));
        assert_eq!(v[1][..2], [1.0, 2.0]);
        assert!(v[1][2].is_nan());
        assert_eq!(v[1][3..], [4.0, 5.0]);
    }
}
