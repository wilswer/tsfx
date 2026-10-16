//! Column-wise scaling of extracted features across groups.
//!
//! The scalers follow scikit-learn's `StandardScaler` and `MinMaxScaler`
//! (`fit_transform` on the feature matrix): NaN and null are ignored when
//! fitting and kept in the output, as are infinities (which scikit-learn
//! rejects), and near-constant features get a scale of
//! 1, so they map to 0 (`Standard`) or the lower end of the range (`MinMax`).

use polars::prelude::*;

use crate::error::ExtractionError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scaling {
    /// Zero mean and unit (population) variance.
    Standard,
    /// Linear map of each feature's `[min, max]` onto `feature_range`.
    MinMax { feature_range: (f64, f64) },
}

impl Scaling {
    pub fn validate(&self) -> Result<(), ExtractionError> {
        match *self {
            Scaling::Standard => Ok(()),
            Scaling::MinMax {
                feature_range: (lo, hi),
            } => {
                if !(lo.is_finite() && hi.is_finite()) {
                    Err(ExtractionError::InvalidSettings(format!(
                        "MinMax feature_range must be finite, got ({lo}, {hi})"
                    )))
                } else if lo >= hi {
                    Err(ExtractionError::InvalidSettings(format!(
                        "MinMax feature_range minimum must be smaller than its maximum, got ({lo}, {hi})"
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Expression replacing column `name` with its scaled Float64 values.
    pub fn expr(&self, name: &str) -> Expr {
        let x = col(name).cast(DataType::Float64);
        // Fit on the finite values: NaN counts as missing, as in scikit-learn
        // (which rejects infinities; here they stay infinite, sign kept)
        let fit = when(x.clone().is_finite())
            .then(x.clone())
            .otherwise(lit(NULL));
        let scaled = match *self {
            Scaling::Standard => {
                let mean = fit.clone().mean();
                let var = fit.clone().var(0);
                let n = fit.count().cast(DataType::Float64);
                // scikit-learn's `_is_constant_feature`: the variance is
                // within the rounding error of the two-pass algorithm
                let eps = lit(f64::EPSILON);
                let bound = n.clone() * eps.clone() * var.clone() + (n * mean.clone() * eps).pow(2);
                let scale = when(var.clone().lt_eq(bound))
                    .then(lit(1.0))
                    .otherwise(var.sqrt());
                (x.clone() - mean) / scale
            }
            Scaling::MinMax {
                feature_range: (lo, hi),
            } => {
                let data_min = fit.clone().min();
                let range = fit.max() - data_min.clone();
                // scikit-learn's `_handle_zeros_in_scale`
                let range = when(range.clone().lt(lit(10.0 * f64::EPSILON)))
                    .then(lit(1.0))
                    .otherwise(range);
                let scale = lit(hi - lo) / range;
                let offset = lit(lo) - data_min * scale.clone();
                x.clone() * scale + offset
            }
        };
        // Non-finite values pass through, also where a feature has no finite
        // value to fit on (the scaled value would be null then, not NaN)
        when(x.clone().is_finite().not())
            .then(x)
            .otherwise(scaled)
            .alias(name)
    }
}

/// Scale every column of `df` except those in `keep` (grouping and time
/// columns).
pub fn scale_features(df: DataFrame, scaling: &Scaling, keep: &[String]) -> LazyFrame {
    let exprs: Vec<Expr> = df
        .get_column_names()
        .into_iter()
        .filter(|name| !keep.iter().any(|k| k == name.as_str()))
        .map(|name| scaling.expr(name))
        .collect();
    df.lazy().with_columns(exprs)
}

#[cfg(test)]
mod tests {
    use super::{Scaling, scale_features};
    use polars::prelude::*;

    fn scaled(values: &[f64], scaling: Scaling) -> Vec<Option<f64>> {
        let df = df![
            "id" => (0..values.len() as i64).collect::<Vec<_>>(),
            "v" => values,
        ]
        .unwrap();
        let out = scale_features(df, &scaling, &["id".to_string()])
            .collect()
            .unwrap();
        assert_eq!(
            out.column("id").unwrap().i64().unwrap().to_vec(),
            (0..values.len() as i64).map(Some).collect::<Vec<_>>()
        );
        out.column("v").unwrap().f64().unwrap().to_vec()
    }

    #[test]
    fn test_standard() {
        let s = 1.25_f64.sqrt();
        assert_eq!(
            scaled(&[1.0, 2.0, 3.0, 4.0], Scaling::Standard),
            [-1.5 / s, -0.5 / s, 0.5 / s, 1.5 / s].map(Some)
        );
        assert_eq!(scaled(&[7.0, 7.0], Scaling::Standard), [Some(0.0); 2]);
    }

    #[test]
    fn test_min_max() {
        let scaling = Scaling::MinMax {
            feature_range: (-1.0, 1.0),
        };
        assert_eq!(
            scaled(&[2.0, 4.0, 6.0], scaling),
            [-1.0, 0.0, 1.0].map(Some)
        );
        assert_eq!(scaled(&[7.0, 7.0], scaling), [Some(-1.0); 2]);
    }

    #[test]
    fn test_non_finite_pass_through() {
        for scaling in [
            Scaling::Standard,
            Scaling::MinMax {
                feature_range: (-1.0, 1.0),
            },
        ] {
            let out = scaled(&[1.0, f64::NAN, 3.0, f64::INFINITY], scaling);
            assert_eq!(out[0], Some(-1.0));
            assert!(out[1].unwrap().is_nan());
            assert_eq!(out[2], Some(1.0));
            assert_eq!(out[3], Some(f64::INFINITY));
            assert!(scaled(&[f64::NAN], scaling)[0].unwrap().is_nan());
        }
    }

    #[test]
    fn test_validate() {
        assert!(Scaling::Standard.validate().is_ok());
        for (lo, hi, ok) in [
            (0.0, 1.0, true),
            (1.0, 1.0, false),
            (2.0, 1.0, false),
            (0.0, f64::INFINITY, false),
            (f64::NAN, 1.0, false),
        ] {
            let scaling = Scaling::MinMax {
                feature_range: (lo, hi),
            };
            assert_eq!(scaling.validate().is_ok(), ok, "({lo}, {hi})");
        }
    }
}
