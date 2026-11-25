//! Reduction of the extracted features before scaling: drop features that are
//! mostly missing, almost always one value, near-duplicates of an earlier
//! feature, or noise (no better than shuffled "shadow" features at respecting
//! the neighbor structure of the rows).
//!
//! The steps decide which columns remain; the values of the columns kept are
//! not changed. Missing means null, NaN or ±inf. Thresholds are strict: a
//! feature is dropped when its value is above the threshold.

mod duplicates;
mod matrix;
mod noise;
mod rng;
mod shares;

use crate::error::ExtractionError;

/// With fewer features than this reaching the noise step, the fit warns: the
/// shadow features then make up much of the distance between rows, and the
/// step drops real features by mistake.
pub const NOISE_WARN_MIN_FEATURES: usize = 50;

/// Fewest rows (groups or windows) any reduction is fitted on.
const MIN_ROWS: usize = 10;

/// Largest `noise_sensitivity`: 999 shadow features.
const MAX_NOISE_SENSITIVITY: f64 = 0.999;

/// Settings of the feature reduction. Each step is turned off with `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    /// Drop a feature whose share of missing values is above this, in [0, 1).
    pub max_missing: Option<f64>,
    /// Drop a feature whose most frequent value covers more than this share
    /// of its non-missing values, in (0, 1).
    pub max_top_share: Option<f64>,
    /// Drop a feature whose |Spearman correlation| with an earlier kept
    /// feature is above this, in (0, 1).
    pub max_correlation: Option<f64>,
    /// About this share of pure-noise features is dropped by the noise step,
    /// in (0, 0.999].
    pub noise_sensitivity: Option<f64>,
    /// Nearest rows per row in the noise step's neighbor graph, at least 1.
    pub noise_neighbors: usize,
    /// Row cap for the correlation and noise steps, above `noise_neighbors`.
    pub max_samples: usize,
    /// Seed of the row sample and the shadow features.
    pub seed: u64,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction {
            max_missing: Some(0.05),
            max_top_share: Some(0.95),
            max_correlation: Some(0.95),
            noise_sensitivity: Some(0.99),
            noise_neighbors: 10,
            max_samples: 5000,
            seed: 0,
        }
    }
}

impl Reduction {
    /// Checks every setting's range; the error names the field and its value.
    pub fn validate(&self) -> Result<(), ExtractionError> {
        let invalid = |name: &str, range: &str, value: f64| {
            Err(ExtractionError::InvalidSettings(format!(
                "ReductionSetting {name} must be in {range}, got {value}"
            )))
        };
        // NaN fails every range
        let open_unit = |v: f64| v > 0.0 && v < 1.0;
        match self.max_missing {
            Some(v) if !(0.0..1.0).contains(&v) => return invalid("max_missing", "[0, 1)", v),
            _ => {}
        }
        match self.max_top_share {
            Some(v) if !open_unit(v) => return invalid("max_top_share", "(0, 1)", v),
            _ => {}
        }
        match self.max_correlation {
            Some(v) if !open_unit(v) => return invalid("max_correlation", "(0, 1)", v),
            _ => {}
        }
        match self.noise_sensitivity {
            Some(v) if !(v > 0.0 && v <= MAX_NOISE_SENSITIVITY) => {
                return invalid("noise_sensitivity", "(0, 0.999]", v);
            }
            _ => {}
        }
        if self.noise_neighbors < 1 {
            return Err(ExtractionError::InvalidSettings(
                "ReductionSetting noise_neighbors must be at least 1, got 0".to_string(),
            ));
        }
        if self.max_samples <= self.noise_neighbors {
            return Err(ExtractionError::InvalidSettings(format!(
                "ReductionSetting max_samples must be larger than noise_neighbors ({}), got {}",
                self.noise_neighbors, self.max_samples
            )));
        }
        Ok(())
    }

    /// Number of shadow features of the noise step, `None` without it: a
    /// pure-noise feature beats all of them with probability
    /// 1 / (shadows + 1) = 1 - `noise_sensitivity`.
    pub fn shadow_count(&self) -> Option<usize> {
        self.noise_sensitivity.map(|s| {
            ((1.0 / (1.0 - s)).round() as usize)
                .saturating_sub(1)
                .max(1)
        })
    }

    /// Fewest rows (groups or windows) this reduction is fitted on.
    pub fn min_rows(&self) -> usize {
        match self.noise_sensitivity {
            Some(_) => MIN_ROWS.max(self.noise_neighbors + 1),
            None => MIN_ROWS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Reduction;
    use crate::error::ExtractionError;

    #[test]
    fn test_shadow_count() {
        for (s, n) in [(0.99, 99), (0.95, 19), (0.5, 1), (0.2, 1), (0.999, 999)] {
            let r = Reduction {
                noise_sensitivity: Some(s),
                ..Default::default()
            };
            assert_eq!(r.shadow_count(), Some(n), "{s}");
        }
        let r = Reduction {
            noise_sensitivity: None,
            ..Default::default()
        };
        assert_eq!(r.shadow_count(), None);
    }

    #[test]
    fn test_validate() {
        assert!(Reduction::default().validate().is_ok());
        let bad = [
            Reduction {
                max_missing: Some(1.0),
                ..Default::default()
            },
            Reduction {
                max_missing: Some(-0.1),
                ..Default::default()
            },
            Reduction {
                max_top_share: Some(0.0),
                ..Default::default()
            },
            Reduction {
                max_top_share: Some(1.0),
                ..Default::default()
            },
            Reduction {
                max_correlation: Some(1.0),
                ..Default::default()
            },
            Reduction {
                noise_sensitivity: Some(0.0),
                ..Default::default()
            },
            Reduction {
                noise_sensitivity: Some(0.9995),
                ..Default::default()
            },
            Reduction {
                noise_sensitivity: Some(f64::NAN),
                ..Default::default()
            },
            Reduction {
                max_missing: Some(f64::NAN),
                ..Default::default()
            },
            Reduction {
                noise_neighbors: 0,
                ..Default::default()
            },
            Reduction {
                max_samples: 10,
                noise_neighbors: 10,
                ..Default::default()
            },
        ];
        for r in bad {
            assert!(
                matches!(r.validate(), Err(ExtractionError::InvalidSettings(_))),
                "{r:?}"
            );
        }
        assert!(
            Reduction {
                max_missing: Some(0.0),
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        let all_off = Reduction {
            max_missing: None,
            max_top_share: None,
            max_correlation: None,
            noise_sensitivity: None,
            ..Default::default()
        };
        assert!(all_off.validate().is_ok());
    }

    #[test]
    fn test_validate_message_names_the_field() {
        let r = Reduction {
            max_missing: Some(1.0),
            ..Default::default()
        };
        let err = r.validate().unwrap_err().to_string();
        assert!(
            err.contains("max_missing must be in [0, 1), got 1"),
            "{err}"
        );
    }

    #[test]
    fn test_min_rows() {
        assert_eq!(Reduction::default().min_rows(), 11);
        let no_noise = Reduction {
            noise_sensitivity: None,
            ..Default::default()
        };
        assert_eq!(no_noise.min_rows(), 10);
        let many_neighbors = Reduction {
            noise_neighbors: 30,
            ..Default::default()
        };
        assert_eq!(many_neighbors.min_rows(), 31);
    }
}
