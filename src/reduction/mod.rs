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

use std::collections::HashSet;

use polars::prelude::*;
use rayon::prelude::*;

use self::duplicates::{correlation_matrix, greedy_duplicates};
use self::matrix::{rank_matrix, tie_broken_rank_matrix};
use self::noise::noise_scores;
use self::rng::SplitMix64;
use self::shares::{feature_values, missing_share, top_share};
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
    /// Seed of the row sample and of the noise step (its tie-breaking and
    /// shadow features).
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

    /// Decides which feature columns of `features` to keep: every column not
    /// in `key_cols` (the grouping and time columns) is a feature. The rows
    /// are sorted by `key_cols` first, so the decision does not depend on
    /// their order as long as the keys identify the rows, as they do for
    /// extracted features (one row per group or window). Errors on invalid
    /// settings and on fewer rows than
    /// [`Reduction::min_rows`]: the reduction is fitted across the rows.
    pub fn fit(
        &self,
        features: &DataFrame,
        key_cols: &[String],
    ) -> Result<ReductionFit, ExtractionError> {
        self.validate()?;
        let n = features.height();
        let min_rows = self.min_rows();
        if n < min_rows {
            return Err(ExtractionError::InvalidSettings(format!(
                "feature reduction needs at least {min_rows} rows (groups or windows) to fit, \
                 got {n}; it is fitted across the extracted rows, so it cannot be fitted on \
                 a single series"
            )));
        }
        let names: Vec<String> = features
            .get_column_names()
            .into_iter()
            .filter(|name| !key_cols.iter().any(|key| key == name.as_str()))
            .map(|name| name.to_string())
            .collect();
        // The sorted copy lives only until the values are read
        let values = if key_cols.is_empty() {
            feature_values(features, &names)?
        } else {
            let sorted = features.sort(
                key_cols,
                SortMultipleOptions::default()
                    .with_nulls_last(true)
                    .with_maintain_order(true),
            )?;
            feature_values(&sorted, &names)?
        };
        let d = names.len();
        let mut dropped: Vec<Option<Dropped>> = vec![None; d];

        // Step 1: mostly missing, on all rows
        let missing: Option<Vec<f64>> = self.max_missing.map(|max| {
            let shares: Vec<f64> = values.par_iter().map(|v| missing_share(v)).collect();
            for (drop, &share) in dropped.iter_mut().zip(&shares) {
                if share > max {
                    *drop = Some(Dropped::Missing);
                }
            }
            shares
        });

        // Step 2: almost one value, on all rows
        let mut top: Vec<Option<f64>> = vec![None; d];
        if let Some(max) = self.max_top_share {
            let alive = remaining(&dropped);
            let shares: Vec<f64> = alive.par_iter().map(|&j| top_share(&values[j])).collect();
            for (&j, &share) in alive.iter().zip(&shares) {
                top[j] = Some(share);
                if share > max {
                    dropped[j] = Some(Dropped::OneValue);
                }
            }
        }

        // Steps 3 and 4 run on at most max_samples rows; the generator draws
        // the sample first, then the noise step's tie-breaking seeds and
        // shadows
        let mut rng = SplitMix64::new(self.seed);
        let rows: Vec<usize> = if n > self.max_samples {
            rng.sample_sorted(n, self.max_samples)
        } else {
            (0..n).collect()
        };

        // Step 3: near-duplicates
        let mut duplicate_of: Vec<Option<(usize, f64)>> = vec![None; d];
        if let Some(max) = self.max_correlation {
            let alive = remaining(&dropped);
            if !alive.is_empty() {
                let columns: Vec<&[f64]> = alive.iter().map(|&j| values[j].as_slice()).collect();
                let corr = correlation_matrix(&rank_matrix(&columns, &rows));
                // Extraction order decides who is kept (the follow-up with an
                // outcome label orders by relevance instead)
                let order: Vec<usize> = (0..alive.len()).collect();
                let duplicates = greedy_duplicates(&corr, &order, max);
                for (&j, duplicate) in alive.iter().zip(duplicates) {
                    if let Some(duplicate) = duplicate {
                        dropped[j] = Some(Dropped::Duplicate);
                        duplicate_of[j] = Some((alive[duplicate.of], duplicate.correlation));
                    }
                }
            }
        }

        // Step 4: noise
        let mut noise_score: Vec<Option<f64>> = vec![None; d];
        let mut noise_cutoff: Option<f64> = None;
        let mut warnings = Vec::new();
        if let Some(n_shadows) = self.shadow_count() {
            let alive = remaining(&dropped);
            if !alive.is_empty() {
                if alive.len() < NOISE_WARN_MIN_FEATURES {
                    warnings.push(format!(
                        "the noise step ran on {} features (fewer than \
                         {NOISE_WARN_MIN_FEATURES}): with few features it may drop real \
                         features as noise; consider a lower noise_sensitivity, or \
                         noise_sensitivity=None",
                        alive.len()
                    ));
                }
                let columns: Vec<&[f64]> = alive.iter().map(|&j| values[j].as_slice()).collect();
                // Ties broken at random, one seed per column, drawn after the
                // row sample and before the shadows: a discrete noise feature
                // is then as random as a shadow (with average ranks, its tie
                // blocks let it beat the shadows far more often)
                let seeds: Vec<u64> = columns.iter().map(|_| rng.next_u64()).collect();
                let result = noise_scores(
                    &tie_broken_rank_matrix(&columns, &rows, &seeds),
                    n_shadows,
                    self.noise_neighbors,
                    &mut rng,
                );
                for (&j, &score) in alive.iter().zip(&result.scores) {
                    noise_score[j] = Some(score);
                    // NaN is never below the cutoff
                    let beats_shadows = score < result.cutoff;
                    if !beats_shadows {
                        dropped[j] = Some(Dropped::Noise);
                    }
                }
                noise_cutoff = Some(result.cutoff);
            }
        }

        let kept: Vec<String> = names
            .iter()
            .zip(&dropped)
            .filter(|(_, drop)| drop.is_none())
            .map(|(name, _)| name.clone())
            .collect();
        let kept_flags: Vec<bool> = dropped.iter().map(Option::is_none).collect();
        let dropped_by: Vec<Option<&str>> =
            dropped.iter().map(|drop| drop.map(Dropped::name)).collect();
        let missing_share: Vec<Option<f64>> = match missing {
            Some(shares) => shares.into_iter().map(Some).collect(),
            None => vec![None; d],
        };
        let duplicate_names: Vec<Option<&str>> = duplicate_of
            .iter()
            .map(|dup| dup.map(|(k, _)| names[k].as_str()))
            .collect();
        let correlation: Vec<Option<f64>> =
            duplicate_of.iter().map(|dup| dup.map(|(_, r)| r)).collect();
        let cutoff: Vec<Option<f64>> = noise_score
            .iter()
            .map(|score| score.and(noise_cutoff))
            .collect();
        let report = df![
            "feature" => names.as_slice(),
            "kept" => kept_flags,
            "dropped_by" => dropped_by,
            "missing_share" => missing_share,
            "top_share" => top,
            "duplicate_of" => duplicate_names,
            "correlation" => correlation,
            "noise_score" => noise_score,
            "noise_cutoff" => cutoff,
        ]?;
        Ok(ReductionFit {
            kept,
            report,
            warnings,
        })
    }
}

/// Which step dropped a feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dropped {
    Missing,
    OneValue,
    Duplicate,
    Noise,
}

impl Dropped {
    /// The name in the report's `dropped_by` column.
    fn name(self) -> &'static str {
        match self {
            Dropped::Missing => "missing",
            Dropped::OneValue => "one_value",
            Dropped::Duplicate => "duplicate",
            Dropped::Noise => "noise",
        }
    }
}

/// Indices of the features no step has dropped yet.
fn remaining(dropped: &[Option<Dropped>]) -> Vec<usize> {
    (0..dropped.len())
        .filter(|&j| dropped[j].is_none())
        .collect()
}

/// A fitted reduction: which feature columns to keep and why. Plain data, so
/// it can be kept and applied to other frames with the same columns.
#[derive(Clone, Debug)]
pub struct ReductionFit {
    /// The kept feature columns, in the frame's column order.
    pub kept: Vec<String>,
    /// One row per feature column, in the frame's column order: `feature`,
    /// `kept`, `dropped_by` (`missing`, `one_value`, `duplicate`, `noise` or
    /// null), `missing_share`, `top_share`, `duplicate_of`, `correlation`
    /// (signed Spearman ρ with `duplicate_of`), `noise_score` and
    /// `noise_cutoff`. A step that is turned off, or that a feature did not
    /// reach, leaves its columns null.
    pub report: DataFrame,
    /// Warnings of the fit, e.g. too few features for the noise step.
    pub warnings: Vec<String>,
}

impl ReductionFit {
    /// `features` with only the `key_cols` and the kept feature columns, in
    /// the frame's column order and with their dtypes unchanged.
    pub fn apply(&self, features: DataFrame, key_cols: &[String]) -> PolarsResult<DataFrame> {
        let wanted: HashSet<&str> = key_cols
            .iter()
            .chain(&self.kept)
            .map(String::as_str)
            .collect();
        let selected: Vec<PlSmallStr> = features
            .get_column_names()
            .into_iter()
            .filter(|name| wanted.contains(name.as_str()))
            .cloned()
            .collect();
        features.select(selected)
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

#[cfg(test)]
mod fit_tests {
    use super::Reduction;
    use super::rng::SplitMix64;
    use polars::prelude::*;

    fn uniform(g: &mut SplitMix64) -> f64 {
        (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn keys() -> Vec<String> {
        vec!["id".to_string()]
    }

    /// 40 rows: "miss" 10% NaN, "const", "a", "a_sq" (monotone in a), "b".
    fn crafted() -> DataFrame {
        let mut g = SplitMix64::new(7);
        let n = 40;
        let a: Vec<f64> = (0..n).map(|_| uniform(&mut g)).collect();
        let b: Vec<f64> = (0..n).map(|_| uniform(&mut g)).collect();
        let miss: Vec<f64> = (0..n)
            .map(|i| if i % 10 == 0 { f64::NAN } else { i as f64 })
            .collect();
        let a_sq: Vec<f64> = a.iter().map(|x| x * x).collect();
        df![
            "id" => (0..n as i64).collect::<Vec<_>>(),
            "miss" => miss,
            "const" => vec![1.0; n],
            "a" => a,
            "a_sq" => a_sq,
            "b" => b,
        ]
        .unwrap()
    }

    fn strs(df: &DataFrame, name: &str) -> Vec<Option<String>> {
        df.column(name)
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|s| s.map(str::to_string))
            .collect()
    }

    #[test]
    fn test_fit_steps_and_report() {
        let r = Reduction {
            noise_sensitivity: None,
            ..Default::default()
        };
        let fit = r.fit(&crafted(), &keys()).unwrap();
        assert_eq!(fit.kept, ["a", "b"]);
        let rep = &fit.report;
        let features: Vec<String> = strs(rep, "feature").into_iter().flatten().collect();
        assert_eq!(features, ["miss", "const", "a", "a_sq", "b"]);
        let some = |s: &str| Some(s.to_string());
        assert_eq!(
            strs(rep, "dropped_by"),
            [
                some("missing"),
                some("one_value"),
                None,
                some("duplicate"),
                None
            ]
        );
        assert_eq!(strs(rep, "duplicate_of")[3], some("a"));
        let corr = rep
            .column("correlation")
            .unwrap()
            .f64()
            .unwrap()
            .get(3)
            .unwrap();
        assert!((corr - 1.0).abs() < 1e-12, "{corr}");
        let missing = rep.column("missing_share").unwrap().f64().unwrap();
        assert_eq!(missing.get(0), Some(0.1));
        assert_eq!(missing.get(2), Some(0.0));
        let top = rep.column("top_share").unwrap().f64().unwrap();
        assert_eq!(top.get(0), None); // dropped before step 2
        assert_eq!(top.get(1), Some(1.0));
        assert_eq!(rep.column("noise_score").unwrap().null_count(), 5);
        assert_eq!(rep.column("noise_cutoff").unwrap().null_count(), 5);
        assert!(fit.warnings.is_empty());
        let kept = rep.column("kept").unwrap().bool().unwrap();
        assert_eq!(
            kept.iter().collect::<Vec<_>>(),
            [false, false, true, false, true].map(Some)
        );
    }

    #[test]
    fn test_apply_keeps_keys_and_order() {
        let r = Reduction {
            noise_sensitivity: None,
            ..Default::default()
        };
        let df = crafted();
        let fit = r.fit(&df, &keys()).unwrap();
        let out = fit.apply(df, &keys()).unwrap();
        assert_eq!(column_names(&out), ["id", "a", "b"]);
    }

    fn column_names(df: &DataFrame) -> Vec<String> {
        df.get_column_names()
            .into_iter()
            .map(|name| name.to_string())
            .collect()
    }

    #[test]
    fn test_all_dropped_keeps_keys() {
        let df = df![
            "id" => (0..20i64).collect::<Vec<_>>(),
            "c" => [1.0; 20],
        ]
        .unwrap();
        let fit = Reduction::default().fit(&df, &keys()).unwrap();
        assert!(fit.kept.is_empty());
        assert!(fit.warnings.is_empty());
        assert_eq!(column_names(&fit.apply(df, &keys()).unwrap()), ["id"]);
    }

    #[test]
    fn test_only_key_columns() {
        let df = df!["id" => (0..20i64).collect::<Vec<_>>()].unwrap();
        let fit = Reduction::default().fit(&df, &keys()).unwrap();
        assert!(fit.kept.is_empty());
        assert_eq!(fit.report.height(), 0);
        assert_eq!(fit.report.width(), 9);
        let dtypes: Vec<DataType> = fit.report.dtypes();
        assert_eq!(
            dtypes,
            [
                DataType::String,
                DataType::Boolean,
                DataType::String,
                DataType::Float64,
                DataType::Float64,
                DataType::String,
                DataType::Float64,
                DataType::Float64,
                DataType::Float64,
            ]
        );
    }

    #[test]
    fn test_too_few_rows() {
        let df = df!["id" => [0i64, 1, 2, 3, 4], "x" => [1.0, 2.0, 3.0, 4.0, 5.0]].unwrap();
        let err = Reduction::default()
            .fit(&df, &keys())
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 11 rows"), "{err}");
        assert!(err.contains("got 5"), "{err}");
    }

    #[test]
    fn test_invalid_settings_fail_the_fit() {
        let r = Reduction {
            max_missing: Some(2.0),
            ..Default::default()
        };
        assert!(r.fit(&crafted(), &keys()).is_err());
    }

    #[test]
    fn test_noise_warning() {
        // crafted() leaves 2 features for the noise step
        let fit = Reduction::default().fit(&crafted(), &keys()).unwrap();
        assert_eq!(fit.warnings.len(), 1);
        assert!(
            fit.warnings[0].contains("noise_sensitivity"),
            "{}",
            fit.warnings[0]
        );
        assert_eq!(fit.report.column("noise_score").unwrap().null_count(), 3);
    }

    #[test]
    fn test_disabled_steps_leave_nulls() {
        let r = Reduction {
            max_missing: None,
            max_top_share: None,
            max_correlation: None,
            noise_sensitivity: None,
            ..Default::default()
        };
        let fit = r.fit(&crafted(), &keys()).unwrap();
        assert_eq!(fit.kept, ["miss", "const", "a", "a_sq", "b"]);
        for name in [
            "dropped_by",
            "missing_share",
            "top_share",
            "duplicate_of",
            "correlation",
            "noise_score",
            "noise_cutoff",
        ] {
            assert_eq!(fit.report.column(name).unwrap().null_count(), 5, "{name}");
        }
    }

    /// The report's columns as comparable values: floats as bit patterns
    /// (any NaN as one value), others as strings.
    fn report_bits(df: &DataFrame) -> Vec<Vec<Option<String>>> {
        df.columns()
            .iter()
            .map(|c| match c.dtype() {
                DataType::Float64 => c
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(|x| if x.is_nan() { u64::MAX } else { x.to_bits() }.to_string()))
                    .collect(),
                _ => c
                    .cast(&DataType::String)
                    .unwrap()
                    .str()
                    .unwrap()
                    .iter()
                    .map(|s| s.map(str::to_string))
                    .collect(),
            })
            .collect()
    }

    /// Same kept columns and a bit-identical report for 1 and 4 threads and
    /// any row order, on the sampled path (120 rows > max_samples = 50).
    #[test]
    fn test_deterministic_across_threads_and_row_order() {
        let (n, d) = (120, 60);
        let mut g = SplitMix64::new(9);
        let mut columns = vec![Column::new("id".into(), (0..n as i64).collect::<Vec<_>>())];
        for j in 0..d {
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    let u = uniform(&mut g);
                    if j % 3 == 0 {
                        (i % 4) as f64 + u
                    } else if j % 7 == 1 {
                        // discrete: the noise step breaks its ties at random
                        f64::from(u < 0.3)
                    } else {
                        u
                    }
                })
                .collect();
            columns.push(Column::new(format!("f{j}").into(), values));
        }
        let df = DataFrame::new_infer_height(columns).unwrap();
        let mut order: Vec<IdxSize> = (0..n as IdxSize).collect();
        SplitMix64::new(10).shuffle(&mut order);
        let shuffled = df.take(&IdxCa::from_vec("idx".into(), order)).unwrap();

        let r = Reduction {
            noise_neighbors: 5,
            max_samples: 50,
            ..Default::default()
        };
        let fit_in = |threads: usize, frame: &DataFrame| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| r.fit(frame, &keys()).unwrap())
        };
        let one = fit_in(1, &df);
        let four = fit_in(4, &shuffled);
        // Not vacuous: features reached the noise step
        assert!(one.report.column("noise_score").unwrap().null_count() < d);
        assert_eq!(one.kept, four.kept);
        assert_eq!(report_bits(&one.report), report_bits(&four.report));
        assert_eq!(one.warnings, four.warnings);
    }

    /// A constant and an all-missing feature that reach the noise step (the
    /// earlier steps off) score NaN and are dropped as noise. Regression:
    /// random tie-breaking turned them into random permutations, which got
    /// through 1% of the time.
    #[test]
    fn test_constant_features_are_noise() {
        let n = 60;
        let mut g = SplitMix64::new(11);
        let mut columns = vec![Column::new("id".into(), (0..n as i64).collect::<Vec<_>>())];
        for j in 0..50 {
            let values: Vec<f64> = (0..n)
                .map(|i| (i % 3) as f64 + 0.3 * uniform(&mut g) + j as f64)
                .collect();
            columns.push(Column::new(format!("f{j}").into(), values));
        }
        columns.push(Column::new("constant".into(), vec![2.0; n]));
        columns.push(Column::new("missing".into(), vec![f64::NAN; n]));
        let df = DataFrame::new_infer_height(columns).unwrap();
        let r = Reduction {
            max_missing: None,
            max_top_share: None,
            max_correlation: None,
            ..Default::default()
        };
        let fit = r.fit(&df, &keys()).unwrap();
        let rep = &fit.report;
        let dropped_by = strs(rep, "dropped_by");
        let scores = rep.column("noise_score").unwrap().f64().unwrap();
        for i in [50, 51] {
            assert_eq!(dropped_by[i].as_deref(), Some("noise"), "row {i}");
            assert!(scores.get(i).unwrap().is_nan(), "row {i}");
        }
    }
}
