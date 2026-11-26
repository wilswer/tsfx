//! Reduction of the extracted features before scaling: drop features that are
//! mostly missing, almost always one value, near-duplicates of an earlier
//! feature, or noise (no better than shuffled "shadow" features at respecting
//! the neighbor structure of the rows); then, with a cap, keep only the most
//! informative of the rest (greedy column subset selection).
//!
//! The steps decide which columns remain; the values of the columns kept are
//! not changed. Missing means null, NaN or ±inf. Thresholds are strict: a
//! feature is dropped when its value is above the threshold.

mod cap;
mod duplicates;
mod matrix;
mod noise;
mod rng;
mod shares;

use ndarray::Array2;
use polars::prelude::*;
use rayon::prelude::*;

use self::cap::{CapStop, greedy_subset};
use self::duplicates::{correlation_matrix, greedy_duplicates};
use self::matrix::{rank_matrix, tie_broken_rank_matrix};
use self::noise::noise_scores;
use self::rng::SplitMix64;
use self::shares::{column_values, missing_share, top_share};
use crate::error::ExtractionError;

/// With fewer features than this reaching the noise step, the fit warns: the
/// shadow features then make up much of the distance between rows, and the
/// step drops real features by mistake.
pub const NOISE_WARN_MIN_FEATURES: usize = 50;

/// With fewer rows than this in the noise step, the fit warns: the neighbor
/// graph is then coarse, and the step drops many real features as noise (in
/// tests, features of medium strength: 45% at 20 rows, 6% at 40, 2% at 60).
pub const NOISE_WARN_MIN_ROWS: usize = 50;

/// Fewest rows (groups or windows) any reduction is fitted on.
const MIN_ROWS: usize = 10;

/// Smallest `noise_sensitivity`: 1 shadow feature.
const MIN_NOISE_SENSITIVITY: f64 = 0.5;

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
    /// in [0.5, 0.999].
    pub noise_sensitivity: Option<f64>,
    /// Nearest rows per row in the noise step's neighbor graph, at least 1.
    pub noise_neighbors: usize,
    /// Row cap for the correlation and noise steps, above `noise_neighbors`.
    pub max_samples: usize,
    /// Seed of the row sample and of the noise step (its tie-breaking and
    /// shadow features).
    pub seed: u64,
    /// The cap (step 5), `None` without it.
    pub cap: Option<Cap>,
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
            cap: None,
        }
    }
}

/// Settings of the cap (step 5): keep the most informative features by greedy
/// column subset selection, at most `max_features` and until `min_explained`
/// of the candidates' spread is explained (1.0: until nothing is left to
/// explain), never more than the rows used − 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Cap {
    /// Keep at most this many features, at least 1; `None`: no count limit.
    pub max_features: Option<usize>,
    /// Stop once the kept features explain this share of the candidates'
    /// spread, in (0, 1]; 1.0: until nothing is left to explain.
    pub min_explained: f64,
}

impl Default for Cap {
    fn default() -> Self {
        Cap {
            max_features: None,
            min_explained: 1.0,
        }
    }
}

impl Cap {
    /// Checks both settings' ranges; the error names the field and its value.
    pub fn validate(&self) -> Result<(), ExtractionError> {
        if self.max_features == Some(0) {
            return Err(ExtractionError::InvalidSettings(
                "CapSetting max_features must be at least 1, got 0".to_string(),
            ));
        }
        // NaN fails the range
        if !(self.min_explained > 0.0 && self.min_explained <= 1.0) {
            return Err(ExtractionError::InvalidSettings(format!(
                "CapSetting min_explained must be in (0, 1], got {}",
                self.min_explained
            )));
        }
        Ok(())
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
            Some(v) if !(MIN_NOISE_SENSITIVITY..=MAX_NOISE_SENSITIVITY).contains(&v) => {
                return invalid("noise_sensitivity", "[0.5, 0.999]", v);
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
        if let Some(cap) = &self.cap {
            cap.validate()?;
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
    /// in `key_cols` (the grouping and time columns) is a feature. Steps 3 to
    /// 5 read the rows in the order of `key_cols`, so the decision does not
    /// depend on the input row order as long as the keys identify the rows,
    /// as they do for extracted features (one row per group or window).
    /// Memory: the shares of steps 1 and 2 are computed column by column, and
    /// steps 3 to 5 hold only the sampled rows (at most `max_samples`) of the
    /// features left, plus step 3's correlation matrix: 8 · d² bytes for the d
    /// features left after step 2 (about 1.8 GB at 15,000 features). With a
    /// cap, that matrix is kept through step 4 (or computed for the cap's
    /// candidates when step 3 is off), and the cap adds one vector of the
    /// candidates' length per pick. Errors on invalid settings and on fewer
    /// rows than [`Reduction::min_rows`]: the reduction is fitted across the
    /// rows.
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
        let d = names.len();
        let mut dropped: Vec<Option<Dropped>> = vec![None; d];

        // Steps 1 and 2 do not depend on the row order: per column, on all
        // rows, in parallel, keeping only the shares (not the values)
        let shares: Vec<(f64, Option<f64>)> = names
            .par_iter()
            .map(|name| {
                let values = column_values(features, name)?;
                let missing = missing_share(&values);
                let reaches_step_2 = self.max_missing.is_none_or(|max| missing <= max);
                let top =
                    (self.max_top_share.is_some() && reaches_step_2).then(|| top_share(&values));
                Ok((missing, top))
            })
            .collect::<PolarsResult<_>>()?;
        // Step 1: mostly missing
        if let Some(max) = self.max_missing {
            for (drop, (missing, _)) in dropped.iter_mut().zip(&shares) {
                if *missing > max {
                    *drop = Some(Dropped::Missing);
                }
            }
        }
        // Step 2: almost one value
        if let Some(max) = self.max_top_share {
            for (drop, (_, top)) in dropped.iter_mut().zip(&shares) {
                if top.is_some_and(|share| share > max) {
                    *drop = Some(Dropped::OneValue);
                }
            }
        }

        // Steps 3 to 5 run on at most max_samples rows, in the order of the
        // key columns; the generator draws the sample first, then the noise
        // step's tie-breaking seeds and shadows
        let order = key_order(features, key_cols)?;
        let mut rng = SplitMix64::new(self.seed);
        let rows: Vec<usize> = if n > self.max_samples {
            rng.sample_sorted(n, self.max_samples)
                .into_iter()
                .map(|position| order[position])
                .collect()
        } else {
            order
        };
        // Only the sampled rows of the features left after step 2 are read
        let candidates = remaining(&dropped);
        let sampled: Vec<Vec<f64>> = if self.max_correlation.is_some()
            || self.noise_sensitivity.is_some()
            || self.cap.is_some()
        {
            candidates
                .par_iter()
                .map(|&j| {
                    let values = column_values(features, &names[j])?;
                    Ok(rows.iter().map(|&i| values[i]).collect())
                })
                .collect::<PolarsResult<_>>()?
        } else {
            Vec::new()
        };
        let mut position: Vec<Option<usize>> = vec![None; d];
        for (k, &j) in candidates.iter().enumerate() {
            position[j] = Some(k);
        }
        let sampled_columns = |subset: &[usize]| -> Vec<&[f64]> {
            subset
                .iter()
                .map(|&j| sampled[position[j].expect("read after step 2")].as_slice())
                .collect()
        };
        let sample_rows: Vec<usize> = (0..rows.len()).collect();

        // Step 3: near-duplicates
        let mut duplicate_of: Vec<Option<(usize, f64)>> = vec![None; d];
        // Step 3's features and correlation matrix, kept for the cap
        let mut step_3_correlations: Option<(Vec<usize>, Array2<f64>)> = None;
        if let Some(max) = self.max_correlation {
            let alive = remaining(&dropped);
            if !alive.is_empty() {
                let corr = correlation_matrix(&rank_matrix(&sampled_columns(&alive), &sample_rows));
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
                if self.cap.is_some() {
                    step_3_correlations = Some((alive, corr));
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
                if rows.len() < NOISE_WARN_MIN_ROWS {
                    warnings.push(format!(
                        "the noise step ran on {} rows (fewer than {NOISE_WARN_MIN_ROWS}): with \
                         few rows it drops many real features as noise; consider more groups \
                         or windows (or a larger max_samples), a lower noise_sensitivity, or \
                         noise_sensitivity=None",
                        rows.len()
                    ));
                }
                let columns = sampled_columns(&alive);
                // Ties broken at random, one seed per column, drawn after the
                // row sample and before the shadows: a discrete noise feature
                // is then as random as a shadow (with average ranks, its tie
                // blocks let it beat the shadows far more often)
                let seeds: Vec<u64> = columns.iter().map(|_| rng.next_u64()).collect();
                let result = noise_scores(
                    &tie_broken_rank_matrix(&columns, &sample_rows, &seeds),
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

        // Step 5: cap
        let mut cap_rank: Vec<Option<i64>> = vec![None; d];
        let mut cap_gain: Vec<Option<f64>> = vec![None; d];
        let mut cap_explained: Vec<Option<f64>> = vec![None; d];
        if let Some(cap) = &self.cap {
            let candidates = remaining(&dropped);
            if !candidates.is_empty() {
                if self.noise_sensitivity.is_none() {
                    warnings.push(
                        "the cap ran without the noise step (noise_sensitivity=None): it cannot \
                         tell noise from a feature unrelated to the others, and may keep noise; \
                         consider turning the noise step on"
                            .to_string(),
                    );
                }
                // The candidates' Spearman correlations: step 3's, or their own
                let corr = match step_3_correlations.take() {
                    Some((alive, corr)) => {
                        let mut at = vec![0; d];
                        for (k, &j) in alive.iter().enumerate() {
                            at[j] = k;
                        }
                        let positions: Vec<usize> = candidates.iter().map(|&j| at[j]).collect();
                        corr.select(ndarray::Axis(0), &positions)
                            .select(ndarray::Axis(1), &positions)
                    }
                    None => correlation_matrix(&rank_matrix(
                        &sampled_columns(&candidates),
                        &sample_rows,
                    )),
                };
                let result = greedy_subset(&corr, cap);
                drop(corr);
                for (k, &p) in result.picks.iter().enumerate() {
                    let j = candidates[p];
                    cap_rank[j] = Some(k as i64 + 1);
                    cap_gain[j] = Some(result.gains[k]);
                    cap_explained[j] = Some(result.explained[k]);
                }
                for (&j, next_gain) in candidates.iter().zip(&result.next_gain) {
                    if let Some(gain) = *next_gain {
                        cap_gain[j] = Some(gain);
                        dropped[j] = Some(if gain == 0.0 {
                            Dropped::Explained
                        } else {
                            Dropped::Cap
                        });
                    }
                }
                let kept = result.picks.len();
                let rows_limited = result.stop == CapStop::NothingLeft && kept < candidates.len();
                let r = rows.len();
                if rows_limited && n <= self.max_samples {
                    if let Some(max) = cap.max_features.filter(|&max| kept < max) {
                        warnings.push(format!(
                            "the cap kept {kept} features, fewer than max_features={max}: {r} \
                             rows span at most {} directions, and the kept features explain the \
                             other candidates fully on them",
                            r - 1
                        ));
                    }
                } else if rows_limited && kept == r - 1 {
                    warnings.push(format!(
                        "the cap kept {kept} features: it works on a sample of max_samples={} of \
                         the {n} rows, which spans at most {} directions; raise max_samples to \
                         keep more (the noise step's time grows with the square of the rows)",
                        self.max_samples,
                        self.max_samples - 1
                    ));
                }
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
        let missing_share: Vec<Option<f64>> = shares
            .iter()
            .map(|&(missing, _)| self.max_missing.map(|_| missing))
            .collect();
        let top: Vec<Option<f64>> = shares.iter().map(|&(_, top)| top).collect();
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
            "cap_rank" => cap_rank,
            "cap_gain" => cap_gain,
            "cap_explained" => cap_explained,
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
    /// Cut by the cap's `max_features` or `min_explained`.
    Cap,
    /// Nothing left to explain when the cap stopped: the kept features
    /// explain it fully on the rows used.
    Explained,
}

impl Dropped {
    /// The name in the report's `dropped_by` column.
    fn name(self) -> &'static str {
        match self {
            Dropped::Missing => "missing",
            Dropped::OneValue => "one_value",
            Dropped::Duplicate => "duplicate",
            Dropped::Noise => "noise",
            Dropped::Cap => "cap",
            Dropped::Explained => "explained",
        }
    }
}

/// Positions of the rows of `features` in the order of `key_cols` (nulls
/// last; the input order without key columns).
fn key_order(features: &DataFrame, key_cols: &[String]) -> PolarsResult<Vec<usize>> {
    if key_cols.is_empty() {
        return Ok((0..features.height()).collect());
    }
    const ROW: &str = "__tsfx_row";
    let keys = features
        .select(key_cols.iter().map(String::as_str))?
        .with_row_index(ROW.into(), None)?
        .sort(
            key_cols,
            SortMultipleOptions::default()
                .with_nulls_last(true)
                .with_maintain_order(true),
        )?;
    Ok(keys
        .column(ROW)?
        .idx()?
        .into_no_null_iter()
        .map(|i| i as usize)
        .collect())
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
    /// `kept`, `dropped_by` (`missing`, `one_value`, `duplicate`, `noise`,
    /// `cap`, `explained` or null), `missing_share`, `top_share`,
    /// `duplicate_of`, `correlation` (signed Spearman ρ with
    /// `duplicate_of`), `noise_score`, `noise_cutoff`, and the cap's
    /// `cap_rank` (pick order, from 1), `cap_gain` (a pick's gain in
    /// features' worth; for a feature the cap dropped, the gain it would have
    /// added next, 0 if `explained`) and `cap_explained` (the share explained
    /// after a pick). A step that is turned off, or that a feature did not
    /// reach, leaves its columns null.
    pub report: DataFrame,
    /// Warnings of the fit: too few features or rows for the noise step, the
    /// cap without the noise step, and the cap stopped by the rows or the
    /// sample before `max_features`.
    pub warnings: Vec<String>,
}

impl ReductionFit {
    /// `features` with only the `key_cols`, then the kept feature columns (in
    /// the order they were fitted), dtypes unchanged. A missing column is an
    /// error.
    pub fn apply(&self, features: DataFrame, key_cols: &[String]) -> PolarsResult<DataFrame> {
        features.select(key_cols.iter().chain(&self.kept).map(String::as_str))
    }
}

#[cfg(test)]
mod tests {
    use super::{Cap, Reduction};
    use crate::error::ExtractionError;

    #[test]
    fn test_cap_defaults_and_validate() {
        assert_eq!(Reduction::default().cap, None);
        assert_eq!(
            Cap::default(),
            Cap {
                max_features: None,
                min_explained: 1.0
            }
        );
        for ok in [
            Cap::default(),
            Cap {
                max_features: Some(1),
                min_explained: 0.5,
            },
        ] {
            assert!(
                Reduction {
                    cap: Some(ok),
                    ..Default::default()
                }
                .validate()
                .is_ok()
            );
        }
        let msg = |cap: Cap| {
            Reduction {
                cap: Some(cap),
                ..Default::default()
            }
            .validate()
            .unwrap_err()
            .to_string()
        };
        assert!(
            msg(Cap {
                max_features: Some(0),
                ..Default::default()
            })
            .contains("CapSetting max_features must be at least 1, got 0")
        );
        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            assert!(
                msg(Cap {
                    min_explained: bad,
                    ..Default::default()
                })
                .contains("CapSetting min_explained must be in (0, 1]"),
                "{bad}"
            );
        }
    }

    #[test]
    fn test_shadow_count() {
        for (s, n) in [(0.99, 99), (0.95, 19), (0.5, 1), (0.7, 2), (0.999, 999)] {
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
                noise_sensitivity: Some(0.49),
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
    use super::rng::SplitMix64;
    use super::{Cap, Reduction, ReductionFit};
    use polars::prelude::*;

    fn uniform(g: &mut SplitMix64) -> f64 {
        (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn keys() -> Vec<String> {
        vec!["id".to_string()]
    }

    /// 60 rows: "miss" 10% NaN, "const", "a", "a_sq" (monotone in a), "b".
    fn crafted() -> DataFrame {
        let mut g = SplitMix64::new(7);
        let n = 60;
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
        for name in ["cap_rank", "cap_gain", "cap_explained"] {
            assert_eq!(rep.column(name).unwrap().null_count(), 5, "{name}");
        }
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
        assert_eq!(fit.report.width(), 12);
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
                DataType::Int64,
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
            fit.warnings[0].contains("2 features"),
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
            "cap_rank",
            "cap_gain",
            "cap_explained",
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
        let one = assert_deterministic(&r, &df, &shuffled);
        // Not vacuous: features reached the noise step
        assert!(one.report.column("noise_score").unwrap().null_count() < d);

        let capped = Reduction {
            cap: Some(Cap {
                max_features: Some(10),
                min_explained: 0.9,
            }),
            ..r
        };
        let one = assert_deterministic(&capped, &df, &shuffled);
        // Not vacuous: features reached the cap
        assert!(one.report.column("cap_rank").unwrap().null_count() < d);
    }

    /// Fits `r` on `df` with 1 thread and on `shuffled` (the same rows in
    /// another order) with 4, asserts the same kept columns, a bit-identical
    /// report and the same warnings, and returns the first fit.
    fn assert_deterministic(r: &Reduction, df: &DataFrame, shuffled: &DataFrame) -> ReductionFit {
        let fit_in = |threads: usize, frame: &DataFrame| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| r.fit(frame, &keys()).unwrap())
        };
        let one = fit_in(1, df);
        let four = fit_in(4, shuffled);
        assert_eq!(one.kept, four.kept);
        assert_eq!(report_bits(&one.report), report_bits(&four.report));
        assert_eq!(one.warnings, four.warnings);
        one
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

    /// Equal to a threshold is kept: 1 missing of 20 is 0.05, one value on
    /// 19 of 20 is 0.95; one more of either is dropped.
    #[test]
    fn test_thresholds_are_strict() {
        // distinct values, the first `k` of them missing
        let missing_first = |k: usize| -> Vec<f64> {
            (0..20)
                .map(|i| if i < k { f64::NAN } else { i as f64 })
                .collect()
        };
        // all 0.0 but the first `k`, which are distinct
        let zeros_but = |k: usize| -> Vec<f64> {
            (0..20)
                .map(|i| if i < k { 1.0 + i as f64 } else { 0.0 })
                .collect()
        };
        let df = df![
            "id" => (0..20i64).collect::<Vec<_>>(),
            "missing_1" => missing_first(1),
            "missing_2" => missing_first(2),
            "top_19" => zeros_but(1),
            "top_20" => zeros_but(0),
            "spread" => (0..20).map(|i| 1.5 * i as f64).collect::<Vec<_>>(),
        ]
        .unwrap();
        let r = Reduction {
            max_correlation: None,
            noise_sensitivity: None,
            ..Default::default()
        };
        let fit = r.fit(&df, &keys()).unwrap();
        assert_eq!(
            strs(&fit.report, "dropped_by"),
            [
                None,
                Some("missing".to_string()),
                None,
                Some("one_value".to_string()),
                None
            ]
        );
    }

    /// The fit's warnings for `rows` rows and `features` features that all
    /// reach the noise step (steps 1 to 3 off).
    fn noise_warnings(rows: usize, features: usize) -> Vec<String> {
        let mut g = SplitMix64::new(13);
        let mut columns = vec![Column::new(
            "id".into(),
            (0..rows as i64).collect::<Vec<_>>(),
        )];
        for j in 0..features {
            let values: Vec<f64> = (0..rows).map(|_| uniform(&mut g)).collect();
            columns.push(Column::new(format!("f{j}").into(), values));
        }
        let df = DataFrame::new_infer_height(columns).unwrap();
        let r = Reduction {
            max_missing: None,
            max_top_share: None,
            max_correlation: None,
            ..Default::default()
        };
        r.fit(&df, &keys()).unwrap().warnings
    }

    #[test]
    fn test_noise_warning_edges() {
        let features_49 = noise_warnings(60, 49);
        assert_eq!(features_49.len(), 1);
        assert!(features_49[0].contains("49 features"), "{}", features_49[0]);
        assert!(noise_warnings(60, 50).is_empty());
        let rows_49 = noise_warnings(49, 60);
        assert_eq!(rows_49.len(), 1);
        assert!(rows_49[0].contains("49 rows"), "{}", rows_49[0]);
        assert!(noise_warnings(50, 60).is_empty());
    }

    /// Applying a fit to a frame without one of its kept columns is an error,
    /// not a silently smaller frame.
    #[test]
    fn test_apply_errors_on_a_missing_column() {
        let r = Reduction {
            noise_sensitivity: None,
            ..Default::default()
        };
        let fit = r.fit(&crafted(), &keys()).unwrap();
        let without_b = crafted().drop("b").unwrap();
        assert!(fit.apply(without_b, &keys()).is_err());
    }

    /// `rows` rows of `features` independent uniform columns f0, f1, ...
    fn uniform_frame(rows: usize, features: usize, seed: u64) -> DataFrame {
        let mut g = SplitMix64::new(seed);
        let mut columns = vec![Column::new(
            "id".into(),
            (0..rows as i64).collect::<Vec<_>>(),
        )];
        for j in 0..features {
            let values: Vec<f64> = (0..rows).map(|_| uniform(&mut g)).collect();
            columns.push(Column::new(format!("f{j}").into(), values));
        }
        DataFrame::new_infer_height(columns).unwrap()
    }

    /// Steps 1 and 2 at their defaults, 3 and 4 off, and `cap`.
    fn cap_reduction(cap: Cap) -> Reduction {
        Reduction {
            max_correlation: None,
            noise_sensitivity: None,
            cap: Some(cap),
            ..Default::default()
        }
    }

    fn cap_ranks(df: &DataFrame) -> Vec<Option<i64>> {
        df.column("cap_rank")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .collect()
    }

    fn floats(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
        df.column(name).unwrap().f64().unwrap().iter().collect()
    }

    fn count(values: &[Option<String>], value: &str) -> usize {
        values
            .iter()
            .filter(|v| v.as_deref() == Some(value))
            .count()
    }

    /// The picks' explained shares, in pick order.
    fn explained_by_rank(df: &DataFrame) -> Vec<f64> {
        let mut picks: Vec<(i64, f64)> = cap_ranks(df)
            .into_iter()
            .zip(floats(df, "cap_explained"))
            .filter_map(|(rank, share)| Some((rank?, share?)))
            .collect();
        picks.sort_by_key(|&(rank, _)| rank);
        picks.into_iter().map(|(_, share)| share).collect()
    }

    const NO_NOISE: &str = "the cap ran without the noise step";

    #[test]
    fn test_cap_rows_limit() {
        let fit = cap_reduction(Cap::default())
            .fit(&uniform_frame(12, 40, 21), &keys())
            .unwrap();
        assert_eq!(fit.kept.len(), 11);
        let rep = &fit.report;
        let dropped = strs(rep, "dropped_by");
        assert_eq!(count(&dropped, "explained"), 29);
        for (drop, gain) in dropped.iter().zip(floats(rep, "cap_gain")) {
            if drop.as_deref() == Some("explained") {
                assert_eq!(gain, Some(0.0));
            }
        }
        let mut ranks: Vec<i64> = cap_ranks(rep).into_iter().flatten().collect();
        ranks.sort_unstable();
        assert_eq!(ranks, (1..=11).collect::<Vec<_>>());
        let kept = rep.column("kept").unwrap().bool().unwrap();
        for (k, rank) in kept.iter().zip(cap_ranks(rep)) {
            assert_eq!(k, Some(rank.is_some()));
        }
        assert_eq!(fit.warnings.len(), 1, "{:?}", fit.warnings);
        assert!(fit.warnings[0].starts_with(NO_NOISE), "{}", fit.warnings[0]);
    }

    #[test]
    fn test_cap_rows_limit_with_max_features() {
        let cap = Cap {
            max_features: Some(200),
            ..Default::default()
        };
        let fit = cap_reduction(cap)
            .fit(&uniform_frame(12, 40, 21), &keys())
            .unwrap();
        assert_eq!(fit.kept.len(), 11);
        assert!(
            fit.warnings
                .iter()
                .any(|w| w.contains("fewer than max_features=200")
                    && w.contains("explain the other 29 candidates fully on these 12 rows")
                    && w.contains("at most 11")),
            "{:?}",
            fit.warnings
        );
    }

    /// Exact (rank) copies with step 3 off run out before max_features while
    /// the rows allow far more. Regression: the warning blamed the rows ("40
    /// rows span at most 39 directions"); it says the kept features explain
    /// the rest.
    #[test]
    fn test_cap_copies_run_out_warning() {
        let mut df = uniform_frame(40, 10, 31);
        for j in 0..10 {
            let copy = df.column(&format!("f{j}")).unwrap().clone();
            df.with_column(copy.with_name(format!("copy{j}").into()))
                .unwrap();
        }
        let cap = Cap {
            max_features: Some(30),
            ..Default::default()
        };
        let fit = cap_reduction(cap).fit(&df, &keys()).unwrap();
        assert_eq!(fit.kept.len(), 10);
        let warning = fit
            .warnings
            .iter()
            .find(|w| w.contains("fewer than max_features=30"))
            .expect("a max_features warning");
        assert!(
            warning.contains("explain the other 10 candidates fully on these 40 rows"),
            "{warning}"
        );
        assert!(!warning.contains("span"), "{warning}");
    }

    /// Sampled rows with step 3 off: the cap ranks the sample itself, and the
    /// sample's size is what stops it.
    #[test]
    fn test_cap_sample_limit() {
        let r = Reduction {
            max_samples: 20,
            ..cap_reduction(Cap::default())
        };
        let fit = r.fit(&uniform_frame(30, 40, 23), &keys()).unwrap();
        assert_eq!(fit.kept.len(), 19);
        assert!(
            fit.warnings
                .iter()
                .any(|w| w.contains("max_samples=20 of the 30 rows")
                    && w.contains("raise max_samples")),
            "{:?}",
            fit.warnings
        );
    }

    #[test]
    fn test_cap_max_features() {
        let cap = Cap {
            max_features: Some(5),
            ..Default::default()
        };
        let fit = cap_reduction(cap)
            .fit(&uniform_frame(60, 30, 24), &keys())
            .unwrap();
        assert_eq!(fit.kept.len(), 5);
        let rep = &fit.report;
        let dropped = strs(rep, "dropped_by");
        assert_eq!(count(&dropped, "cap"), 25);
        for (drop, gain) in dropped.iter().zip(floats(rep, "cap_gain")) {
            if drop.as_deref() == Some("cap") {
                assert!(gain.is_some_and(|g| g > 0.0), "{gain:?}");
            }
        }
        let shares = explained_by_rank(rep);
        assert_eq!(shares.len(), 5);
        assert!(shares.windows(2).all(|w| w[0] < w[1]), "{shares:?}");
        assert!(!fit.warnings.iter().any(|w| w.contains("fewer than")));
    }

    #[test]
    fn test_cap_min_explained() {
        let cap = Cap {
            min_explained: 0.6,
            ..Default::default()
        };
        let fit = cap_reduction(cap)
            .fit(&uniform_frame(60, 30, 24), &keys())
            .unwrap();
        let shares = explained_by_rank(&fit.report);
        let k = shares.len();
        assert_eq!(fit.kept.len(), k);
        assert!(k >= 2, "{shares:?}");
        assert!(shares[k - 1] >= 0.6 && shares[k - 2] < 0.6, "{shares:?}");
    }

    #[test]
    fn test_cap_fewer_candidates_than_rows() {
        let fit = cap_reduction(Cap::default())
            .fit(&uniform_frame(60, 30, 24), &keys())
            .unwrap();
        assert_eq!(fit.kept.len(), 30);
        assert_eq!(cap_ranks(&fit.report).iter().flatten().count(), 30);
        assert_eq!(fit.warnings.len(), 1, "{:?}", fit.warnings);
    }

    #[test]
    fn test_cap_step3_off_matches_on() {
        let df = uniform_frame(60, 30, 26);
        let off = cap_reduction(Cap::default()).fit(&df, &keys()).unwrap();
        let on = Reduction {
            max_correlation: Some(0.999999),
            ..cap_reduction(Cap::default())
        }
        .fit(&df, &keys())
        .unwrap();
        assert_eq!(count(&strs(&on.report, "dropped_by"), "duplicate"), 0);
        assert_eq!(cap_ranks(&off.report), cap_ranks(&on.report));
    }

    #[test]
    fn test_cap_all_missing_column_is_explained() {
        let mut df = uniform_frame(40, 20, 27);
        df.with_column(Column::new("nan".into(), vec![f64::NAN; 40]))
            .unwrap();
        let r = Reduction {
            max_missing: None,
            max_top_share: None,
            ..cap_reduction(Cap::default())
        };
        let fit = r.fit(&df, &keys()).unwrap();
        let rep = &fit.report;
        assert_eq!(strs(rep, "dropped_by")[20].as_deref(), Some("explained"));
        assert_eq!(floats(rep, "cap_gain")[20], Some(0.0));
        assert_eq!(fit.kept.len(), 20);
        for name in ["cap_gain", "cap_explained"] {
            assert!(
                floats(rep, name).into_iter().flatten().all(f64::is_finite),
                "{name}"
            );
        }
    }

    /// Rows 6..12 repeat rows 0..6: the rows span at most 5 directions, so the
    /// cap stops early, and the rows warning still fires.
    #[test]
    fn test_cap_duplicated_rows() {
        let base = uniform_frame(6, 40, 28);
        let index = IdxCa::from_vec("idx".into(), (0..12).map(|i| i % 6).collect());
        let mut df = base.take(&index).unwrap();
        df.with_column(Column::new("id".into(), (0..12i64).collect::<Vec<_>>()))
            .unwrap();
        let cap = Cap {
            max_features: Some(20),
            ..Default::default()
        };
        let fit = cap_reduction(cap).fit(&df, &keys()).unwrap();
        assert!(fit.kept.len() <= 5, "{}", fit.kept.len());
        let dropped = strs(&fit.report, "dropped_by");
        assert_eq!(count(&dropped, "explained"), 40 - fit.kept.len());
        assert!(
            fit.warnings
                .iter()
                .any(|w| w.contains("fewer than max_features=20")),
            "{:?}",
            fit.warnings
        );
    }

    #[test]
    fn test_cap_noise_on_no_cap_warning() {
        let n = 60;
        let mut g = SplitMix64::new(29);
        let mut columns = vec![Column::new("id".into(), (0..n as i64).collect::<Vec<_>>())];
        for j in 0..50 {
            let values: Vec<f64> = (0..n)
                .map(|i| (i % 3) as f64 + 0.3 * uniform(&mut g) + j as f64)
                .collect();
            columns.push(Column::new(format!("f{j}").into(), values));
        }
        let df = DataFrame::new_infer_height(columns).unwrap();
        let r = Reduction {
            cap: Some(Cap::default()),
            ..Default::default()
        };
        let fit = r.fit(&df, &keys()).unwrap();
        assert!(
            !fit.warnings.iter().any(|w| w.contains("cap")),
            "{:?}",
            fit.warnings
        );
    }

    /// With step 3 off, a near-copy (a feature equal to another but for one
    /// tied pair of neighboring values) reaches the cap with a residual
    /// variance just above the tolerance at 1000 rows, where its running
    /// score can round below zero. Regression: the fit panicked for about
    /// half the seeds (the final review's ports); all 20 must fit, keeping
    /// a and the two uniform columns.
    #[test]
    fn test_cap_near_copy_without_step_3() {
        let n = 1000;
        for seed in 0..20 {
            let mut g = SplitMix64::new(300 + seed);
            let a: Vec<f64> = (0..n).map(|_| uniform(&mut g)).collect();
            // a with its two smallest values tied
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&x, &y| a[x].total_cmp(&a[y]));
            let mut tied = a.clone();
            tied[order[1]] = a[order[0]];
            let u1: Vec<f64> = (0..n).map(|_| uniform(&mut g)).collect();
            let u2: Vec<f64> = (0..n).map(|_| uniform(&mut g)).collect();
            let df = df![
                "id" => (0..n as i64).collect::<Vec<_>>(),
                "a" => a,
                "a_tied" => tied,
                "u1" => u1,
                "u2" => u2,
            ]
            .unwrap();
            let fit = cap_reduction(Cap::default()).fit(&df, &keys()).unwrap();
            for name in ["a", "u1", "u2"] {
                assert!(
                    fit.kept.iter().any(|k| k == name),
                    "seed {seed}: {:?}",
                    fit.kept
                );
            }
        }
    }
}
