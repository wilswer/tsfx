//! Step 5: the cap. Greedy column subset selection (Farahat, Ghodsi and
//! Kamel, "An Efficient Greedy Method for Unsupervised Feature Selection",
//! ICDM 2011) on the candidates' Spearman correlation matrix: each pick is the
//! candidate that explains the most of all candidates' spread not yet explained
//! by the ones picked before it. Its gain, in features' worth (each candidate's
//! spread is 1, or 0 if it is constant), is the information it adds per
//! dimension. With r rows everything is explained after at most r − 1 picks
//! (centered rows span at most r − 1 directions), so the cap never keeps more.
//!
//! The recursive form keeps only the correlation matrix C and the picked
//! vectors ω: for the residual matrix G = C − Σ ω ωᵀ it updates each
//! candidate's ‖G eᵢ‖² (`f`) and Gᵢᵢ (`g`) after every pick, at O(d²) per pick
//! for d candidates, without a second d × d matrix.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::Cap;

/// A candidate whose residual variance is at most this is fully explained
/// (each candidate's own variance is 1, or 0 if it is constant).
pub(crate) const EXPLAINED_TOLERANCE: f64 = 1e-9;

/// Gains within this relative distance of the best are equal, and the
/// earliest candidate wins. Rounding alone separates gains that are equal by
/// construction, such as the last pick at the rows limit (one direction is
/// left, and every remaining candidate explains all of it).
pub(crate) const GAIN_TIE: f64 = 1e-6;

/// When two or more candidates' running scores are within this relative
/// distance of the best, those that have shrunk (see [`SHRUNK`]) are
/// recomputed exactly before the choice, so a tie at [`GAIN_TIE`] can be
/// told. With a clear best there is no tie to tell.
const NEAR_TOP: f64 = 1e-3;

/// The running score f shrinks by subtraction from its start f₀, with an
/// absolute error of about ε · picks · f₀: once f < SHRUNK · f₀, its
/// relative error can come near [`GAIN_TIE`] (ε · 5,000 picks / 1e-4 =
/// 5e-9 at most above it), and it is recomputed where it may decide a tie
/// or is reported as a dropped candidate's next gain. A pick's gain is
/// always computed exactly.
const SHRUNK: f64 = 1e-4;

/// Why the cap stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapStop {
    /// `max_features` candidates were picked.
    MaxFeatures,
    /// The picks explain at least `min_explained` (below 1.0).
    MinExplained,
    /// Every candidate left is fully explained.
    NothingLeft,
}

/// The cap's picks.
#[derive(Clone, Debug)]
pub(crate) struct CapResult {
    /// Candidate indices (columns of `c`) in pick order.
    pub picks: Vec<usize>,
    /// Gain of each pick, in features' worth.
    pub gains: Vec<f64>,
    /// Explained share after each pick.
    pub explained: Vec<f64>,
    /// Per candidate: `None` if picked; `Some(0.0)` if fully explained at the
    /// stop; else `Some(the gain it would add as the next pick)`.
    pub next_gain: Vec<Option<f64>>,
    /// Why the cap stopped.
    pub stop: CapStop,
}

/// Gᵢ: candidate i's column of the residual matrix G = C − Σ ω ωᵀ, exactly,
/// from C and the picked ω (O(d · picks)). ‖Gᵢ‖² is its running score f.
fn residual_column(c: &Array2<f64>, omegas: &[Array1<f64>], i: usize) -> Array1<f64> {
    let mut column = c.column(i).to_owned();
    for w in omegas {
        column.scaled_add(-w[i], w);
    }
    column
}

/// Picks candidates greedily, the highest gain first (the first in column
/// order on ties, gains within a relative [`GAIN_TIE`] counting as equal),
/// until `cap.max_features` are picked, the explained share reaches
/// `cap.min_explained` (checked only below 1.0), or every candidate left is
/// fully explained. `c`: the candidates' correlation matrix (d × d,
/// symmetric, diagonal 1, or 0 for a constant candidate). The products with
/// `c` run in parallel over its rows, each a sequential dot product, so the
/// result does not depend on the thread count.
pub(crate) fn greedy_subset(c: &Array2<f64>, cap: &Cap) -> CapResult {
    let d = c.nrows();
    let mut f: Vec<f64> = c
        .columns()
        .into_iter()
        .map(|column| column.iter().map(|x| x * x).sum())
        .collect();
    let f0 = f.clone();
    let mut g: Vec<f64> = c.diag().to_vec();
    let trace: f64 = g.iter().sum();
    let mut picked = vec![false; d];
    let mut omegas: Vec<Array1<f64>> = Vec::new();
    let (mut picks, mut gains, mut explained) = (Vec::new(), Vec::new(), Vec::new());
    let mut share = 0.0;
    let open = |i: usize, picked: &[bool], g: &[f64]| !picked[i] && g[i] > EXPLAINED_TOLERANCE;
    let stop = if trace <= 0.0 {
        CapStop::NothingLeft
    } else {
        loop {
            // The highest gain; gains within GAIN_TIE of it are equal, and the
            // earliest candidate wins
            let mut options: Vec<(usize, f64)> = (0..d)
                .filter(|&i| open(i, &picked, &g))
                .map(|i| (i, f[i] / g[i]))
                .collect();
            let Some(top) = options.iter().map(|&(_, gain)| gain).reduce(f64::max) else {
                break CapStop::NothingLeft;
            };
            // A possible tie near the top: f exactly there where it has
            // shrunk, ‖G eᵢ‖² from C and the picked ω (O(d · picks) each)
            let near_top = |gain: f64| gain >= top * (1.0 - NEAR_TOP);
            let possible_tie = options.iter().filter(|&&(_, gain)| near_top(gain)).count() > 1;
            for (i, gain) in options.iter_mut() {
                if possible_tie && near_top(*gain) && f[*i] < SHRUNK * f0[*i] {
                    let column = residual_column(c, &omegas, *i);
                    f[*i] = column.dot(&column);
                    *gain = f[*i] / g[*i];
                }
            }
            let top = options
                .iter()
                .map(|&(_, gain)| gain)
                .reduce(f64::max)
                .expect("options is not empty");
            let (l, _) = options
                .into_iter()
                .find(|&(_, gain)| gain >= top * (1.0 - GAIN_TIE))
                .expect("the best gain is among the options");

            // The residual column of l: its exact gain, and ω, scaled to
            // unit residual variance
            let delta = residual_column(c, &omegas, l);
            let gain = delta.dot(&delta) / delta[l];
            picked[l] = true;
            share += gain / trace;
            picks.push(l);
            gains.push(gain);
            explained.push(share);
            let omega = &delta / delta[l].sqrt();
            // h = G ω with the residual matrix before this pick
            let c_omega: Vec<f64> = (0..d)
                .into_par_iter()
                .map(|i| c.row(i).dot(&omega))
                .collect();
            let mut h = Array1::from(c_omega);
            for w in &omegas {
                h.scaled_add(-w.dot(&omega), w);
            }
            let norm2 = omega.dot(&omega);
            for ((fi, gi), (&o, &hi)) in f
                .iter_mut()
                .zip(g.iter_mut())
                .zip(omega.iter().zip(h.iter()))
            {
                *fi += -2.0 * o * hi + norm2 * o * o;
                *gi -= o * o;
            }
            omegas.push(omega);

            if cap.max_features == Some(picks.len()) {
                break CapStop::MaxFeatures;
            }
            if cap.min_explained < 1.0 && share >= cap.min_explained {
                break CapStop::MinExplained;
            }
            if !(0..d).any(|i| open(i, &picked, &g)) {
                break CapStop::NothingLeft;
            }
        }
    };
    // Reported: a shrunk running score exactly (each independent, in parallel)
    let next_gain = (0..d)
        .into_par_iter()
        .map(|i| {
            if picked[i] {
                None
            } else if g[i] <= EXPLAINED_TOLERANCE {
                Some(0.0)
            } else if f[i] < SHRUNK * f0[i] {
                let column = residual_column(c, &omegas, i);
                Some(column.dot(&column) / g[i])
            } else {
                Some(f[i] / g[i])
            }
        })
        .collect();
    CapResult {
        picks,
        gains,
        explained,
        next_gain,
        stop,
    }
}

#[cfg(test)]
mod tests {
    use super::{CapStop, greedy_subset};
    use crate::reduction::Cap;
    use crate::reduction::duplicates::correlation_matrix;
    use crate::reduction::matrix::rank_matrix;
    use crate::reduction::rng::SplitMix64;
    use ndarray::{Array1, Array2, ArrayView1, array};

    fn uniform(g: &mut SplitMix64) -> f64 {
        (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standardized ranks of `d` uniform columns of `n` rows.
    fn ranked(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut g = SplitMix64::new(seed);
        let columns: Vec<Vec<f64>> = (0..d)
            .map(|_| (0..n).map(|_| uniform(&mut g)).collect())
            .collect();
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        rank_matrix(&refs, &(0..n).collect::<Vec<_>>())
    }

    /// `v` minus its projection onto the orthonormal `basis` (twice, for
    /// stability).
    fn residual(v: ArrayView1<f64>, basis: &[Array1<f64>]) -> Array1<f64> {
        let mut r = v.to_owned();
        for _ in 0..2 {
            for q in basis {
                let p = q.dot(&r);
                r = r - p * q;
            }
        }
        r
    }

    /// Σᵢ ‖P zᵢ‖², P the projection onto the span of `basis`.
    fn captured(z: &Array2<f64>, basis: &[Array1<f64>]) -> f64 {
        z.columns()
            .into_iter()
            .map(|column| basis.iter().map(|q| q.dot(&column).powi(2)).sum::<f64>())
            .sum()
    }

    /// The cap's greedy by definition: at each step, the unpicked column
    /// (residual variance above 1e-9) whose unit residual q captures the most
    /// spread, Σᵢ (q · zᵢ)² / Σᵢ ‖zᵢ‖² (computed directly, not as a
    /// difference of totals, so small gains stay accurate; gains within a
    /// relative 1e-6 of the best are ties, won by the lowest index), with the
    /// same stop rules. Returns the picks and the explained share after each.
    fn brute_force(z: &Array2<f64>, cap: &Cap) -> (Vec<usize>, Vec<f64>) {
        let (n, d) = z.dim();
        let total: f64 = z.iter().map(|x| x * x).sum();
        let (mut basis, mut picks, mut explained) = (Vec::new(), Vec::new(), Vec::new());
        loop {
            let current = explained.last().copied().unwrap_or(0.0);
            let mut options: Vec<(usize, f64, Array1<f64>)> = Vec::new();
            for j in (0..d).filter(|j| !picks.contains(j)) {
                let r = residual(z.column(j), &basis);
                let norm2 = r.dot(&r);
                if norm2 / n as f64 <= 1e-9 {
                    continue;
                }
                let q = &r / norm2.sqrt();
                let gain = captured(z, std::slice::from_ref(&q)) / total;
                options.push((j, gain, q));
            }
            let Some(top) = options.iter().map(|o| o.1).reduce(f64::max) else {
                break;
            };
            let (j, gain, q) = options
                .into_iter()
                .find(|o| o.1 >= top * (1.0 - 1e-6))
                .unwrap();
            let share = current + gain;
            picks.push(j);
            explained.push(share);
            basis.push(q);
            if cap.max_features == Some(picks.len())
                || (cap.min_explained < 1.0 && share >= cap.min_explained)
            {
                break;
            }
        }
        (picks, explained)
    }

    fn trace(c: &Array2<f64>) -> f64 {
        c.diag().sum()
    }

    fn caps() -> [Cap; 3] {
        [
            Cap::default(),
            Cap {
                max_features: Some(3),
                ..Default::default()
            },
            Cap {
                min_explained: 0.7,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn matches_brute_force() {
        for (case, (n, d)) in [(30, 8), (12, 20), (40, 40)].into_iter().enumerate() {
            let z = ranked(n, d, 100 + case as u64);
            let c = correlation_matrix(&z);
            for cap in caps() {
                let got = greedy_subset(&c, &cap);
                let (picks, explained) = brute_force(&z, &cap);
                assert_eq!(got.picks, picks, "n {n}, d {d}, {cap:?}");
                for k in 0..picks.len() {
                    assert!(
                        (got.explained[k] - explained[k]).abs() < 1e-9,
                        "n {n}, d {d}, {cap:?}, pick {k}"
                    );
                    let before = if k == 0 { 0.0 } else { explained[k - 1] };
                    let gain = (explained[k] - before) * trace(&c);
                    assert!((got.gains[k] - gain).abs() < 1e-9, "pick {k}");
                }
                assert!(got.explained.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    /// At the rows limit the last pick is a tie by construction: one direction
    /// is left, and every remaining candidate explains all of it. Regression:
    /// rounding gave it to candidate 9 here, not to the earliest (1).
    #[test]
    fn rows_limit_tie_goes_to_the_earliest() {
        let c = correlation_matrix(&ranked(12, 20, 101));
        let r = greedy_subset(&c, &Cap::default());
        assert_eq!(r.picks.len(), 11);
        let earliest_left = (0..20).find(|j| !r.picks[..10].contains(j)).unwrap();
        assert_eq!(r.picks[10], earliest_left, "{:?}", r.picks);
    }

    /// Columns v and |v| of a few repeated values (as the fuzz test draws):
    /// near the rows limit, nearly explained candidates tie, and the
    /// recursive scores, which shrink by subtraction, are too imprecise for
    /// them. Regression: rounding decided these ties; each case's picks now
    /// match the accurate reference (the earliest candidate wins). Found with
    /// a numpy port of `greedy_subset`, after the fuzz test failed.
    #[test]
    fn near_ties_of_nearly_explained_candidates() {
        let cases: [(&[&[f64]], &[usize]); 4] = [
            (
                &[
                    &[
                        0.5, 0.0, 0.5, 2.0, 0.5, 2.0, 3.0, -0.5, 3.0, 3.0, -0.5, -0.5,
                    ],
                    &[
                        0.0, 0.0, 0.0, -0.5, -1.0, 2.0, 0.0, 1.0, 3.0, -0.5, 3.0, -0.5,
                    ],
                    &[
                        3.0, 2.0, 0.5, -0.5, -0.5, -0.5, 3.0, -0.5, -1.0, 0.5, 2.0, 2.0,
                    ],
                    &[
                        -1.0, 2.0, -0.5, 2.0, 1.0, -1.0, 0.5, -1.0, -1.0, 1.0, 1.0, 0.0,
                    ],
                    &[0.0, 0.0, 0.5, 3.0, 0.0, 3.0, 2.0, 2.0, 0.0, 3.0, 3.0, -0.5],
                    &[1.0, 2.0, 2.0, 3.0, 1.0, -1.0, 2.0, 0.5, -1.0, 3.0, 0.5, 1.0],
                ],
                &[10, 8, 4, 1, 7, 3, 2, 0, 6, 9, 5],
            ),
            (
                &[
                    &[-0.5, 2.0, -0.5, -1.0, 3.0, 2.0, 0.5, 1.0, -1.0, -0.5],
                    &[0.5, 0.5, 1.0, -1.0, 0.0, 2.0, 0.5, 2.0, 2.0, 0.0],
                    &[0.5, 1.0, 3.0, 2.0, 1.0, 0.5, 2.0, 3.0, -1.0, 1.0],
                    &[-0.5, 2.0, 0.5, -0.5, 0.5, 2.0, -1.0, -0.5, -1.0, 2.0],
                    &[-1.0, 1.0, -1.0, 2.0, 2.0, 3.0, 0.5, 3.0, 1.0, 1.0],
                ],
                &[8, 5, 3, 7, 0, 6, 1, 2, 4],
            ),
            (
                &[
                    &[
                        0.0, -1.0, 0.0, -0.5, -0.5, 2.0, -0.5, 3.0, 2.0, 0.0, 3.0, 0.5,
                    ],
                    &[
                        3.0, 3.0, -1.0, -0.5, 0.5, 0.0, 2.0, -0.5, -1.0, 3.0, 1.0, 3.0,
                    ],
                    &[
                        -0.5, 3.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5, 0.5, 1.0, 2.0, 1.0,
                    ],
                    &[
                        2.0, 2.0, 0.0, 0.5, 0.5, 0.5, -0.5, 0.5, -1.0, -1.0, -0.5, -0.5,
                    ],
                    &[
                        0.0, -1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, -0.5, 3.0, 3.0, -1.0,
                    ],
                    &[1.0, 0.5, 2.0, 3.0, 2.0, -0.5, -0.5, 0.5, 0.0, 2.0, 0.5, 3.0],
                ],
                &[2, 11, 9, 6, 4, 0, 7, 8, 1, 3, 5],
            ),
            (
                &[
                    &[2.0, 1.0, -1.0, 2.0, 0.0, 0.5, 0.0, 3.0, -1.0, -1.0],
                    &[3.0, 1.0, 0.0, 0.5, 2.0, -1.0, 0.5, 3.0, 3.0, 1.0],
                    &[2.0, 0.5, 0.0, 0.5, 2.0, 2.0, 0.0, -0.5, 0.0, -0.5],
                    &[2.0, 0.5, 0.5, 3.0, -1.0, 2.0, 1.0, 2.0, 2.0, 0.0],
                    &[0.0, 0.0, -0.5, -0.5, -0.5, 3.0, 2.0, 0.0, 1.0, 1.0],
                    &[-1.0, -1.0, 1.0, -0.5, 3.0, 2.0, -0.5, 3.0, 1.0, 1.0],
                ],
                &[0, 11, 9, 3, 4, 7, 6, 1, 2],
            ),
        ];
        for (case, (values, want)) in cases.into_iter().enumerate() {
            // The candidates: v and |v| of each column
            let columns: Vec<Vec<f64>> = values
                .iter()
                .flat_map(|v| [v.to_vec(), v.iter().map(|x| x.abs()).collect()])
                .collect();
            let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
            let rows: Vec<usize> = (0..values[0].len()).collect();
            let c = correlation_matrix(&rank_matrix(&refs, &rows));
            let got = greedy_subset(&c, &Cap::default());
            assert_eq!(got.picks, want, "case {case}");
        }
    }

    /// A reported gain, of a pick or of a candidate the cap dropped, comes
    /// from the running score; where that score had shrunk far from its start
    /// it was up to 2e-4 relative off in these cases (columns v and |v|, as
    /// the fuzz test draws). Regression: every reported gain matches the
    /// exact one (Gram-Schmidt on z) to a relative 1e-6, the most the
    /// correlation matrix allows for nearly explained candidates. Found with
    /// a numpy port of `greedy_subset`, after the fuzz test failed.
    #[test]
    fn reported_gains_are_accurate() {
        let cases: [(&[&[f64]], usize); 3] = [
            (
                &[
                    &[
                        0.0, 3.0, 1.0, 3.0, 3.0, 0.0, -1.0, 1.0, 2.0, 1.0, 0.0, 0.5, 1.0,
                    ],
                    &[
                        2.0, -1.0, 0.5, 2.0, 0.5, 2.0, 3.0, 0.0, 1.0, 3.0, -1.0, 0.5, 1.0,
                    ],
                    &[
                        2.0, 2.0, -1.0, 1.0, -1.0, 3.0, 2.0, 1.0, 3.0, -1.0, -0.5, 0.0, -0.5,
                    ],
                    &[
                        2.0, 0.5, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, -1.0,
                    ],
                    &[
                        -1.0, 0.0, 1.0, 0.5, 3.0, -0.5, 1.0, -1.0, -0.5, -0.5, -0.5, 0.0, 3.0,
                    ],
                    &[
                        -0.5, 3.0, 3.0, 3.0, -1.0, -0.5, 0.5, 3.0, 2.0, 1.0, 2.0, 2.0, 1.0,
                    ],
                ],
                12,
            ),
            (
                &[
                    &[
                        1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 3.0, 2.0, -1.0, 0.5, 3.0, -1.0, -1.0,
                    ],
                    &[
                        -0.5, 0.0, 0.5, 2.0, 1.0, -1.0, -1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0,
                    ],
                    &[
                        1.0, -1.0, -1.0, 1.0, -0.5, 2.0, 0.0, -1.0, 0.0, -0.5, 3.0, 1.0, 0.5,
                    ],
                    &[
                        1.0, 1.0, -1.0, 3.0, -0.5, -1.0, -1.0, 2.0, 0.5, 0.5, 2.0, 1.0, 0.0,
                    ],
                    &[
                        -1.0, 2.0, -1.0, -1.0, -1.0, 0.0, -1.0, 3.0, 2.0, -1.0, 3.0, 1.0, 1.0,
                    ],
                    &[
                        -1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 3.0, -1.0, 0.0, 3.0, 2.0,
                    ],
                ],
                11,
            ),
            (
                &[
                    &[2.0, -1.0, 1.0, -1.0, -1.0, 3.0, -0.5, 3.0, 3.0, 0.0, 1.0],
                    &[-1.0, 3.0, -1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0],
                    &[2.0, -1.0, 3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5],
                    &[-1.0, -1.0, -0.5, 0.5, 0.0, -1.0, 1.0, -0.5, 2.0, 1.0, -1.0],
                    &[2.0, 1.0, 0.0, -1.0, -0.5, 0.0, -0.5, -0.5, 2.0, -0.5, 1.0],
                ],
                10,
            ),
        ];
        for (case, (values, max_features)) in cases.into_iter().enumerate() {
            // The candidates: v and |v| of each column
            let columns: Vec<Vec<f64>> = values
                .iter()
                .flat_map(|v| [v.to_vec(), v.iter().map(|x| x.abs()).collect()])
                .collect();
            let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
            let rows: Vec<usize> = (0..values[0].len()).collect();
            let z = rank_matrix(&refs, &rows);
            let n = z.nrows() as f64;
            let cap = Cap {
                max_features: Some(max_features),
                ..Default::default()
            };
            let r = greedy_subset(&correlation_matrix(&z), &cap);
            let close = |got: f64, exact: f64| (got - exact).abs() <= 1e-9 + 1e-6 * exact.abs();
            // The gain of q, a unit residual: the spread it captures
            let gain_of = |q: &Array1<f64>| captured(&z, std::slice::from_ref(q)) / n;
            let mut basis: Vec<Array1<f64>> = Vec::new();
            for (&p, &gain) in r.picks.iter().zip(&r.gains) {
                let res = residual(z.column(p), &basis);
                let q = &res / res.dot(&res).sqrt();
                let exact = gain_of(&q);
                assert!(
                    close(gain, exact),
                    "case {case}, pick {p}: {gain} vs {exact}"
                );
                basis.push(q);
            }
            for (p, next) in r.next_gain.iter().enumerate() {
                let res = residual(z.column(p), &basis);
                // Fully explained (gain 0) or picked: nothing to compare
                let (Some(gain), true) = (*next, res.dot(&res) / n > 1e-7) else {
                    continue;
                };
                let exact = gain_of(&(&res / res.dot(&res).sqrt()));
                assert!(
                    close(gain, exact),
                    "case {case}, candidate {p}: {gain} vs {exact}"
                );
            }
        }
    }

    /// The cap fuzz test's failing example from the server (13 rows, 20 value
    /// columns, `max_features=11`): the step-3 survivors among the Minimal
    /// features of one-row groups, each a value column itself (kind 0) or its
    /// root mean square sqrt(v²) (kind 1). A dropped candidate's reported next
    /// gain was 1.2e-9 off, its running score having shrunk to 7e-7 of its
    /// start. Regression: every reported next gain is within 5e-10 of the
    /// exact one (what is left is the residual variance's own error, about
    /// 1.5e-10 here).
    #[test]
    fn fuzz_case_next_gains_are_exact() {
        let values: [&[f64]; 20] = [
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                509676.69039311935,
                1.0,
                -802519.9929700459,
                -807932.9426823203,
                -6.103515625e-05,
                -2.3490934740301657e-49,
                -2.1422631469964986e-114,
                -7.936770407619299e-290,
            ],
            &[
                0.0,
                0.0,
                323883.0360722027,
                376928.558960614,
                417508.59140582173,
                603254.0696183534,
                724528.9598331284,
                7.720892410901334e-24,
                0.0,
                0.0,
                -632398.6881668966,
                -757376.6478153225,
                0.0,
            ],
            &[
                0.0,
                1.0,
                282831.84254235146,
                0.0,
                -138872.99014007114,
                -155401.98627069418,
                -241407.48365913134,
                -267348.47353935754,
                -389064.3867234854,
                -487009.4218578606,
                -783009.8026924266,
                -816791.6536526026,
                -9.940149120488192e-38,
            ],
            &[
                0.0,
                1.9,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -559742.6781409974,
                0.0,
                821982.6229607952,
                3.2526553628590448e-192,
                1.1,
            ],
            &[
                0.0,
                1.0,
                85200.68770180992,
                2.3212531302849813e-184,
                -7503.242583489169,
                -712454.5786140847,
                -732381.3120328663,
                -843329.3799263302,
                -855284.1292134796,
                -961677.5427362447,
                -0.17394156975081732,
                -6.433435931383957e-26,
                -2.4038780861102895e-208,
            ],
            &[
                0.0,
                194556.04722908605,
                933038.5558773666,
                972467.2793628906,
                8.317811600238448e-282,
                -1.0,
                -312325.3855180555,
                -711821.1421997524,
                -0.3333333333333333,
                -1.0794587701032884e-129,
                -5.519444870435659e-161,
                2.2250738585e-313,
                2.2250738585e-313,
            ],
            &[
                999999.0,
                58393.69282905455,
                187334.81009663967,
                709121.0972493859,
                1e-05,
                1.0,
                9.997567704226483e-43,
                2.6482583804449476e-84,
                -68239.0198698612,
                -445807.49314794934,
                -621824.5327180875,
                -2.220446049250313e-16,
                -2.9942308805039834e-291,
            ],
            &[
                978685.0,
                135739.76022778684,
                203114.0086441175,
                274565.35130117065,
                587626.6071624584,
                632028.1920093792,
                7.595717962262004e-70,
                9.525311766108848e-98,
                0.0,
                -533277.301045432,
                -978684.6782176886,
                -2.3453140531990792e-137,
                2.2250738585e-313,
            ],
            &[
                1.0,
                9740.278487357311,
                270385.65034914156,
                -1.0,
                -35944.02720920555,
                -499571.43442746496,
                -548521.0201452894,
                -835470.5899702472,
                -998511.0615454023,
                0.0,
                -1.401298464324817e-45,
                -1.573308636239996e-261,
                -5e-324,
            ],
            &[
                1.0,
                403290.3784913111,
                424258.4855242744,
                0.0,
                -551107.1083371898,
                -993083.6922066595,
                -9.13279012053771e-47,
                8555.273649993003,
                828773.8965487347,
                2.7538058195051377e-157,
                150403.42229655245,
                289694.8530842301,
                -151877.65237929905,
            ],
            &[
                109957.24787377115,
                487560.90848439396,
                762894.3627899382,
                813701.7432996144,
                869595.3839041605,
                0.0,
                -446531.29685662605,
                -473834.1257760993,
                -1.1754943508222875e-38,
                -1.307968409828948e-186,
                -556738.3502192537,
                5356.670508669573,
                -19546.093118222314,
            ],
            &[
                217187.16775000026,
                -545101.5647308716,
                564389.9605327484,
                999999.9999999999,
                2.225073858507e-311,
                -847589.3024230214,
                -777023.557939088,
                -790940.0123129799,
                388912.9153634412,
                -1.0,
                -818821.7817232681,
                -471529.998085995,
                793927.3216778471,
            ],
            &[
                286029.6324752609,
                379758.29174656747,
                636327.3626993075,
                0.011189734190317503,
                5.211548881734184e-200,
                5.53520463829265e-214,
                -1.0,
                0.0,
                -157361.32402468054,
                -205156.39742347668,
                -619730.9173303384,
                -1.192092896e-07,
                -4.9223147029372016e-203,
            ],
            &[
                829760.0743309972,
                -631755.2263487899,
                -1.546061824100847e-270,
                0.0,
                153910.164678975,
                911110.8205374952,
                3.16917577718583e-36,
                3.567092823142443e-196,
                -1.1,
                -68774.716310055,
                -298129.87046889623,
                -376241.70207020524,
                -718783.5590153541,
            ],
            &[
                961014.5994625113,
                2.1141971074152292e-125,
                0.0,
                667841.3005756498,
                -259614.3282151305,
                0.0,
                299782.42183288164,
                -162595.7717761317,
                490486.7428400975,
                -4.0709308393771084e-187,
                -5.960464477539063e-08,
                298459.0741635035,
                -2.0996989887090903e-28,
            ],
            &[
                1.1754943508222875e-38,
                2.555478008703844e-113,
                5.479653316271118e-206,
                0.0,
                0.0,
                -89968.30105074437,
                -393765.6295242661,
                -482329.6178931743,
                -587590.454075915,
                -623706.2610993845,
                -761694.0125082494,
                -846587.3509280795,
                -3.411032844574461e-241,
            ],
            &[
                1.1754943508222875e-38,
                0.0,
                -6.874392312438261e-88,
                -152839.6984175695,
                295421.9064075502,
                938236.1031619841,
                -257884.37685674988,
                2.2250738585e-313,
                -730797.6441825596,
                312137.7995877196,
                1.2422275210781286e-19,
                5.3909151525118376e-67,
                1.0,
            ],
            &[
                -479736.33440234384,
                -503638.0,
                5.256923785437841e-194,
                2.2250738585072014e-308,
                -2.0,
                -5.424009132190237e-273,
                1.0,
                -2027.0,
                -503642.0,
                0.0,
                -2027.0,
                615179.0,
                503641.0,
            ],
            &[
                -377776.0,
                -2.0,
                -658922.0,
                321103.0,
                1.0,
                208183.0,
                377776.0,
                5.275334033123803e-234,
                377775.0,
                -321102.0,
                0.0,
                -377774.0,
                -658922.0,
            ],
        ];
        let candidates: [(usize, usize); 37] = [
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
            (10, 0),
            (10, 1),
            (11, 0),
            (11, 1),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (15, 0),
            (15, 1),
            (16, 1),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
        ];
        let columns: Vec<Vec<f64>> = candidates
            .iter()
            .map(|&(column, kind)| {
                let v = values[column].iter();
                if kind == 0 {
                    v.copied().collect()
                } else {
                    v.map(|&x| (x * x).sqrt()).collect()
                }
            })
            .collect();
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let z = rank_matrix(&refs, &(0..13).collect::<Vec<_>>());
        let n = z.nrows() as f64;
        let cap = Cap {
            max_features: Some(11),
            ..Default::default()
        };
        let r = greedy_subset(&correlation_matrix(&z), &cap);
        let mut basis: Vec<Array1<f64>> = Vec::new();
        for &p in &r.picks {
            let res = residual(z.column(p), &basis);
            basis.push(&res / res.dot(&res).sqrt());
        }
        let mut compared = 0;
        for (p, next) in r.next_gain.iter().enumerate() {
            let res = residual(z.column(p), &basis);
            let (Some(gain), true) = (*next, res.dot(&res) / n > 1e-7) else {
                continue;
            };
            let exact = captured(&z, std::slice::from_ref(&(&res / res.dot(&res).sqrt()))) / n;
            assert!(
                (gain - exact).abs() < 5e-10,
                "candidate {p}: {gain} vs {exact}"
            );
            compared += 1;
        }
        assert!(compared > 0);
    }

    #[test]
    fn rows_limit() {
        let c = correlation_matrix(&ranked(12, 20, 7));
        let r = greedy_subset(&c, &Cap::default());
        assert_eq!(r.picks.len(), 11);
        assert_eq!(r.stop, CapStop::NothingLeft);
        let left: Vec<Option<f64>> = (0..20)
            .filter(|j| !r.picks.contains(j))
            .map(|j| r.next_gain[j])
            .collect();
        assert_eq!(left, vec![Some(0.0); 9]);
        assert!((r.explained[10] - 1.0).abs() < 1e-9, "{}", r.explained[10]);
    }

    #[test]
    fn fewer_candidates_than_rows() {
        let c = correlation_matrix(&ranked(30, 8, 8));
        let r = greedy_subset(&c, &Cap::default());
        assert_eq!(r.picks.len(), 8);
        assert!(r.next_gain.iter().all(Option::is_none));
    }

    #[test]
    fn max_features_stop() {
        let c = correlation_matrix(&ranked(30, 8, 9));
        let cap = Cap {
            max_features: Some(3),
            ..Default::default()
        };
        let r = greedy_subset(&c, &cap);
        assert_eq!(r.picks.len(), 3);
        assert_eq!(r.stop, CapStop::MaxFeatures);
        for j in (0..8).filter(|j| !r.picks.contains(j)) {
            assert!(r.next_gain[j].is_some_and(|g| g > 0.0), "{j}");
        }
    }

    #[test]
    fn min_explained_crossing_pick_kept() {
        let c = correlation_matrix(&ranked(30, 8, 10));
        let cap = Cap {
            min_explained: 0.7,
            ..Default::default()
        };
        let r = greedy_subset(&c, &cap);
        let k = r.picks.len();
        assert!(k >= 2);
        assert!(r.explained[k - 2] < 0.7 && 0.7 <= r.explained[k - 1]);
        assert_eq!(r.stop, CapStop::MinExplained);
    }

    #[test]
    fn exact_copy_never_picked() {
        let mut z = ranked(30, 5, 11);
        let first = z.column(0).to_owned();
        z.column_mut(1).assign(&first);
        let r = greedy_subset(&correlation_matrix(&z), &Cap::default());
        assert_eq!(r.picks[0], 0);
        assert!(!r.picks.contains(&1));
        assert_eq!(r.next_gain[1], Some(0.0));
    }

    #[test]
    fn all_constant_candidates() {
        let r = greedy_subset(&Array2::zeros((3, 3)), &Cap::default());
        assert!(r.picks.is_empty());
        assert_eq!(r.next_gain, vec![Some(0.0); 3]);
        assert_eq!(r.stop, CapStop::NothingLeft);
    }

    #[test]
    fn single_candidate() {
        let r = greedy_subset(&array![[1.0]], &Cap::default());
        assert_eq!(r.picks, vec![0]);
        assert_eq!(r.gains, vec![1.0]);
        assert_eq!(r.explained, vec![1.0]);
    }

    #[test]
    fn stop_at_one_is_nothing_left() {
        for (case, (n, d)) in [(30, 8), (12, 20), (40, 40)].into_iter().enumerate() {
            let c = correlation_matrix(&ranked(n, d, 200 + case as u64));
            let r = greedy_subset(&c, &Cap::default());
            assert_eq!(r.stop, CapStop::NothingLeft, "n {n}, d {d}");
        }
    }
}
