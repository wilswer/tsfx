//! Similarity of the series to a given query pattern.

use ndarray::ArrayView1;

use super::kernel::{GroupContext, Kernel, extremes};
use crate::utils::stats::{numpy_mean, population_var};

/// Below this, stumpy treats `σ_Q σ_T m` as this value
/// (`config.STUMPY_DENOM_THRESHOLD`).
const DENOM_THRESHOLD: f64 = 1e-14;

/// One query of [`QuerySimilarityCount`].
#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    /// The query subsequence $Q$.
    pub query: Vec<f64>,
    /// Largest distance that counts as a match.
    pub threshold: f64,
    /// z-normalise $Q$ and each subsequence before comparing.
    pub normalize: bool,
}

/// Query similarity count feature.
///
/// The number of length-$m$ subsequences of the series whose distance to
/// the query $Q$ (of length $m$) is at most `threshold`. With `normalize`,
/// the distance is the z-normalised Euclidean distance computed as
/// `stumpy.core.mass` does,
/// $$ d = \sqrt{\left|2m\left(1 - \min\left(1, \frac{Q \cdot T_i - m\,\mu_Q\,\mu_{T_i}}{\max(m\,\sigma_Q\,\sigma_{T_i}, 10^{-14})}\right)\right)\right|}, $$
/// with population standard deviations, $d = 0$ if both $Q$ and the
/// subsequence $T_i$ are constant and $d = \sqrt{m}$ if exactly one is;
/// without, it is the plain Euclidean distance $\lVert Q - T_i \rVert_2$.
///
/// # Parameters
/// - `query`: the query subsequence, at least 3 values.
/// - `threshold`: largest distance that counts (default 0, exact matches).
/// - `normalize`: z-normalised distance (default `true`).
///
/// Config:
/// `[query_similarity_count] parameters = [{ query = [1.0, 2.0, 3.0], threshold = 0.5, normalize = true }, ...]`;
/// one column per entry. Off by default: the feature needs a query, and
/// tsfresh's default (no query) only gives NaN.
///
/// # Output column
/// `{name}__query_similarity_count__query_{i}`, where `i` is the 0-based
/// position of the entry in the config's `parameters` list.
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A query with fewer than 3 values gives NaN.
/// - A series shorter than the query gives NaN (tsfresh raises an error).
/// - A query with NaN or $\pm\infty$ matches nothing; a subsequence with NaN
///   or $\pm\infty$ is at infinite distance.
/// - The default threshold of 0 asks for exact matches; tsfresh computes
///   the dot products with an FFT and `fastmath`, so its distance of an
///   exact match is often a tiny positive number instead of 0. tsfx
///   computes them directly, so a count at a threshold within rounding of
///   a distance can differ from tsfresh's.
///
/// # Cost
/// $O(n\,m)$ per query.
///
/// # tsfresh
/// `feature_calculators.query_similarity_count` (v0.21.2), with stumpy's
/// `core.mass` / `core.mass_absolute`.
pub struct QuerySimilarityCount {
    pub queries: Vec<Query>,
}

impl Kernel for QuerySimilarityCount {
    fn names(&self) -> Vec<String> {
        (0..self.queries.len())
            .map(|i| format!("query_similarity_count__query_{i}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let t = ctx.values();
        for query in &self.queries {
            let q = query.query.as_slice();
            let m = q.len();
            if m < 3 || t.len() < m {
                out.push(f64::NAN);
                continue;
            }
            let count = if q.iter().all(|v| v.is_finite()) {
                let distance = if query.normalize {
                    z_normalized_distances(q, t)
                } else {
                    euclidean_distances(q, t)
                };
                distance.iter().filter(|&&d| d <= query.threshold).count()
            } else {
                // stumpy: every distance to a non-finite query is infinite
                if f64::INFINITY <= query.threshold {
                    t.len() - m + 1
                } else {
                    0
                }
            };
            out.push(count as f64);
        }
    }
}

/// z-normalised Euclidean distance of the finite query `q` to every
/// length-`q.len()` subsequence of `t`, as `stumpy.core.mass`.
fn z_normalized_distances(q: &[f64], t: &[f64]) -> Vec<f64> {
    let m = q.len();
    let mf = m as f64;
    let mean_q = numpy_mean(q);
    let std_q = population_var(&ArrayView1::from(q)).sqrt();
    let (min_q, max_q) = extremes(q.iter().copied());
    let q_constant = max_q - min_q == 0.0;
    t.windows(m)
        .map(|w| {
            if !w.iter().all(|v| v.is_finite()) {
                return f64::INFINITY;
            }
            let (min_t, max_t) = extremes(w.iter().copied());
            let t_constant = max_t - min_t == 0.0;
            let d_squared = if q_constant && t_constant {
                0.0
            } else if q_constant || t_constant {
                mf
            } else {
                let mean_t = numpy_mean(w);
                let std_t = population_var(&ArrayView1::from(w)).sqrt();
                let qt = q.iter().zip(w).map(|(a, b)| a * b).sum::<f64>();
                let denom = (std_q * std_t * mf).max(DENOM_THRESHOLD);
                let rho = ((qt - mean_q * mean_t * mf) / denom).min(1.0);
                (2.0 * mf * (1.0 - rho)).abs()
            };
            d_squared.sqrt()
        })
        .collect()
}

/// Euclidean distance of the finite query `q` to every length-`q.len()`
/// subsequence of `t`, as `stumpy.core.mass_absolute`.
fn euclidean_distances(q: &[f64], t: &[f64]) -> Vec<f64> {
    t.windows(q.len())
        .map(|w| {
            if !w.iter().all(|v| v.is_finite()) {
                return f64::INFINITY;
            }
            q.iter()
                .zip(w)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_is_zero() {
        let q = [1.0, 3.0, 2.0];
        let t = [0.0, 1.0, 3.0, 2.0, 5.0, 2.0, 6.0, 4.0];
        let d = z_normalized_distances(&q, &t);
        assert!(d[1] < 1e-7);
        // [2, 6, 4] is [1, 3, 2] scaled by 2: z-normalised, identical
        assert!(d[5] < 1e-7);
        assert_eq!(euclidean_distances(&q, &t)[1], 0.0);
    }

    #[test]
    fn test_constant_subsequences() {
        let q = [1.0, 1.0, 1.0];
        let t = [2.0, 2.0, 2.0, 3.0];
        let d = z_normalized_distances(&q, &t);
        assert_eq!(d, vec![0.0, 3.0_f64.sqrt()]);
    }
}
