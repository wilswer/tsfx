//! Step 4: noise. The Laplacian score (He, Cai and Niyogi, NeurIPS 2005) of
//! each feature on a nearest-neighbor graph of the rows, compared with the
//! scores of shuffled "shadow" features scored in the same graph. A feature
//! that varies no less between neighboring rows than the best shadow carries
//! no neighbor structure: it is noise.

use ndarray::{Array2, ArrayView1, s};
use rayon::prelude::*;

use super::matrix::rank_matrix;
use super::rng::SplitMix64;

/// Rows per block in the parallel nearest-neighbor search: each block holds a
/// 64 × n matrix, so the memory in use is about threads · 64 · n floats.
const ROW_BLOCK: usize = 64;

/// For each row of `m`, its `k` (at least 1) nearest other rows with their
/// squared Euclidean distances, sorted by (distance, index): equal distances
/// go to the lower index. Distances come from ‖a‖² + ‖b‖² − 2a·b; one at most
/// 1e-12 · (‖a‖² + ‖b‖²) is rounding noise of (nearly) equal rows and counts
/// as exactly 0. Parallel over blocks of 64 rows; each block's distances to
/// all rows come from one single-threaded matrix product, so no n × n matrix
/// is held and the result does not depend on the thread count.
pub(crate) fn knn(m: &Array2<f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
    debug_assert!(k >= 1, "knn needs at least one neighbor");
    let n = m.nrows();
    let norms: Vec<f64> = m.rows().into_iter().map(|row| row.dot(&row)).collect();
    let starts: Vec<usize> = (0..n).step_by(ROW_BLOCK).collect();
    let blocks: Vec<Vec<Vec<(usize, f64)>>> = starts
        .par_iter()
        .map(|&start| {
            let end = (start + ROW_BLOCK).min(n);
            let gram = m.slice(s![start..end, ..]).dot(&m.t());
            (start..end)
                .map(|i| nearest(i, gram.row(i - start), &norms, k))
                .collect()
        })
        .collect();
    blocks.into_iter().flatten().collect()
}

/// The `k` nearest rows of row `i`, from its row of the Gram matrix.
fn nearest(i: usize, gram_row: ArrayView1<f64>, norms: &[f64], k: usize) -> Vec<(usize, f64)> {
    let by_distance = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    let mut candidates: Vec<(usize, f64)> = (0..norms.len())
        .filter(|&j| j != i)
        .map(|j| {
            let scale = norms[i] + norms[j];
            let d2 = scale - 2.0 * gram_row[j];
            // Rounding noise where rows are (nearly) equal, of either sign:
            // snap it to 0, so equal rows tie exactly
            (j, if d2 <= 1e-12 * scale { 0.0 } else { d2 })
        })
        .collect();
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k - 1, by_distance);
        candidates.truncate(k);
    }
    candidates.sort_by(by_distance);
    candidates
}

/// The Laplacian score of every column of `m` on the graph of the neighbor
/// lists `nn` (from [`knn`]): edges are the union of the neighbor pairs, with
/// weight `exp(-d² / t)`, `t` the lower median of all neighbor distances (1.0
/// if that is 0). With degrees `D` and `L = D − W`, the score of a column `f`
/// is `f̃ᵀ L f̃ / f̃ᵀ D f̃`, `f̃ = f − (fᵀD1 / 1ᵀD1) 1`: lower means the column
/// varies less between neighbors. NaN for a constant column. Parallel over
/// columns; every sum runs in a fixed order.
pub(crate) fn laplacian_scores(m: &Array2<f64>, nn: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let n = m.nrows();
    let mut distances: Vec<f64> = nn.iter().flatten().map(|&(_, d2)| d2).collect();
    distances.sort_by(f64::total_cmp);
    let t = match distances.get(distances.len().saturating_sub(1) / 2) {
        Some(&median) if median > 0.0 => median,
        _ => 1.0,
    };

    // Each undirected edge once, as (lower, upper, weight); a stable sort
    // keeps the copy found from the lower row first
    let mut edges: Vec<(usize, usize, f64)> = nn
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().map(move |&(j, d2)| (i.min(j), i.max(j), d2)))
        .collect();
    edges.sort_by_key(|edge| (edge.0, edge.1));
    edges.dedup_by(|later, first| (later.0, later.1) == (first.0, first.1));
    for edge in edges.iter_mut() {
        edge.2 = (-edge.2 / t).exp();
    }
    let mut degree = vec![0.0; n];
    for &(i, j, w) in &edges {
        degree[i] += w;
        degree[j] += w;
    }
    let total_degree: f64 = degree.iter().sum();

    (0..m.ncols())
        .into_par_iter()
        .map(|c| {
            // A contiguous copy: the edge loop reads it in random order
            let f = m.column(c).to_vec();
            if f.iter().all(|&x| x == f[0]) {
                // The weighted mean can miss a constant by an ulp, which
                // would score it 0, the best score
                return f64::NAN;
            }
            let mean = degree.iter().zip(&f).map(|(d, x)| d * x).sum::<f64>() / total_degree;
            // Centering cancels in the differences of the numerator
            let numerator: f64 = edges
                .iter()
                .map(|&(i, j, w)| w * (f[i] - f[j]).powi(2))
                .sum();
            let denominator: f64 = degree
                .iter()
                .zip(&f)
                .map(|(d, x)| d * (x - mean).powi(2))
                .sum();
            if denominator > 0.0 {
                numerator / denominator
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// The noise step's scores.
#[derive(Clone, Debug)]
pub(crate) struct NoiseResult {
    /// Laplacian score of each column of the input matrix.
    pub scores: Vec<f64>,
    /// The lowest shadow score: a feature is kept only if its score is below.
    pub cutoff: f64,
}

/// Scores the columns of the rank matrix `z` against `n_shadows` shadow
/// columns, each a permutation (from `rng`, in shadow order) of the
/// standardized ranks of a tie-free column, which is what any noise column
/// becomes after the rank transform. The graph is built on the real and the
/// shadow columns together, with `neighbors` neighbors per row, so that a
/// pure-noise feature and the shadows are interchangeable: it beats them all
/// with probability 1 / (shadows + 1).
pub(crate) fn noise_scores(
    z: &Array2<f64>,
    n_shadows: usize,
    neighbors: usize,
    rng: &mut SplitMix64,
) -> NoiseResult {
    let (n, d) = z.dim();
    let positions: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let all_rows: Vec<usize> = (0..n).collect();
    let base = rank_matrix(&[&positions], &all_rows).column(0).to_vec();
    let mut m = Array2::zeros((n, d + n_shadows));
    m.slice_mut(s![.., ..d]).assign(z);
    for shadow in 0..n_shadows {
        let mut column = base.clone();
        rng.shuffle(&mut column);
        m.column_mut(d + shadow)
            .assign(&ArrayView1::from(column.as_slice()));
    }
    let scores = laplacian_scores(&m, &knn(&m, neighbors));
    let cutoff = scores[d..].iter().copied().fold(f64::INFINITY, f64::min);
    NoiseResult {
        scores: scores[..d].to_vec(),
        cutoff,
    }
}

#[cfg(test)]
mod tests {
    use super::{knn, laplacian_scores, noise_scores};
    use crate::reduction::matrix::{rank_matrix, tie_broken_rank_matrix};
    use crate::reduction::rng::SplitMix64;
    use ndarray::{Array2, array};

    fn uniform(g: &mut SplitMix64) -> f64 {
        (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn test_laplacian_exact() {
        // rows 0, 1, 10, 11 on a line, k = 1: edges (0,1), (2,3), d² = 1, t = 1
        let m = array![[0.0], [1.0], [10.0], [11.0]];
        let s = laplacian_scores(&m, &knn(&m, 1));
        assert!((s[0] - 2.0 / 101.0).abs() < 1e-12, "{}", s[0]);
    }

    #[test]
    fn test_knn_ties_go_to_lower_index() {
        let m = array![[0.0], [1.0], [-1.0]];
        assert_eq!(knn(&m, 1)[0], vec![(1, 1.0)]);
        // equal rows are at distance exactly 0
        let dup = array![[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]];
        assert_eq!(knn(&dup, 2)[2], vec![(0, 0.0), (1, 0.0)]);
    }

    #[test]
    fn test_constant_column_scores_nan() {
        // Rows 0, 1, 3, 10, 11 with k = 1: unequal degrees, where a weighted
        // mean of a constant can be off by an ulp
        let constants = [0.1, 5.0, 0.3, 7.7, 1e-3, 123.456, -2.2];
        let m = Array2::from_shape_fn((5, 1 + constants.len()), |(i, j)| {
            if j == 0 {
                [0.0, 1.0, 3.0, 10.0, 11.0][i]
            } else {
                constants[j - 1]
            }
        });
        let scores = laplacian_scores(&m, &knn(&m, 1));
        assert!(scores[0].is_finite());
        for (c, score) in constants.iter().zip(&scores[1..]) {
            assert!(score.is_nan(), "constant {c} scored {score}");
        }
    }

    /// Informative columns (3 clusters) all beat the cutoff; pure-noise
    /// columns get through at about 1 / (shadows + 1) = 1%.
    #[test]
    fn test_noise_rule_rates() {
        let (n, informative, noise) = (400, 30, 20);
        let mut noise_kept = 0;
        for seed in 0..20 {
            let mut g = SplitMix64::new(100 + seed);
            let cols: Vec<Vec<f64>> = (0..informative + noise)
                .map(|j| {
                    (0..n)
                        .map(|i| {
                            let u = uniform(&mut g);
                            if j < informative {
                                (i % 3) as f64 + 0.3 * u
                            } else {
                                u
                            }
                        })
                        .collect()
                })
                .collect();
            let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
            let z = rank_matrix(&refs, &(0..n).collect::<Vec<_>>());
            let r = noise_scores(&z, 99, 10, &mut SplitMix64::new(seed));
            assert!(
                r.scores[..informative].iter().all(|s| *s < r.cutoff),
                "seed {seed}"
            );
            noise_kept += r.scores[informative..]
                .iter()
                .filter(|s| **s < r.cutoff)
                .count();
        }
        assert!(noise_kept <= 12, "{noise_kept} of 400 noise columns kept");
    }

    /// Dense references: neighbors from all pairwise distances computed
    /// directly, and scores from full W, D and L matrices.
    fn dense_knn(m: &Array2<f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
        (0..m.nrows())
            .map(|i| {
                let mut all: Vec<(usize, f64)> = (0..m.nrows())
                    .filter(|&j| j != i)
                    .map(|j| {
                        let diff = &m.row(i) - &m.row(j);
                        (j, diff.dot(&diff))
                    })
                    .collect();
                all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                all.truncate(k);
                all
            })
            .collect()
    }

    fn dense_scores(m: &Array2<f64>, nn: &[Vec<(usize, f64)>]) -> Vec<f64> {
        let n = m.nrows();
        let mut d2: Vec<f64> = nn.iter().flatten().map(|p| p.1).collect();
        d2.sort_by(f64::total_cmp);
        let t = match d2[(d2.len() - 1) / 2] {
            x if x > 0.0 => x,
            _ => 1.0,
        };
        let mut w = Array2::<f64>::zeros((n, n));
        for (i, row) in nn.iter().enumerate() {
            for &(j, dist) in row {
                let (a, b) = (i.min(j), i.max(j));
                if w[[a, b]] == 0.0 {
                    let weight = (-dist / t).exp();
                    w[[a, b]] = weight;
                    w[[b, a]] = weight;
                }
            }
        }
        let deg = w.sum_axis(ndarray::Axis(1));
        let lap = Array2::from_diag(&deg) - &w;
        m.columns()
            .into_iter()
            .map(|f| {
                let mean = deg.dot(&f) / deg.sum();
                let ft = f.mapv(|x| x - mean);
                let den = (&ft * &deg).dot(&ft);
                if den > 0.0 {
                    ft.dot(&lap.dot(&ft)) / den
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    #[test]
    fn test_knn_and_scores_match_dense_reference() {
        let mut case = 0;
        // 63, 64, 65 and 300 rows cross the 64-row blocks of knn
        for n in [3, 11, 63, 64, 65, 300] {
            for d in [1, 3, 5, 8, 2] {
                case += 1;
                let mut g = SplitMix64::new(1000 + case);
                let mut m = Array2::from_shape_fn((n, d), |_| uniform(&mut g));
                if case % 3 == 0 {
                    // duplicated rows: exact ties
                    for i in (1..n).step_by(4) {
                        let row = m.row(i - 1).to_owned();
                        m.row_mut(i).assign(&row);
                    }
                }
                let k = 10.min(n - 1);
                let got = knn(&m, k);
                let want = dense_knn(&m, k);
                for (row_got, row_want) in got.iter().zip(&want) {
                    let idx_got: Vec<usize> = row_got.iter().map(|p| p.0).collect();
                    let idx_want: Vec<usize> = row_want.iter().map(|p| p.0).collect();
                    assert_eq!(idx_got, idx_want, "case {case} (n {n}, d {d})");
                    for (a, b) in row_got.iter().zip(row_want) {
                        assert!((a.1 - b.1).abs() <= 1e-9 * (1.0 + b.1), "case {case}");
                    }
                }
                let scores = laplacian_scores(&m, &got);
                let dense = dense_scores(&m, &got);
                for (a, b) in scores.iter().zip(&dense) {
                    assert!(
                        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-10 * b.abs() + 1e-15,
                        "case {case}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// Rare-flag noise (0/1 columns with about 10% ones) among informative
    /// columns, with average ranks and with ties broken at random: how many
    /// of 200 (20 columns, 10 seeds) beat every shadow, and whether a boolean
    /// aligned with the clusters does in every seed.
    fn rare_flag_noise_kept(tie_break: bool) -> (usize, bool) {
        let (n, informative, flags) = (300, 40, 20);
        let (mut kept, mut aligned_always) = (0, true);
        for seed in 0..10 {
            let mut g = SplitMix64::new(700 + seed);
            let mut cols: Vec<Vec<f64>> = (0..informative)
                .map(|_| {
                    (0..n)
                        .map(|i| (i % 3) as f64 + 0.3 * uniform(&mut g))
                        .collect()
                })
                .collect();
            cols.push((0..n).map(|i| f64::from(i % 3 == 0)).collect());
            for _ in 0..flags {
                cols.push((0..n).map(|_| f64::from(uniform(&mut g) < 0.1)).collect());
            }
            let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
            let rows: Vec<usize> = (0..n).collect();
            let z = if tie_break {
                let seeds: Vec<u64> = (0..refs.len() as u64).map(|j| 1000 * seed + j).collect();
                tie_broken_rank_matrix(&refs, &rows, &seeds)
            } else {
                rank_matrix(&refs, &rows)
            };
            let r = noise_scores(&z, 99, 10, &mut SplitMix64::new(seed));
            aligned_always &= r.scores[informative] < r.cutoff;
            kept += r.scores[informative + 1..]
                .iter()
                .filter(|s| **s < r.cutoff)
                .count();
        }
        (kept, aligned_always)
    }

    /// With ties broken at random, rare-flag noise gets through about as
    /// rarely as continuous noise (1%, ~2 of 200); with average ranks its tie
    /// blocks let it beat the shadows far more often (a numpy check of this
    /// setup: 32 of 200), which this test also confirms, so that it detects a
    /// return to average ranks. A cluster-aligned boolean is kept either way.
    #[test]
    fn test_discrete_noise_needs_tie_breaking() {
        let (with_ties_broken, aligned) = rare_flag_noise_kept(true);
        assert!(
            with_ties_broken <= 8,
            "{with_ties_broken} of 200 flags kept"
        );
        assert!(aligned, "the cluster-aligned boolean was dropped");
        let (with_average_ranks, _) = rare_flag_noise_kept(false);
        assert!(
            with_average_ranks >= 16,
            "only {with_average_ranks} of 200 flags kept with average ranks: \
             the setup no longer shows why ties are broken"
        );
    }
}
