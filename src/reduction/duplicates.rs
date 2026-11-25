//! Step 3: near-duplicates. Spearman correlations between the features (from
//! the rank matrix), then a greedy walk that keeps a feature unless it is too
//! correlated with one kept before it.

use ndarray::{Array2, s};
use rayon::prelude::*;

/// Columns per block in the correlation matrix's parallel product.
const BLOCK: usize = 64;

/// The correlation matrix `zᵀ z / n` of the columns of `z` (n rows). With
/// standardized ranks in `z` (see `rank_matrix`) this is Spearman's
/// correlation. Computed in parallel over blocks of 64 columns, each block of
/// rows of the result a single-threaded matrix product, placed by index.
pub(crate) fn correlation_matrix(z: &Array2<f64>) -> Array2<f64> {
    let (n, d) = z.dim();
    let starts: Vec<usize> = (0..d).step_by(BLOCK).collect();
    let blocks: Vec<Array2<f64>> = starts
        .par_iter()
        .map(|&start| {
            let end = (start + BLOCK).min(d);
            z.slice(s![.., start..end]).t().dot(z)
        })
        .collect();
    let mut corr = Array2::zeros((d, d));
    for (&start, block) in starts.iter().zip(&blocks) {
        corr.slice_mut(s![start..start + block.nrows(), ..])
            .assign(block);
    }
    let n = n as f64;
    corr.mapv_inplace(|x| x / n);
    corr
}

/// A feature dropped as a near-duplicate of a kept one.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Duplicate {
    /// Index of the kept feature it copies.
    pub of: usize,
    /// Their signed correlation.
    pub correlation: f64,
}

/// Walks the features in `order` and keeps each unless its |correlation| with
/// an already kept feature is above `max_correlation`; then it is a
/// [`Duplicate`] of the kept feature it correlates with most (the one kept
/// first on ties). Indexed by feature; `None` means kept. `order` is the
/// priority, a permutation of the feature indices: earlier features win
/// (extraction order in stage 1).
pub(crate) fn greedy_duplicates(
    corr: &Array2<f64>,
    order: &[usize],
    max_correlation: f64,
) -> Vec<Option<Duplicate>> {
    let mut result: Vec<Option<Duplicate>> = vec![None; corr.nrows()];
    let mut kept: Vec<usize> = Vec::new();
    for &j in order {
        let mut best: Option<Duplicate> = None;
        for &k in &kept {
            let r = corr[[j, k]];
            if r.abs() > max_correlation
                && best.as_ref().is_none_or(|b| r.abs() > b.correlation.abs())
            {
                best = Some(Duplicate {
                    of: k,
                    correlation: r,
                });
            }
        }
        match best {
            Some(duplicate) => result[j] = Some(duplicate),
            None => kept.push(j),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{correlation_matrix, greedy_duplicates};
    use crate::reduction::matrix::rank_matrix;
    use crate::reduction::rng::SplitMix64;
    use ndarray::{Array2, array};

    #[test]
    fn test_spearman_monotone_and_negative() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let sq = [1.0, 4.0, 9.0, 16.0];
        let rev = [4.0, 3.0, 2.0, 1.0];
        let z = rank_matrix(&[&x, &sq, &rev], &[0, 1, 2, 3]);
        let c = correlation_matrix(&z);
        assert!((c[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((c[[0, 2]] + 1.0).abs() < 1e-12);
        assert!((c[[1, 1]] - 1.0).abs() < 1e-12);
    }

    /// Blocks of 64 columns put together give the plain product, also for a
    /// width that is not a multiple of 64.
    #[test]
    fn test_correlation_blocks_match_direct_product() {
        let mut g = SplitMix64::new(5);
        let (n, d) = (37, 130);
        let z = Array2::from_shape_fn((n, d), |_| {
            (g.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        });
        let c = correlation_matrix(&z);
        let direct = z.t().dot(&z) / n as f64;
        assert_eq!(c.dim(), (d, d));
        assert!(
            c.iter()
                .zip(direct.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    fn corr3(r01: f64, r02: f64, r12: f64) -> Array2<f64> {
        array![[1.0, r01, r02], [r01, 1.0, r12], [r02, r12, 1.0]]
    }

    #[test]
    fn test_greedy() {
        let d = greedy_duplicates(&corr3(0.97, 0.5, 0.96), &[0, 1, 2], 0.95);
        assert!(d[0].is_none() && d[2].is_none());
        assert_eq!(d[1].as_ref().unwrap().of, 0);
        // equal to the threshold is kept; negative correlation counts by |ρ|
        assert!(greedy_duplicates(&corr3(0.95, 0.0, 0.0), &[0, 1, 2], 0.95)[1].is_none());
        let neg = greedy_duplicates(&corr3(-0.97, 0.0, 0.0), &[0, 1, 2], 0.95);
        assert_eq!(neg[1].as_ref().unwrap().correlation, -0.97);
        // the order decides who is kept
        let rev = greedy_duplicates(&corr3(0.97, 0.5, 0.96), &[2, 1, 0], 0.95);
        assert!(rev[2].is_none() && rev[0].is_none());
        assert_eq!(rev[1].as_ref().unwrap().of, 2);
    }

    #[test]
    fn test_greedy_names_the_closest_kept_feature() {
        // 2 copies both 0 (0.96) and 1 (0.99): it is a duplicate of 1
        let d = greedy_duplicates(&corr3(0.1, 0.96, 0.99), &[0, 1, 2], 0.95);
        assert_eq!(d[2].as_ref().unwrap().of, 1);
        // a tie goes to the feature kept first
        let tie = greedy_duplicates(&corr3(0.1, 0.97, -0.97), &[0, 1, 2], 0.95);
        assert_eq!(tie[2].as_ref().unwrap().of, 0);
    }
}
