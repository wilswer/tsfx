//! The rank matrices the correlation and noise steps work on: per feature, the
//! missing values filled with the median, then ranks standardized to mean 0
//! and variance 1. Ranks make both steps independent of scale and keep a few
//! extreme values from dominating. The correlation step uses average ranks
//! (Spearman); the noise step breaks ties at random instead, so that every
//! pure-noise feature, discrete or not, becomes a uniformly random permutation
//! like the shadow features it is compared with.

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

use super::rng::SplitMix64;

/// 1-based ranks of `xs` (no NaN), ties given their average rank.
pub(crate) fn average_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && xs[order[j]] == xs[order[i]] {
            j += 1;
        }
        // Sorted positions i..j share the ranks i + 1 ..= j
        let average = (i + 1 + j) as f64 / 2.0;
        for &k in &order[i..j] {
            ranks[k] = average;
        }
        i = j;
    }
    ranks
}

/// The `rows.len() × columns.len()` matrix of the columns' values at `rows`:
/// missing values (NaN) replaced by the median of the others (the mean of the
/// two middle values for an even count; 0.0 if all are missing), then average
/// ranks standardized to mean 0 and population variance 1. A column of equal
/// values becomes all 0.0. Every index in `rows` must be a row of every
/// column. Columns are processed in parallel.
pub(crate) fn rank_matrix(columns: &[&[f64]], rows: &[usize]) -> Array2<f64> {
    let ranked: Vec<Vec<f64>> = columns
        .par_iter()
        .map(|column| {
            let mut ranks = average_ranks(&filled(column, rows));
            standardize(&mut ranks);
            ranks
        })
        .collect();
    assemble(rows.len(), &ranked)
}

/// Like [`rank_matrix`], but ties are broken at random, from one seed per
/// column (`seeds`, as many as `columns`): the ranks of every column are a
/// permutation of 1..=n, standardized. The noise step works on this matrix.
pub(crate) fn tie_broken_rank_matrix(
    columns: &[&[f64]],
    rows: &[usize],
    seeds: &[u64],
) -> Array2<f64> {
    assert_eq!(columns.len(), seeds.len(), "one seed per column");
    let ranked: Vec<Vec<f64>> = columns
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(column, &seed)| {
            let mut ranks = tie_broken_ranks(&filled(column, rows), &mut SplitMix64::new(seed));
            standardize(&mut ranks);
            ranks
        })
        .collect();
    assemble(rows.len(), &ranked)
}

/// 1-based ranks of `xs` (no NaN) with ties broken by random keys from `rng`.
fn tie_broken_ranks(xs: &[f64], rng: &mut SplitMix64) -> Vec<f64> {
    let keys: Vec<u64> = (0..xs.len()).map(|_| rng.next_u64()).collect();
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]).then(keys[a].cmp(&keys[b])));
    let mut ranks = vec![0.0; xs.len()];
    for (rank, &i) in order.iter().enumerate() {
        ranks[i] = (rank + 1) as f64;
    }
    ranks
}

/// The column's values at `rows`, with missing values (NaN) replaced by the
/// median of the others (0.0 if all are missing) and -0.0 by 0.0.
fn filled(column: &[f64], rows: &[usize]) -> Vec<f64> {
    let mut xs: Vec<f64> = rows.iter().map(|&i| column[i] + 0.0).collect();
    let mut present: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    let fill = median(&mut present).unwrap_or(0.0);
    for x in xs.iter_mut().filter(|x| x.is_nan()) {
        *x = fill;
    }
    xs
}

/// The matrix with `columns` (each `n` long) as its columns.
fn assemble(n: usize, columns: &[Vec<f64>]) -> Array2<f64> {
    let mut z = Array2::zeros((n, columns.len()));
    for (j, column) in columns.iter().enumerate() {
        z.column_mut(j).assign(&ArrayView1::from(column.as_slice()));
    }
    z
}

/// The median of `xs` (sorted in place), `None` for no values.
fn median(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        // As numpy: (a + b) / 2, halving first only where the sum overflows
        // (halving first loses the last bit of subnormal values)
        let (a, b) = (xs[n / 2 - 1], xs[n / 2]);
        let sum = a + b;
        if sum.is_finite() {
            sum / 2.0
        } else {
            a / 2.0 + b / 2.0
        }
    })
}

/// Mean 0 and population variance 1 in place; all 0.0 without variance.
fn standardize(xs: &mut [f64]) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance > 0.0 {
        let sd = variance.sqrt();
        for x in xs.iter_mut() {
            *x = (*x - mean) / sd;
        }
    } else {
        xs.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{average_ranks, median, rank_matrix, tie_broken_rank_matrix};

    #[test]
    fn test_average_ranks() {
        assert_eq!(
            average_ranks(&[10.0, 20.0, 20.0, 30.0]),
            vec![1.0, 2.5, 2.5, 4.0]
        );
        assert_eq!(average_ranks(&[3.0, 1.0, 2.0]), vec![3.0, 1.0, 2.0]);
        assert_eq!(average_ranks(&[0.0, -0.0]), vec![1.5, 1.5]);
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut [f64::MAX, f64::MAX]), Some(f64::MAX));
        // subnormals: as numpy's (a + b) / 2, not a / 2 + b / 2 (which is 0)
        let tiny = f64::from_bits(1);
        assert_eq!(median(&mut [tiny, tiny]), Some(tiny));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn test_rank_matrix() {
        let a = [1.0, f64::NAN, 3.0]; // NaN -> median 2
        let c = [5.0, 5.0, 5.0];
        let z = rank_matrix(&[&a, &c], &[0, 1, 2]);
        // ranks 1, 2, 3: mean 2, population std sqrt(2/3)
        assert!((z[[0, 0]] + 1.0 / (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(z[[1, 0]], 0.0);
        assert_eq!(z.column(1).to_vec(), vec![0.0; 3]);
        let sub = rank_matrix(&[&a], &[0, 2]);
        assert_eq!(sub.dim(), (2, 1));
        assert!(sub[[0, 0]] < 0.0 && sub[[1, 0]] > 0.0);
    }

    #[test]
    fn test_rank_matrix_all_missing_is_zero() {
        let m = [f64::NAN; 4];
        let z = rank_matrix(&[&m], &[0, 1, 2, 3]);
        assert_eq!(z.column(0).to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_tie_broken_ranks_are_a_permutation() {
        let xs = [1.0, 1.0, 1.0, 2.0, f64::NAN, -0.0, 0.0];
        let rows: Vec<usize> = (0..xs.len()).collect();
        let z = tie_broken_rank_matrix(&[&xs], &rows, &[7]);
        let base = rank_matrix(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], &rows);
        let mut got = z.column(0).to_vec();
        let mut want = base.column(0).to_vec();
        got.sort_by(f64::total_cmp);
        want.sort_by(f64::total_cmp);
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-12));
        // 2.0 is the largest value: always the highest rank
        assert_eq!(z[[3, 0]], *want.last().unwrap());
        // the same seed gives the same ranks, another seed breaks ties anew
        assert_eq!(z, tie_broken_rank_matrix(&[&xs], &rows, &[7]));
        let others: Vec<_> = (0..20)
            .map(|seed| tie_broken_rank_matrix(&[&xs], &rows, &[seed]))
            .collect();
        assert!(others.iter().any(|other| *other != z));
    }
}
