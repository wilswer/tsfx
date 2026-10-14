//! Complexity and entropy measures.

use std::collections::{BTreeMap, HashSet};

use super::kernel::{GroupContext, Kernel, extremes, scalar_kernel};
use crate::utils::stats::{numpy_linspace, numpy_searchsorted_left, numpy_sum};

/// Numbers of unordered pairs of length-`m` and of length-`(m + 1)` sliding
/// windows of `x` within Chebyshev distance `r` of each other (tsfresh counts
/// a match at distance <= tolerance).
///
/// Both counts come from one pass over the pairs, comparing values in place:
/// a pair of length-`(m + 1)` windows can only match if its first `m` values
/// do.
fn _count_matches(x: &[f64], m: usize, r: f64) -> (usize, usize) {
    // Numbers of windows of length m and m + 1
    let n_m = (x.len() + 1).saturating_sub(m);
    let n_m_plus_1 = x.len().saturating_sub(m);
    let close = |i: usize, j: usize, k: usize| (x[i + k] - x[j + k]).abs() <= r;
    let (mut matches_m, mut matches_m_plus_1) = (0, 0);
    for i in 0..n_m {
        for j in (i + 1)..n_m {
            if (0..m).all(|k| close(i, j, k)) {
                matches_m += 1;
                if j < n_m_plus_1 && close(i, j, m) {
                    matches_m_plus_1 += 1;
                }
            }
        }
    }
    (matches_m, matches_m_plus_1)
}

scalar_kernel! {
/// Sample entropy feature.
///
/// The sample entropy (SampEn) of the time series, a measure of its
/// complexity, with embedding dimension $m = 2$ and tolerance
/// $r = 0.2\,\sigma$, where $\sigma$ is the population standard deviation:
/// $$ \text{SampEn} = -\ln \frac{A}{B}, $$
/// where $B$ is the number of pairs of length-$m$ windows and $A$ the number
/// of pairs of length-$(m+1)$ windows whose Chebyshev (maximum) distance is
/// at most $r$. A regular series scores low, an irregular one high.
///
/// # Output column
/// `{name}__sample_entropy`
///
/// # Edge cases
/// - Unlike the other features, nulls are **not** dropped; a null or NaN
///   anywhere in the series gives NaN.
/// - Fewer than 4 values give NaN (no matching pairs of windows).
/// - When no length-$(m+1)$ windows match, the result is $+\infty$.
/// - **Deliberate deviation from tsfresh:** a series containing $\pm\infty$
///   gives NaN. tsfresh's tolerance becomes NaN, every match count turns
///   negative after subtracting self-matches, and the ratio of two negative
///   counts returns a finite but meaningless value.
///
/// # Cost
/// $O(n^2)$ in the series length; only computed for
/// `FeatureSetting::Comprehensive`, and only with a `[sample_entropy]`
/// config section (in the default config).
///
/// # tsfresh
/// `feature_calculators.sample_entropy` (v0.21.2).
    SampleEntropy, "sample_entropy", |ctx| {
        // Nulls count as NaN here, and any NaN makes the result NaN
        if ctx.has_nulls() {
            f64::NAN
        } else {
            let m = 2;
            let r = 0.2 * ctx.std();
            let (matches_m, matches_m_plus_1) = _count_matches(ctx.values(), m, r);
            ((matches_m as f64) / (matches_m_plus_1 as f64)).ln()
        }
    }
}

/// The entropy of the values binned into `max_bins` equal-width bins, as
/// tsfresh's `binned_entropy`: `np.histogram(values, bins=max_bins)` (bin
/// edges and the assignment of values to bins reproduced exactly, the last
/// bin closed), then $-\sum_k p_k \ln p_k$ over the non-empty bins.
///
/// NaN if `values` is empty, contains NaN, `max_bins` is 0, or numpy's
/// histogram raises (an infinite value, or a range too small for
/// `max_bins` distinct edges).
pub(crate) fn binned_entropy(values: &[f64], max_bins: usize) -> f64 {
    if values.is_empty() || max_bins == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let (mut first, mut last) = extremes(values.iter().copied());
    // np.histogram: "autodetected range is not finite"
    if !first.is_finite() || !last.is_finite() {
        return f64::NAN;
    }
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    let edges = numpy_linspace(first, last, max_bins + 1);
    // np.histogram: "Too many bins for data range"
    if edges.windows(2).any(|w| w[0] >= w[1]) {
        return f64::NAN;
    }
    let denom = last - first;
    let mut hist = vec![0_usize; max_bins];
    for &v in values {
        // numpy's fast path for equal bins, including its ±1 corrections
        let mut idx = (((v - first) / denom * max_bins as f64) as usize).min(max_bins);
        if idx == max_bins {
            idx -= 1;
        }
        if idx > 0 && v < edges[idx] {
            idx -= 1;
        }
        if v >= edges[idx + 1] && idx != max_bins - 1 {
            idx += 1;
        }
        hist[idx] += 1;
    }
    let n = values.len() as f64;
    let terms: Vec<f64> = hist
        .iter()
        .map(|&count| {
            let p = count as f64 / n;
            let p = if p == 0.0 { 1.0 } else { p };
            p * p.ln()
        })
        .collect();
    -numpy_sum(terms.len(), |i| terms[i])
}

/// Binned entropy feature.
///
/// Bin the values into `max_bins` equal-width bins between the minimum and
/// the maximum and return the entropy of the bin frequencies:
/// $$ -\sum_{k} p_k \ln p_k \cdot \mathbf{1}_{(p_k > 0)}, $$
/// where $p_k$ is the share of values in bin $k$. Bins follow
/// `np.histogram` exactly: each bin includes its left edge, the last bin
/// also its right edge, and a constant series is binned over
/// $[x - 0.5, x + 0.5]$.
///
/// # Parameters
/// - `max_bins`: number of bins. Config:
///   `[binned_entropy] parameters = [{ max_bins = 10 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__binned_entropy__max_bins_{max_bins}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A NaN anywhere gives NaN, as tsfresh.
/// - `max_bins = 0` gives NaN.
/// - **Deliberate deviation from tsfresh:** a series containing $\pm\infty$,
///   or whose range is too small for `max_bins` distinct bin edges, gives
///   NaN; tsfresh raises numpy's `ValueError` from `np.histogram`.
///
/// # tsfresh
/// `feature_calculators.binned_entropy` (v0.21.2). tsfresh's `np.log`
/// (numpy's SIMD logarithm) and Rust's `ln` can differ in the last bit, so
/// results can differ from tsfresh's in the last bits.
pub struct BinnedEntropy {
    pub max_bins: Vec<usize>,
}

impl Kernel for BinnedEntropy {
    fn names(&self) -> Vec<String> {
        self.max_bins
            .iter()
            .map(|max_bins| format!("binned_entropy__max_bins_{max_bins}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        out.extend(
            self.max_bins
                .iter()
                .map(|&max_bins| binned_entropy(x, max_bins)),
        );
    }
}

/// Lempel-Ziv complexity feature.
///
/// Bin the values into `bins` equal-width bins between the minimum and the
/// maximum (bin $k$ holds the values $v$ with $e_k < v \leq e_{k+1}$, as
/// `np.searchsorted(np.linspace(min, max, bins + 1)[1:], x, side="left")`),
/// then parse the sequence of bin indices from left to right into the
/// shortest sub-words not seen before. The complexity is the number of
/// distinct sub-words divided by the series length; for example "100111"
/// splits into 1, 0, 01 and 11, giving $4/6$.
///
/// # Parameters
/// - `bins`: number of bins. Config:
///   `[lempel_ziv_complexity] parameters = [{ bins = 2 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__lempel_ziv_complexity__bins_{bins}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A NaN makes every bin edge NaN; values are then binned as numpy's
///   `searchsorted` does with NaN sorted last, exactly as in tsfresh.
///
/// # Cost
/// $O(n^2)$ in the worst case (long repeated sub-words), typically close to
/// linear.
///
/// # tsfresh
/// `feature_calculators.lempel_ziv_complexity` (v0.21.2).
pub struct LempelZivComplexity {
    pub bins: Vec<usize>,
}

impl Kernel for LempelZivComplexity {
    fn names(&self) -> Vec<String> {
        self.bins
            .iter()
            .map(|bins| format!("lempel_ziv_complexity__bins_{bins}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let (min, max) = (ctx.min(), ctx.max());
        for &bins in &self.bins {
            let edges = numpy_linspace(min, max, bins + 1);
            let sequence = numpy_searchsorted_left(&edges[1..], x);
            let n = sequence.len();
            let mut sub_words: HashSet<&[usize]> = HashSet::new();
            let (mut ind, mut inc) = (0, 1);
            while ind + inc <= n {
                let sub_word = &sequence[ind..ind + inc];
                if sub_words.contains(sub_word) {
                    inc += 1;
                } else {
                    sub_words.insert(sub_word);
                    ind += inc;
                    inc = 1;
                }
            }
            out.push(sub_words.len() as f64 / n as f64);
        }
    }
}

/// Permutation entropy feature.
///
/// Cut the series into windows of `dimension` values, starting every `tau`
/// values, replace each window by the ordinal pattern of its values (the
/// rank of each value within the window), and return the entropy of the
/// pattern frequencies, $-\sum_\pi p_\pi \ln p_\pi$ (Bandt & Pompe, 2002).
///
/// # Parameters
/// - `tau`: step between window starts.
/// - `dimension`: window length.
///
/// Config:
/// `[permutation_entropy] parameters = [{ tau = 1, dimension = 3 }, ...]`; one
/// column per entry.
///
/// # Output column
/// `{name}__permutation_entropy__tau_{tau}__dimension_{dimension}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - Fewer than `dimension` values (no complete window) give NaN, as
///   tsfresh.
/// - NaN ranks above every number; `-0.0` and `0.0` are equal.
/// - `dimension < 2` or `tau = 0` give NaN (tsfresh fails an assertion).
/// - **Deliberate deviation from tsfresh:** equal values within a window
///   (including two NaN) are ranked by position, as a stable sort does.
///   tsfresh ranks with `np.argsort`, whose order of ties depends on the
///   sorting algorithm numpy picks for the CPU (with AVX-512 it is not
///   stable), so its result for windows with ties varies between machines.
///
/// # tsfresh
/// `feature_calculators.permutation_entropy` (v0.21.2). tsfresh's `np.log`
/// and Rust's `ln` can differ in the last bit.
pub struct PermutationEntropy {
    /// `(tau, dimension)` per config entry.
    pub params: Vec<(usize, usize)>,
}

impl Kernel for PermutationEntropy {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(tau, dimension)| {
                format!("permutation_entropy__tau_{tau}__dimension_{dimension}")
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        for &(tau, dimension) in &self.params {
            out.push(permutation_entropy(x, tau, dimension));
        }
    }
}

fn permutation_entropy(x: &[f64], tau: usize, dimension: usize) -> f64 {
    if dimension < 2 || tau == 0 || x.len() < dimension {
        return f64::NAN;
    }
    let less = |a: f64, b: f64| a < b || (b.is_nan() && !a.is_nan());
    let equal = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    let windows = (x.len() - dimension) / tau + 1;
    // The rank of each value of window k
    let ranks = |k: usize| {
        let window = &x[k * tau..k * tau + dimension];
        window.iter().enumerate().map(move |(i, &v)| {
            window
                .iter()
                .enumerate()
                .filter(|&(j, &w)| less(w, v) || (j < i && equal(w, v)))
                .count()
        })
    };
    let terms = match (dimension as u64).checked_pow(dimension as u32) {
        // A pattern as one base-`dimension` number, which orders the
        // patterns as their rank sequences: no allocation per window
        Some(_) => pattern_terms(
            windows,
            (0..windows).map(|k| ranks(k).fold(0u64, |code, r| code * dimension as u64 + r as u64)),
        ),
        None => pattern_terms(windows, (0..windows).map(|k| ranks(k).collect::<Vec<_>>())),
    };
    -numpy_sum(terms.len(), |i| terms[i])
}

/// $p \ln p$ of each pattern's frequency, in increasing pattern order (as
/// `np.unique(axis=0)` counts them, lexicographically).
fn pattern_terms<K: Ord>(windows: usize, patterns: impl Iterator<Item = K>) -> Vec<f64> {
    let mut counts: BTreeMap<K, usize> = BTreeMap::new();
    for pattern in patterns {
        *counts.entry(pattern).or_insert(0) += 1;
    }
    counts
        .values()
        .map(|&count| {
            let p = count as f64 / windows as f64;
            p * p.ln()
        })
        .collect()
}

/// Approximate entropy feature.
///
/// The approximate entropy (ApEn) of the series (Pincus, 1991), a measure of
/// its regularity, with run length `m` and tolerance $r\,\sigma$, where
/// $\sigma$ is the population standard deviation:
/// $$ \text{ApEn} = \left| \Phi_m - \Phi_{m+1} \right|, \quad
///    \Phi_k = \frac{1}{N_k} \sum_{j=1}^{N_k} \ln \frac{C_j^k}{N_k}, $$
/// where $N_k = n - k + 1$ is the number of length-$k$ windows and $C_j^k$
/// the number of them (window $j$ itself included) within Chebyshev
/// (maximum) distance $r\,\sigma$ of window $j$.
///
/// # Parameters
/// - `m`: run length.
/// - `r`: tolerance as a multiple of the standard deviation.
///
/// Config:
/// `[approximate_entropy] parameters = [{ m = 2, r = 0.1 }, ...]`; one column
/// per entry.
///
/// # Output column
/// `{name}__approximate_entropy__m_{m}__r_{r}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - $n \leq m + 1$ gives 0, as tsfresh.
/// - NaN or $\pm\infty$ values are kept: windows at NaN distance never
///   match, exactly as in tsfresh (typically giving NaN).
/// - A negative tolerance $r\,\sigma < 0$ or `m = 0` gives NaN (tsfresh
///   raises).
///
/// # Cost
/// $O(n^2)$ in the series length; all `r` of one `m` share a single pass
/// over the pairs of windows. Only computed for
/// `FeatureSetting::Comprehensive`.
///
/// # tsfresh
/// `feature_calculators.approximate_entropy` (v0.21.2). tsfresh's `np.log`
/// and Rust's `ln` can differ in the last bit.
pub struct ApproximateEntropy {
    /// `(m, r)` per config entry.
    pub params: Vec<(usize, f64)>,
}

impl Kernel for ApproximateEntropy {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(m, r)| format!("approximate_entropy__m_{m}__r_{r}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let n = x.len();
        let std = ctx.std();
        let mut results = vec![f64::NAN; self.params.len()];
        // Parameters that need the pass over the windows, by run length
        let mut pending: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
        for (k, &(m, r)) in self.params.iter().enumerate() {
            let r = r * std;
            if r < 0.0 || m == 0 {
                continue;
            }
            if n <= m + 1 {
                results[k] = 0.0;
                continue;
            }
            pending.entry(m).or_default().push((k, r));
        }
        for (m, params) in pending {
            let rs: Vec<f64> = params.iter().map(|&(_, r)| r).collect();
            for (&(k, _), apen) in params.iter().zip(approximate_entropies(x, m, &rs)) {
                results[k] = apen;
            }
        }
        out.extend(results);
    }
}

/// NaN if either is NaN, else the larger: `np.max` of two values.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Approximate entropy for run length `m` (with `n > m + 1`) and each
/// absolute tolerance in `rs`, in one pass over the pairs of windows.
fn approximate_entropies(x: &[f64], m: usize, rs: &[f64]) -> Vec<f64> {
    let n = x.len();
    let (w_m, w_m1) = (n - m + 1, n - m);
    let mut counts_m = vec![vec![0_u32; w_m]; rs.len()];
    let mut counts_m1 = vec![vec![0_u32; w_m1]; rs.len()];
    for i in 0..w_m {
        for j in i..w_m {
            let d_m = (0..m).fold(0.0, |d, k| nan_max(d, (x[i + k] - x[j + k]).abs()));
            for (r, counts) in rs.iter().zip(counts_m.iter_mut()) {
                if d_m <= *r {
                    counts[i] += 1;
                    if i != j {
                        counts[j] += 1;
                    }
                }
            }
            if j < w_m1 {
                let d_m1 = nan_max(d_m, (x[i + m] - x[j + m]).abs());
                for (r, counts) in rs.iter().zip(counts_m1.iter_mut()) {
                    if d_m1 <= *r {
                        counts[i] += 1;
                        if i != j {
                            counts[j] += 1;
                        }
                    }
                }
            }
        }
    }
    let phi = |counts: &[u32]| {
        let w = counts.len() as f64;
        numpy_sum(counts.len(), |j| (counts[j] as f64 / w).ln()) / w
    };
    counts_m
        .iter()
        .zip(&counts_m1)
        .map(|(c_m, c_m1)| (phi(c_m) - phi(c_m1)).abs())
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayView1;

    use super::*;

    // Expected values from tsfresh v0.21.2 / numpy 2.4.4.
    #[test]
    fn test_binned_entropy() {
        assert_eq!(binned_entropy(&[10.0; 100], 10), 0.0);
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        assert!((binned_entropy(&x, 100) - 10_f64.ln()).abs() < 1e-12);
        // the maximum falls in the last (closed) bin
        assert!((binned_entropy(&[0.0, 1.0], 2) - 2_f64.ln()).abs() < 1e-12);
        assert!(binned_entropy(&[1.0, f64::NAN], 10).is_nan());
        assert!(binned_entropy(&[1.0, f64::INFINITY], 10).is_nan());
    }

    #[test]
    fn test_permutation_entropy() {
        let x = [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0];
        assert!((permutation_entropy(&x, 1, 3) - 1.054920167).abs() < 1e-9);
        assert!(permutation_entropy(&[1.0, -1.0], 1, 3).is_nan());
    }

    #[test]
    fn test_approximate_entropy() {
        let x: Vec<f64> = [12.0, 13.0, 15.0, 16.0, 17.0].repeat(10);
        let std = crate::utils::stats::population_std(&ArrayView1::from(&x));
        let apen = approximate_entropies(&x, 2, &[0.9 * std]);
        assert!((apen[0] - 0.282456191).abs() < 1e-9);
    }
}
