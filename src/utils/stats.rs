//! Shared statistical conventions for the feature extractors.

use std::borrow::Cow;

use itertools::Itertools;
use ndarray::ArrayView1;

/// Sum of `value(0), ..., value(n - 1)` exactly as numpy's `add.reduce` sums
/// a contiguous `float64` array (numpy 2.x): pairwise summation with eight
/// accumulators for blocks of up to 128 values, split in halves above that,
/// added to an initial `0.0`. Results equal numpy's bit for bit, which a
/// straight or Welford sum does not.
pub fn numpy_sum(n: usize, value: impl Fn(usize) -> f64) -> f64 {
    0.0 + pairwise_sum(0, n, &value)
}

fn pairwise_sum(start: usize, n: usize, value: &impl Fn(usize) -> f64) -> f64 {
    if n < 8 {
        (start..start + n).fold(0.0, |res, i| res + value(i))
    } else if n <= 128 {
        let mut r: [f64; 8] = std::array::from_fn(|j| value(start + j));
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += value(start + i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for k in i..n {
            res += value(start + k);
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(start, n2, value) + pairwise_sum(start + n2, n - n2, value)
    }
}

/// Arithmetic mean, exactly as `np.mean` (see [`numpy_sum`]). NaN for an
/// empty series or if any value is NaN.
pub fn numpy_mean(x: &[f64]) -> f64 {
    numpy_sum(x.len(), |i| x[i]) / x.len() as f64
}

/// Population variance (`ddof = 0`) of a series.
///
/// Exactly `np.var` as used by tsfresh: the mean, then the mean of the
/// squared deviations, both summed as numpy does (see [`numpy_sum`]).
/// Returns the raw value; how a zero variance is handled is up to each
/// feature.
pub fn population_var(arr: &ArrayView1<f64>) -> f64 {
    let n = arr.len();
    let mean = numpy_sum(n, |i| arr[i]) / n as f64;
    numpy_sum(n, |i| {
        let d = arr[i] - mean;
        d * d
    }) / n as f64
}

/// Population standard deviation (`ddof = 0`) of a series: the square root
/// of [`population_var`], exactly `np.std` as used by tsfresh. Returns the
/// raw value; how a zero standard deviation is handled is up to each
/// feature.
pub fn population_std(arr: &ArrayView1<f64>) -> f64 {
    population_var(arr).sqrt()
}

/// Pairwise leaf of pandas' moment sums: 128 SSE2 batches of two lanes.
const PANDAS_MOMENTS_LEAF: usize = 256;

/// The two lane sums and the non-NaN count of an even-length `x`, as pandas'
/// `accumulate_mean_pairwise` (x86_64 build: SSE2, two `f64` lanes): within a
/// leaf, lane `j` adds the values at positions `j, j + 2, ...`; longer slices
/// are split near the middle and their lane sums added. A NaN adds `+0.0` with
/// `skipna`, and ends the sum (`None`) without it.
fn pandas_mean_lanes(x: &[f64], skipna: bool) -> Option<([f64; 2], usize)> {
    if x.len() > PANDAS_MOMENTS_LEAF {
        let mid = x.len() / 2 / 2 * 2;
        let (left, left_count) = pandas_mean_lanes(&x[..mid], skipna)?;
        let (right, right_count) = pandas_mean_lanes(&x[mid..], skipna)?;
        let sum = [left[0] + right[0], left[1] + right[1]];
        return Some((sum, left_count + right_count));
    }
    let mut sum = [0.0; 2];
    let mut count = 0;
    for pair in x.as_chunks::<2>().0 {
        if !skipna && pair.iter().any(|v| v.is_nan()) {
            return None;
        }
        for (lane, &v) in pair.iter().enumerate() {
            sum[lane] += if v.is_nan() { 0.0 } else { v };
            count += usize::from(!v.is_nan());
        }
    }
    Some((sum, count))
}

/// Lane sums of $d, d^2, d^3, d^4$ with $d = x_i - \text{mean}$ (0 for a NaN)
/// of an even-length `x`, as pandas' `accumulate_central_diffs_pairwise`, in
/// the same lanes and leaves as [`pandas_mean_lanes`].
fn pandas_central_lanes(x: &[f64], mean: f64) -> [[f64; 2]; 4] {
    if x.len() > PANDAS_MOMENTS_LEAF {
        let mid = x.len() / 2 / 2 * 2;
        let left = pandas_central_lanes(&x[..mid], mean);
        let right = pandas_central_lanes(&x[mid..], mean);
        return std::array::from_fn(|k| [left[k][0] + right[k][0], left[k][1] + right[k][1]]);
    }
    let mut acc = [[0.0; 2]; 4];
    for pair in x.as_chunks::<2>().0 {
        for (lane, &v) in pair.iter().enumerate() {
            let d = if v.is_nan() { 0.0 } else { v - mean };
            let d2 = d * d;
            acc[0][lane] += d;
            acc[1][lane] += d2;
            acc[2][lane] += d2 * d;
            acc[3][lane] += d2 * d2;
        }
    }
    acc
}

/// `std::max(v, 0.0)`: a negative becomes 0, NaN stays NaN (unlike
/// [`f64::max`]).
fn clamp_negative(v: f64) -> f64 {
    if v < 0.0 { 0.0 } else { v }
}

/// The number of non-NaN values of `x` and its central moment sums
/// $m_k = \sum_i (x_i - \mu)^k$ for $k = 2, 3, 4$, bit for bit as pandas 3.1
/// computes them for `Series.skew` and `Series.kurtosis` (`moments_reduce`,
/// x86_64 build).
///
/// pandas sums in two passes: a trial mean, then the powers of the
/// deviations from it, both in two SSE2 lanes with pairwise sums (see
/// [`pandas_mean_lanes`]) and the odd last value added on its own. The
/// trial mean's rounding error is then corrected (Pébay et al., 2016), so a
/// constant series gives exactly $m_2 = 0$. With `skipna`, NaN values are
/// skipped where they are; without it, a NaN makes every sum NaN.
fn pandas_moments(x: &[f64], skipna: bool) -> (usize, [f64; 3]) {
    let undefined = (x.len(), [f64::NAN; 3]);
    let (front, last) = x.split_at(x.len() / 2 * 2);
    let Some((lanes, front_count)) = pandas_mean_lanes(front, skipna) else {
        return undefined;
    };
    if !skipna && last.iter().any(|v| v.is_nan()) {
        return undefined;
    }
    // The odd last value, unless it is a skipped NaN; pandas sums it on its
    // own from 0.0 and adds that to the two lanes
    let last = last.first().copied().filter(|v| !v.is_nan());
    let count = front_count + usize::from(last.is_some());
    if count == 0 {
        return (0, [0.0; 3]);
    }
    let n = count as f64;
    let mean = (lanes[0] + lanes[1] + last.map_or(0.0, |v| 0.0 + v)) / n;

    let lanes = pandas_central_lanes(front, mean);
    let last_acc = last.map_or([0.0; 4], |v| {
        let d = v - mean;
        let d2 = d * d;
        [0.0 + d, 0.0 + d2, 0.0 + d2 * d, 0.0 + d2 * d2]
    });
    let [m1, m2, m3, m4]: [f64; 4] =
        std::array::from_fn(|k| lanes[k][0] + lanes[k][1] + last_acc[k]);

    // Correct for the trial mean's error c = m1 / n, as pandas'
    // compute_moments_with_correction does
    let c = m1 / n;
    let c2 = c * c;
    let term1 = m1 * c;
    let corrected = [
        clamp_negative(m2 - term1),
        m3 - 3.0 * m2 * c + 2.0 * c * term1,
        clamp_negative(m4 - 4.0 * m3 * c + 6.0 * m2 * c2 - 3.0 * term1 * c2),
    ];
    (count, corrected)
}

/// `arr` as a slice, copied only if it is not contiguous.
fn contiguous<'a>(arr: &'a ArrayView1<f64>) -> Cow<'a, [f64]> {
    arr.as_slice()
        .map_or_else(|| Cow::Owned(arr.to_vec()), Cow::Borrowed)
}

/// Bias-adjusted sample skewness (Fisher-Pearson G1) of a series.
///
/// Matches `pandas.Series.skew(skipna=False)` (pandas 3.1), which tsfresh
/// uses, bit for bit on x86_64:
/// $$ G_1 = \frac{n \sqrt{n - 1}}{n - 2} \frac{m_3}{m_2^{3/2}}, $$
/// with $m_k = \sum_{i=1}^{n} (x_i - \mu)^k$ computed as pandas does.
/// Returns NaN for fewer than 3 values, for a constant series ($0 / 0$) and
/// if any value is NaN.
pub fn sample_skewness(arr: &ArrayView1<f64>) -> f64 {
    let (count, [m2, m3, _]) = pandas_moments(&contiguous(arr), false);
    if count < 3 {
        return f64::NAN;
    }
    let n = count as f64;
    m3 / (m2 * m2.sqrt()) * (n * (n - 1.0).sqrt() / (n - 2.0))
}

/// Bias-adjusted sample excess kurtosis of a series.
///
/// Matches `pandas.Series.kurtosis()` (pandas 3.1, `skipna=True`), which
/// tsfresh uses, bit for bit on x86_64:
/// $$ G_2 = \frac{n - 1}{(n - 2) (n - 3)}
///        \left( \frac{n (n + 1) m_4}{m_2^2} - 3 (n - 1) \right), $$
/// with $m_k = \sum_{i=1}^{n} (x_i - \mu)^k$ computed as pandas does over
/// the $n$ non-NaN values. A normal distribution scores 0. Returns NaN for
/// fewer than 4 non-NaN values and for a constant series ($0 / 0$).
pub fn sample_excess_kurtosis(arr: &ArrayView1<f64>) -> f64 {
    let (count, [m2, _, m4]) = pandas_moments(&contiguous(arr), true);
    if count < 4 {
        return f64::NAN;
    }
    let n = count as f64;
    let inner = n * (n + 1.0) * (m4 / (m2 * m2)) - 3.0 * (n - 1.0);
    (n - 1.0) / ((n - 2.0) * (n - 3.0)) * inner
}

/// Ordinary least squares intercept and slope of `y` against its index
/// $0, 1, \dots, n - 1$, as `scipy.stats.linregress` (which tsfresh uses)
/// computes them: centred sums,
/// $b = \sum (i - \bar{i})(y_i - \bar{y}) / \sum (i - \bar{i})^2$ and
/// $a = \bar{y} - b\,\bar{i}$, summed as numpy does. (A one-pass
/// $\sum i\,y_i - n\,\bar{i}\,\bar{y}$ loses digits to cancellation.)
/// NaN for fewer than 2 values.
pub(crate) fn calculate_sequential_ols(y: &[f64]) -> (f64, f64) {
    let n = y.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mean_x = (n as f64 - 1.0) / 2.0;
    let mean_y = numpy_mean(y);
    let ss_xx = numpy_sum(n, |i| (i as f64 - mean_x) * (i as f64 - mean_x));
    let ss_xy = numpy_sum(n, |i| (i as f64 - mean_x) * (y[i] - mean_y));
    let slope = ss_xy / ss_xx;
    (mean_y - slope * mean_x, slope)
}

pub(crate) fn get_length_sequences_where(x: &ndarray::Array1<bool>) -> Vec<usize> {
    let mut group_lengths = Vec::new();
    for (key, group) in &x.into_iter().chunk_by(|elt| *elt) {
        if *key {
            group_lengths.push(group.count());
        }
    }
    group_lengths
}

/// The `q`-quantile of `sorted` (ascending, not empty), exactly as
/// `np.quantile` computes it (method "linear"): index $h = (n - 1)\,q$,
/// neighbours $a = x_{(\lfloor h \rfloor)}$, $b = x_{(\lfloor h \rfloor + 1)}$
/// and weight $t = h - \lfloor h \rfloor$, then numpy's `_lerp`:
/// $a + (b - a)\,t$ for $t < 0.5$, $b - (b - a)(1 - t)$ otherwise. At or
/// beyond the last index numpy uses the last value for both neighbours with
/// $t = h + 1$ (it computes the weight from the replacement index $-1$).
/// So an infinite neighbour can give NaN, exactly as in numpy.
pub fn numpy_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let h = (n as f64 - 1.0) * q;
    let (i, j, t) = if h >= (n - 1) as f64 {
        (n - 1, n - 1, h + 1.0)
    } else {
        let previous = h.floor();
        (previous as usize, previous as usize + 1, h - previous)
    };
    let (a, b) = (sorted[i], sorted[j]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// The median of `values` (not empty, reordered in place), exactly as
/// `np.median` computes it: the `np.mean` of the middle value, or of the two
/// middle values for an even count. Found with a selection, $O(n)$.
pub fn numpy_median(values: &mut [f64]) -> f64 {
    let n = values.len();
    let (left, &mut upper, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        return numpy_mean(&[upper]);
    }
    let lower = left.iter().copied().max_by(f64::total_cmp).unwrap_or(upper);
    numpy_mean(&[lower, upper])
}

/// `num` evenly spaced values from `start` to `stop` (both included),
/// exactly as `np.linspace(start, stop, num)` computes them: $i \cdot s +
/// \text{start}$ with step $s = (\text{stop} - \text{start}) / (\text{num} - 1)$
/// (or $(i / (\text{num} - 1)) \cdot \Delta + \text{start}$ if the step
/// underflows to 0), the last value set to `stop`.
pub(crate) fn numpy_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let delta = stop - start;
    let mut y: Vec<f64> = if num > 1 {
        let div = (num - 1) as f64;
        let step = delta / div;
        if step == 0.0 {
            (0..num).map(|i| i as f64 / div * delta + start).collect()
        } else {
            (0..num).map(|i| i as f64 * step + start).collect()
        }
    } else {
        (0..num).map(|i| i as f64 * delta + start).collect()
    };
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// numpy's ordering of floats for sorting and searching: NaN sorts last.
fn numpy_less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `np.searchsorted(arr, keys, side="left")`, reproducing numpy's binary
/// search step for step (including how it reuses the bounds of the previous
/// key), so that the result also equals numpy's when `arr` is not sorted
/// (for example edges that came out NaN).
pub(crate) fn numpy_searchsorted_left(arr: &[f64], keys: &[f64]) -> Vec<usize> {
    let n = arr.len();
    let mut out = Vec::with_capacity(keys.len());
    let (mut min_idx, mut max_idx) = (0, n);
    let Some(&first) = keys.first() else {
        return out;
    };
    let mut last_key = first;
    for &key in keys {
        if numpy_less(last_key, key) {
            max_idx = n;
        } else {
            min_idx = 0;
            max_idx = if max_idx < n { max_idx + 1 } else { n };
        }
        last_key = key;
        while min_idx < max_idx {
            let mid = min_idx + ((max_idx - min_idx) >> 1);
            if numpy_less(arr[mid], key) {
                min_idx = mid + 1;
            } else {
                max_idx = mid;
            }
        }
        out.push(min_idx);
    }
    out
}

/// Index of the first maximum, like `np.argmax`: if the series contains NaN,
/// the index of the first NaN. `None` for an empty series.
pub(crate) fn np_argmax(x: &ArrayView1<f64>) -> Option<usize> {
    np_arg_extreme(x, |a, b| a > b)
}

/// Index of the first minimum, like `np.argmin`: if the series contains NaN,
/// the index of the first NaN. `None` for an empty series.
pub(crate) fn np_argmin(x: &ArrayView1<f64>) -> Option<usize> {
    np_arg_extreme(x, |a, b| a < b)
}

fn np_arg_extreme(x: &ArrayView1<f64>, better: fn(f64, f64) -> bool) -> Option<usize> {
    if let Some(i) = x.iter().position(|v| v.is_nan()) {
        return Some(i);
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in x.iter().enumerate() {
        match best {
            Some((_, b)) if !better(v, b) => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_population_var_and_std() {
        let arr = array![1.0, 2.0, 3.0, 4.0];
        assert_eq!(population_var(&arr.view()), 1.25);
        assert_eq!(population_std(&arr.view()), 1.25_f64.sqrt());
    }

    #[test]
    fn test_single_value_is_zero() {
        let arr = array![5.0];
        assert_eq!(population_var(&arr.view()), 0.0);
        assert_eq!(population_std(&arr.view()), 0.0);
    }

    fn skew(x: &[f64]) -> f64 {
        sample_skewness(&ArrayView1::from(x))
    }

    fn kurt(x: &[f64]) -> f64 {
        sample_excess_kurtosis(&ArrayView1::from(x))
    }

    /// `n` exactly representable values, long enough for pandas' pairwise
    /// sums (split above 256 values), with an odd last value when `n` is odd.
    fn spread(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 1013) as f64 / 8.0 - 60.0)
            .collect()
    }

    /// `x` with every 37th value NaN.
    fn with_nans(x: Vec<f64>) -> Vec<f64> {
        let nan = |(i, v): (usize, f64)| if i % 37 == 0 { f64::NAN } else { v };
        x.into_iter().enumerate().map(nan).collect()
    }

    // Expected values from pandas 3.1.1 (tsfresh v0.21.2), bit for bit.
    #[test]
    fn test_sample_skewness() {
        assert_eq!(skew(&[1.0, 1.0, 1.0, 2.0, 2.0]), 0.6085806194501846);
        assert!(skew(&[1.0, 2.0]).is_nan());
        // Constant: m2 is exactly 0, also where the mean is not exactly 0.1,
        // and 0 / 0 is NaN (pandas 3.0 gave 0)
        assert!(skew(&[1.0, 1.0, 1.0]).is_nan());
        assert!(skew(&[0.1, 0.1, 0.1]).is_nan());
        // Not constant in the last bit: its real value (pandas 3.0 gave 0)
        assert_eq!(
            skew(&[2.0, 2.0, 2.0, 2.0 + 2f64.powi(-51)]),
            1.9999999999999998
        );
        assert_eq!(skew(&spread(601)), -0.004372880111837337);
        assert_eq!(skew(&spread(1030)), 0.00044984524990832395);
        // skipna=False: a NaN makes the result NaN
        assert!(skew(&with_nans(spread(601))).is_nan());
    }

    #[test]
    fn test_sample_excess_kurtosis() {
        assert_eq!(kurt(&[1.0, 1.0, 1.0, 2.0, 2.0]), -3.333333333333334);
        assert!(kurt(&[1.0, 1.0, 2.0]).is_nan());
        assert!(kurt(&[1.0; 4]).is_nan());
        assert!(kurt(&[0.1; 4]).is_nan());
        assert_eq!(
            kurt(&[2.0, 2.0, 2.0, 2.0 + 2f64.powi(-51)]),
            4.000000000000002
        );
        assert_eq!(kurt(&spread(601)), -1.1968627379174224);
        assert_eq!(kurt(&spread(1030)), -1.1996285017289967);
        // skipna=True: NaN values are skipped where they are, not removed
        // first, which would move the later values to other lanes
        assert_eq!(kurt(&with_nans(spread(601))), -1.1970558807795155);
        assert_eq!(kurt(&with_nans(spread(1030))), -1.1999598316528708);
        assert!(kurt(&[1.0, f64::NAN, 2.0, 3.0]).is_nan());
    }
}
