//! Continuous wavelet transforms with the Ricker ("Mexican hat") wavelet.
//!
//! Both features convolve the series with sampled wavelets, using one
//! portable direct convolution, so results are the same on every platform.
//! They equal tsfresh's within rounding. [`NumberCwtPeaks`] compares
//! neighbouring coefficients; where two are tied in real arithmetic (for
//! example on a series symmetric about a point between two samples),
//! rounding decides, and the count can differ from tsfresh's. numpy's own
//! result there depends on its BLAS and CPU.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::kernel::{GroupContext, Kernel};

/// `np.convolve(a, v, mode)` for `mode` `"full"` or `"same"`: each output
/// is a plain sequential sum over the overlapping samples.
fn np_convolve(a: &[f64], v: &[f64], full: bool) -> Vec<f64> {
    let (a, v) = if v.len() > a.len() { (v, a) } else { (a, v) };
    let (n1, n2) = (a.len(), v.len());
    // Index into the full convolution (length n1 + n2 - 1) of the first
    // output, and the number of outputs
    let (first, len) = if full {
        (0, n1 + n2 - 1)
    } else {
        ((n2 - 1) / 2, n1)
    };
    (first..first + len)
        .map(|k| {
            // full[k] = sum over j of a[k - j] * v[j], for valid indices
            let lo = k.saturating_sub(n1 - 1);
            let hi = k.min(n2 - 1);
            let mut r = 0.0;
            for j in lo..=hi {
                r += a[k - j] * v[j];
            }
            r
        })
        .collect()
}

/// Number of samples of the integrated wavelet table (`precision = 12`).
const PSI_POINTS: usize = 1 << 12;

/// pywt's integrated Mexican hat wavelet: `mexh` sampled at 4096 points on
/// its support $[-8, 8]$ (`np.linspace`), integrated by cumulative sum
/// times the step (`pywt.integrate_wavelet`). Returns the table and step;
/// computed once.
fn integrated_mexh() -> &'static (Vec<f64>, f64) {
    static TABLE: std::sync::OnceLock<(Vec<f64>, f64)> = std::sync::OnceLock::new();
    TABLE.get_or_init(compute_integrated_mexh)
}

fn compute_integrated_mexh() -> (Vec<f64>, f64) {
    let (lo, hi) = (-8.0_f64, 8.0_f64);
    let delta = (PSI_POINTS - 1) as f64;
    let step = (hi - lo) / delta;
    let grid = |i: usize| {
        if i == PSI_POINTS - 1 {
            hi
        } else {
            i as f64 * step + lo
        }
    };
    let norm = 3.0_f64.sqrt() * PI.sqrt().sqrt();
    let dx = grid(1) - grid(0);
    let mut acc = 0.0;
    let table = (0..PSI_POINTS)
        .map(|i| {
            let t = grid(i);
            let t2 = t * t;
            acc += (1.0 - t2) * (-t2 / 2.0).exp() * 2.0 / norm;
            acc * dx
        })
        .collect();
    (table, dx)
}

/// `pywt.cwt(x, scales=widths, wavelet="mexh")` with its defaults
/// (`method="conv"`, `precision=12`): one row of `x.len()` coefficients per
/// width.
fn pywt_cwt_mexh(x: &[f64], widths: &[usize]) -> Vec<Vec<f64>> {
    let (int_psi, step) = integrated_mexh();
    let range = 16.0; // x[-1] - x[0] of the sampling grid
    widths
        .iter()
        .map(|&scale| {
            let scale_f = scale as f64;
            // np.arange(scale * range + 1) / (scale * step), truncated
            let count = (scale_f * range + 1.0).ceil() as usize;
            let filter: Vec<f64> = (0..count)
                .map(|i| (i as f64 / (scale_f * step)) as usize)
                .filter(|&j| j < int_psi.len())
                .map(|j| int_psi[j])
                .rev()
                .collect();
            let conv = np_convolve(x, &filter, true);
            let factor = -scale_f.sqrt();
            let coef: Vec<f64> = conv.windows(2).map(|w| factor * (w[1] - w[0])).collect();
            let d = (coef.len() - x.len()) as f64 / 2.0;
            let start = d.floor() as usize;
            coef[start..start + x.len()].to_vec()
        })
        .collect()
}

/// Continuous wavelet transform coefficients feature.
///
/// The continuous wavelet transform of the series with the Mexican hat
/// (Ricker) wavelet
/// $$ \psi(t) = \frac{2}{\sqrt{3}\,\pi^{1/4}} (1 - t^2)\, e^{-t^2/2} $$
/// at each width (scale) $a$ of `widths`, computed as pywt does (convolving
/// with the integrated wavelet and differencing). Returns coefficient
/// `coeff` (a position in the series) at width `w`.
///
/// # Parameters
/// - `widths`: the scales of one transform (positive integers).
/// - `coeff`: position of the coefficient, from 0.
/// - `w`: the width to read, one of `widths`.
///
/// Config: `[cwt_coefficients] parameters = [{ widths = [2, 5, 10, 20], coeff = 0, w = 2 }, ...]`;
/// one column per entry. The transform is computed once per distinct
/// `widths` (and group). A `w` that is not in `widths` is rejected when the
/// features are set up.
///
/// # Output column
/// `{name}__cwt_coefficients__coeff_{coeff}__w_{w}__widths_{widths}`, with
/// the widths joined by `_` (tsfresh: `widths_(2, 5, 10, 20)`), e.g.
/// `val__cwt_coefficients__coeff_0__w_2__widths_2_5_10_20`.
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - `coeff` $\geq n$ gives NaN.
/// - NaN and $\pm\infty$ propagate to the coefficients whose filter window
///   covers them, as in pywt.
///
/// # Cost
/// $O(n \cdot \sum a)$ per distinct `widths` (the filter of width $a$ has
/// $16a + 1$ taps).
///
/// # tsfresh
/// `feature_calculators.cwt_coefficients` (v0.21.2), which calls
/// `pywt.cwt(x, widths, "mexh")`.
pub struct CwtCoefficients {
    /// `(widths, coeff, w)` per config entry.
    pub params: Vec<(Vec<usize>, usize, usize)>,
}

impl Kernel for CwtCoefficients {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(widths, coeff, w)| {
                let widths = widths.iter().map(|w| w.to_string()).collect::<Vec<_>>();
                format!(
                    "cwt_coefficients__coeff_{coeff}__w_{w}__widths_{}",
                    widths.join("_")
                )
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let mut transforms: HashMap<&[usize], Vec<Vec<f64>>> = HashMap::new();
        for (widths, coeff, w) in &self.params {
            let cwt = transforms
                .entry(widths.as_slice())
                .or_insert_with(|| pywt_cwt_mexh(x, widths));
            // Validated when the kernel is built: w is one of the widths
            let row = widths.iter().position(|v| v == w).unwrap_or(0);
            out.push(if *coeff < x.len() {
                cwt[row][*coeff]
            } else {
                f64::NAN
            });
        }
    }
}

/// tsfresh's `_ricker(points, a)` (scipy's former `ricker`).
fn ricker(points: usize, a: usize) -> Vec<f64> {
    let a = a as f64;
    let amp = 2.0 / ((3.0 * a).sqrt() * PI.powf(0.25));
    let wsq = a * a;
    let half = (points as f64 - 1.0) / 2.0;
    (0..points)
        .map(|i| {
            let v = i as f64 - half;
            let xsq = v * v;
            let modulation = 1.0 - xsq / wsq;
            let gauss = (-xsq / (2.0 * wsq)).exp();
            amp * modulation * gauss
        })
        .collect()
}

/// `scipy.stats.scoreatpercentile(values, per)` (`interpolation_method =
/// "fraction"`); NaN sorts last, as in `np.sort`. Selects the two order
/// statistics it needs instead of sorting; `buf` is scratch space.
fn score_at_percentile(values: &[f64], per: f64, buf: &mut Vec<f64>) -> f64 {
    let cmp = |a: &f64, b: &f64| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap(),
    };
    buf.clear();
    buf.extend_from_slice(values);
    let idx = per / 100.0 * (buf.len() - 1) as f64;
    let i = idx as usize;
    let (_, &mut lo, above) = buf.select_nth_unstable_by(i, cmp);
    if i as f64 == idx {
        lo * 1.0 / 1.0
    } else {
        // The next order statistic is the smallest value above position i
        let hi = *above.iter().min_by(|a, b| cmp(a, b)).unwrap();
        let (w0, w1) = ((i + 1) as f64 - idx, idx - i as f64);
        (lo * w0 + hi * w1) / (w0 + w1)
    }
}

/// One ridge line of `_identify_ridge_lines`: its rows and columns in the
/// order found (rows descending) and the number of rows since it last grew.
struct RidgeLine {
    rows: Vec<usize>,
    cols: Vec<usize>,
    gap: usize,
}

/// The number of peaks `scipy.signal.find_peaks_cwt(x, widths=1..=n,
/// wavelet=_ricker)` finds, with all its defaults.
fn count_cwt_peaks(x: &[f64], n: usize) -> usize {
    let len = x.len();
    let widths: Vec<usize> = (1..=n).collect();
    // scipy's _cwt: convolve(x, ricker(min(10 w, len), w)[::-1], mode="same"),
    // direct method (see NumberCwtPeaks' docs)
    let cwt: Vec<Vec<f64>> = widths
        .iter()
        .map(|&w| {
            let mut wavelet = ricker((10 * w).min(len), w);
            wavelet.reverse();
            np_convolve(x, &wavelet, false)
        })
        .collect();
    // argrelmax along each row, mode "clip": the ends are never maxima
    let is_max = |row: &[f64], i: usize| -> bool {
        let prev = row[i.saturating_sub(1)];
        let next = row[(i + 1).min(len - 1)];
        row[i] > prev && row[i] > next
    };
    let max_cols: Vec<Vec<usize>> = cwt
        .iter()
        .map(|row| (0..len).filter(|&i| is_max(row, i)).collect())
        .collect();

    // _identify_ridge_lines(cwt, widths / 4, gap_thresh = ceil(widths[0]))
    let Some(start_row) = (0..n).rev().find(|&r| !max_cols[r].is_empty()) else {
        return 0;
    };
    let gap_thresh = 1; // ceil(widths[0])
    let mut ridge_lines: Vec<RidgeLine> = max_cols[start_row]
        .iter()
        .map(|&col| RidgeLine {
            rows: vec![start_row],
            cols: vec![col],
            gap: 0,
        })
        .collect();
    let mut final_lines = Vec::new();
    for row in (0..start_row).rev() {
        let max_distance = widths[row] as f64 / 4.0;
        for line in ridge_lines.iter_mut() {
            line.gap += 1;
        }
        // The lines' last columns at the start of the row, sorted, keeping
        // the first line at each column (np.argmin picks the first closest)
        let mut ends: Vec<(usize, usize)> = ridge_lines
            .iter()
            .enumerate()
            .map(|(i, l)| (*l.cols.last().unwrap(), i))
            .collect();
        ends.sort_unstable();
        ends.dedup_by_key(|e| e.0);
        for &col in &max_cols[row] {
            // The closest line is at the nearest end column on either side;
            // on a tie in distance, the first line wins
            let p = ends.partition_point(|&(c, _)| c < col);
            let left = p.checked_sub(1).map(|j| ends[j]);
            let right = ends.get(p).copied();
            let closest = [left, right]
                .into_iter()
                .flatten()
                .min_by_key(|&(c, i)| (c.abs_diff(col), i))
                .filter(|&(c, _)| c.abs_diff(col) as f64 <= max_distance)
                .map(|(_, i)| i);
            match closest {
                Some(i) => {
                    let line = &mut ridge_lines[i];
                    line.cols.push(col);
                    line.rows.push(row);
                    line.gap = 0;
                }
                None => ridge_lines.push(RidgeLine {
                    rows: vec![row],
                    cols: vec![col],
                    gap: 0,
                }),
            }
        }
        // End lines that skipped too many rows, keeping the others in order
        let (ended, alive): (Vec<_>, Vec<_>) = std::mem::take(&mut ridge_lines)
            .into_iter()
            .partition(|line| line.gap > gap_thresh);
        final_lines.extend(ended);
        ridge_lines = alive;
    }

    // _filter_ridge_lines(cwt, lines, min_length = ceil(n / 4),
    // window_size = ceil(len / 20), min_snr = 1, noise_perc = 10)
    let min_length = (n as f64 / 4.0).ceil();
    let window_size = (len as f64 / 20.0).ceil() as usize;
    let (hf_window, odd) = (window_size / 2, window_size % 2);
    let row_one = &cwt[0];
    let mut buf = Vec::new();
    let mut noise = |ind: usize| {
        let start = ind.saturating_sub(hf_window);
        let end = (ind + hf_window + odd).min(len);
        score_at_percentile(&row_one[start..end], 10.0, &mut buf)
    };
    final_lines
        .iter()
        .chain(ridge_lines.iter())
        .filter(|line| {
            if (line.rows.len() as f64) < min_length {
                return false;
            }
            // scipy reorders each line's points by row; the first point is
            // then the last one found (the line's lowest row)
            let (row, col) = (*line.rows.last().unwrap(), *line.cols.last().unwrap());
            let snr = (cwt[row][col] / noise(col)).abs();
            // NaN compares False, so a NaN SNR keeps the line
            snr >= 1.0 || snr.is_nan()
        })
        .count()
}

/// Number of CWT peaks feature.
///
/// The number of peaks of the series that are visible in its continuous
/// wavelet transform with the Ricker wavelet at widths $1, \dots, n$ and
/// have a high enough signal-to-noise ratio, as found by
/// `scipy.signal.find_peaks_cwt` with its defaults: local maxima of each
/// width's transform are joined into ridge lines across widths (from the
/// largest width with a maximum down to width 1, joining a maximum to the
/// closest line within $w / 4$ columns and ending lines that skip more than
/// one width); a line counts if it spans at least $\lceil n / 4 \rceil$
/// widths and its width-1 coefficient is at least the 10th percentile of
/// the width-1 coefficients in a window of $\lceil \text{len} / 20 \rceil$
/// around it (in absolute ratio).
///
/// # Parameters
/// - `n`: the largest width. Config:
///   `[number_cwt_peaks] parameters = [{ n = 1 }, ...]`; one column per entry.
///
/// # Output column
/// `{name}__number_cwt_peaks__n_{n}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series with fewer than 3 values gives 0 (no interior maxima).
/// - NaN and $\pm\infty$ spread to the transform coefficients whose window
///   covers them; NaN coefficients are never maxima, and a NaN
///   signal-to-noise ratio keeps a line, as in scipy.
/// - scipy picks direct or FFT convolution by an operation-count heuristic.
///   tsfx always convolves directly, which is what scipy picks unless the
///   wavelet is very long (widths of roughly 30 and more on long series);
///   there, scipy's FFT rounding noise (and its spreading of NaN or
///   $\pm\infty$ over the whole row) can change the count.
/// - **Deliberate deviation from tsfresh:** where two neighbouring
///   coefficients are tied in real arithmetic, rounding decides which is the
///   maximum. tsfx uses one portable convolution and the standard `exp`, so
///   its counts are the same on every platform, while numpy's rounding (and
///   so tsfresh's count) depends on its BLAS and CPU. Ties are common for
///   smooth series symmetric about a point between two samples: a Gaussian
///   bump centred on a sample counts 2 peaks with `n = 5`, tsfresh 1.
///   Irregular series are not affected.
///
/// # Cost
/// $O(\text{len} \cdot n^2)$ for the transform and $O(n \cdot \text{len}
/// \log \text{len})$ to join ridge lines, plus a selection over a window of
/// $\lceil \text{len} / 20 \rceil$ for each ridge line's noise: up to
/// $O(\text{len}^2)$ in all, which dominates on long series.
///
/// # tsfresh
/// `feature_calculators.number_cwt_peaks` (v0.21.2).
pub struct NumberCwtPeaks {
    pub ns: Vec<usize>,
}

impl Kernel for NumberCwtPeaks {
    fn names(&self) -> Vec<String> {
        self.ns
            .iter()
            .map(|n| format!("number_cwt_peaks__n_{n}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        out.extend(self.ns.iter().map(|&n| count_cwt_peaks(x, n) as f64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_np_convolve_same_length() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = [1.0, 0.0, -1.0];
        assert_eq!(
            np_convolve(&a, &v, true),
            vec![1.0, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0]
        );
        assert_eq!(np_convolve(&a, &v, false), vec![2.0, 2.0, 2.0, 2.0, -4.0]);
        // longer kernel: inputs are swapped
        assert_eq!(np_convolve(&v, &a, true), np_convolve(&a, &v, true));
    }

    #[test]
    fn test_cwt_peaks_of_bump() {
        // Off-centre, so no two neighbouring coefficients are tied
        let x: Vec<f64> = (0..50)
            .map(|i| (-((i as f64 - 25.3).powi(2)) / 20.0).exp())
            .collect();
        assert_eq!(count_cwt_peaks(&x, 1), 1);
        assert_eq!(count_cwt_peaks(&x, 5), 1);
        assert_eq!(count_cwt_peaks(&[1.0, 2.0], 5), 0);
    }
}
