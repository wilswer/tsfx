//! Continuous wavelet transforms with the Ricker ("Mexican hat") wavelet.
//!
//! Both features convolve the series with sampled wavelets. The convolution
//! reproduces `np.convolve` as numpy (with OpenBLAS) evaluates it on x86-64,
//! summation order and fused multiply-adds included, and the wavelets are
//! sampled with numpy's and pywt's exact `exp` / `pow` results, so that
//! tsfresh's comparisons between neighbouring coefficients (which decide
//! [`NumberCwtPeaks`], and which ties in real arithmetic leave to rounding)
//! come out the same. On that platform both features equal tsfresh's bit
//! for bit; with another BLAS or CPU, numpy's own rounding differs, and so
//! may tied comparisons.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::kernel::{GroupContext, Kernel};

/// Dot product as OpenBLAS's x86-64 `ddot` kernels (Haswell / SkylakeX)
/// evaluate it: blocks of 32 values in four 8-wide FMA accumulators (each
/// folded to 4 wide afterwards), then blocks of 16 in four 4-wide ones,
/// a fixed reduction tree, and the tail with scalar FMAs. numpy adds the
/// result to `0.0`.
#[inline(always)]
fn ddot_impl(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let n1 = n & !15;
    let mut i = 0;
    let mut dot = 0.0;
    if n1 > 0 {
        let mut wide = [[0.0_f64; 8]; 4];
        let n32 = n1 & !31;
        while i < n32 {
            for (r, acc) in wide.iter_mut().enumerate() {
                for (l, a) in acc.iter_mut().enumerate() {
                    let k = i + 8 * r + l;
                    *a = x[k].mul_add(y[k], *a);
                }
            }
            i += 32;
        }
        let mut acc: [[f64; 4]; 4] =
            std::array::from_fn(|r| std::array::from_fn(|l| wide[r][l] + wide[r][l + 4]));
        while i < n1 {
            for (r, a4) in acc.iter_mut().enumerate() {
                for (l, a) in a4.iter_mut().enumerate() {
                    let k = i + 4 * r + l;
                    *a = x[k].mul_add(y[k], *a);
                }
            }
            i += 16;
        }
        let a0: [f64; 4] =
            std::array::from_fn(|l| ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l]);
        dot = (a0[0] + a0[2]) + (a0[1] + a0[3]);
    }
    for k in i..n {
        dot = y[k].mul_add(x[k], dot);
    }
    0.0 + dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
fn ddot_fma(x: &[f64], y: &[f64]) -> f64 {
    ddot_impl(x, y)
}

/// See [`ddot_impl`]; uses hardware FMA when available (same results,
/// `mul_add` is exact either way).
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports FMA
        return unsafe { ddot_fma(x, y) };
    }
    ddot_impl(x, y)
}

/// `np.convolve(a, v, mode)` for `mode` `"full"` or `"same"`.
///
/// numpy swaps the inputs so that `a` is the longer one, then correlates
/// `a` with the reversed `v`: edge outputs (and all outputs for kernels
/// longer than 11 values) with BLAS `ddot`, the rest with a plain
/// sequential loop (numpy's `small_correlate`).
fn np_convolve(a: &[f64], v: &[f64], full: bool) -> Vec<f64> {
    let (a, v) = if v.len() > a.len() { (v, a) } else { (a, v) };
    let k: Vec<f64> = v.iter().rev().copied().collect();
    let (n1, n2) = (a.len(), k.len());
    let (n_left, n_right) = if full {
        (n2 - 1, n2 - 1)
    } else {
        (n2 / 2, n2 - n2 / 2 - 1)
    };
    let mut out = Vec::with_capacity(n1 + n_left + n_right + 1 - n2);
    let mut ip2 = n_left;
    let mut n = n2 - n_left;
    let mut p1 = 0;
    for _ in 0..n_left {
        out.push(ddot(&a[p1..p1 + n], &k[ip2..ip2 + n]));
        n += 1;
        ip2 -= 1;
    }
    if n2 <= 11 {
        for i in 0..=n1 - n2 {
            let mut r = 0.0;
            for (j, kj) in k.iter().enumerate() {
                r += a[i + j] * kj;
            }
            out.push(r);
        }
    } else {
        for i in 0..=n1 - n2 {
            out.push(ddot(&a[i..i + n], &k));
        }
    }
    p1 += n1 - n2 + 1;
    for _ in 0..n_right {
        n -= 1;
        out.push(ddot(&a[p1..p1 + n], &k[ip2..ip2 + n]));
        p1 += 1;
    }
    out
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
    // Evaluated as pywt's C code does (bitwise equal to its table), with
    // t^2 from the C library's pow, which is not always t * t; black_box
    // keeps the compiler from turning pow(t, 2) into t * t
    let norm = 3.0_f64.sqrt() * PI.sqrt().sqrt();
    let two = std::hint::black_box(2.0);
    let dx = grid(1) - grid(0);
    let mut acc = 0.0;
    let table = (0..PSI_POINTS)
        .map(|i| {
            let t = grid(i);
            let t2 = t.powf(two);
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
            let gauss = np_exp(-xsq / (2.0 * wsq));
            amp * modulation * gauss
        })
        .collect()
}

/// `np.exp` of one value as numpy evaluates arrays: on x86-64 CPUs with
/// AVX-512 (Skylake-X and later) numpy calls Intel SVML's `__svml_exp8_ha`,
/// which this reproduces bit for bit (its 16-entry table and polynomial,
/// from numpy's vendored SVML); elsewhere numpy uses the C library's `exp`,
/// as does this.
///
/// The Ricker wavelet's exact bits decide which of two tied neighbouring
/// coefficients wins (for example on a series symmetric around a point
/// between two samples), so they must match numpy's.
fn np_exp(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static SVML: OnceLock<bool> = OnceLock::new();
        let svml = *SVML.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512cd")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512dq")
                && std::arch::is_x86_feature_detected!("avx512vl")
        });
        // Beyond SVML's threshold it calls out to a scalar routine
        if svml && x.abs() < 707.0 {
            return svml_exp_ha(x);
        }
    }
    x.exp()
}

/// `__svml_exp8_ha` (one lane) for $|x| < 707$: $x = (M + j/16 + ...)\ln 2$
/// with $N = M + j/16$ rounded towards $-\infty$ to 1/16, then
/// $e^x = 2^M\, T_h[j]\,(1 + p)$ with
/// $p = T_l[j] + r\,(c_1 + c_2 r + \dots + c_6 r^5)$, $r = x - N \ln 2$.
fn svml_exp_ha(x: f64) -> f64 {
    const TH: [u64; 16] = [
        0x3ff0000000000000,
        0x3ff0b5586cf9890f,
        0x3ff172b83c7d517b,
        0x3ff2387a6e756238,
        0x3ff306fe0a31b715,
        0x3ff3dea64c123422,
        0x3ff4bfdad5362a27,
        0x3ff5ab07dd485429,
        0x3ff6a09e667f3bcd,
        0x3ff7a11473eb0187,
        0x3ff8ace5422aa0db,
        0x3ff9c49182a3f090,
        0x3ffae89f995ad3ad,
        0x3ffc199bdd85529c,
        0x3ffd5818dcfba487,
        0x3ffea4afa2a490da,
    ];
    const TLR: [u64; 16] = [
        0x0000000000000000,
        0x3c979aa65d837b6d,
        0xbc801b15eaa59348,
        0x3c968efde3a8a894,
        0x3c834d754db0abb6,
        0x3c859f48a72a4c6d,
        0x3c7690cebb7aafb0,
        0x3c9063e1e21c5409,
        0xbc93b3efbf5e2228,
        0xbc7b32dcb94da51d,
        0x3c8db72fc1f0eab4,
        0x3c71affc2b91ce27,
        0x3c8c1a7792cb3387,
        0x3c736eae30af0cb3,
        0x3c74a385a63d07a7,
        0xbc8ff7128fd391f0,
    ];
    let c = f64::from_bits;
    let log2e = c(0x3ff71547652b82fe);
    let (ln2_hi, ln2_lo) = (c(0x3fe62e42fefa39ef), c(0x3c7abc9e3b39803f));
    let [c6, c5, c4, c3, c2, c1] = [
        0x3f57411836940c04,
        0x3f81101cbbc265c0,
        0x3fa55557242d68fe,
        0x3fc5555553939732,
        0x3fe000000000d008,
        0x3fefffffffffff70,
    ]
    .map(c);
    // floor(16 x log2(e)) of the exact product (SVML: a round-to-zero FMA
    // onto a large positive shifter)
    let hi = x * log2e;
    let lo = x.mul_add(log2e, -hi);
    let mut f = (hi * 16.0).floor();
    if hi * 16.0 == f && lo < 0.0 {
        f -= 1.0;
    }
    let f = f as i64;
    let n = f as f64 / 16.0;
    let j = (f & 15) as usize;
    let m = f >> 4;
    let r = (-n).mul_add(ln2_hi, x);
    let r = (-ln2_lo).mul_add(n, r);
    // SVML masks the exponent so that |r| < 2 even for special inputs
    let r = c(r.to_bits() & 0xbfffffffffffffff);
    let r2 = r * r;
    let p56 = c6.mul_add(r, c5);
    let p34 = c4.mul_add(r, c3);
    let p12 = c2.mul_add(r, c1);
    let p = r2.mul_add(p56, p34);
    let p = r2.mul_add(p, p12);
    let p = p.mul_add(r, c(TLR[j]));
    let th = c(TH[j]);
    let res = th.mul_add(p, th);
    // vscalefpd: res * 2^m, exact for normal results
    res * 2.0_f64.powi(m as i32)
}

/// `scipy.stats.scoreatpercentile(values, per)` (`interpolation_method =
/// "fraction"`); NaN sorts last, as in `np.sort`.
fn score_at_percentile(values: &[f64], per: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap(),
    });
    let idx = per / 100.0 * (sorted.len() - 1) as f64;
    let i = idx as usize;
    if i as f64 == idx {
        sorted[i] * 1.0 / 1.0
    } else {
        let (w0, w1) = ((i + 1) as f64 - idx, idx - i as f64);
        (sorted[i] * w0 + sorted[i + 1] * w1) / (w0 + w1)
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
        let prev_cols: Vec<usize> = ridge_lines
            .iter()
            .map(|l| *l.cols.last().unwrap())
            .collect();
        for &col in &max_cols[row] {
            // np.argmin: the first closest line
            let closest = prev_cols
                .iter()
                .enumerate()
                .min_by_key(|&(i, &c)| (c.abs_diff(col), i))
                .filter(|&(_, &c)| c.abs_diff(col) as f64 <= max_distance)
                .map(|(i, _)| i);
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
        for i in (0..ridge_lines.len()).rev() {
            if ridge_lines[i].gap > gap_thresh {
                final_lines.push(ridge_lines.remove(i));
            }
        }
    }

    // _filter_ridge_lines(cwt, lines, min_length = ceil(n / 4),
    // window_size = ceil(len / 20), min_snr = 1, noise_perc = 10)
    let min_length = (n as f64 / 4.0).ceil();
    let window_size = (len as f64 / 20.0).ceil() as usize;
    let (hf_window, odd) = (window_size / 2, window_size % 2);
    let row_one = &cwt[0];
    let noise = |ind: usize| {
        let start = ind.saturating_sub(hf_window);
        let end = (ind + hf_window + odd).min(len);
        score_at_percentile(&row_one[start..end], 10.0)
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
///
/// # Cost
/// $O(\text{len} \cdot n^2)$.
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
        let x: Vec<f64> = (0..50)
            .map(|i| (-((i as f64 - 25.0).powi(2)) / 20.0).exp())
            .collect();
        assert_eq!(count_cwt_peaks(&x, 1), 1);
        assert_eq!(count_cwt_peaks(&x, 5), 1);
        assert_eq!(count_cwt_peaks(&[1.0, 2.0], 5), 0);
    }
}
