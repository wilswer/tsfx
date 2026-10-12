//! Frequency-domain features: the discrete Fourier transform and Welch's
//! power spectral density.

use std::cell::RefCell;
use std::f64::consts::PI;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::kernel::{GroupContext, Kernel};
use crate::utils::stats::{numpy_mean, numpy_sum};
use crate::utils::toml_reader::{FftAggType, FftAttr};

thread_local! {
    /// FFT plans are cached per length; one planner per worker thread.
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// The full complex DFT of a real sequence (unnormalised, forward).
fn dft(x: &[f64]) -> Vec<Complex64> {
    let mut buffer: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    PLANNER.with(|planner| {
        planner
            .borrow_mut()
            .plan_fft_forward(buffer.len())
            .process(&mut buffer);
    });
    buffer
}

/// The one-sided DFT of a real series, as `np.fft.rfft`: coefficients
/// $0, \dots, \lfloor n/2 \rfloor$. The imaginary parts of the DC (and, for
/// even $n$, Nyquist) coefficients are exactly 0, as in numpy.
///
/// If any value is NaN or $\pm\infty$, every coefficient is NaN (see the
/// features' docs for why this deviates from numpy).
pub(crate) fn rfft(x: &[f64]) -> Vec<Complex64> {
    let n_out = x.len() / 2 + 1;
    if x.iter().any(|v| !v.is_finite()) {
        return vec![Complex64::new(f64::NAN, f64::NAN); n_out];
    }
    let mut spectrum = dft(x);
    spectrum.truncate(n_out);
    spectrum[0].im = 0.0;
    if x.len().is_multiple_of(2) {
        spectrum[n_out - 1].im = 0.0;
    }
    spectrum
}

/// The periodic Hann window of length `m`, as
/// `scipy.signal.get_window("hann", m)`.
fn hann_periodic(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    // scipy: the symmetric window of length m + 1, last point dropped;
    // fac = np.linspace(-pi, pi, m + 1), w = 0.5 + 0.5 * cos(fac)
    let step = 2.0 * PI / m as f64;
    (0..m)
        .map(|i| 0.5 + 0.5 * (i as f64 * step - PI).cos())
        .collect()
}

/// Longest Welch segment tsfresh uses.
const MAX_SEGMENT_LENGTH: usize = 256;

/// Welch's power spectral density estimate, as tsfresh calls
/// `scipy.signal.welch(x, nperseg=min(len(x), 256))`: periodic Hann window,
/// 50% overlap (`noverlap = nperseg // 2`), each segment's mean removed,
/// sampling frequency 1, density scaling, one-sided (bins other than DC and
/// Nyquist doubled), averaged over segments.
///
/// NaN or $\pm\infty$ anywhere gives NaN in every bin, as scipy.
pub(crate) fn welch_psd(x: &[f64]) -> Vec<f64> {
    let len = x.len().min(MAX_SEGMENT_LENGTH);
    let n_freqs = len / 2 + 1;
    if x.iter().any(|v| !v.is_finite()) {
        return vec![f64::NAN; n_freqs];
    }
    let window = hann_periodic(len);
    let scale = 1.0 / numpy_sum(len, |i| window[i] * window[i]);
    let step = len - len / 2;
    let n_segments = (x.len() - len) / step + 1;
    let mut psd = vec![0.0; n_freqs];
    for s in 0..n_segments {
        let segment = &x[s * step..s * step + len];
        // detrend="constant": subtract the segment's np.mean
        let mean = numpy_mean(segment);
        let windowed: Vec<f64> = segment
            .iter()
            .zip(&window)
            .map(|(v, w)| (v - mean) * w)
            .collect();
        let spectrum = dft(&windowed);
        for (k, p) in psd.iter_mut().enumerate() {
            let mut power = spectrum[k].norm_sqr() * scale;
            // one-sided: double everything but DC and (even length) Nyquist
            if k > 0 && !(len.is_multiple_of(2) && k == n_freqs - 1) {
                power *= 2.0;
            }
            *p += power;
        }
    }
    for p in psd.iter_mut() {
        *p /= n_segments as f64;
    }
    psd
}

/// FFT coefficient feature.
///
/// One attribute of coefficient $k$ of the one-sided discrete Fourier
/// transform of the series,
/// $$ A_k = \sum_{m=0}^{n-1} x_m \exp\left(-2\pi i \frac{mk}{n}\right), \quad k = 0, \dots, \lfloor n/2 \rfloor: $$
/// its real part, imaginary part, absolute value, or angle in degrees
/// (in $(-180, 180]$, as `np.angle(..., deg=True)`).
///
/// # Parameters
/// - `attr`: `real`, `imag`, `abs` or `angle`; any other name is rejected
///   when the config is loaded.
/// - `coeff`: the coefficient $k \geq 0$.
///
/// Config: `[fft_coefficient] parameters = [{ attr = "real", coeff = 0 }, ...]`;
/// one column per entry.
///
/// # Output column
/// `{name}__fft_coefficient__attr_{attr}__coeff_{coeff}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - $k > \lfloor n/2 \rfloor$ gives NaN.
/// - **Deliberate deviation from tsfresh:** NaN or $\pm\infty$ anywhere in
///   the series gives NaN for every attribute. numpy's real FFT keeps some
///   parts finite (for example the imaginary part of $A_1$ of
///   `[nan, 1, 2, 3]` is 2) depending on the order of its butterflies.
/// - Coefficients equal to 0 in exact arithmetic come out as tiny rounding
///   noise, whose angle is arbitrary (and differs from numpy's).
///
/// # Cost
/// One FFT per group, shared with [`FftAggregated`].
///
/// # tsfresh
/// `feature_calculators.fft_coefficient` (v0.21.2).
pub struct FftCoefficient {
    /// `(attr, coeff)` per config entry.
    pub params: Vec<(FftAttr, usize)>,
}

impl Kernel for FftCoefficient {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(attr, coeff)| format!("fft_coefficient__attr_{attr}__coeff_{coeff}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let spectrum = ctx.rfft();
        for (attr, coeff) in &self.params {
            out.push(match spectrum.get(*coeff) {
                None => f64::NAN,
                Some(c) => match attr {
                    FftAttr::Real => c.re,
                    FftAttr::Imag => c.im,
                    FftAttr::Abs => c.re.hypot(c.im),
                    FftAttr::Angle => c.im.atan2(c.re).to_degrees(),
                },
            });
        }
    }
}

/// Aggregated FFT feature.
///
/// Treats the absolute values $y_k = |A_k|$ of the one-sided Fourier
/// transform (see [`FftCoefficient`]) as a distribution over the index $k$,
/// with moments $\mu_j = \sum_k k^j y_k / \sum_k y_k$, and returns its
/// - `centroid`: $c = \mu_1$,
/// - `variance`: $v = \mu_2 - c^2$,
/// - `skew`: $(\mu_3 - 3cv - c^3) / v^{3/2}$,
/// - `kurtosis`: $(\mu_4 - 4c\mu_3 + 6\mu_2c^2 - 3c) / v^2$ (tsfresh's
///   formula, kept as is).
///
/// # Parameters
/// - `aggtype`: `centroid`, `variance`, `skew` or `kurtosis`; any other name
///   is rejected when the config is loaded. Config:
///   `[fft_aggregated] parameters = [{ aggtype = "centroid" }, ...]`; one
///   column per entry.
///
/// # Output column
/// `{name}__fft_aggregated__aggtype_{aggtype}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A series of zeros gives NaN (empty distribution).
/// - `skew` and `kurtosis` are NaN when $v < 0.5$ (the spectrum is too
///   concentrated for them to mean anything), as tsfresh.
/// - NaN or $\pm\infty$ in the series gives NaN.
///
/// # tsfresh
/// `feature_calculators.fft_aggregated` (v0.21.2).
pub struct FftAggregated {
    pub aggtypes: Vec<FftAggType>,
}

impl Kernel for FftAggregated {
    fn names(&self) -> Vec<String> {
        self.aggtypes
            .iter()
            .map(|aggtype| format!("fft_aggregated__aggtype_{aggtype}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let y: Vec<f64> = ctx.rfft().iter().map(|c| c.re.hypot(c.im)).collect();
        let total = numpy_sum(y.len(), |k| y[k]);
        // y.dot(np.arange(len(y)) ** j) / y.sum()
        let moment = |j: i32| {
            y.iter()
                .enumerate()
                .map(|(k, v)| v * (k as f64).powi(j))
                .sum::<f64>()
                / total
        };
        let centroid = moment(1);
        let variance = moment(2) - centroid.powf(2.0);
        for aggtype in &self.aggtypes {
            out.push(match aggtype {
                FftAggType::Centroid => centroid,
                FftAggType::Variance => variance,
                FftAggType::Skew if variance < 0.5 => f64::NAN,
                FftAggType::Skew => {
                    (moment(3) - 3.0 * centroid * variance - centroid.powf(3.0))
                        / variance.powf(1.5)
                }
                FftAggType::Kurtosis if variance < 0.5 => f64::NAN,
                FftAggType::Kurtosis => {
                    (moment(4) - 4.0 * centroid * moment(3) + 6.0 * moment(2) * centroid.powf(2.0)
                        - 3.0 * centroid)
                        / variance.powf(2.0)
                }
            });
        }
    }
}

/// Welch spectral density feature.
///
/// Bin `coeff` of the power spectral density of the series estimated with
/// Welch's method: overlapping segments of $\min(n, 256)$ values (50%
/// overlap), each with its mean removed and a periodic Hann window applied,
/// one-sided density scaling with sampling frequency 1, averaged over the
/// segments; exactly the defaults of `scipy.signal.welch`.
///
/// # Parameters
/// - `coeff`: frequency bin. Config:
///   `[spkt_welch_density] parameters = [{ coeff = 2 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__spkt_welch_density__coeff_{coeff}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - Bins past the last frequency ($\lfloor \min(n, 256) / 2 \rfloor$) give
///   NaN.
/// - NaN or $\pm\infty$ in the series gives NaN.
///
/// # Cost
/// One Welch estimate per group, shared with [`FourierEntropy`].
///
/// # tsfresh
/// `feature_calculators.spkt_welch_density` (v0.21.2).
pub struct SpktWelchDensity {
    pub coeffs: Vec<usize>,
}

impl Kernel for SpktWelchDensity {
    fn names(&self) -> Vec<String> {
        self.coeffs
            .iter()
            .map(|coeff| format!("spkt_welch_density__coeff_{coeff}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let psd = ctx.welch_psd();
        for &coeff in &self.coeffs {
            out.push(psd.get(coeff).copied().unwrap_or(f64::NAN));
        }
    }
}

/// Fourier entropy feature.
///
/// The binned entropy of the Welch power spectral density (see
/// [`SpktWelchDensity`]) scaled to a maximum of 1: the values $p_j / \max p$
/// are put into `bins` equal-width bins between their minimum and maximum
/// (as `np.histogram`), and with $q_k$ the share of values in bin $k$,
/// $$ -\sum_{k:\, q_k > 0} q_k \ln q_k. $$
///
/// # Parameters
/// - `bins`: number of bins. Config:
///   `[fourier_entropy] parameters = [{ bins = 2 }, ...]`; one column per
///   entry.
///
/// # Output column
/// `{name}__fourier_entropy__bins_{bins}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - A spectrum without power (for example a constant series) gives NaN.
/// - NaN or $\pm\infty$ in the series gives NaN.
/// - When the scaled spectrum spans too small a range for `bins` distinct
///   bin edges, tsfresh raises (`np.histogram`'s "Too many bins"); TSFX gives
///   NaN.
///
/// # tsfresh
/// `feature_calculators.fourier_entropy` (v0.21.2).
pub struct FourierEntropy {
    pub bins: Vec<usize>,
}

impl Kernel for FourierEntropy {
    fn names(&self) -> Vec<String> {
        self.bins
            .iter()
            .map(|bins| format!("fourier_entropy__bins_{bins}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let psd = ctx.welch_psd();
        // np.max: NaN if any value is NaN
        let max = psd
            .iter()
            .copied()
            .reduce(|a, b| if a.is_nan() || b > a { b } else { a })
            .unwrap_or(f64::NAN);
        let scaled: Vec<f64> = psd.iter().map(|p| p / max).collect();
        for &bins in &self.bins {
            out.push(binned_entropy(&scaled, bins));
        }
    }
}

/// tsfresh's `binned_entropy`: the entropy of an `np.histogram` of the
/// values into `max_bins` equal-width bins; NaN if any value is NaN (or the
/// histogram cannot be built).
///
/// Stand-in for the shared `binned_entropy` feature helper, to be replaced by
/// it.
fn binned_entropy(x: &[f64], max_bins: usize) -> f64 {
    if x.is_empty() || x.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let Some(hist) = numpy_histogram(x, max_bins) else {
        return f64::NAN;
    };
    let n = x.len() as f64;
    // probs[probs == 0] = 1.0; -np.sum(probs * np.log(probs))
    let probs: Vec<f64> = hist
        .iter()
        .map(|&count| if count == 0 { 1.0 } else { count as f64 / n })
        .collect();
    -numpy_sum(probs.len(), |k| probs[k] * probs[k].ln())
}

/// Counts of `np.histogram(x, bins)` for finite, non-empty `x`, with numpy's
/// bin edges and index computation; `None` where numpy raises (an infinite
/// range, or too many bins for the range).
fn numpy_histogram(x: &[f64], bins: usize) -> Option<Vec<usize>> {
    let (mut first, mut last) = super::kernel::extremes(x.iter().copied());
    if !first.is_finite() || !last.is_finite() || bins == 0 {
        return None;
    }
    if first == last {
        first -= 0.5;
        last += 0.5;
    }
    // np.linspace(first, last, bins + 1)
    let delta = last - first;
    let step = delta / bins as f64;
    let mut edges: Vec<f64> = (0..=bins)
        .map(|i| {
            if step == 0.0 {
                i as f64 / bins as f64 * delta + first
            } else {
                i as f64 * step + first
            }
        })
        .collect();
    edges[bins] = last;
    if edges.windows(2).any(|w| w[0] >= w[1]) {
        return None;
    }
    let mut counts = vec![0; bins];
    for &v in x {
        let mut index = (((v - first) / delta) * bins as f64) as usize;
        if index == bins {
            index -= 1;
        }
        if v < edges[index] {
            index -= 1;
        }
        if index != bins - 1 && v >= edges[index + 1] {
            index += 1;
        }
        counts[index] += 1;
    }
    Some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfft_matches_numpy_doc_example() {
        // np.fft.rfft([0, 1, 0, 0]) == [1, -1j, -1]
        let spectrum = rfft(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(spectrum.len(), 3);
        let expected = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)];
        for (c, (re, im)) in spectrum.iter().zip(expected) {
            assert!((c.re - re).abs() < 1e-12 && (c.im - im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hann_periodic() {
        assert_eq!(hann_periodic(1), vec![1.0]);
        let w = hann_periodic(4);
        for (a, b) in w.iter().zip([0.0, 0.5, 1.0, 0.5]) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_welch_single_value() {
        assert_eq!(welch_psd(&[5.0]), vec![0.0]);
    }

    #[test]
    fn test_histogram_edges() {
        // last bin includes the right edge
        assert_eq!(numpy_histogram(&[0.0, 0.5, 1.0], 2), Some(vec![1, 2]));
        // a constant series is centred in a unit range
        assert_eq!(numpy_histogram(&[3.0, 3.0], 3), Some(vec![0, 2, 0]));
    }
}
