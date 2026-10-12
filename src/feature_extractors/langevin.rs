//! The deterministic dynamics of a Langevin model fitted to the series.

use nalgebra::{Complex, DMatrix};

use super::kernel::{GroupContext, Kernel};
use crate::utils::stats::{numpy_linspace, numpy_quantile, numpy_searchsorted_left};

/// Coefficients of the polynomial $h$ of degree `m` (highest power first,
/// as `np.polyfit`) fitted to the deterministic dynamics
/// $\dot{x}(t) = h(x(t)) + \mathcal{N}(0, R)$ of a Langevin model
/// (Friedrich et al., 2000), exactly as tsfresh's
/// `_estimate_friedrich_coefficients` computes them:
///
/// 1. pair each value $x_t$ (all but the last) with the change
///    $x_{t+1} - x_t$;
/// 2. bin the values into `r` quantile bins, as `pd.qcut(x[:-1], r)`;
/// 3. average values and changes per bin, as pandas' group-by mean;
/// 4. fit the polynomial through the bin averages by least squares, as
///    `np.polyfit(..., deg=m)`.
///
/// `None` where tsfresh returns NaN coefficients: duplicate quantile bin
/// edges (`pd.qcut` raises), or a fit that fails (`np.linalg.lstsq` raises
/// on non-finite input). Also `None` (tsfresh raises instead) for fewer than
/// 2 values, `m = 0`, `r = 0`, or when every bin average is NaN.
fn estimate_friedrich_coefficients(x: &[f64], m: usize, r: usize) -> Option<Vec<f64>> {
    if x.len() < 2 || m == 0 || r == 0 {
        return None;
    }
    let signal = &x[..x.len() - 1];
    let delta: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();

    // pd.qcut: quantile levels rounded up where not exact, then the edges
    // as np.quantile of the non-NaN values
    let levels = numpy_linspace(0.0, 1.0, r + 1)
        .into_iter()
        .enumerate()
        .map(|(i, q)| {
            if r as f64 * q != i as f64 {
                q.next_up().min(1.0)
            } else {
                q
            }
        });
    let mut sorted: Vec<f64> = signal.iter().copied().filter(|v| !v.is_nan()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_unstable_by(f64::total_cmp);
    let edges: Vec<f64> = levels.map(|q| numpy_quantile(&sorted, q)).collect();
    // "Bin edges must be unique" (pandas lets two equal edges pass)
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    if edges.len() != 2
        && (0..edges.len()).any(|i| (i + 1..edges.len()).any(|j| same(edges[i], edges[j])))
    {
        return None;
    }

    // pd.qcut labels each bin with an interval of rounded edges; building
    // them can fail, and bins with equal labels form one group
    let group_of_bin = interval_label_groups(&edges)?;
    let groups = group_of_bin.last().map_or(0, |&g| g + 1);

    // Bin of each value (right-closed, the lowest edge included), then the
    // group-by mean of values and changes per group, with pandas' Kahan sums
    let ids = numpy_searchsorted_left(&edges, signal);
    let mut sums = vec![[Kahan::default(), Kahan::default()]; groups];
    let mut observed = vec![false; groups];
    for (k, (&v, &id)) in signal.iter().zip(&ids).enumerate() {
        let id = if v == edges[0] { 1 } else { id };
        if v.is_nan() || id == 0 || id == edges.len() {
            continue;
        }
        let group = group_of_bin[id - 1];
        observed[group] = true;
        sums[group][0].add(v);
        sums[group][1].add(delta[k]);
    }
    let (xs, ys): (Vec<f64>, Vec<f64>) = sums
        .iter()
        .zip(&observed)
        .filter(|&(_, &observed)| observed)
        .map(|(sums, _)| (sums[0].mean(), sums[1].mean()))
        .filter(|(x, y)| !x.is_nan() && !y.is_nan())
        .unzip();
    if xs.is_empty() {
        return None;
    }
    polyfit(&xs, &ys, m)
}

/// The group of each quantile bin, as `pd.qcut` labels the bins: intervals
/// between the edges rounded by `_round_frac` (to the smallest precision
/// from 3 that keeps the edges distinct, else 3), the lowest edge lowered by
/// `10 ** -precision`. Bins with equal intervals share a group. `None` where
/// pandas fails to build the intervals (an edge that rounds to NaN, or the
/// intervals not ordered).
fn interval_label_groups(edges: &[f64]) -> Option<Vec<usize>> {
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    let distinct = |levels: &[f64]| {
        (0..levels.len()).all(|i| (i + 1..levels.len()).all(|j| !same(levels[i], levels[j])))
    };
    let round = |precision: i32| -> Vec<f64> {
        edges
            .iter()
            .map(|&edge| round_frac(edge, precision))
            .collect()
    };
    let precision = (3..20).find(|&p| distinct(&round(p))).unwrap_or(3);
    let mut breaks = round(precision);
    breaks[0] -= 10_f64.powf(-f64::from(precision));
    // IntervalIndex.from_breaks validation
    for w in breaks.windows(2) {
        if w[0].is_nan() != w[1].is_nan() || w[0] > w[1] {
            return None;
        }
    }
    let mut groups = Vec::with_capacity(breaks.len() - 1);
    for i in 0..breaks.len() - 1 {
        // Intervals are ordered, so equal ones are neighbours
        let group = match groups.last() {
            Some(&g) if same(breaks[i - 1], breaks[i]) && same(breaks[i], breaks[i + 1]) => g,
            Some(&g) => g + 1,
            None => 0,
        };
        groups.push(group);
    }
    Some(groups)
}

/// pandas' `_round_frac`: round to `precision` digits after the decimal
/// point, or for $|x| < 1$ to `precision` significant digits.
fn round_frac(x: f64, precision: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let digits = if x.trunc() == 0.0 {
        -(x.fract().abs().log10().floor() as i32) - 1 + precision
    } else {
        precision
    };
    numpy_around(x, digits)
}

/// `np.around(x, decimals)`: scale by a power of ten (computed by repeated
/// multiplication, as numpy), round half to even, scale back.
fn numpy_around(x: f64, decimals: i32) -> f64 {
    let power_of_ten = |n: i32| {
        const P10: [f64; 9] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8];
        if n < 9 {
            P10[n as usize]
        } else {
            (9..n).fold(1e9, |p, _| p * 10.0)
        }
    };
    if decimals >= 0 {
        let f = power_of_ten(decimals);
        (x * f).round_ties_even() / f
    } else {
        let f = power_of_ten(-decimals);
        (x / f).round_ties_even() * f
    }
}

/// pandas' group-by sum: Kahan summation skipping NaN, the compensation
/// reset when it turns NaN (an infinite value).
#[derive(Clone, Copy, Default)]
struct Kahan {
    sum: f64,
    compensation: f64,
    count: usize,
}

impl Kahan {
    fn add(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        self.count += 1;
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = t - self.sum - y;
        if self.compensation.is_nan() {
            self.compensation = 0.0;
        }
        self.sum = t;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

/// Least-squares polynomial of degree `deg` through `(x, y)`, highest power
/// first, as `np.polyfit`: Vandermonde columns scaled to unit norm, solved
/// by SVD with singular values up to `len(x) * eps` times the largest
/// treated as zero (the minimum-norm solution when underdetermined). `None`
/// for non-finite input, where `np.linalg.lstsq` raises.
fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Option<Vec<f64>> {
    let (rows, order) = (x.len(), deg + 1);
    // np.vander: powers by repeated multiplication, highest first
    let mut lhs = DMatrix::from_fn(rows, order, |_, _| 1.0);
    for i in 0..rows {
        for j in (0..deg).rev() {
            lhs[(i, j)] = lhs[(i, j + 1)] * x[i];
        }
    }
    let scale: Vec<f64> = (0..order)
        .map(|j| {
            (0..rows)
                .fold(0.0, |s, i| s + lhs[(i, j)] * lhs[(i, j)])
                .sqrt()
        })
        .collect();
    for j in 0..order {
        for i in 0..rows {
            lhs[(i, j)] /= scale[j];
        }
    }
    if lhs.iter().chain(y).any(|v| !v.is_finite()) {
        return None;
    }
    let rhs = DMatrix::from_column_slice(rows, 1, y);
    let svd = lhs.clone().svd(true, true);
    let largest = svd.singular_values.max();
    let cutoff = rows as f64 * f64::EPSILON * largest;
    let full_rank = rows >= order && svd.singular_values.iter().all(|&s| s > cutoff);
    let solution = if full_rank {
        // Householder QR: nalgebra's SVD loses digits on small singular
        // values that LAPACK's keeps, QR does not
        let qr = lhs.qr();
        let qtb = qr.q().transpose() * &rhs;
        qr.r().solve_upper_triangular(&qtb)?
    } else {
        svd.solve(&rhs, cutoff).ok()?
    };
    Some((0..order).map(|j| solution[(j, 0)] / scale[j]).collect())
}

/// The largest real part of the roots of the polynomial `p` (highest power
/// first), as `np.max(np.real(np.roots(p)))`: leading zeros dropped,
/// trailing zeros giving roots at 0, the rest as the eigenvalues of the
/// companion matrix. NaN if there are no roots or the companion matrix is
/// not finite (where numpy raises).
fn max_real_root(p: &[f64]) -> f64 {
    let Some(first) = p.iter().position(|&c| c != 0.0) else {
        return f64::NAN;
    };
    let last = p.iter().rposition(|&c| c != 0.0).unwrap_or(first);
    let trailing_zeros = p.len() - last - 1;
    let p = &p[first..=last];
    let mut real_parts = Vec::new();
    if p.len() > 1 {
        let n = p.len() - 1;
        let mut companion = DMatrix::zeros(n, n);
        for j in 0..n {
            companion[(0, j)] = -p[j + 1] / p[0];
        }
        for i in 1..n {
            companion[(i, i - 1)] = 1.0;
        }
        if companion.iter().any(|v: &f64| !v.is_finite()) {
            return f64::NAN;
        }
        // Balanced as LAPACK's dgeev does (nalgebra does not), then the
        // eigenvalues polished on the polynomial
        balance(&mut companion);
        real_parts.extend(
            companion
                .complex_eigenvalues()
                .iter()
                .map(|&z| polish_root(p, z).re),
        );
    }
    real_parts.extend(std::iter::repeat_n(0.0, trailing_zeros));
    if real_parts.is_empty() {
        return f64::NAN;
    }
    // np.max: NaN if any is NaN
    real_parts
        .into_iter()
        .reduce(|a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
        .unwrap_or(f64::NAN)
}

/// Balance a square matrix in place by a diagonal similarity transform with
/// powers of 2 (Parlett & Reinsch), so that each row and its column have
/// similar norms. The eigenvalues are unchanged but computed far more
/// accurately for badly scaled matrices such as companion matrices.
fn balance(a: &mut DMatrix<f64>) {
    let n = a.nrows();
    let mut converged = false;
    while !converged {
        converged = true;
        for i in 0..n {
            let mut c: f64 = (0..n).filter(|&j| j != i).map(|j| a[(j, i)].abs()).sum();
            let r: f64 = (0..n).filter(|&j| j != i).map(|j| a[(i, j)].abs()).sum();
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let s = c + r;
            let mut f = 1.0;
            while c < r / 2.0 {
                f *= 2.0;
                c *= 4.0;
            }
            while c >= r * 2.0 {
                f /= 2.0;
                c /= 4.0;
            }
            if (c + r) / f < 0.95 * s {
                converged = false;
                for j in 0..n {
                    a[(i, j)] /= f;
                    a[(j, i)] *= f;
                }
            }
        }
    }
}

/// Refine an approximate root `z` of the polynomial `p` (highest power
/// first) by Newton steps, keeping a step only while it lowers $|p(z)|$.
fn polish_root(p: &[f64], mut z: Complex<f64>) -> Complex<f64> {
    let eval = |z: Complex<f64>| {
        p.iter().fold(
            (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
            |(value, slope), &c| (value * z + c, slope * z + value),
        )
    };
    let (mut value, mut slope) = eval(z);
    for _ in 0..50 {
        if value.norm() == 0.0 || slope.norm() == 0.0 {
            break;
        }
        let step = value / slope;
        let next = z - step;
        let (next_value, next_slope) = eval(next);
        if next_value.norm().is_nan() || next_value.norm() >= value.norm() {
            break;
        }
        (z, value, slope) = (next, next_value, next_slope);
        if step.norm() <= f64::EPSILON * z.norm() {
            break;
        }
    }
    z
}

/// Friedrich coefficients feature.
///
/// The coefficients of the polynomial $h(x) = \sum_{k=0}^{m} c_k\,x^{m-k}$
/// fitted to the deterministic dynamics of the Langevin model
/// $$ \dot{x}(t) = h(x(t)) + \mathcal{N}(0, R) $$
/// (Friedrich et al., 2000): the values (all but the last) are binned into
/// `r` quantile bins, the values and their changes $x_{t+1} - x_t$ are
/// averaged per bin, and $h$ is the least-squares polynomial of degree `m`
/// through the bin averages. Coefficient `coeff` is that of $x^{m - \text{coeff}}$
/// (`np.polyfit` order). For short series the result depends strongly on
/// the parameters.
///
/// # Parameters
/// - `coeff`: index of the returned coefficient (0 is the highest power).
/// - `m`: degree of the polynomial.
/// - `r`: number of quantile bins.
///
/// Config:
/// `[friedrich_coefficients] parameters = [{ coeff = 0, m = 3, r = 30 }, ...]`;
/// one column per entry. The fit is done once per distinct `(m, r)`.
///
/// # Output column
/// `{name}__friedrich_coefficients__coeff_{coeff}__m_{m}__r_{r}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - `coeff > m` gives NaN.
/// - Duplicate quantile bin edges (for example a series with many equal
///   values, or fewer distinct values than bins) give NaN, as tsfresh.
/// - Non-finite bin averages ($\pm\infty$ in the series) give NaN, as
///   tsfresh. NaN values are left out of the bins and the averages, as
///   pandas does.
/// - Fewer than 2 values, every bin average NaN, `m = 0` or `r = 0` give NaN
///   (tsfresh raises).
///
/// # tsfresh
/// `feature_calculators.friedrich_coefficients` (v0.21.2). The least-squares
/// solution comes from a different SVD implementation than numpy's LAPACK,
/// so results differ from tsfresh's in the last bits (more for
/// ill-conditioned fits).
pub struct FriedrichCoefficients {
    /// `(coeff, m, r)` per config entry.
    pub params: Vec<(usize, usize, usize)>,
}

impl Kernel for FriedrichCoefficients {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(coeff, m, r)| format!("friedrich_coefficients__coeff_{coeff}__m_{m}__r_{r}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        // Coefficients per distinct (m, r), fitted once
        let mut fits: Vec<(usize, usize, Option<Vec<f64>>)> = Vec::new();
        for &(coeff, m, r) in &self.params {
            let fit = match fits.iter().position(|f| (f.0, f.1) == (m, r)) {
                Some(i) => &fits[i].2,
                None => {
                    fits.push((m, r, estimate_friedrich_coefficients(x, m, r)));
                    &fits[fits.len() - 1].2
                }
            };
            out.push(
                fit.as_ref()
                    .and_then(|c| c.get(coeff).copied())
                    .unwrap_or(f64::NAN),
            );
        }
    }
}

/// Maximum Langevin fixed point feature.
///
/// The largest fixed point $\max \{x : h(x) = 0\}$ (largest real part of
/// the roots) of the polynomial $h$ fitted to the deterministic dynamics of
/// the Langevin model, as in [`FriedrichCoefficients`].
///
/// # Parameters
/// - `m`: degree of the polynomial.
/// - `r`: number of quantile bins.
///
/// Config: `[max_langevin_fixed_point] parameters = [{ m = 3, r = 30 }, ...]`;
/// one column per entry.
///
/// # Output column
/// `{name}__max_langevin_fixed_point__m_{m}__r_{r}`
///
/// # Edge cases
/// - NaN wherever [`FriedrichCoefficients`] gives NaN coefficients.
/// - A polynomial without roots (constant) gives NaN, as tsfresh.
///
/// # tsfresh
/// `feature_calculators.max_langevin_fixed_point` (v0.21.2). Roots are the
/// eigenvalues of the companion matrix, as `np.roots`, from a different
/// eigenvalue solver and refined by Newton steps on the polynomial, so they
/// agree with numpy's to about the last bits. When the fitted leading
/// coefficient is (numerically) zero the largest root is ill-conditioned and
/// can differ substantially from tsfresh's.
pub struct MaxLangevinFixedPoint {
    /// `(m, r)` per config entry.
    pub params: Vec<(usize, usize)>,
}

impl Kernel for MaxLangevinFixedPoint {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(m, r)| format!("max_langevin_fixed_point__m_{m}__r_{r}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        for &(m, r) in &self.params {
            out.push(
                estimate_friedrich_coefficients(x, m, r)
                    .map(|c| max_real_root(&c))
                    .unwrap_or(f64::NAN),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_real_root() {
        // Badly scaled: (t + 1)(t^2 + 1) with t = x / a, roots -a and ±ia
        let a = 774366.0_f64;
        let p = [1.0 / (a * a * a), 1.0 / (a * a), 1.0 / a, 1.0];
        assert!(max_real_root(&p).abs() < 1e-6, "{}", max_real_root(&p));
        // (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6
        let root = max_real_root(&[1.0, 0.0, -7.0, 6.0]);
        assert!((root - 2.0).abs() < 1e-12);
        // trailing zero: x^2 - x = x (x - 1); leading zero dropped
        assert!((max_real_root(&[0.0, 1.0, -1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(max_real_root(&[0.0, 0.0, 2.0, 0.0]), 0.0);
        assert!(max_real_root(&[0.0, 0.0, 0.0, 2.0]).is_nan());
        assert!(max_real_root(&[0.0; 4]).is_nan());
    }

    #[test]
    fn test_polyfit_exact() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v * v - v + 0.5).collect();
        let c = polyfit(&x, &y, 2).unwrap();
        for (got, want) in c.iter().zip([2.0, -1.0, 0.5]) {
            assert!((got - want).abs() < 1e-12, "{c:?}");
        }
    }

    // tsfresh v0.21.2 test_friedrich_equal_to_snapshot
    #[test]
    fn test_friedrich_snapshot() {
        let x = [
            -0.53, -0.61, -1.26, -0.88, -0.34, 0.58, 2.86, -0.47, 0.78, -0.45, -0.27, 0.43, 1.72,
            0.26, 1.02, -0.09, 0.65, 1.49, -0.95, -1.02, -0.64, -1.63, -0.71, -0.43, -1.69, 0.05,
            1.58, 1.1, 0.55, -1.02,
        ];
        let c = estimate_friedrich_coefficients(&x, 2, 30).unwrap();
        let expected = [-0.24536975738843042, -0.533309548662685, 0.2759399238199404];
        for (got, want) in c.iter().zip(expected) {
            assert!((got - want).abs() < 1e-7, "{c:?}");
        }
    }

    #[test]
    fn test_friedrich_duplicate_edges() {
        assert!(estimate_friedrich_coefficients(&[0.0; 100], 3, 30).is_none());
    }
}
