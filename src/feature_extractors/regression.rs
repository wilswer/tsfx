//! Autoregressive models and unit-root tests fitted by least squares.

use super::kernel::{GroupContext, Kernel};
use crate::utils::toml_reader::{AdfAttr, AdfAutolag};

/// An ordinary least squares fit, as statsmodels' `OLS(y, X).fit()` (the
/// default `method="pinv"`).
struct Ols {
    params: Vec<f64>,
    /// Sum of squared residuals.
    ssr: f64,
    rank: usize,
    nobs: usize,
    /// Diagonal of $(X^T X)^+$, statsmodels' `normalized_cov_params`.
    normalized_cov_diag: Vec<f64>,
}

/// Residuals at most this fraction of the scale of the fitted terms count as
/// an exact fit (see [`Ols::fit`]).
const PERFECT_FIT_TOL: f64 = 1e-10;

/// Information criteria closer than this (relative to their magnitude, at
/// least 1) count as equal when choosing the ADF lag.
const IC_TIE_TOL: f64 = 1e-9;

impl Ols {
    /// Fit `y` against the columns of `x`, see [`Ols::fit_prefixes`].
    fn fit(x: &[Vec<f64>], y: &[f64]) -> Option<Self> {
        Self::fit_prefixes(x, y, x.len()).and_then(|mut fits| fits.pop())
    }

    /// Fit `y` against the first `p` columns of `x`, for every `p` from
    /// `min_cols` to all of them (sharing one QR decomposition, as the first
    /// `p` columns of a QR decomposition are one of the first `p` columns).
    ///
    /// Each fit is statsmodels' pseudoinverse solution: singular values
    /// below $10^{-15}$ times the largest are dropped, and the rank counts
    /// singular values above $\max(s) \cdot p \cdot \varepsilon$
    /// (`np.linalg.matrix_rank` of the diagonal matrix of singular values).
    /// Like LAPACK's SVD (which reduces a tall matrix by QR first), the
    /// Householder QR followed by a Jacobi SVD of the triangular factor
    /// resolves exactly collinear columns (a constant stretch of the series)
    /// to negligible singular values, so the minimum-norm solution is found.
    ///
    /// A fit whose root mean squared residual is at most
    /// [`PERFECT_FIT_TOL`] times the scale of the terms (largest
    /// $|y_i| + \sum_j |X_{ij}\,\beta_j|$) is exact up to rounding, and its
    /// residual sum of squares is set to 0: the rounding noise would
    /// otherwise decide t-values and information criteria.
    ///
    /// `None` if the design is not finite (numpy's SVD raises
    /// `LinAlgError`); a non-finite `y` propagates into the results.
    fn fit_prefixes(x: &[Vec<f64>], y: &[f64], min_cols: usize) -> Option<Vec<Self>> {
        if x.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let (nobs, k) = (y.len(), x.len());
        let (r, c) = householder_qr(x, y);
        // tails[p] = |c[p..]|^2, the part of y no combination of the first
        // p columns can reach
        let mut tails = vec![0.0; k + 1];
        tails[k] = c[k.min(nobs)..].iter().map(|v| v * v).sum();
        for p in (0..k).rev() {
            tails[p] = tails[p + 1] + if p < nobs { c[p] * c[p] } else { 0.0 };
        }
        let fits = (min_cols..=k)
            .map(|p| {
                // The p x p triangular factor of the first p columns
                let r_p: Vec<Vec<f64>> = r[..p].iter().map(|col| col[..p].to_vec()).collect();
                let (u, s, v) = jacobi_svd(r_p.clone());
                let s_max = s.iter().copied().fold(0.0, f64::max);
                let cutoff = 1e-15 * s_max;
                // pinv = V diag(1/s) U^T with unit columns u_j = U_j / s_j,
                // so beta = sum_j v_j (U_j . c) / s_j^2
                let inv_sq: Vec<f64> = s
                    .iter()
                    .map(|&sj| if sj > cutoff { 1.0 / (sj * sj) } else { 0.0 })
                    .collect();
                let mut params = vec![0.0; p];
                for j in 0..p {
                    if inv_sq[j] == 0.0 {
                        continue;
                    }
                    let uc = u[j].iter().zip(&c).map(|(a, b)| a * b).sum::<f64>() * inv_sq[j];
                    for (b, vij) in params.iter_mut().zip(&v[j]) {
                        *b += vij * uc;
                    }
                }
                let normalized_cov_diag = (0..p)
                    .map(|i| (0..p).map(|j| v[j][i] * v[j][i] * inv_sq[j]).sum())
                    .collect();
                // Residual: the unreachable tail plus what the (possibly
                // rank-deficient) solution misses of the reachable part
                let mut ssr = tails[p];
                for i in 0..p.min(nobs) {
                    let fitted: f64 = (i..p).map(|j| r_p[j][i] * params[j]).sum();
                    ssr += (c[i] - fitted) * (c[i] - fitted);
                }
                let scale = (0..nobs)
                    .map(|row| {
                        y[row].abs()
                            + x[..p]
                                .iter()
                                .zip(&params)
                                .map(|(col, b)| (col[row] * b).abs())
                                .sum::<f64>()
                    })
                    .fold(0.0, f64::max);
                if (ssr / nobs as f64).sqrt() <= PERFECT_FIT_TOL * scale {
                    ssr = 0.0;
                }
                let tol = s_max * p as f64 * f64::EPSILON;
                Self {
                    params,
                    ssr,
                    rank: s.iter().filter(|&&v| v > tol).count(),
                    nobs,
                    normalized_cov_diag,
                }
            })
            .collect();
        Some(fits)
    }

    /// The t-value of parameter `i`: the estimate over its standard error,
    /// with the error variance estimated as `ssr / (nobs - rank)`.
    fn tvalue(&self, i: usize) -> f64 {
        let scale = self.ssr / (self.nobs - self.rank) as f64;
        self.params[i] / (self.normalized_cov_diag[i] * scale).sqrt()
    }

    /// The Gaussian log-likelihood at the estimate (statsmodels' `llf`).
    fn llf(&self) -> f64 {
        let nobs2 = self.nobs as f64 / 2.0;
        -nobs2 * (2.0 * std::f64::consts::PI).ln()
            - nobs2 * (self.ssr / self.nobs as f64).ln()
            - nobs2
    }

    /// Akaike's information criterion, counting `rank` parameters.
    fn aic(&self) -> f64 {
        -2.0 * self.llf() + 2.0 * self.rank as f64
    }

    /// Bayes' information criterion, counting `rank` parameters.
    fn bic(&self) -> f64 {
        -2.0 * self.llf() + (self.nobs as f64).ln() * self.rank as f64
    }
}

/// Householder QR decomposition $X = QR$ of the matrix with columns `x`
/// (at least as many rows as columns). Returns the columns of the upper
/// triangular $R$ (each of length $k$) and $Q^T y$.
fn householder_qr(x: &[Vec<f64>], y: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (n, k) = (y.len(), x.len());
    debug_assert!(n >= k);
    let mut a: Vec<Vec<f64>> = x.to_vec();
    let mut c = y.to_vec();
    for j in 0..k {
        let norm = a[j][j..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        // Reflect a[j][j..] onto alpha e_1, alpha of opposite sign to the
        // leading entry to avoid cancellation
        let alpha = -norm.copysign(a[j][j]);
        let mut v = a[j][j..].to_vec();
        v[0] -= alpha;
        let v_norm_sq: f64 = v.iter().map(|e| e * e).sum();
        let reflect = |col: &mut [f64]| {
            let s = 2.0 * v.iter().zip(col.iter()).map(|(a, b)| a * b).sum::<f64>() / v_norm_sq;
            for (e, vi) in col.iter_mut().zip(&v) {
                *e -= s * vi;
            }
        };
        for col in a[j + 1..].iter_mut() {
            reflect(&mut col[j..]);
        }
        reflect(&mut c[j..]);
        a[j][j] = alpha;
        a[j][j + 1..].iter_mut().for_each(|e| *e = 0.0);
    }
    for col in a.iter_mut() {
        col.truncate(k);
    }
    (a, c)
}

/// One-sided Jacobi (Hestenes) SVD of the matrix with columns `a`: rotate
/// pairs of columns until all are orthogonal, applying the same rotations
/// to the identity. Returns the rotated columns $U_j = s_j u_j$ (not
/// normalised), the singular values $s_j = \lVert U_j \rVert$ (unsorted) and
/// the right singular vectors $v_j$ as columns, so that $A = \sum_j U_j v_j^T$.
fn jacobi_svd(mut a: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let k = a.len();
    let mut v: Vec<Vec<f64>> = (0..k)
        .map(|j| (0..k).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let rotate = |cols: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64| {
        let (left, right) = cols.split_at_mut(q);
        for (xp, xq) in left[p].iter_mut().zip(right[0].iter_mut()) {
            let (vp, vq) = (*xp, *xq);
            *xp = c * vp - s * vq;
            *xq = s * vp + c * vq;
        }
    };
    for _sweep in 0..100 {
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let alpha = dot(&a[p], &a[p]);
                let beta = dot(&a[q], &a[q]);
                let gamma = dot(&a[p], &a[q]);
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = c * t;
                rotate(&mut a, p, q, c, s);
                rotate(&mut v, p, q, c, s);
            }
        }
        if !rotated {
            break;
        }
    }
    let s = a.iter().map(|col| dot(col, col).sqrt()).collect();
    (a, s, v)
}

/// Autoregressive coefficient feature.
///
/// Fits an AR(k) process with an intercept by conditional maximum likelihood
/// (ordinary least squares on the lagged values, as statsmodels'
/// `AutoReg(x, lags=k, trend="c").fit()`),
/// $$ x_t = \varphi_0 + \sum_{i=1}^{k} \varphi_i\, x_{t-i} + \varepsilon_t, $$
/// and returns the coefficient $\varphi_{\text{coeff}}$ ($\varphi_0$ is the
/// intercept).
///
/// # Parameters
/// - `coeff`: which coefficient $i$ to return.
/// - `k`: the order of the process.
///
/// Config: `[ar_coefficient] parameters = [{ coeff = 0, k = 10 }, ...]`; one
/// column per entry. One fit per distinct `k`.
///
/// # Output column
/// `{name}__ar_coefficient__coeff_{coeff}__k_{k}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - `coeff` $> k$ gives NaN.
/// - Fewer than $2k + 2$ values (statsmodels needs more observations than
///   regressors and raises otherwise): NaN for `coeff` $< k$ and
///   0 for `coeff` $= k$, as tsfresh (whose fallback has only $k$ entries).
/// - A constant or otherwise collinear series gives the minimum-norm least
///   squares solution, as statsmodels' pseudoinverse.
/// - **Deliberate deviation from tsfresh:** a series containing NaN or
///   $\pm\infty$ gives NaN for every coefficient. tsfresh raises
///   (statsmodels' `MissingDataError` is not caught), failing the whole
///   extraction.
///
/// # tsfresh
/// `feature_calculators.ar_coefficient` (v0.21.2).
pub struct ArCoefficient {
    /// `(coeff, k)` per config entry.
    pub params: Vec<(usize, usize)>,
}

/// The AR(k) coefficients $\varphi_0, \dots, \varphi_k$, or `None` if the
/// model cannot be fitted.
fn ar_fit(x: &[f64], k: usize) -> Option<Vec<f64>> {
    let n = x.len();
    // statsmodels needs more observations (n - k) than regressors (k + 1):
    // ValueError if fewer, ZeroDivisionError (a degree-of-freedom correction)
    // if as many
    if n < 2 * k + 2 {
        return None;
    }
    let nobs = n - k;
    // Columns: intercept, then x lagged by 1, ..., k
    let design: Vec<Vec<f64>> = (0..=k)
        .map(|c| {
            if c == 0 {
                vec![1.0; nobs]
            } else {
                x[k - c..n - c].to_vec()
            }
        })
        .collect();
    Ols::fit(&design, &x[k..]).map(|fit| fit.params)
}

impl Kernel for ArCoefficient {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(coeff, k)| format!("ar_coefficient__coeff_{coeff}__k_{k}"))
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        if x.iter().any(|v| !v.is_finite()) {
            out.extend(self.params.iter().map(|_| f64::NAN));
            return;
        }
        let mut fits: Vec<(usize, Option<Vec<f64>>)> = Vec::new();
        for &(coeff, k) in &self.params {
            let fit = match fits.iter().find(|(fk, _)| *fk == k) {
                Some((_, fit)) => fit,
                None => {
                    fits.push((k, ar_fit(x, k)));
                    &fits.last().unwrap().1
                }
            };
            out.push(match fit {
                _ if coeff > k => f64::NAN,
                Some(params) => params[coeff],
                // tsfresh's fallback is [nan] * k, and indexing past it gives 0
                None if coeff == k => 0.0,
                None => f64::NAN,
            });
        }
    }
}

/// MacKinnon's (1994, 2010 tables) approximate p-value of an augmented
/// Dickey-Fuller test statistic for a regression with a constant and one
/// integrated series, as statsmodels' `mackinnonp(teststat, "c", N=1)`.
fn mackinnon_p(teststat: f64) -> f64 {
    // statsmodels' tau_max_c, tau_min_c, tau_star_c and the scaled
    // tau_c_smallp / tau_c_largep coefficients for N = 1
    const TAU_MAX: f64 = 2.74;
    const TAU_MIN: f64 = -18.83;
    const TAU_STAR: f64 = -1.61;
    const SMALL_P: [f64; 3] = [2.1659, 1.4412, 0.038269000000000004];
    const LARGE_P: [f64; 4] = [1.7339, 0.9320200000000001, -0.12745, -0.010368];
    if teststat > TAU_MAX {
        return 1.0;
    } else if teststat < TAU_MIN {
        return 0.0;
    }
    let coef: &[f64] = if teststat <= TAU_STAR {
        &SMALL_P
    } else {
        &LARGE_P
    };
    // np.polyval on the reversed (ascending) coefficients: Horner's scheme
    let z = coef.iter().rev().fold(0.0, |acc, c| acc * teststat + c);
    normal_cdf(z)
}

/// The standard normal CDF, as scipy's `ndtr` computes it from erf / erfc.
fn normal_cdf(a: f64) -> f64 {
    let x = a * std::f64::consts::FRAC_1_SQRT_2;
    let z = x.abs();
    if z < std::f64::consts::FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(x)
    } else {
        let y = 0.5 * libm::erfc(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

/// The ADF regression of the last `nobs` differences (the response, second)
/// on the lagged level, `lags` lagged differences and a constant (the
/// design columns, first): `[constant, level, lag 1, ..., lag lags]` if
/// `const_first`, else `[level, lag 1, ..., lag lags, constant]`.
fn adf_regression(
    x: &[f64],
    lags: usize,
    nobs: usize,
    const_first: bool,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    // Row r explains the difference x[t + 1] - x[t] with t = n - 1 - nobs + r
    let start = n - 1 - nobs;
    let diffs = |from: usize| (from..from + nobs).map(|i| x[i + 1] - x[i]).collect();
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(lags + 2);
    design.push(x[start..start + nobs].to_vec());
    design.extend((1..=lags).map(|lag| diffs(start - lag)));
    if const_first {
        design.insert(0, vec![1.0; nobs]);
    } else {
        design.push(vec![1.0; nobs]);
    }
    (design, diffs(start))
}

/// Test statistic, p-value and used lag of the augmented Dickey-Fuller test
/// with a constant, as statsmodels' `adfuller(x, autolag=...)`; NaN for all
/// three where statsmodels raises (constant or too short series, a
/// non-finite value in a design matrix).
fn adfuller(x: &[f64], autolag: AdfAutolag) -> [f64; 3] {
    const FAILED: [f64; 3] = [f64::NAN; 3];
    let n = x.len();
    // x.max() == x.min(): never true with a NaN
    let (min, max) = super::kernel::extremes(x.iter().copied());
    if min == max {
        return FAILED;
    }
    // Schwert's rule, capped at nobs // 2 - ntrend - 1 (ntrend = 1)
    let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as i64;
    let maxlag = schwert.min(n as i64 / 2 - 2);
    if maxlag < 0 {
        return FAILED;
    }
    let maxlag = maxlag as usize;

    let usedlag = if autolag == AdfAutolag::None {
        maxlag
    } else {
        // Every candidate is fitted on the same observations, so their
        // information criteria are comparable
        let (full, y) = adf_regression(x, maxlag, n - 1 - maxlag, true);
        // fits[lags]: constant, level and the first `lags` lags
        let Some(fits) = Ols::fit_prefixes(&full, &y, 2) else {
            return FAILED;
        };
        match autolag {
            AdfAutolag::Aic | AdfAutolag::Bic => {
                let ic = |fit: &Ols| {
                    if autolag == AdfAutolag::Aic {
                        fit.aic()
                    } else {
                        fit.bic()
                    }
                };
                // Python's min over (ic, lag): the first strict minimum.
                // Criteria within rounding of each other tie (an extra lag
                // collinear with the others adds neither rank nor fit), and
                // ties go to the smaller lag
                let mut best = 0;
                for lags in 1..=maxlag {
                    let best_ic = ic(&fits[best]);
                    if ic(&fits[lags]) < best_ic - IC_TIE_TOL * best_ic.abs().max(1.0) {
                        best = lags;
                    }
                }
                best
            }
            _ => {
                // t-stat: drop lags until the last one is significant at 5%
                const STOP: f64 = 1.6448536269514722;
                let mut best = 0;
                for lags in (0..=maxlag).rev() {
                    best = lags;
                    if fits[lags].tvalue(lags + 1).abs() >= STOP {
                        break;
                    }
                }
                best
            }
        }
    };
    let (design, y) = adf_regression(x, usedlag, n - 1 - usedlag, false);
    let Some(fit) = Ols::fit(&design, &y) else {
        return FAILED;
    };
    let teststat = fit.tvalue(0);
    [teststat, mackinnon_p(teststat), usedlag as f64]
}

/// Augmented Dickey-Fuller feature.
///
/// The augmented Dickey-Fuller test for a unit root, with a constant, as
/// statsmodels' `adfuller(x, autolag=...)`: regress the differences on the
/// lagged level, $p$ lagged differences and a constant,
/// $$ \Delta x_t = \alpha + \gamma\, x_{t-1} + \sum_{i=1}^{p} \beta_i\, \Delta x_{t-i} + \varepsilon_t, $$
/// where $p$ is chosen among $0, \dots, \lceil 12 (n/100)^{1/4} \rceil$
/// (capped at $\lfloor n/2 \rfloor - 2$) by `autolag`. Returns the test
/// statistic (the t-value of $\gamma$), its MacKinnon approximate p-value or
/// the used lag $p$. A small p-value rejects the unit root.
///
/// # Parameters
/// - `attr`: `teststat`, `pvalue` or `usedlag`.
/// - `autolag` (optional, default `"AIC"`): how $p$ is chosen: `"AIC"` or
///   `"BIC"` (smallest criterion, all candidates fitted on the same
///   observations), `"t-stat"` (from the largest $p$ down until the last lag
///   is significant at 5%) or `"None"` (the largest $p$).
///
/// Config: `[augmented_dickey_fuller] parameters = [{ attr = "teststat" }, { attr = "pvalue", autolag = "BIC" }, ...]`;
/// one column per entry. One test per distinct `autolag`.
///
/// # Output column
/// `{name}__augmented_dickey_fuller__attr_{attr}__autolag_{autolag}`
///
/// # Edge cases
/// - Nulls are dropped first; a group with no non-null values gives NaN.
/// - Fewer than 4 values or a constant series give NaN (statsmodels raises,
///   tsfresh returns NaN).
/// - NaN or $\pm\infty$ values give NaN. Only the last value enters no
///   design matrix, so if it is the only non-finite one, statsmodels gets
///   as far as choosing a lag: the test statistic and p-value are NaN, the
///   used lag is 0 (`"AIC"`, `"BIC"`, `"t-stat"`, as every criterion is NaN)
///   or the largest lag (`"None"`).
/// - **Deliberate deviation from tsfresh:** when the regression fits exactly
///   (root mean squared residual at most $10^{-10}$ of the scale of the
///   fitted terms, e.g. a periodic or linear series), the residual variance
///   is taken as 0: the test statistic is $\pm\infty$ (p-value 0 or 1) and
///   AIC / BIC pick the smallest exactly fitting lag. statsmodels divides by
///   the rounding noise of the residuals instead, so its statistic (around
///   $\pm 10^{15}$) and chosen lag are arbitrary.
/// - **Deliberate deviation from tsfresh:** when two candidate lags fit
///   equally well because the extra lagged difference is collinear with the
///   other regressors (e.g. after a constant stretch), their AIC / BIC tie
///   (to within a relative $10^{-9}$) and the smaller lag is used, as
///   Python's `min` does for exact ties. statsmodels breaks the tie by
///   rounding noise.
///
/// # Cost
/// $O(n\,p^2)$ per distinct `autolag`, with $p$ the largest lag (68 for
/// $n = 10^5$): the $p + 1$ candidate fits share one QR decomposition.
///
/// # tsfresh
/// `feature_calculators.augmented_dickey_fuller` (v0.21.2).
pub struct AugmentedDickeyFuller {
    /// `(attr, autolag)` per config entry.
    pub params: Vec<(AdfAttr, AdfAutolag)>,
}

impl Kernel for AugmentedDickeyFuller {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .map(|(attr, autolag)| {
                format!("augmented_dickey_fuller__attr_{attr}__autolag_{autolag}")
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.values();
        let mut tests: Vec<(AdfAutolag, [f64; 3])> = Vec::new();
        for &(attr, autolag) in &self.params {
            let result = match tests.iter().find(|(a, _)| *a == autolag) {
                Some((_, result)) => *result,
                None => {
                    let result = adfuller(x, autolag);
                    tests.push((autolag, result));
                    result
                }
            };
            out.push(match attr {
                AdfAttr::Teststat => result[0],
                AdfAttr::Pvalue => result[1],
                AdfAttr::Usedlag => result[2],
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mackinnon_p() {
        // statsmodels.tsa.adfvalues.mackinnonp(t, "c", 1)
        assert!((mackinnon_p(1.7320508075688792) - 0.9982042849867909).abs() < 1e-15);
        assert!((mackinnon_p(-1.732050807568878) - 0.4146791580296235).abs() < 1e-15);
        assert_eq!(mackinnon_p(3.0), 1.0);
        assert_eq!(mackinnon_p(-20.0), 0.0);
        assert!(mackinnon_p(f64::NAN).is_nan());
    }

    #[test]
    fn test_ar_fit_constant_is_minimum_norm() {
        // statsmodels' pinv: AutoReg([1.0] * 10, lags=3).fit().params
        let params = ar_fit(&[1.0; 10], 3).unwrap();
        for p in params {
            assert!((p - 0.25).abs() < 1e-12, "{p}");
        }
        assert!(ar_fit(&[1.0, 2.0, 3.0, 7.0], 3).is_none());
    }
}
