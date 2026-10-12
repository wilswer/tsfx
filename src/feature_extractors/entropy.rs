//! Complexity and entropy measures.

use itertools::Itertools;
use ndarray::{Array1, ArrayView1, Axis};

use super::kernel::scalar_kernel;

/// All length-`chunk_size` sliding windows of `x`, one step apart.
fn _into_subchunks(x: &ArrayView1<f64>, chunk_size: usize) -> Vec<Array1<f64>> {
    let mut subchunks = Vec::with_capacity(x.len());
    for chunk in x.axis_windows(Axis(0), chunk_size) {
        subchunks.push(chunk.to_owned());
    }
    subchunks
}

/// Number of unordered pairs of templates within Chebyshev distance `r`.
fn _get_matches(templates: Vec<Array1<f64>>, r: f64) -> usize {
    let mut matches = 0;
    for combo in templates.into_iter().combinations(2) {
        let a = combo[0].to_owned();
        let b = combo[1].to_owned();
        let diff = a - b;
        // tsfresh counts a match at distance <= tolerance
        let dist_check = diff.mapv(|x| if x.abs() <= r { 1 } else { 0 }).sum();
        if dist_check == diff.len() {
            matches += 1;
        }
    }
    matches
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
/// `FeatureSetting::Comprehensive`, where it is always on (the config file
/// does not switch it off).
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
            let matches_m = _get_matches(_into_subchunks(ctx.x(), m), r);
            let matches_m_plus_1 = _get_matches(_into_subchunks(ctx.x(), m + 1), r);
            ((matches_m as f64) / (matches_m_plus_1 as f64)).ln()
        }
    }
}
