//! Linear fits over time.

use super::kernel::{GroupContext, Kernel};
use crate::utils::stats::{
    aggregate_on_chunks, calculate_sequential_ols, skip_nan_mean, skip_nan_median, skip_nan_reduce,
    skip_nan_sample_var,
};
use crate::utils::toml_reader::ChunkAggregator;

/// Linear trend feature.
///
/// The intercept $a$ and slope $b$ of the least-squares line through the
/// values against their 0-based position, $x_i \approx a + b\,i$:
/// $$ b = \frac{\sum_i (i - \bar{i})(x_i - \bar{x})}{\sum_i (i - \bar{i})^2}, \quad a = \bar{x} - b\,\bar{i}. $$
/// Only these two of tsfresh's `linear_trend` attributes are computed
/// (not `pvalue`, `rvalue` or `stderr`).
///
/// # Output column
/// `{name}__linear_trend_intercept` and `{name}__linear_trend_slope`
///
/// # Edge cases
/// - Nulls are dropped first.
/// - Fewer than 2 values give NaN.
/// - NaN values are kept and make both results NaN.
///
/// # tsfresh
/// `feature_calculators.linear_trend` (v0.21.2) with `attr` = `"intercept"`
/// or `"slope"`.
pub struct LinearTrend;

impl Kernel for LinearTrend {
    fn names(&self) -> Vec<String> {
        vec![
            "linear_trend_intercept".to_string(),
            "linear_trend_slope".to_string(),
        ]
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let (intercept, slope) = calculate_sequential_ols(ctx.values().iter().copied(), ctx.len());
        out.extend([intercept, slope]);
    }
}

/// Aggregated linear trend feature.
///
/// Split the series into consecutive chunks of `chunk_size` values (the last
/// chunk may be shorter), aggregate each chunk to one number, and fit a
/// least-squares line through the aggregates against the chunk index, as in
/// [`LinearTrend`]. Returns the intercept and slope of that line.
///
/// # Parameters
/// - `chunk_size`: values per chunk.
/// - `aggregator`: how each chunk is summarised: `mean`, `median`, `min`,
///   `max` or `var` (sample variance, `ddof = 1`, as pandas). Any other name
///   is rejected when the config is loaded.
///
/// Config: `[agg_linear_trend] parameters = [{ chunk_size = 5, aggregator = "mean" }, ...]`;
/// two columns per entry.
///
/// # Output column
/// `{name}__agg_linear_trend_intercept__chunk_size_{chunk_size}__agg_{aggregator}`
/// and `{name}__agg_linear_trend_slope__chunk_size_{chunk_size}__agg_{aggregator}`
///
/// # Edge cases
/// - Nulls are dropped first. A group with fewer than `chunk_size` values gives NaN.
/// - Fewer than 2 chunks give NaN (no line can be fitted).
/// - Each chunk aggregate skips NaN, like pandas; a chunk that is all NaN
///   (or, for `var`, has fewer than 2 non-NaN values) gives NaN, and so does
///   the fit.
///
/// # tsfresh
/// `feature_calculators.agg_linear_trend` (v0.21.2) with `chunk_len`,
/// `f_agg` and `attr` = `"intercept"` or `"slope"`.
pub struct AggLinearTrend {
    /// `(chunk_size, aggregator)` per config entry.
    pub params: Vec<(usize, ChunkAggregator)>,
}

impl Kernel for AggLinearTrend {
    fn names(&self) -> Vec<String> {
        self.params
            .iter()
            .flat_map(|(chunk_size, aggregator)| {
                ["intercept", "slope"].map(|attr| {
                    format!("agg_linear_trend_{attr}__chunk_size_{chunk_size:.1}__agg_{aggregator}")
                })
            })
            .collect()
    }

    fn compute(&self, ctx: &GroupContext, out: &mut Vec<f64>) {
        let x = ctx.x();
        for (chunk_size, aggregator) in &self.params {
            let chunk_size = *chunk_size;
            if x.len() < chunk_size {
                out.extend([f64::NAN, f64::NAN]);
                continue;
            }
            let aggregates = match aggregator {
                ChunkAggregator::Mean => aggregate_on_chunks(x, chunk_size, skip_nan_mean),
                ChunkAggregator::Median => aggregate_on_chunks(x, chunk_size, skip_nan_median),
                ChunkAggregator::Max => {
                    aggregate_on_chunks(x, chunk_size, |c| skip_nan_reduce(c, f64::max))
                }
                ChunkAggregator::Min => {
                    aggregate_on_chunks(x, chunk_size, |c| skip_nan_reduce(c, f64::min))
                }
                // ddof=1 on purpose: tsfresh aggregates chunks with pandas' Series.var
                ChunkAggregator::Var => aggregate_on_chunks(x, chunk_size, skip_nan_sample_var),
            };
            let n = aggregates.len();
            let (intercept, slope) = calculate_sequential_ols(aggregates, n);
            out.extend([intercept, slope]);
        }
    }
}
