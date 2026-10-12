# Changelog

All notable changes to TSFX are documented here.

## 0.4.0 (2026-10-12)

### Changed

- **Output column order.** `length` comes first, then all features of the
  first value column, then all features of the next, each in tier order
  (Minimal, then Efficient, then Comprehensive). Previously each tier looped
  over all value columns. Column names are unchanged. **If you select
  feature columns by position** (for example `df.to_numpy()` feeding a
  trained model), select them by name instead or retrain: the positions
  differ from earlier releases.
- `median` and `quantile` of a group without non-null values are now NaN
  (they were null), like every other feature and like tsfresh.
- `quantile` of a group that contains nulls now equals `np.quantile` of its
  non-null values exactly; it could differ in the last bits before.
- `mean`, `variance`, `standard_deviation` and every feature built on the
  mean or variance (for example `variation_coefficient`,
  `ratio_beyond_r_sigma`, `large_standard_deviation`, `autocorrelation`,
  normalised `cid_ce`, `count_above_mean`) now use numpy's summation and
  equal tsfresh's `np.mean` / `np.var` / `np.std` bit for bit (before: in
  29% of cases for the variance). Most values change only in the last bits,
  but threshold features can change where a value sits exactly on the
  threshold: `ratio_beyond_r_sigma` of `[1, 3, 3, -2, 7, 7, 0, 1]` with
  `r = 1.5` is now 0, as in tsfresh (it was 0.375).
- Found by fuzzing every feature against tsfresh:
  - `mean_n_absolute_max` is NaN unless there are more than `n` values, and
    NaN if the series contains NaN (tsfresh sorts NaN last, as the largest
    absolute value).
  - `sum_values`, `absolute_sum_of_changes`, `mean_absolute_change`,
    `root_mean_square`, `autocorrelation`, `c3`,
    `time_reversal_asymmetry_statistic`, `median` and `quantile` now equal
    tsfresh's numpy results bit for bit (they differed in the last bits).
  - **`agg_linear_trend` follows tsfresh's `extract_features`**, which hands
    the calculator a numpy array: the `var` aggregator is the population
    variance (`ddof = 0`, was `ddof = 1`), and a NaN in a chunk now makes the
    result NaN instead of being skipped. The `median` aggregator (a tsfx
    addition) follows `np.median` likewise.
  - `linear_trend` and `agg_linear_trend` fit the line from centred sums,
    as `scipy.stats.linregress` does; the one-pass formula lost digits when
    the values span many orders of magnitude.
- Feature values now match tsfresh v0.21.2 for every feature, which changes
  results for many features compared with 0.3.3, and invalid config files
  raise `ValueError`. See
  [PR #16](https://github.com/wilswer/tsfx/pull/16) for the full list.

### Performance

- Every feature except `length` is computed in one pass per group: a single
  `apply` per value column instead of one per feature (about 60 for
  Efficient), so Polars collects each group's values once, and statistics
  that several features need (mean, variance, differences, sorted values)
  are computed once. Measured on a 2× Xeon Gold 6230 (80 threads):
  Efficient on the stocks data 113 → 43 ms (−62%), on 100M rows of the
  1BRC data 8.8 → 4.6 s (−48%); Minimal on 1BRC 5.0 → 4.1 s (−17%).
  Minimal on the small, already sorted stocks data is slower, 19 → 23 ms,
  where grouping itself takes about 13 ms.
- `extract_features` on in-memory data (for example `DataFrame.lazy()`) no
  longer serialises the data to hand it to Rust: about 0.5 s less per
  million rows.
- The reoccurrence features share one sort per group: about 25% faster
  Efficient extraction.
- `number_peaks` computes all configured `n` in one pass instead of copying
  and rotating the series per shift: about 16% faster Efficient extraction.
- `agg_linear_trend` aggregates chunks through views instead of copying each
  chunk, and computes the chunk variance without allocating: about 14%
  faster Efficient extraction.
