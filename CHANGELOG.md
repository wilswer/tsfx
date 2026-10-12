# Changelog

All notable changes to TSFX are documented here.

## Unreleased

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
