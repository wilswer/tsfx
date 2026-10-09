# Changelog

All notable changes to TSFX are documented here.

## Unreleased

### Changed

- **Output column order.** The counting-based reoccurrence features
  (`has_duplicate`, `ratio_value_number_to_time_series_length`,
  `sum_of_reoccurring_values`, `sum_of_reoccurring_data_points`,
  `percentage_of_reoccurring_values_to_all_values`,
  `percentage_of_reoccurring_values_to_all_datapoints`) are now computed
  together and appear next to each other; `variation_coefficient` and
  `mean_change` now come after them. Column names and values are unchanged.
  **If you select feature columns by position** (for example
  `df.to_numpy()` feeding a trained model), select them by name instead or
  retrain: the positions differ from earlier releases.
- Feature values now match tsfresh v0.21.2 for every feature, which changes
  results for many features compared with 0.3.3, and invalid config files
  raise `ValueError`. See
  [PR #16](https://github.com/wilswer/tsfx/pull/16) for the full list.

### Performance

- `extract_features` on in-memory data (for example `DataFrame.lazy()`) no
  longer serialises the data to hand it to Rust: about 0.5 s less per
  million rows.
- The reoccurrence features share one sort per group: about 25% faster
  Efficient extraction.
