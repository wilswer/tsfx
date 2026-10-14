# Changelog

All notable changes to TSFX are documented here.

## Unreleased

### Changed

- **A `config_path` that does not exist raises `ValueError`** ("could not
  read config file …") instead of silently using the default config, so a
  mistyped path no longer goes unnoticed. Without a `config_path`, a
  `.tsfx-config.toml` in the working directory is still used if present,
  else the default config.
- tsfx no longer prints "No config file detected. Using default config."
  when it uses the default config (the message went to the process's
  standard output, which Python cannot capture or silence).

## 0.6.0 (2026-10-13)

### Added

- **30 more tsfresh feature calculators**, each with tsfresh's default
  parameters, its own config section and parity tests against tsfresh
  0.21.2 (its own test cases plus differential fuzzing):
  - Changes and counts: `mean_second_derivative_central`, `value_count`,
    `energy_ratio_by_chunks`, `change_quantiles`, `benford_correlation`,
    `query_similarity_count`.
  - Entropy and complexity: `binned_entropy`, `lempel_ziv_complexity`,
    `permutation_entropy`, `approximate_entropy`.
  - Langevin model: `friedrich_coefficients`, `max_langevin_fixed_point`.
  - Spectral: `fft_coefficient`, `fft_aggregated`, `spkt_welch_density`,
    `fourier_entropy`.
  - Wavelets: `cwt_coefficients`, `number_cwt_peaks`.
  - Autoregression: `agg_autocorrelation`, `partial_autocorrelation`,
    `ar_coefficient`, `augmented_dickey_fuller` (`autolag` AIC, BIC,
    t-stat or None).

  All are in the `Efficient` tier, except `approximate_entropy`
  (`Comprehensive`, as tsfresh marks it expensive).
  `query_similarity_count` is off unless configured, since it needs a query
  (tsfresh's default, no query, only gives NaN). Unknown string parameters
  (for example an `f_agg` or `attr`) are rejected when the config is loaded.
  Not implemented: `linear_trend_timewise` (needs the time index) and
  `matrix_profile` (its `matrixprofile` dependency is unmaintained; tsfresh
  disables it too).

### Changed

- **Fair TSFresh benchmark.** `python_benchmarks/tsfresh_compare.py`
  replaces `minimal_bench.py`, `efficient_bench.py`, `1brc/benchmark.py` and
  `efficient.json`. TSFresh now runs exactly the calculators and parameters
  TSFX runs for the tier (derived from `.tsfx-config.toml` and checked by
  feature count), both libraries get the same pre-sorted in-memory data with
  CSV parsing outside the timer, TSFresh's `n_jobs` matches the Polars
  thread pool, and both get a warm-up and repeated runs. Previously TSFX's
  time included reading the CSV, the stocks series were not sorted by date
  for TSFX, and the feature sets differed.
- **Many more output columns by default.** The `Efficient` tier now gives
  about 750 columns per value column (was about 155), 400 of them from
  `fft_coefficient`. Existing columns keep their names, but the new ones
  come after them for each value column, so **positions of existing columns
  change**: select feature columns by name. To keep the old set, list the
  features you want in `.tsfx-config.toml`. `Efficient` extraction takes
  longer accordingly (about 6× on the 505-series stocks benchmark).
- Deliberate deviations from tsfresh, each documented in the feature's docs
  and tested:
  - Where tsfresh raises (and so fails the whole extraction), tsfx gives NaN:
    `binned_entropy` with ±inf, `ar_coefficient` with NaN or ±inf,
    `query_similarity_count` of a series shorter than the query, and the
    entropy and Langevin features on inputs numpy cannot handle.
  - `fft_coefficient` of a series with NaN or ±inf is NaN for every column
    (numpy keeps some parts finite, depending on its internal order).
  - `permutation_entropy` ranks ties within a window by position; numpy's
    sort is not stable on some CPUs, so tsfresh's result depends on the CPU.
  - `number_cwt_peaks`: where neighbouring wavelet coefficients are tied in
    real arithmetic (for example a smooth bump centred on a sample),
    rounding decides the maximum. tsfx's arithmetic is portable, so counts
    are the same on every platform but can differ from tsfresh's (whose
    rounding depends on numpy's BLAS and CPU).
  - `augmented_dickey_fuller`: an exact fit gives a test statistic of ±inf
    (statsmodels divides by rounding noise), and collinear extra lags whose
    information criteria tie resolve to the smaller lag.
- Spectral, wavelet, regression, entropy and Langevin features equal
  tsfresh's to within rounding (relative and absolute 1e-9, or a tolerance
  scaled to the series or fit) rather than bit for bit, since tsfx does not
  reproduce numpy's FFT, LAPACK and SIMD `log` arithmetic.

### Fixed

- **Linux wheels for free-threaded CPython 3.14 and PyPy 3.11 again.** 0.5.0
  shipped only the abi3 wheels, which those interpreters cannot use, so
  installing there built tsfx from source (with a Rust toolchain).
- **`ar_coefficient` matches statsmodels 0.15.** A series of exactly
  2k + 1 values (as many observations as AR(k) parameters) now gives the
  exact fit, as tsfresh does with statsmodels 0.15, instead of NaN (or 0 for
  `coeff = k`), which matched statsmodels 0.14. For example, a single value
  with `k = 0` gives that value. The tsfresh fuzz tests in CI now pin
  `statsmodels==0.15.0`.

### Performance

- **`number_cwt_peaks` is much faster on long series,** with the same
  results bit for bit: on 50M rows of 1BRC data (series of about 5,600
  values) it takes 2.6 s instead of 10.9 s. Its noise estimate
  now selects the 10th percentile of each window instead of sorting it, and
  ridge lines are matched by binary search instead of comparing every pair.
  The `Efficient` tier on 50M rows of 1BRC data takes 13.0 s instead of
  21.9 s.
- **`augmented_dickey_fuller`, `permutation_entropy` and
  `change_quantiles` are faster,** with the same results bit for bit.
  Together they make the `Efficient` tier on 100M rows of 1BRC data take
  25.5 s instead of 35.3 s (−28%), the features alone measured as follows:
  - `augmented_dickey_fuller` 16.6 → about 10 s: the check for an exact fit
    first compares against a cheap upper bound of the fit's scale, and only
    computes the exact scale (a pass over every row and regressor) when the
    residual is small enough to matter.
  - `permutation_entropy` 3.7 → 2.1 s: each window's pattern is counted as
    one integer instead of an allocated list.
  - `change_quantiles` 3.0 → 2.5 s: each corridor's changes are computed
    once and shared by its aggregations.
- **`augmented_dickey_fuller` and `ar_coefficient` solve well-conditioned
  regressions from the Gram matrix.** A Householder QR sweeps the whole
  design once per regressor and was limited by memory bandwidth on long
  series; the Gram matrix needs one pass. On 100M rows of 1BRC data,
  `augmented_dickey_fuller` alone takes 4.4 s instead of 12.6 s. **Not bit
  for bit:** results agree with the previous ones to within about
  1e-10 (ADF test statistics and p-values within 1e-12 relative on
  1BRC and edge-case series, the same chosen lags, AR coefficients within
  2e-11 of the fit's largest coefficient). Ill-conditioned,
  near-exact and exactly fitting regressions still use the QR decomposition,
  so their results, including the deviations from tsfresh documented for
  exact fits, are unchanged.

## 0.5.0 (2026-10-12)

### Changed

- **Python 3.11 or newer is required** (3.9 and 3.10 have reached end of
  life). On 3.9 or 3.10, pip keeps installing 0.4.0.
- Wheels use the stable ABI (abi3): one wheel per platform covers CPython
  3.11 and every later version, including 3.15. Free-threaded builds still
  get version-specific wheels. Release builds strip symbols, so wheels are
  smaller (PyPI limits a project's total size).

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
