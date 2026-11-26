from __future__ import annotations

import polars as pl

from tsfx import tsfx


def extract_features(
    lf: pl.LazyFrame,
    settings: tsfx.ExtractionSettings,
    streaming: bool = False,
) -> pl.DataFrame:
    """Extract features from a Polars `LazyFrame`.

    Short wrapper around `tsfx.extract_features` that runs feature extraction on
    a Polars `LazyFrame` and returns a flattened `pl.DataFrame`. Any `Struct`
    columns produced are unnested to top-level columns.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input lazy frame.
    settings : tsfx.ExtractionSettings
        Extraction configuration.
    streaming : bool, optional
        Enable streaming mode.

    Returns
    -------
    pl.DataFrame
        Flattened DataFrame with extracted features.

    Raises
    ------
    ValueError
        If the settings are invalid, or `settings.reduction` cannot be fitted
        (fewer groups or windows than it needs).

    Warns
    -----
    UserWarning
        If the reduction's noise step runs on fewer than 50 features or rows:
        it may then drop real features as noise.

    Example
    -------
    ```python
    import polars as pl
    from tsfx import ExtractionSettings, FeatureSetting, extract_features

    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
        },
    ).lazy()

    settings = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val", "value"],
    )

    gdf = extract_features(df, settings)
    ```

    """
    if streaming:
        fdf = tsfx.extract_features(lf=lf, settings=settings, streaming=True)
    else:
        # Collect only the needed columns here and hand Rust a DataFrame:
        # that crosses without copying, whereas a LazyFrame is serialised
        # together with any in-memory data in its plan.
        df = lf.select(settings._input_columns).collect()
        fdf = tsfx.extract_features_df(df=df, settings=settings)

    return fdf.unnest(
        [
            col_name
            for col_name, dtype in zip(
                fdf.columns,
                fdf.dtypes,
                strict=True,
            )
            if dtype == pl.Struct
        ],
    )


def reduction_report(
    lf: pl.LazyFrame,
    settings: tsfx.ExtractionSettings,
) -> pl.DataFrame:
    """Explain which features the reduction keeps and why.

    Extracts the features of `lf` as `settings` say (without scaling, which
    the reduction does not depend on), fits `settings.reduction` and returns
    its report: one row per extracted feature, in extraction order, with the
    columns

    - `feature`: the feature's column name;
    - `kept`: whether the reduction keeps it;
    - `dropped_by`: the step that dropped it (`"missing"`, `"one_value"`,
      `"duplicate"`, `"noise"`, or the cap's `"cap"` and `"explained"`), null
      if kept;
    - `missing_share`: its share of missing values (null, NaN, ±inf);
    - `top_share`: the share of its most frequent value among its non-missing
      values;
    - `duplicate_of` and `correlation`: the kept feature it copies, and their
      Spearman correlation;
    - `noise_score` and `noise_cutoff`: its Laplacian score and the best
      shadow feature's score; it is noise unless its score is below;
    - `cap_rank`, `cap_gain` and `cap_explained` (with a `CapSetting`): its
      pick order (from 1), the information it added in features' worth (for
      a feature the cap dropped: what it would have added next, 0 if
      `"explained"`), and the share of the spread explained after its pick.

    A step that is turned off, or that a feature did not reach, leaves its
    columns null.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input lazy frame.
    settings : tsfx.ExtractionSettings
        Extraction configuration with a `reduction`.

    Returns
    -------
    pl.DataFrame
        The report.

    Raises
    ------
    ValueError
        If `settings` has no reduction, or the reduction cannot be fitted
        (e.g. too few groups or windows).

    Warns
    -----
    UserWarning
        If the noise step runs on fewer than 50 features or rows: it may then
        drop real features as noise. If the cap runs without the noise step,
        or the rows (or the `max_samples` sample) stop it before
        `max_features`.

    """
    df = lf.select(settings._input_columns).collect()
    return tsfx.reduction_report_df(df=df, settings=settings)
