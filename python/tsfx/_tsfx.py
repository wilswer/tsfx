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
    fdf = tsfx.extract_features(
        lf=lf,
        settings=settings,
        streaming=streaming,
    )

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
