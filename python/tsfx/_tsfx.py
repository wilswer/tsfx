from __future__ import annotations

import polars as pl

from tsfx import tsfx


def extract_features(
    lf: pl.LazyFrame,
    settings: tsfx.ExtractionSettings,
    streaming: bool = False,
) -> pl.DataFrame:
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
