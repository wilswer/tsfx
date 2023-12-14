import time

import pandas as pd
import polars as pl
import tsfresh
import tsfx
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfx import ExtractionSettings, FeatureSetting


def tsfresh_minimal() -> None:
    """Benchmark TSFRESH minimal extraction."""
    df = pd.read_csv("./test_data/all_stocks_5yr.csv")
    df = df.dropna(axis=0, how="any")
    settings = MinimalFCParameters()
    start = time.time()
    fdf = tsfresh.extract_features(
        df,
        column_id="Name",
        column_sort="date",
        default_fc_parameters=settings,
    )
    end = time.time()
    print(f"TSFRESH Minimal extraction took {end - start} s")
    print(fdf.head())


def tsfx_minimal() -> None:
    """Benchmark TSFX minimal extraction."""
    lf = pl.scan_csv("test_data/all_stocks_5yr.csv")
    lf = lf.drop_nulls()
    opts = ExtractionSettings(
        grouping_col="Name",
        feature_setting=FeatureSetting.Minimal,
        value_cols=["open", "high", "low", "close", "volume"],
    )
    start = time.time()
    fdf = tsfx.extract_features(lf, opts)
    end = time.time()
    print(f"TSFX Minimal extraction took {end - start} s")
    print(fdf.sort(pl.col("Name")).head())


if __name__ == "__main__":
    tsfresh_minimal()
    tsfx_minimal()
