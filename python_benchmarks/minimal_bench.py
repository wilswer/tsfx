import time

import pandas as pd
import polars as pl
import tsfresh
import tsfx
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfx import ExtractionSettings, FeatureSetting


def tsfresh_minimal() -> None:
    """Benchmark TSFRESH minimal features extraction."""
    print("Benchmarking TSFRESH minimal features extraction...")
    print("Loading data...")
    df = pd.read_csv("./test_data/all_stocks_5yr.csv")
    df = df.dropna(axis=0, how="any")
    print("Data loaded")
    settings = MinimalFCParameters()
    print("Starting TSFRESH minimal features extraction...")
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
    """Benchmark TSFX minimal features extraction."""
    print("Benchmarking TSFX minimal features extraction...")
    print("Loading data...")
    lf = pl.scan_csv("test_data/all_stocks_5yr.csv")
    lf = lf.drop_nulls()
    print("Data loaded")
    opts = ExtractionSettings(
        grouping_cols=["Name"],
        feature_setting=FeatureSetting.Minimal,
        value_cols=["open", "high", "low", "close", "volume"],
    )
    print("Starting TSFX minimal features extraction...")
    start = time.time()
    fdf = tsfx.extract_features(lf, opts)
    end = time.time()
    print(f"TSFX Minimal extraction took {end - start} s")
    print(fdf.sort(pl.col("Name")).head())


if __name__ == "__main__":
    tsfx_minimal()
    tsfresh_minimal()
