import time

import pandas as pd
import polars as pl
import tsfresh
import tsfx
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfx import ExtractionSettings, FeatureSetting


def tsfresh_efficient() -> None:
    """Benchmark TSFRESH efficient features extraction."""
    print("Benchmarking TSFRESH efficient features extraction...")
    print("Loading data...")
    df = pd.read_csv("./test_data/all_stocks_5yr.csv")
    df = df.dropna(axis=0, how="any")
    print("Data loaded")
    settings = EfficientFCParameters()
    print("Starting TSFRESH efficient features extraction...")
    start = time.time()
    fdf = tsfresh.extract_features(
        df[["date", "open", "Name"]],
        column_id="Name",
        column_sort="date",
        default_fc_parameters=settings,
    )
    end = time.time()
    print(f"TSFRESH Efficient extraction took {end - start} s")
    print(fdf.head())


def tsfx_efficient() -> None:
    """Benchmark TSFX efficient features extraction."""
    print("Benchmarking TSFX efficient features extraction...")
    print("Loading data...")
    lf = pl.scan_csv("test_data/all_stocks_5yr.csv")
    lf = lf.drop_nulls()
    print("Data loaded")
    opts = ExtractionSettings(
        grouping_col="Name",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["open"],
    )
    print("Starting TSFX efficient features extraction...")
    start = time.time()
    fdf = tsfx.extract_features(lf, opts)
    end = time.time()
    print(f"TSFX Efficient extraction took {end - start} s")
    print(fdf.sort(pl.col("Name")).head())


if __name__ == "__main__":
    tsfx_efficient()
    tsfresh_efficient()
