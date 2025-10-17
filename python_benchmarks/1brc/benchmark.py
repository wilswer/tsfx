import json
import time

import pandas as pd
import polars as pl
import tsfresh
import tsfx
from tsfx import ExtractionSettings, FeatureSetting


def tsfresh_efficient() -> float:
    """Benchmark TSFRESH efficient features extraction."""
    print("Benchmarking TSFRESH efficient features extraction...")
    print("Loading data...")
    df = pd.read_csv(
        "./test_data/measurements.txt",
        sep=";",
        usecols=[0, 1],
        names=["location", "temperature"],
    )
    df = df.dropna(axis=0, how="any")
    print("Data loaded")
    with open("./python_benchmarks/efficient.json") as f:
        settings = json.load(f)
    print("Starting TSFRESH efficient features extraction...")
    start = time.time()
    fdf = tsfresh.extract_features(
        df[["location", "temperature"]],
        column_id="location",
        column_sort="location",
        default_fc_parameters=settings,
    )
    end = time.time()
    print(f"TSFRESH Efficient extraction took {end - start} s")
    print(fdf.head())
    return end - start


def tsfx_efficient() -> float:
    """Benchmark TSFX efficient features extraction."""
    print("Benchmarking TSFX efficient features extraction...")
    print("Loading data...")
    lf = pl.scan_csv(
        "./test_data/measurements.txt",
        separator=";",
        has_header=False,
    )
    lf = lf.with_columns(
        pl.col("column_1").alias("location"),
        pl.col("column_2").alias("temperature"),
    )
    lf = lf.select(pl.col("location"), pl.col("temperature"))
    lf = lf.drop_nulls()
    print("Data loaded")
    opts = ExtractionSettings(
        grouping_cols=["location"],
        feature_setting=FeatureSetting.Efficient,
        value_cols=["temperature"],
    )
    print("Starting TSFX efficient features extraction...")
    start = time.time()
    fdf = tsfx.extract_features(lf, opts)
    end = time.time()
    print(f"TSFX Efficient extraction took {end - start} s")
    print(fdf.sort(pl.col("location")).head())
    return end - start


if __name__ == "__main__":
    time1 = tsfx_efficient()
    time2 = tsfresh_efficient()
    print(f"Speed-up ~ {time2 / time1:.1f}x")
