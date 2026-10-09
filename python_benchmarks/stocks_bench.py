"""End-to-end TSFX benchmark on the stocks dataset, for use with hyperfine.

Usage: python python_benchmarks/stocks_bench.py {minimal,efficient} {memory,scan}

"memory" passes ``DataFrame.lazy()`` (in-memory data), "scan" passes a
``scan_csv`` LazyFrame. Example:

    hyperfine --warmup 2 \\
        'python python_benchmarks/stocks_bench.py efficient memory'
"""

import sys

import polars as pl
from tsfx import ExtractionSettings, FeatureSetting, extract_features

VALUE_COLS = ["open", "high", "low", "close", "volume"]
TIERS = {"minimal": FeatureSetting.Minimal, "efficient": FeatureSetting.Efficient}


def main(tier: str, source: str) -> None:
    lf = (
        pl.scan_csv("test_data/all_stocks_5yr.csv")
        .drop_nulls()
        .with_columns([pl.col(c).cast(pl.Float64) for c in VALUE_COLS])
    )
    if source == "memory":
        lf = lf.collect().lazy()
    settings = ExtractionSettings(
        grouping_cols=["Name"],
        value_cols=VALUE_COLS,
        feature_setting=TIERS[tier],
    )
    extract_features(lf, settings)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
