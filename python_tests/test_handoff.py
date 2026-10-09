"""The non-streaming path (data collected in Python and handed to Rust as a
DataFrame) must give exactly the same result as the streaming path (the
LazyFrame plan handed to Rust)."""

import math
import random

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)


def _frame() -> pl.DataFrame:
    rng = random.Random(0)
    rows = []
    for gid in ("a", "b", "c"):
        for day in range(40):
            value = rng.gauss(0, 1)
            if gid == "c" and day % 7 == 0:
                value = math.nan
            rows.append(
                {
                    "id": gid,
                    "time": f"2020-{1 + day // 28:02d}-{1 + day % 28:02d}",
                    "val": value,
                    "other": rng.gauss(5, 2),
                    "unused": "x",
                },
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    "tier",
    [FeatureSetting.Minimal, FeatureSetting.Efficient, FeatureSetting.Comprehensive],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_streaming_and_in_memory_paths_agree(tier, dynamic):
    dynamic_settings = (
        DynamicGroupBySettings(
            time_col="time",
            every="1mo",
            period="1mo",
            offset="0d",
            datetime_format="%Y-%m-%d",
        )
        if dynamic
        else None
    )
    settings = ExtractionSettings(
        grouping_cols=["id"],
        value_cols=["val", "other"],
        feature_setting=tier,
        dynamic_settings=dynamic_settings,
    )
    sort_cols = ["id", "time"] if dynamic else ["id"]
    in_memory = extract_features(_frame().lazy(), settings).sort(sort_cols)
    streamed = extract_features(_frame().lazy(), settings, streaming=True).sort(
        sort_cols,
    )
    assert_frame_equal(in_memory, streamed)
