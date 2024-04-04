import polars as pl
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)


def test_empty_config():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
        config_path="./python_tests/data/.tsfx-config-empty.toml",
    )
    gdf = extract_features(df, opts)
    print(gdf.head())
    assert gdf.shape == (4, 1)
