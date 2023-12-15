import polars as pl
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)


def test_empty_df():
    df = pl.DataFrame({"id": [], "val": []}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    assert fdf.is_empty()


def test_nan_df():
    df = pl.DataFrame({"id": ["a", "b", "c"], "val": [1.0, 2.0, None]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b"]


def test_has_duplicate():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val_has_duplicate").to_list() == [0.0, 0.0, 0.0, 1.0]
