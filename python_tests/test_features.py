import math

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

def test_only_nan_group_dropped():
    df = pl.DataFrame({"id": ["a", "b", "c"], "val": [1.0, 2.0, None]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b"]

def test_nan_df():
    df = pl.DataFrame({"id": ["a", "b", "c", "c"], "val": [1.0, 2.0, 1.0, None]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b", "c"]
    assert fdf.get_column("val__mean").to_list() == [1.0, 2.0, 1.0]


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

    assert fdf.get_column("val__has_duplicate").to_list() == [0.0, 0.0, 0.0, 1.0]

def test_ratio_value_number_to_time_series_length():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0],
            "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__ratio_value_number_to_time_series_length").to_list() == [1.0, 1.0, 1.0, 0.25]

def test_variation_coefficient():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0],
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

    assert math.isnan(fdf.get_column("val__variation_coefficient").to_list()[-1])

def test_sum_of_reoccurring_values():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__sum_of_reoccurring_values").to_list() == [0.0, 0.0, 0.0, 3.0]

def test_sum_of_reoccurring_values():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__sum_of_reoccurring_data_points").to_list() == [0.0, 0.0, 0.0, 7.0]
