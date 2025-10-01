import math
import pytest

import polars as pl
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    extract_features,
    DynamicGroupBySettings,
)


def test_empty_df():
    df = pl.DataFrame({"id": [], "val": []})
    df = df.select([pl.col("id").cast(pl.Utf8), pl.col("val").cast(pl.Float64)])
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df.lazy(), opts)
    assert fdf.is_empty()


def test_unit_length_df():
    df = pl.DataFrame({"id": ["a"], "val": [1.0]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    assert fdf.shape[0] == 1


def test_only_nan_group_dropped():
    df = pl.DataFrame({"id": ["a", "b", "c", "c"], "val": [1.0, 2.0, None, None]})
    df = df.select([pl.col("id").cast(pl.Utf8), pl.col("val").cast(pl.Float64)])
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df.lazy(), opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b", "c"]

def test_only_nan_group_dropped_with_date():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "time": [
                "2001-01-01",
                "2002-01-01",
                "2003-01-01",
                "2001-01-01",
                "2002-01-01",
                "2003-01-01",
                "2001-01-01",
                "2002-01-01",
                "2003-01-01",
            ],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None, None, None],
            "value": [4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 4.0, 5.0, 6.0],
        },
    ).lazy()

    dyn_settings = DynamicGroupBySettings(
        time_col="time",
        every="1y",
        period="1y",
        offset="0y",
        datetime_format="%Y-%m-%d",
    )
    settings = ExtractionSettings(
        grouping_col="id",
        value_cols=["val", "value"],
        feature_setting=FeatureSetting.Efficient,
        dynamic_settings=dyn_settings,
    )
    fdf = extract_features(df, settings)
    assert fdf.get_column("id").to_list() == ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    assert fdf["val__mean"].fill_nan(None).to_list()[0] == 1.0
    assert fdf["val__mean"].fill_nan(None).to_list()[1] == 2.0
    assert fdf["val__mean"].fill_nan(None).to_list()[2] == 3.0
    assert fdf["val__mean"].fill_nan(None).to_list()[-1] == None

def test_long_constant_df():
    N = 100_000
    df = pl.DataFrame({"id": ["a"] * (N + 1) + ["b"] * (N + 1), "val": [0.0] * N + [1.0] + [1.0] * N + [2.0]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b"]
    assert fdf.get_column("val__mean").to_list() == pytest.approx([0.0, 1.0], abs=1e-4)



def test_nan_df():
    df = pl.DataFrame({"id": ["a", "b", "c", "c"], "val": [1.0, 2.0, 1.0, None]}).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b", "c"]
    assert fdf.get_column("val__mean").to_list() == [1.0, 2.0, 1.0]


def test_nan_df2():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c"] + ["d"] * 10,
            "val": [1.0, 2.0, 1.0, None, None, 1.0] + [None] * 8 + [1.0, 1.0],
            "val2": [1.0, None, None, 1.0, None, 1.0] + [1.0] * 8 + [None, None],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("id").to_list() == ["a", "b", "c", "d"]
    assert fdf.get_column("length").to_list() == [2.0, 1.0, 1.0, 2.0]


def test_length():
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
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("length").to_list() == [1.0, 2.0, 3.0, 4.0]


def test_sum_values():
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
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__sum_values").to_list() == [1.0, 3.0, 6.0, 4.0]


def test_minimum():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__minimum").to_list() == [1.0, 1.0, 1.0, -1.0]


def test_maximum():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__maximum").to_list() == [1.0, 2.0, 3.0, 1.0]


def test_absolute_energy():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, -2.0, 1.0, -2.0, 3.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__absolute_energy").to_list() == [1.0, 5.0, 14.0, 4.0]


def test_median1():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__median").to_list() == [1.0, 1.5, 2.0, 0.0]


def test_median2():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__median").to_list() == [1.0, 1.5, 2.0, 0.0]


def test_linear_trend_intercept():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__linear_trend_intercept").to_list() == pytest.approx(
        [1.0, 1.0, 3.0, -1.0], abs=1e-6,
    )


def test_linear_trend_slope():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__linear_trend_slope").to_list() == pytest.approx(
        [0.0, 1.0, -1.0, 0.0], abs=1e-6,
    )


def test_has_duplicate_max():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__has_duplicate_max").to_list() == [0.0, 0.0, 1.0, 1.0]


def test_has_duplicate_min():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__has_duplicate_min").to_list() == [0.0, 0.0, 0.0, 1.0]


def test_has_duplicate1():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__has_duplicate").to_list() == [0.0, 0.0, 1.0, 1.0]


def test_has_duplicate2():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
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
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column(
        "val__ratio_value_number_to_time_series_length",
    ).to_list() == [1.0, 1.0, 1.0, 0.25]


def test_variation_coefficient():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
            "val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert math.isnan(fdf.get_column("val__variation_coefficient").to_list()[-1])


def test_sum_of_reoccurring_values():
    df = pl.DataFrame(
        {
            "id": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "d",
                "d",
                "d",
                "d",
                "d",
                "d",
            ],
            "val": [
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                3.0,
            ],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__sum_of_reoccurring_values").to_list() == [
        0.0,
        0.0,
        0.0,
        3.0,
    ]


def test_sum_of_reoccurring_data_points():
    df = pl.DataFrame(
        {
            "id": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
                "d",
                "d",
                "d",
                "d",
                "d",
                "d",
            ],
            "val": [
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                3.0,
            ],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__sum_of_reoccurring_data_points").to_list() == [
        0.0,
        0.0,
        0.0,
        7.0,
    ]


def test_standard_deviation():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c"],
            "val": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__standard_deviation").to_list()[0] == 0
    assert math.isnan(fdf.get_column("val__standard_deviation").to_list()[-1])


def test_variance():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c"],
            "val": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__variance").to_list()[0] == 0
    assert math.isnan(fdf.get_column("val__variance").to_list()[-1])

def test_variance_larger_than_standard_deviation():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "val": [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.1, 0.1, 0.1],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__variance_larger_than_standard_deviation").to_list()[0] == 1.0
    assert fdf.get_column("val__variance_larger_than_standard_deviation").to_list()[1] == 0.0


def test_large_standard_deviation():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a"],
            "val": [-1.0, -1.0, 1.0, 1.0],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__large_standard_deviation__r_0.25").to_list()[0] == 1.0
    assert fdf.get_column("val__large_standard_deviation__r_0.30").to_list()[0] == 1.0
    assert fdf.get_column("val__large_standard_deviation__r_0.50").to_list()[0] == 1.0
    assert fdf.get_column("val__large_standard_deviation__r_0.70").to_list()[0] == 0.0

def test_symmetry_looking():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c", "d", "d"],
            "val": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -0.9, -0.900001],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__symmetry_looking__r_0.75").to_list()[0] == 1.0
    assert fdf.get_column("val__symmetry_looking__r_0.05").to_list() == [1.0, 0.0, 1.0, 1.0]

def test_percentage_of_reoccurring_values_to_all_values():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "b", "b", "c", "d", "d", "d", "d"],
            "val": [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 4.0, 1.0, 1.0, 1.0, -1.0, -1.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column(
        "val__percentage_of_reoccurring_values_to_all_values",
    ).to_list() == [0.5, 0.25, 0.0, 1.0]


def test_percentage_of_reoccurring_values_to_all_datapoints():
    df = pl.DataFrame(
        {
            "id": [
                "a",
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "b",
                "c",
                "d",
                "d",
                "d",
                "d",
            ],
            "val": [
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                3.0,
                4.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
            ],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column(
        "val__percentage_of_reoccurring_values_to_all_datapoints",
    ).to_list() == pytest.approx([0.25, 0.2, 0.0, 0.5])


def test_agg_linear_trend_intercept():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert math.isnan(
        fdf.get_column(
            "val__agg_linear_trend_intercept__chunk_size_5__agg_mean",
        ).to_list()[0],
    )
    assert fdf.get_column(
        "val__agg_linear_trend_intercept__chunk_size_5__agg_mean",
    ).to_list()[1] == pytest.approx(2.0, abs=1e-6)


def test_agg_linear_trend_slope():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert math.isnan(
        fdf.get_column("val__agg_linear_trend_slope__chunk_size_5__agg_mean").to_list()[
            0
        ],
    )
    assert fdf.get_column(
        "val__agg_linear_trend_slope__chunk_size_5__agg_mean",
    ).to_list()[1] == pytest.approx(1.0, abs=1e-6)


def test_mean_n_absolute_max():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert math.isnan(fdf.get_column("val__mean_n_absolute_max__n_7").to_list()[0])
    assert fdf.get_column("val__mean_n_absolute_max__n_7").to_list()[1] == 4.0


def test_mean_change():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__mean_change").to_list() == [0.0, 1.0]


def test_number_crossing_m1():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, -1.0, 2.0, 3.0, -4.0, -5.0, -6.0, 7.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__number_crossing_m__m_0.0").to_list() == [0.0, 3.0]


def test_number_crossing_m2():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [0.0, 1.0, 0.0, 1.0, -3.0, -4.0, -5.0, 6.0, 7.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__number_crossing_m__m_1.0").to_list() == [0.0, 1.0]


def test_number_crossing_m3():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [-1.0, -1.0, 1.0, 2.0, -3.0, -4.0, -5.0, 6.0, 7.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__number_crossing_m__m_-1.0").to_list() == [0.0, 2.0]


def test_number_crossing_m4():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__number_crossing_m__m_0.0").to_list() == [1.0, 1.0]


def test_range_count():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "b", "b", "b", "b"],
            "val": [2.0, 0.0, -2.0, 1.0, 0.0, 1.0, 0.0, -2.0, 0.0, -2.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__range_count__min_-1.0__max_1.0").to_list() == [1.0, 5.0]


def test_index_mass_quantile():
    df = pl.DataFrame(
        {
            "id": ["a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "val": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        },
    ).lazy()

    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")

    assert fdf.get_column("val__index_mass_quantile__q_0.5").to_list() == pytest.approx(
        [0.5, 0.2], abs=1e-6,
    )


def test_c3():
    df = pl.DataFrame(
        {
            "id": ["a"] * 10,
            "val": [1.0] * 10,
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__c3__lag_1").to_list() == [1.0]
    assert fdf.get_column("val__c3__lag_2").to_list() == [1.0]
    assert fdf.get_column("val__c3__lag_3").to_list() == [1.0]


def test_c3_2():
    df = pl.DataFrame(
        {
            "id": ["a"] * 4,
            "val": [1, 2, -3, 4],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__c3__lag_1").to_list() == [-15.0]
    assert fdf.get_column("val__c3__lag_2").to_list() == [0.0]
    assert fdf.get_column("val__c3__lag_3").to_list() == [0.0]


def test_time_reversal_asymmetry_statistic():
    df = pl.DataFrame(
        {
            "id": ["a"] * 10,
            "val": [1.0] * 10,
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_1",
    ).to_list() == [0.0]
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_2",
    ).to_list() == [0.0]
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_3",
    ).to_list() == [0.0]


def test_time_reversal_asymmetry_statistic_2():
    df = pl.DataFrame(
        {
            "id": ["a"] * 4,
            "val": [1, 2, -3, 4],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_1",
    ).to_list() == [-10.0]
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_2",
    ).to_list() == [0.0]
    assert fdf.get_column(
        "val__time_reversal_asymmetry_statistic__lag_3",
    ).to_list() == [0.0]


def test_number_peaks():
    df = pl.DataFrame(
        {
            "id": ["a"] * 14,
            "val": [0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
        },
    ).lazy()
    opts = ExtractionSettings(
        grouping_col="id",
        feature_setting=FeatureSetting.Efficient,
        value_cols=["val"],
    )
    fdf = extract_features(df, opts)
    fdf = fdf.sort("id")
    assert fdf.get_column("val__number_peaks__n_1").to_list() == [2.0]
    assert fdf.get_column("val__number_peaks__n_3").to_list() == [1.0]
    assert fdf.get_column("val__number_peaks__n_5").to_list() == [0.0]
