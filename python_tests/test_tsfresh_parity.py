"""Parity tests against tsfresh's own feature calculator test cases.

Input/expected pairs are copied from tsfresh v0.21.2,
tests/units/feature_extraction/test_feature_calculations.py
(https://github.com/blue-yonder/tsfresh), used under the MIT licence:

    Copyright (c) 2016 Maximilian Christ, Blue Yonder GmbH

Cases tsfresh runs on an empty series are left out: TSFX computes features
per group, and a group cannot be empty.

Any deviation from tsfresh is treated as a bug. Cases that still fail are
marked with ``_bug`` (strict xfail), so the marker must be removed once the
fix lands.
"""

import math

import polars as pl
import pytest
from tsfx import ExtractionSettings, FeatureSetting, extract_features

CONFIG_PATH = "./python_tests/data/.tsfx-config-tsfresh-parity.toml"


def _bug(*case: object, reason: str):
    """Mark a parity case that fails because of a known TSFX bug."""
    return pytest.param(
        *case,
        marks=pytest.mark.xfail(strict=True, reason=f"parity bug: {reason}"),
    )


def _features(values: list[float]) -> dict:
    """Extract the parity features for a single series."""
    df = pl.DataFrame(
        {"id": ["a"] * len(values), "val": [float(v) for v in values]},
    ).lazy()
    opts = ExtractionSettings(
        grouping_cols=["id"],
        feature_setting=FeatureSetting.Comprehensive,
        value_cols=["val"],
        config_path=CONFIG_PATH,
    )
    fdf = extract_features(df, opts)
    assert fdf.shape[0] == 1
    return fdf.row(0, named=True)


def _assert_feature(values: list[float], column: str, expected: float) -> None:
    features = _features(values)
    assert column in features, f"column {column!r} not extracted"
    result = features[column]
    if math.isnan(expected):
        assert result is not None and math.isnan(result), result
    else:
        # tsfresh uses assertEqual / assertAlmostEqual (7 decimal places)
        assert result == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, -1, -1], 1),
        ([1, 2, -2, -1], 1.58113883008),
    ],
)
def test_standard_deviation(values, expected):
    _assert_feature(values, "val__standard_deviation", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, -1, -1], 1),
        ([1, 2, -2, -1], 2.5),
    ],
)
def test_variance(values, expected):
    _assert_feature(values, "val__variance", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, -1, -1], math.nan),
        ([1, 2, -3, -1], -7.681145747868608),
        ([1, 2, 4, -1], 1.2018504251546631),
    ],
)
def test_variation_coefficient(values, expected):
    _assert_feature(values, "val__variation_coefficient", expected)


# tsfresh returns booleans for the next two features; TSFX returns 1.0 / 0.0.
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-1, -1, 1, 1, 1], 0),
        ([-1, -1, 1, 1, 2], 1),
    ],
)
def test_variance_larger_than_standard_deviation(values, expected):
    _assert_feature(values, "val__variance_larger_than_standard_deviation", expected)


@pytest.mark.parametrize(
    ("values", "r", "expected"),
    [
        ([1, 1, 1, 1], "0.00", 0),
        ([-1, -1, 1, 1], "0.00", 1),
        ([-1, -1, 1, 1], "0.25", 1),
        ([-1, -1, 1, 1], "0.30", 1),
        ([-1, -1, 1, 1], "0.50", 0),
    ],
)
def test_large_standard_deviation(values, r, expected):
    _assert_feature(values, f"val__large_standard_deviation__r_{r}", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, 1, 2, 2, 2], 0),
        ([1, 1, 1, 2, 2], 0.6085806194501855),
        ([1, 1, 1], 0),
        ([1, 1], math.nan),
    ],
)
def test_skewness(values, expected):
    _assert_feature(values, "val__skewness", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 1, 1, 2, 2], -3.333333333333333),
        ([1, 1, 1, 1], 0),
        ([1, 1, 1], math.nan),
    ],
)
def test_kurtosis(values, expected):
    _assert_feature(values, "val__kurtosis", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 1, 1, 2, 2], 1.4832396974191), ([0], 0), ([1], 1), ([-1], 1)],
)
def test_root_mean_square(values, expected):
    _assert_feature(values, "val__root_mean_square", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([-2, 2, 5], 3.5), ([1, 2, -1], 2)],
)
def test_mean_absolute_change(values, expected):
    # tsfresh name: mean_abs_change
    _assert_feature(values, "val__mean_absolute_change", expected)


_RATIO_X = [0, 1] * 10 + [10, 20, -30]


@pytest.mark.parametrize(
    ("r", "expected"),
    [
        ("1.00", 3.0 / len(_RATIO_X)),
        ("2.00", 2.0 / len(_RATIO_X)),
        ("3.00", 1.0 / len(_RATIO_X)),
        ("20.00", 0),
    ],
)
def test_ratio_beyond_r_sigma(r, expected):
    _assert_feature(_RATIO_X, f"val__ratio_beyond_r_sigma__r_{r}", expected)


@pytest.mark.parametrize(
    ("values", "normalize", "expected"),
    [
        ([1, 1, 1], "t", 0),
        ([0, 4], "t", 2),
        ([100, 104], "t", 2),
        ([1, 1, 1], "f", 0),
        ([0.5, 3.5, 7.5], "f", 5),
        ([-4.33, -1.33, 2.67], "f", 5),
    ],
)
def test_cid_ce(values, normalize, expected):
    _assert_feature(values, f"val__cid_ce__normalize_{normalize}", expected)


@pytest.mark.parametrize(("values", "expected"), [([-5, 0, 1], 5), ([0], 0)])
def test_absolute_maximum(values, expected):
    _assert_feature(values, "val__absolute_maximum", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 1, 1, 1, 2, 1], 2), ([1, -1, 1, -1], 6), ([1], 0)],
)
def test_absolute_sum_of_changes(values, expected):
    _assert_feature(values, "val__absolute_sum_of_changes", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 1, 2, 1, 2], 3), ([1, 1, 1, 1, 1, 2], 1), ([1, 1, 1, 1, 1], 0)],
)
def test_count_above_mean(values, expected):
    _assert_feature(values, "val__count_above_mean", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 1, 2, 1, 2], 3), ([1, 1, 1, 1, 1, 2], 5), ([1, 1, 1, 1, 1], 0)],
)
def test_count_below_mean(values, expected):
    _assert_feature(values, "val__count_below_mean", expected)


# tsfresh also tests t = nan / ±inf, which the config cannot name as a column
# parameter; those cases are left out.
@pytest.mark.parametrize(
    ("values", "t", "expected"),
    [
        ([1] * 10, "1.0", 1),
        (list(range(10)), "0.0", 1),
        (list(range(10)), "5.0", 0.5),
        ([0.1, 0.2, 0.3] * 3, "0.2", 2 / 3),
        ([math.nan, 0, 1] * 3, "0.0", 2 / 3),
        ([-math.inf, 0, 1] * 3, "0.0", 2 / 3),
        ([math.inf, 0, 1] * 3, "0.0", 1),
    ],
)
def test_count_above(values, t, expected):
    _assert_feature(values, f"val__count_above__t_{t}", expected)


@pytest.mark.parametrize(
    ("values", "t", "expected"),
    [
        ([1] * 10, "1.0", 1),
        (list(range(10)), "0.0", 1 / 10),
        (list(range(10)), "5.0", 6 / 10),
        ([0.1, 0.2, 0.3] * 3, "0.2", 2 / 3),
        ([math.nan, 0, 1] * 3, "0.0", 1 / 3),
        ([-math.inf, 0, 1] * 3, "0.0", 2 / 3),
        ([math.inf, 0, 1] * 3, "0.0", 1 / 3),
    ],
)
def test_count_below(values, t, expected):
    _assert_feature(values, f"val__count_below__t_{t}", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 2, 1], 0.2),
        ([1, 2, 1, 1, 2], 0.2),
        ([2, 1, 1, 1, 1], 0.0),
        ([1, 1, 1, 1, 1], 0.0),
        ([1], 0.0),
    ],
)
def test_first_location_of_maximum(values, expected):
    _assert_feature(values, "val__first_location_of_maximum", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 2, 1], 0.0),
        ([2, 2, 1, 2, 2], 0.4),
        ([2, 1, 1, 1, 2], 0.2),
        ([1, 1, 1, 1, 1], 0.0),
        ([1], 0.0),
    ],
)
def test_first_location_of_minimum(values, expected):
    _assert_feature(values, "val__first_location_of_minimum", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 2, 1], 0.8),
        ([1, 2, 1, 1, 2], 1.0),
        ([2, 1, 1, 1, 1], 0.2),
        ([1, 1, 1, 1, 1], 1.0),
        ([1], 1.0),
    ],
)
def test_last_location_of_maximum(values, expected):
    _assert_feature(values, "val__last_location_of_maximum", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 2, 1], 1.0),
        ([1, 2, 1, 2, 2], 0.6),
        ([2, 1, 1, 1, 2], 0.8),
        ([1, 1, 1, 1, 1], 1.0),
        ([1], 1.0),
    ],
)
def test_last_location_of_minimum(values, expected):
    _assert_feature(values, "val__last_location_of_minimum", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 2, 1, 2, 2, 1], 2),
        ([1, 2, 3, 4, 5, 6], 3),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 1], 1),
    ],
)
def test_longest_strike_above_mean(values, expected):
    _assert_feature(values, "val__longest_strike_above_mean", expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 1, 1, 1, 2, 2, 2], 3),
        ([1, 2, 3, 4, 5, 6], 3),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 1], 1),
    ],
)
def test_longest_strike_below_mean(values, expected):
    _assert_feature(values, "val__longest_strike_below_mean", expected)


@pytest.mark.parametrize(
    ("values", "lag", "expected"),
    [
        ([1, 2, 1, 2, 1, 2], 1, -1),
        ([1, 2, 1, 2, 1, 2], 2, 1),
        ([1, 2, 1, 2, 1, 2], 3, -1),
        ([1, 2, 1, 2, 1, 2], 4, 1),
        ([0, 1, 2, 0, 1, 2], 2, -0.75),
        ([1, 2, 1, 2, 1, 2], 200, math.nan),
        ([math.nan], 0, math.nan),
        ([1], 0, math.nan),
    ],
)
def test_autocorrelation(values, lag, expected):
    _assert_feature(values, f"val__autocorrelation__lag_{lag}", expected)


_SAMPLE_ENTROPY_RANDOM = [
    1, 4, 5, 1, 7, 3, 1, 2, 5, 8, 9, 7, 3, 7, 9, 5, 4, 3, 9, 1,
    2, 3, 4, 2, 9, 6, 7, 4, 9, 2, 9, 9, 6, 5, 1, 3, 8, 1, 5, 3,
    8, 4, 1, 2, 2, 1, 6, 5, 3, 6, 5, 4, 8, 9, 6, 7, 5, 3, 2, 5,
    4, 2, 5, 1, 6, 5, 3, 5, 6, 7, 8, 5, 2, 8, 6, 3, 8, 2, 7, 1,
    7, 3, 5, 6, 2, 1, 3, 7, 3, 5, 3, 7, 6, 7, 7, 2, 3, 1, 7, 8,
]  # fmt: skip


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (_SAMPLE_ENTROPY_RANDOM, 2.38262780),
        ([1] * 10, 0.25131442),
        ([1, 1, 2, 1, 1, 1, 1, 1, 1, 1], 0.74193734),
        ([1, 1, 1, 2, 1, 1, 1, 1, 1, 1], 0.74193734),
        ([1, -1, 1, -1, 1, -1], 0.69314718),
        ([1, -1, 1, math.nan, 1, -1], math.nan),
        (list(range(1000)), 0.0010314596066622707),
    ],
)
def test_sample_entropy(values, expected):
    _assert_feature(values, "val__sample_entropy", expected)


# --- Generated vectors -------------------------------------------------------
# tsfresh's own tests have no case that tells ddof=0 from ddof=1 for these
# features, so the expected values below were generated by running the
# tsfresh v0.21.2 feature calculators on the given input (call in comment).


# agg_linear_trend(pd.Series(x), [{"attr": a, "chunk_len": 3, "f_agg": "var"}])
# tsfresh aggregates chunks with pandas' Series.var, i.e. ddof=1.
@pytest.mark.parametrize(
    ("values", "attr", "expected"),
    [
        (list(range(9)), "intercept", 1.0),
        (list(range(9)), "slope", 0.0),
        ([0, 1, 2, 0, 2, 4, 0, 3, 6], "intercept", 0.666666666666667),
        ([0, 1, 2, 0, 2, 4, 0, 3, 6], "slope", 4.0),
    ],
)
def test_agg_linear_trend_var_generated(values, attr, expected):
    _assert_feature(
        values,
        f"val__agg_linear_trend_{attr}__chunk_size_3__agg_var",
        expected,
    )


# ratio_beyond_r_sigma(np.array([0, 0, 0, 1.0]), r=1.5)
@pytest.mark.parametrize(
    ("values", "r", "expected"),
    [([0, 0, 0, 1], "1.50", 0.25)],
)
def test_ratio_beyond_r_sigma_generated(values, r, expected):
    _assert_feature(values, f"val__ratio_beyond_r_sigma__r_{r}", expected)


# sample_entropy(np.array(x))
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [0.5, 0.7, 0.5, 1.1, -1.0, 1.0, -1.2, 0.2, -1.6, 0.3, -1.7, 0.2],
            0.6931471805599453,
        ),
    ],
)
def test_sample_entropy_generated(values, expected):
    _assert_feature(values, "val__sample_entropy", expected)


# skewness(np.array([0.1, 0.1, 0.1])) with pandas 3.0.2: m2 is ~6e-34 from
# rounding, which pandas' tolerance treats as 0.
@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.1, 0.1, 0.1], 0.0)],
)
def test_skewness_near_constant_generated(values, expected):
    _assert_feature(values, "val__skewness", expected)


# has_duplicate(np.array(x)) with numpy 2.4.4; np.unique merges all NaNs into
# one value (numpy >= 1.21).
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, math.nan, 3, 4, 2, 6], 0),
        ([1, math.nan, math.nan, 2], 1),
    ],
)
def test_has_duplicate_nan_generated(values, expected):
    _assert_feature(values, "val__has_duplicate", expected)


# ratio_value_number_to_time_series_length(np.array(x)) with numpy 2.4.4;
# np.unique merges all NaNs into one value (numpy >= 1.21).
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, math.nan, 3, 4, 2, 6], 1.0),
        ([1, math.nan, math.nan, 2], 0.75),
        ([math.nan, math.nan, 1, 1], 0.5),
    ],
)
def test_ratio_value_number_to_time_series_length_nan_generated(values, expected):
    _assert_feature(
        values,
        "val__ratio_value_number_to_time_series_length",
        expected,
    )
