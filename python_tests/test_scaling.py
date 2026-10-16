"""Scaling of the extracted features across groups (``ExtractionSettings.scaling``).

Hand-computed expectations; the differential fuzzing against scikit-learn is
in ``test_fuzz_sklearn.py``. One row per group makes ``v__sum_values`` (and
mean, minimum, ...) equal the input value, so a feature column is easy to
control here.
"""

import math

import polars as pl
import pytest
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    ScalerSetting,
    extract_features,
)


def _extract(
    values: list[float | None],
    scaling: object = None,
    streaming: bool = False,
) -> pl.DataFrame:
    """Minimal features of one single-row group per value, in input order."""
    df = pl.DataFrame(
        {"id": [f"g{i:03}" for i in range(len(values))], "v": values},
        schema={"id": pl.String, "v": pl.Float64},
    )
    settings = ExtractionSettings(
        grouping_cols=["id"],
        value_cols=["v"],
        feature_setting=FeatureSetting.Minimal,
        scaling=scaling,
    )
    return extract_features(df.lazy(), settings, streaming=streaming).sort("id")


def _same(got: list[float], expected: list[float]) -> bool:
    """Equal up to rounding, with NaN equal to NaN and infinities exact."""
    return len(got) == len(expected) and all(
        (math.isnan(g) and math.isnan(e))
        or (math.isinf(e) and g == e)
        or g == pytest.approx(e, rel=1e-12, abs=1e-12)
        for g, e in zip(got, expected, strict=True)
    )


VALUES = [1.0, 2.0, 3.0, 4.0]


def test_default_is_no_scaling() -> None:
    df = pl.DataFrame({"id": ["g000", "g001"], "v": [1.0, 5.0]}).lazy()
    settings = ExtractionSettings(["id"], ["v"], FeatureSetting.Minimal)
    unscaled = extract_features(df, settings).sort("id")
    assert unscaled["v__sum_values"].to_list() == [1.0, 5.0]
    assert unscaled["length"].dtype == pl.UInt32
    assert unscaled.equals(_extract([1.0, 5.0], scaling=None))


def test_standard() -> None:
    fdf = _extract(VALUES, ScalerSetting.Standard())
    mean = 2.5
    std = math.sqrt(sum((v - mean) ** 2 for v in VALUES) / len(VALUES))
    expected = [(v - mean) / std for v in VALUES]
    for name in ["v__sum_values", "v__mean", "v__minimum", "v__maximum"]:
        assert _same(fdf[name].to_list(), expected), name
    # constant features (a single value has length 1 and variance 0) map to 0
    for name in ["length", "v__variance", "v__standard_deviation"]:
        assert fdf[name].dtype == pl.Float64
        assert fdf[name].to_list() == [0.0] * len(VALUES), name
    assert fdf["id"].to_list() == ["g000", "g001", "g002", "g003"]


def test_min_max_default_range() -> None:
    fdf = _extract(VALUES, ScalerSetting.MinMax())
    assert _same(fdf["v__sum_values"].to_list(), [0.0, 1 / 3, 2 / 3, 1.0])
    # constant features map to the lower end of the range
    assert fdf["length"].to_list() == [0.0] * len(VALUES)


def test_min_max_custom_range() -> None:
    fdf = _extract(VALUES, ScalerSetting.MinMax(feature_range=(-1, 3)))
    assert _same(fdf["v__sum_values"].to_list(), [-1.0, 1 / 3, 5 / 3, 3.0])
    assert fdf["length"].to_list() == [-1.0] * len(VALUES)


def test_min_max_feature_range() -> None:
    assert ScalerSetting.MinMax().feature_range == (0.0, 1.0)
    assert ScalerSetting.MinMax((-2, 5)).feature_range == (-2.0, 5.0)


def test_variant_class_without_arguments() -> None:
    """``ScalerSetting.Standard`` works as ``ScalerSetting.Standard()``."""
    for cls in [ScalerSetting.Standard, ScalerSetting.MinMax]:
        assert _extract(VALUES, cls).equals(_extract(VALUES, cls()))
    assert ScalerSetting.Standard() == ScalerSetting.Standard()
    assert ScalerSetting.MinMax() != ScalerSetting.MinMax((0, 2))


@pytest.mark.parametrize(
    "scaling", [ScalerSetting.Standard(), ScalerSetting.MinMax((-1, 1))]
)
def test_streaming(scaling: ScalerSetting) -> None:
    assert _extract(VALUES, scaling, streaming=True).equals(_extract(VALUES, scaling))


def test_non_finite_ignored_and_kept() -> None:
    """NaN and ±inf are left out of the fit and pass through unchanged."""
    values = [1.0, math.nan, 3.0, math.inf, -math.inf]
    fdf = _extract(values, ScalerSetting.Standard())
    assert _same(
        fdf["v__sum_values"].to_list(), [-1.0, math.nan, 1.0, math.inf, -math.inf]
    )
    fdf = _extract(values, ScalerSetting.MinMax((-1, 1)))
    assert _same(
        fdf["v__sum_values"].to_list(), [-1.0, math.nan, 1.0, math.inf, -math.inf]
    )


@pytest.mark.parametrize(
    "scaling", [ScalerSetting.Standard(), ScalerSetting.MinMax((-1, 1))]
)
def test_feature_without_finite_values_stays_nan(scaling: ScalerSetting) -> None:
    fdf = _extract([math.nan, math.nan, math.inf], scaling)
    got = fdf["v__mean"].to_list()
    assert None not in got
    assert _same(got, [math.nan, math.nan, math.inf])


def test_null_group_gives_nan() -> None:
    """A group without values has NaN features, before and after scaling."""
    fdf = _extract([1.0, None, 3.0], ScalerSetting.Standard())
    assert _same(fdf["v__mean"].to_list(), [-1.0, math.nan, 1.0])


def test_near_constant_feature_is_constant() -> None:
    """As in scikit-learn, a variance within rounding error counts as 0."""
    x = 0.1
    values = [x, math.nextafter(x, 1.0), x]
    fdf = _extract(values, ScalerSetting.Standard())
    assert all(abs(v) < 1e-15 for v in fdf["v__sum_values"].to_list())
    # an actual spread is scaled, however small
    fdf = _extract([1e-100, 2e-100, 3e-100], ScalerSetting.Standard())
    assert _same(fdf["v__sum_values"].to_list(), [-math.sqrt(1.5), 0.0, math.sqrt(1.5)])


def test_key_columns_untouched() -> None:
    """Grouping and dynamic time columns are not scaled."""
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": ["2001-01-01", "2002-01-01", "2003-01-01"] * 2,
            "v": [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
        },
    ).lazy()

    def extract(scaling: ScalerSetting | None) -> pl.DataFrame:
        settings = ExtractionSettings(
            grouping_cols=["id"],
            value_cols=["v"],
            feature_setting=FeatureSetting.Minimal,
            dynamic_settings=DynamicGroupBySettings(
                time_col="time",
                every="1y",
                period="1y",
                offset="0y",
                datetime_format="%Y-%m-%d",
            ),
            scaling=scaling,
        )
        return extract_features(df, settings).sort("id", "time")

    fdf = extract(ScalerSetting.MinMax())
    unscaled = extract(None)
    assert fdf.columns == unscaled.columns
    assert fdf.select("id", "time").equals(unscaled.select("id", "time"))
    assert fdf["id"].dtype == pl.Int64
    assert _same(fdf["v__mean"].to_list(), [0.0, 0.125, 0.25, 0.375, 0.625, 1.0])


def test_several_value_columns_scaled_independently() -> None:
    df = pl.DataFrame(
        {"a": ["x", "y", "z"], "b": [0, 0, 1], "u": [1.0, 2.0, 3.0], "w": [0, 10, 40]},
    ).lazy()
    settings = ExtractionSettings(
        grouping_cols=["a", "b"],
        value_cols=["u", "w"],
        feature_setting=FeatureSetting.Minimal,
        scaling=ScalerSetting.MinMax(),
    )
    fdf = extract_features(df, settings).sort("a")
    assert fdf["a"].to_list() == ["x", "y", "z"]
    assert fdf["b"].to_list() == [0, 0, 1]
    assert _same(fdf["u__mean"].to_list(), [0.0, 0.5, 1.0])
    assert _same(fdf["w__mean"].to_list(), [0.0, 0.25, 1.0])


@pytest.mark.parametrize(
    "feature_range",
    [(1, 1), (2, 1), (0, math.inf), (-math.inf, 0), (math.nan, 1)],
)
def test_invalid_feature_range(feature_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="feature_range"):
        ExtractionSettings(
            ["id"],
            ["v"],
            FeatureSetting.Minimal,
            scaling=ScalerSetting.MinMax(feature_range=feature_range),
        )


@pytest.mark.parametrize("scaling", ["standard", 1, int, FeatureSetting.Minimal])
def test_invalid_scaling(scaling: object) -> None:
    with pytest.raises(TypeError):
        ExtractionSettings(["id"], ["v"], FeatureSetting.Minimal, scaling=scaling)
