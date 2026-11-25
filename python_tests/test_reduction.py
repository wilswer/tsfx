"""Feature reduction (``ExtractionSettings.reduction``) from Python.

The selection itself is tested in Rust (``src/reduction/``) and fuzzed in
``test_fuzz_reduction.py``; these tests check the Python API: the settings
and their validation, that extraction applies the reduction before scaling,
the report, and the warning. The series are deterministic and built with
``math`` (no numpy).
"""

import math
import warnings
from collections.abc import Sequence

import polars as pl
import pytest
from tsfx import (
    ExtractionSettings,
    FeatureSetting,
    ReductionSetting,
    ScalerSetting,
    extract_features,
    reduction_report,
)


def _value(g: int, t: int) -> float:
    """Return a varied, deterministic series value: group `g`, time `t`."""
    return (
        math.sin(0.05 * (g + 3) * t + g) * (1 + g % 7)
        + 0.01 * g * t
        + math.cos(1.7 * t * (g % 5 + 1))
    )


def _series(
    n_groups: int = 60,
    length: int = 100,
    value_cols: Sequence[str] = ("v",),
    short_groups: int = 0,
) -> pl.LazyFrame:
    """Build `n_groups` series, the first `short_groups` with only 2 values.

    Every value column holds the same values.
    """
    ids: list[str] = []
    values: list[float] = []
    for g in range(n_groups):
        for t in range(2 if g < short_groups else length):
            ids.append(f"g{g:03}")
            values.append(_value(g, t))
    return pl.DataFrame({"id": ids, **dict.fromkeys(value_cols, values)}).lazy()


def _settings(
    value_cols: Sequence[str] = ("v",),
    feature_setting: FeatureSetting = FeatureSetting.Efficient,
    scaling: ScalerSetting | None = None,
    reduction: ReductionSetting | None = None,
) -> ExtractionSettings:
    return ExtractionSettings(
        grouping_cols=["id"],
        value_cols=list(value_cols),
        feature_setting=feature_setting,
        scaling=scaling,
        reduction=reduction,
    )


def test_default_is_no_reduction() -> None:
    default = ExtractionSettings(["id"], ["v"], FeatureSetting.Efficient)
    full = extract_features(_series(), default).sort("id")
    assert full.equals(
        extract_features(_series(), _settings(reduction=None)).sort("id"),
    )
    report = reduction_report(_series(), _settings(reduction=ReductionSetting()))
    assert full.columns == ["id", *report["feature"]]


def test_setting_defaults() -> None:
    r = ReductionSetting()
    assert (
        r.max_missing,
        r.max_top_share,
        r.max_correlation,
        r.noise_sensitivity,
        r.noise_neighbors,
        r.max_samples,
        r.seed,
    ) == (0.05, 0.95, 0.95, 0.99, 10, 5000, 0)
    assert r == ReductionSetting()
    assert ReductionSetting(noise_sensitivity=None).noise_sensitivity is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_missing": 1.0},
        {"max_missing": -0.1},
        {"max_top_share": 1.0},
        {"max_correlation": 0.0},
        {"noise_sensitivity": 0.9995},
        {"noise_sensitivity": 0.4},
        {"noise_sensitivity": math.nan},
        {"noise_neighbors": 0},
        {"max_samples": 10},
    ],
)
def test_invalid_setting(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="ReductionSetting"):
        ReductionSetting(**kwargs)  # type: ignore[arg-type]


def test_bare_class_rejected() -> None:
    with pytest.raises(TypeError):
        ExtractionSettings(
            ["id"],
            ["v"],
            FeatureSetting.Minimal,
            reduction=ReductionSetting,  # type: ignore[arg-type]
        )


def test_output_matches_report() -> None:
    settings = _settings(reduction=ReductionSetting())
    out = extract_features(_series(), settings)
    report = reduction_report(_series(), settings)
    kept = report.filter(pl.col("kept"))["feature"].to_list()
    assert out.columns == ["id", *kept]
    assert 0 < len(kept) < report.height
    assert report.columns == [
        "feature",
        "kept",
        "dropped_by",
        "missing_share",
        "top_share",
        "duplicate_of",
        "correlation",
        "noise_score",
        "noise_cutoff",
    ]
    assert set(report["dropped_by"].drop_nulls()) <= {
        "missing",
        "one_value",
        "duplicate",
        "noise",
    }


def test_reduction_then_scaling() -> None:
    reduced = extract_features(_series(), _settings(reduction=ReductionSetting()))
    both = extract_features(
        _series(),
        _settings(reduction=ReductionSetting(), scaling=ScalerSetting.Standard()),
    )
    assert both.columns == reduced.columns
    for name in both.columns[1:]:
        finite = both[name].filter(both[name].is_finite())
        mean = finite.mean()
        assert isinstance(mean, float), name
        assert abs(mean) < 1e-9, name


def test_too_few_rows() -> None:
    with pytest.raises(ValueError, match="at least"):
        extract_features(
            _series(n_groups=3),
            _settings(reduction=ReductionSetting()),
        )


def test_report_requires_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        reduction_report(_series(), _settings())


def test_warning_with_few_features() -> None:
    settings = _settings(
        feature_setting=FeatureSetting.Minimal,
        reduction=ReductionSetting(),
    )
    with pytest.warns(UserWarning, match="noise_sensitivity"):
        extract_features(_series(), settings)
    with pytest.warns(UserWarning, match="noise_sensitivity"):
        reduction_report(_series(), settings)


def test_warning_with_few_rows() -> None:
    settings = _settings(reduction=ReductionSetting())
    with pytest.warns(UserWarning, match="rows"):
        extract_features(_series(n_groups=20), settings)


def test_no_warning_with_many_features() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        extract_features(_series(), _settings(reduction=ReductionSetting()))


def test_duplicates_across_value_columns() -> None:
    # "b" holds the same values as "a": every b__ feature copies an a__ one,
    # or went before step 3 with it
    report = reduction_report(
        _series(value_cols=("a", "b")),
        _settings(
            value_cols=("a", "b"),
            reduction=ReductionSetting(noise_sensitivity=None),
        ),
    )
    b = report.filter(pl.col("feature").str.starts_with("b__"))
    assert b.height > 0
    assert b["dropped_by"].is_in(["duplicate", "one_value", "missing"]).all()
    copies = b.filter(pl.col("dropped_by") == "duplicate")
    assert copies["duplicate_of"].str.starts_with("a__").all()


def test_short_series_nan_features_dropped_as_missing() -> None:
    # 10 of the 40 groups have only 2 values, so some features are NaN there
    report = reduction_report(
        _series(short_groups=10),
        _settings(reduction=ReductionSetting()),
    )
    missing = report.filter(pl.col("missing_share") > 0.05)
    assert missing.height > 0
    assert (missing["dropped_by"] == "missing").all()


def test_streaming_same_output() -> None:
    settings = _settings(reduction=ReductionSetting())
    streamed = extract_features(_series(), settings, streaming=True)
    assert streamed.sort("id").equals(extract_features(_series(), settings).sort("id"))
