"""Differential fuzzing of feature scaling against scikit-learn's scalers.

``ScalerSetting.Standard`` is compared with ``StandardScaler`` and
``ScalerSetting.MinMax`` with ``MinMaxScaler``, both fitted (``fit_transform``)
on the unscaled feature matrix. scikit-learn rejects infinities, so the
reference is fitted with them as NaN and they are put back afterwards: tsfx
leaves non-finite values out of the fit and passes them through.

Opt-in (marker ``fuzz``, needs hypothesis, numpy and scikit-learn):

    pytest -m fuzz -k sklearn

Arbitrary feature matrices come from one-row groups, whose Minimal features
repeat the drawn values; the tests on Efficient features of real series are
slow and run at most 20 examples.

Non-finite entries must match exactly. ``MinMax`` does scikit-learn's
arithmetic and is compared up to a few ulps. ``Standard`` computes the mean
and variance with Polars rather than numpy, so it is compared with a
tolerance from the summation error bound, and features whose variance is so
close to scikit-learn's near-constant threshold that the two can decide
differently are skipped.
"""

import math
import warnings

import polars as pl
import pytest
from tsfx import ExtractionSettings, FeatureSetting, ScalerSetting, extract_features

hypothesis = pytest.importorskip("hypothesis")
np = pytest.importorskip("numpy")
preprocessing = pytest.importorskip("sklearn.preprocessing")
st = hypothesis.strategies

pytestmark = pytest.mark.fuzz

EPS = np.finfo(np.float64).eps
SPECIAL = [math.nan, math.inf, -math.inf, 0.0, -0.0]


@st.composite
def feature_column(draw: st.DrawFn, n: int) -> list[float]:
    """`n` values of one feature: arbitrary, small integers or near-constant."""
    # 1e150 keeps the squares in the variance finite
    finite = st.floats(-1e150, 1e150, allow_nan=False, allow_infinity=False)
    kind = draw(st.sampled_from(["any", "integers", "near_constant", "constant"]))
    if kind == "any":
        value = st.one_of(finite, st.sampled_from(SPECIAL))
        return draw(st.lists(value, min_size=n, max_size=n))
    if kind == "integers":
        value = st.one_of(st.integers(-5, 5).map(float), st.sampled_from(SPECIAL))
        return draw(st.lists(value, min_size=n, max_size=n))
    base = draw(st.floats(-1e12, 1e12, allow_nan=False, allow_infinity=False))
    if kind == "constant":
        return [base] * n
    # a few ulps apart, around scikit-learn's near-constant threshold
    steps = draw(st.lists(st.integers(-4, 4), min_size=n, max_size=n))
    return [base + k * math.ulp(base or 1.0) for k in steps]


@st.composite
def single_row_groups(draw: st.DrawFn) -> pl.DataFrame:
    """One row per group: Minimal features then repeat the drawn values."""
    n = draw(st.integers(1, 40))
    k = draw(st.integers(1, 3))
    columns = {f"v{j}": draw(feature_column(n)) for j in range(k)}
    return pl.DataFrame({"id": list(range(n)), **columns})


@st.composite
def series_groups(draw: st.DrawFn) -> pl.DataFrame:
    """A few series of a few values: a realistic Efficient feature matrix."""
    value = st.one_of(
        st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        st.integers(-5, 5).map(float),
        st.sampled_from(SPECIAL),
    )
    groups = draw(
        st.lists(st.lists(value, min_size=1, max_size=25), min_size=1, max_size=8)
    )
    return pl.DataFrame(
        {
            "id": [i for i, series in enumerate(groups) for _ in series],
            "v": [x for series in groups for x in series],
        },
        schema={"id": pl.Int64, "v": pl.Float64},
    )


min_max = st.tuples(
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    st.floats(1e-3, 1e6, allow_nan=False, allow_infinity=False),
).map(lambda lo_width: (lo_width[0], lo_width[0] + lo_width[1]))


def _extract(
    df: pl.DataFrame,
    feature_setting: FeatureSetting,
    scaling: ScalerSetting | None,
) -> pl.DataFrame:
    value_cols = [c for c in df.columns if c != "id"]
    settings = ExtractionSettings(["id"], value_cols, feature_setting, scaling=scaling)
    return extract_features(df.lazy(), settings).sort("id")


def _reference(X: "np.ndarray", scaler: object) -> "np.ndarray":
    """scikit-learn fitted on the finite values, non-finite passed through."""
    finite = np.isfinite(X)
    with warnings.catch_warnings():
        # all-NaN features
        warnings.simplefilter("ignore", RuntimeWarning)
        Y = scaler.fit_transform(np.where(finite, X, np.nan))
    return np.where(finite, Y, X)


def _check(
    df: pl.DataFrame,
    feature_setting: FeatureSetting,
    scaling: ScalerSetting,
) -> None:
    unscaled = _extract(df, feature_setting, None)
    scaled = _extract(df, feature_setting, scaling)
    assert scaled.columns == unscaled.columns
    assert scaled["id"].equals(unscaled["id"])
    assert all(dtype == pl.Float64 for dtype in scaled.drop("id").dtypes)

    X = unscaled.drop("id").to_numpy().astype(np.float64)
    got = scaled.drop("id").to_numpy()
    finite = np.isfinite(X)
    absX = np.where(finite, np.abs(X), 0.0)

    if isinstance(scaling, ScalerSetting.Standard):
        scaler = preprocessing.StandardScaler()
        expected = _reference(X, scaler)
        n = finite.sum(axis=0)
        var, mean = scaler.var_, scaler.mean_
        bound = n * EPS * var + (n * mean * EPS) ** 2
        # the two implementations may disagree whether this is constant
        ambiguous = (var > bound / 16) & (var < bound * 16)
        # rounding of the sum in the mean and of (x - mean)
        tol = 16 * np.maximum(n, 1) * EPS * absX.max(axis=0) / scaler.scale_
        tol = np.broadcast_to(np.where(ambiguous, np.inf, tol), X.shape)
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=scaling.feature_range)
        expected = _reference(X, scaler)
        tol = 4 * EPS * (absX * scaler.scale_ + np.abs(scaler.min_))

    # non-finite values: exactly as in the reference (NaN, ±inf)
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.array_equal(got[~finite], expected[~finite], equal_nan=True)
    both = finite & ~np.isnan(expected)
    with np.errstate(invalid="ignore"):  # inf - inf
        diff = np.abs(got - expected)
    bad = both & ~(diff <= tol + 1e-9 * np.abs(expected))
    assert not bad.any(), [
        (scaled.columns[1 + j], X[i, j], got[i, j], expected[i, j])
        for i, j in zip(*np.nonzero(bad), strict=True)
    ][:5]


@hypothesis.given(df=single_row_groups())
def test_standard_matches_sklearn(df: pl.DataFrame) -> None:
    _check(df, FeatureSetting.Minimal, ScalerSetting.Standard())


@hypothesis.given(df=single_row_groups(), feature_range=min_max)
def test_min_max_matches_sklearn(
    df: pl.DataFrame,
    feature_range: tuple[float, float],
) -> None:
    _check(df, FeatureSetting.Minimal, ScalerSetting.MinMax(feature_range))


# Efficient extraction takes most of a second: fewer examples
few = hypothesis.settings(max_examples=min(hypothesis.settings().max_examples, 20))


@few
@hypothesis.given(df=series_groups())
def test_standard_matches_sklearn_on_features(df: pl.DataFrame) -> None:
    _check(df, FeatureSetting.Efficient, ScalerSetting.Standard())


@few
@hypothesis.given(df=series_groups(), feature_range=min_max)
def test_min_max_matches_sklearn_on_features(
    df: pl.DataFrame,
    feature_range: tuple[float, float],
) -> None:
    _check(df, FeatureSetting.Efficient, ScalerSetting.MinMax(feature_range))
