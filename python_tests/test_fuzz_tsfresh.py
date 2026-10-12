"""Differential fuzzing against tsfresh, the specification.

Hypothesis generates series (ties, NaN, ±inf, signed zeros, long series for
numpy's pairwise summation, series that hit thresholds exactly, ...), and
every feature of the parity config is compared with tsfresh's feature
calculator. One test per feature, so each
gets its own shrunk counterexample.

Opt-in (marker ``fuzz``, needs hypothesis and tsfresh):

    pytest -m fuzz                              # 200 examples per feature
    HYPOTHESIS_PROFILE=ci pytest -m fuzz        # 10 examples per feature

Features tsfresh computes with numpy must match bit for bit; the families in
``TOLERANT`` (pandas, scipy or BLAS arithmetic tsfx does not reproduce) are
compared with a relative and absolute tolerance of 1e-9. The documented
deliberate deviations are excluded in ``_deviation``, and columns tsfresh has
no counterpart for in ``_tsfx_only``. Calculators get what tsfresh's
``extract_features`` passes them: a numpy array, or a pandas Series where
the calculator asks for one.
"""

import itertools
import math
import re
import warnings

import polars as pl
import pytest
from tsfx import ExtractionSettings, FeatureSetting, extract_features

hypothesis = pytest.importorskip("hypothesis")
fc = pytest.importorskip("tsfresh.feature_extraction.feature_calculators")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
st = hypothesis.strategies

pytestmark = pytest.mark.fuzz

CONFIG = "./python_tests/data/.tsfx-config-tsfresh-parity.toml"

# tsfresh computes these with pandas (skew, kurtosis, chunk aggregates),
# scipy (linregress), BLAS (np.dot) or a long chain of numpy operations
TOLERANT = {
    "skewness",
    "kurtosis",
    "linear_trend",
    "agg_linear_trend",
    "absolute_energy",
    "cid_ce",
    "sample_entropy",
    # statsmodels (np.correlate / BLAS dot, LAPACK SVD least squares)
    "agg_autocorrelation",
    "partial_autocorrelation",
    "ar_coefficient",
    "augmented_dickey_fuller",
}

# tsfx names that tsfresh spells differently
TSFRESH_NAMES = {
    "absolute_energy": "abs_energy",
    "mean_absolute_change": "mean_abs_change",
    "percentage_of_reoccurring_values_to_all_datapoints": (
        "percentage_of_reoccurring_datapoints_to_all_datapoints"
    ),
}

# The largest partial_autocorrelation lag in the parity config
PACF_MAX_LAG = 9

SPECIAL = [0.0, -0.0, 1.0, -1.0, math.nan, math.inf, -math.inf]


@st.composite
def threshold_series(draw: st.DrawFn) -> list[float]:
    """A block of decimals followed by sign-flipped permutations of it, so
    thresholds are hit exactly in real arithmetic (cumulative shares of |x|
    at 1/k, mean and median in exact ratios to the range) and only
    floating-point rounding decides comparisons such as index_mass_quantile's
    and symmetry_looking's."""
    scale = draw(st.sampled_from([10, 100, 4, 8, 3, 7]))
    block = draw(
        st.lists(
            st.integers(-99, 99).map(lambda i: i / scale), min_size=1, max_size=12
        ),
    )
    out = []
    for _ in range(draw(st.sampled_from([2, 5, 10]))):
        signs = draw(st.lists(st.booleans(), min_size=len(block), max_size=len(block)))
        copy = draw(st.permutations(block))
        out += [-v if flip else v for v, flip in zip(copy, signs, strict=True)]
    return out


@st.composite
def series(draw: st.DrawFn) -> list[float]:
    """A series with the edge cases features tend to get wrong."""
    if draw(st.integers(0, 3)) == 0:
        return draw(threshold_series())
    value = st.one_of(
        st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        st.integers(-5, 5).map(float),
        st.integers(-1000, 1000).map(lambda i: i / 10),
        st.sampled_from(SPECIAL),
    )
    clean = draw(st.booleans())
    if clean:
        value = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
    long = draw(st.integers(0, 9)) == 0
    return draw(st.lists(value, min_size=1, max_size=300 if long else 60))


def _tsfx(values: list[float]) -> dict:
    df = pl.DataFrame({"id": ["a"] * len(values), "val": values})
    settings = ExtractionSettings(
        grouping_cols=["id"],
        value_cols=["val"],
        feature_setting=FeatureSetting.Comprehensive,
        config_path=CONFIG,
    )
    row = extract_features(df.lazy(), settings).row(0, named=True)
    return {k.removeprefix("val__"): v for k, v in row.items() if k.startswith("val__")}


def _combiner(fn: object, x: object, param: dict) -> float:
    """The single value a tsfresh combiner returns for one parameter set."""
    ((_, value),) = list(fn(_input(fn, x), [param]))
    return value


def _input(fn: object, x: object) -> object:
    """What tsfresh's extraction passes: a Series where the calculator asks
    for one, else the numpy array."""
    if getattr(fn, "input", None) == "pd.Series":
        return pd.Series(x)
    return x


def _tsfresh(x: object, column: str) -> float:
    """tsfresh's value for a tsfx output column."""
    m = re.fullmatch(
        r"(agg_)?linear_trend_(\w+?)(?:__chunk_size_(\d+)__agg_(\w+))?", column
    )
    if m and m.group(1):
        param = {"attr": m.group(2), "chunk_len": int(m.group(3)), "f_agg": m.group(4)}
        return _combiner(fc.agg_linear_trend, x, param)
    if m:
        return _combiner(fc.linear_trend, x, {"attr": m.group(2)})
    name, _, rest = column.partition("__")
    if name == "agg_autocorrelation":
        f_agg, maxlag = re.fullmatch(r"f_agg_(\w+)__maxlag_(\d+)", rest).groups()
        param = {"f_agg": f_agg, "maxlag": int(maxlag)}
        return _combiner(fc.agg_autocorrelation, x, param)
    params = dict(p.split("_", 1) for p in rest.split("__")) if rest else {}
    if name == "partial_autocorrelation":
        # tsfresh's lag 0 depends on the largest lag requested alongside it
        lags = [{"lag": int(params["lag"])}, {"lag": PACF_MAX_LAG}]
        return list(fc.partial_autocorrelation(x, lags))[0][1]
    if name == "ar_coefficient":
        param = {"coeff": int(params["coeff"]), "k": int(params["k"])}
        return _combiner(fc.ar_coefficient, x, param)
    if name == "augmented_dickey_fuller":
        autolag = None if params["autolag"] == "None" else params["autolag"]
        param = {"attr": params["attr"], "autolag": autolag}
        return _combiner(fc.augmented_dickey_fuller, x, param)
    if name == "symmetry_looking":
        return _combiner(fc.symmetry_looking, x, {"r": float(params["r"])})
    if name == "index_mass_quantile":
        return _combiner(fc.index_mass_quantile, x, {"q": float(params["q"])})
    if name == "cid_ce":
        return fc.cid_ce(x, params["normalize"] == "t")
    if name == "range_count":
        return fc.range_count(x, float(params["min"]), float(params["max"]))
    if name == "mean_n_absolute_max":
        return fc.mean_n_absolute_max(x, int(params["n"]))
    fn = getattr(fc, TSFRESH_NAMES.get(name, name))
    if params:
        ((key, value),) = params.items()
        cast = int if key in ("lag", "n") else float
        return fn(_input(fn, x), cast(value))
    return fn(_input(fn, x))


def _tsfx_only(column: str) -> bool:
    """Columns tsfresh cannot compute: agg_linear_trend's median aggregator
    is a tsfx addition (tsfresh's extraction hands it a numpy array, which
    has no ``median`` method)."""
    return column.endswith("__agg_median")


def _deviation(feature: str, x: list[float]) -> bool:
    """Documented deliberate deviations from tsfresh (see the parity tests)."""
    if feature == "sample_entropy":
        return any(math.isinf(v) for v in x)
    if feature in ("sum_of_reoccurring_values", "sum_of_reoccurring_data_points"):
        # a NaN or ±inf that occurs only once is left out by tsfx
        odd = [v for v in x if not math.isfinite(v)]
        return any(
            sum(1 for w in odd if (math.isnan(v) and math.isnan(w)) or v == w) == 1
            for v in odd
        )
    if feature == "ar_coefficient":
        # tsfresh raises on NaN / ±inf (MissingDataError); tsfx gives NaN
        return not all(math.isfinite(v) for v in x)
    if feature == "augmented_dickey_fuller":
        return _adf_rounding_decides(x)
    return False


def _adf_rounding_decides(values: list[float]) -> bool:
    """Whether rounding noise decides statsmodels' ADF result: the largest
    regression fits (nearly) exactly (tsfx takes the residual variance as 0
    below a relative residual of 1e-10; between that and 1e-5 both compute
    the same, but the residual's rounding noise can exceed the comparison
    tolerance), or an extra lag is collinear with the others, so the
    information criteria of two candidates tie (tsfx takes the smaller
    lag)."""
    x = np.array(values, dtype=float)
    n = len(x)
    if n < 4 or not np.isfinite(x).all():
        return False
    maxlag = min(math.ceil(12 * (n / 100) ** 0.25), n // 2 - 2)
    if maxlag < 0:
        return False
    xd = np.diff(x)
    nobs = n - 1 - maxlag
    cols = [np.ones(nobs), x[maxlag : maxlag + nobs]]
    cols += [xd[maxlag - lag : maxlag - lag + nobs] for lag in range(1, maxlag + 1)]
    design = np.column_stack(cols)
    ranks = [np.linalg.matrix_rank(design[:, : lags + 2]) for lags in range(maxlag + 1)]
    if any(a == b for a, b in itertools.pairwise(ranks)):
        return True
    y = xd[maxlag:]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    terms = np.abs(y) + np.abs(design * beta).sum(axis=1)
    rms = np.sqrt(np.mean((y - design @ beta) ** 2))
    return bool(rms <= 1e-5 * terms.max())


def _same(a: float, b: float, tolerant: bool) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if tolerant:
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _feature(column: str) -> str:
    """The feature a column belongs to, without parameters or attribute."""
    return re.sub(r"_(intercept|slope)$", "", re.sub(r"__.*$", "", column))


FEATURES = sorted({_feature(c) for c in _tsfx([1.0, 2.0, 3.0])})


@pytest.mark.parametrize("feature", FEATURES)
def test_matches_tsfresh(feature: str) -> None:
    tolerant = feature in TOLERANT

    @hypothesis.given(series())
    def check(values: list[float]) -> None:
        hypothesis.assume(not _deviation(feature, values))
        ours = _tsfx(values)
        x = np.array(values, dtype=float)
        mismatches = []
        for column, value in ours.items():
            if _feature(column) != feature or _tsfx_only(column):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = _tsfresh(x, column)
            if not _same(value, expected, tolerant):
                mismatches.append(f"{column}: tsfx {value!r}, tsfresh {expected!r}")
        assert not mismatches, "\n".join(mismatches)

    check()
