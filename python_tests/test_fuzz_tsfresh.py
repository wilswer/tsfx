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
    # np.log (numpy's SIMD logarithm) differs from libm's in the last bit
    "binned_entropy",
    "permutation_entropy",
    "approximate_entropy",
    # LAPACK least squares (np.polyfit) and eigenvalues (np.roots)
    "friedrich_coefficients",
    "max_langevin_fixed_point",
}

# tsfresh raises for these where tsfx returns NaN (documented per feature)
RAISES_IS_NAN = {
    "binned_entropy",
    "lempel_ziv_complexity",
    "permutation_entropy",
    "approximate_entropy",
    "friedrich_coefficients",
    "max_langevin_fixed_point",
}

# tsfx names that tsfresh spells differently
TSFRESH_NAMES = {
    "absolute_energy": "abs_energy",
    "mean_absolute_change": "mean_abs_change",
    "percentage_of_reoccurring_values_to_all_datapoints": (
        "percentage_of_reoccurring_datapoints_to_all_datapoints"
    ),
}

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
    if name in RAISES_IS_NAN:
        try:
            return _tsfresh_multi(x, name, rest)
        except Exception:
            return math.nan
    params = dict(p.split("_", 1) for p in rest.split("__")) if rest else {}
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


def _tsfresh_multi(x: object, name: str, rest: str) -> float:
    """tsfresh's value for features whose parameter names contain "_"."""
    params = dict(
        re.fullmatch(r"(\w+?)_(-?[\d.]+)", p).groups() for p in rest.split("__")
    )
    if name == "binned_entropy":
        return fc.binned_entropy(x, int(params["max_bins"]))
    if name == "lempel_ziv_complexity":
        return fc.lempel_ziv_complexity(x, int(params["bins"]))
    if name == "permutation_entropy":
        return fc.permutation_entropy(x, int(params["tau"]), int(params["dimension"]))
    if name == "approximate_entropy":
        return fc.approximate_entropy(x, int(params["m"]), float(params["r"]))
    if name == "friedrich_coefficients":
        param = {k: int(v) for k, v in params.items()}
        return _combiner(fc.friedrich_coefficients, x, param)
    if name == "max_langevin_fixed_point":
        return fc.max_langevin_fixed_point(x, int(params["r"]), int(params["m"]))
    raise KeyError(name)


def _tsfx_only(column: str) -> bool:
    """Columns tsfresh cannot compute: agg_linear_trend's median aggregator
    is a tsfx addition (tsfresh's extraction hands it a numpy array, which
    has no ``median`` method)."""
    return column.endswith("__agg_median")


def _deviation(feature: str, x: list[float]) -> bool:
    """Documented deliberate deviations from tsfresh (see the parity tests)."""
    if feature == "sample_entropy":
        return any(math.isinf(v) for v in x)
    if feature == "permutation_entropy":
        # ties within a window: tsfresh's tie order depends on numpy's sort
        # for the CPU; tsfx ranks ties by position (windows up to 7 values)
        return any(
            x[i] == x[j] or (math.isnan(x[i]) and math.isnan(x[j]))
            for i in range(len(x))
            for j in range(i + 1, min(i + 7, len(x)))
        )
    if feature in ("sum_of_reoccurring_values", "sum_of_reoccurring_data_points"):
        # a NaN or ±inf that occurs only once is left out by tsfx
        odd = [v for v in x if not math.isfinite(v)]
        return any(
            sum(1 for w in odd if (math.isnan(v) and math.isnan(w)) or v == w) == 1
            for v in odd
        )
    return False


def _same(
    a: float, b: float, tolerant: bool, tol: tuple[float, float] = (1e-9, 1e-9)
) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if tolerant:
        return math.isclose(a, b, rel_tol=tol[0], abs_tol=tol[1])
    return a == b


LANGEVIN = ("friedrich_coefficients", "max_langevin_fixed_point")
# (m, r) of the Langevin entries in the parity config
LANGEVIN_FITS = [(2, 30), (3, 30), (3, 5), (2, 5)]


def _langevin_fit(x: object, m: int, r: int) -> tuple[float, object] | None:
    """Condition number of tsfresh's (column-scaled) polyfit and its
    coefficients, or None where tsfresh cannot fit."""
    df = pd.DataFrame({"signal": x[:-1], "delta": np.diff(x)})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["quantiles"] = pd.qcut(df.signal, r)
            groups = df.groupby("quantiles")
            means = pd.DataFrame(
                {"x": groups.signal.mean(), "y": groups.delta.mean()},
            ).dropna()
            lhs = np.vander(means.x.to_numpy() + 0.0, m + 1)
            lhs /= np.sqrt((lhs * lhs).sum(axis=0))
            s = np.linalg.svd(lhs, compute_uv=False)
            coeffs = np.polyfit(means.x, means.y, deg=m)
    except Exception:
        return None
    # no more bin averages than coefficients: the fit interpolates (or is
    # underdetermined), as ill-posed as an infinite condition number
    if len(means) <= m + 1 or s[-1] <= 0:
        return math.inf, coeffs
    return s[0] / s[-1], coeffs


def _ill_conditioned(feature: str, x: object) -> bool:
    """Langevin fits with at most m + 1 bin averages, or whose condition
    number exceeds 1e6, are not compared: np.polyfit / np.roots (and tsfx)
    are only accurate to about cond * eps there, so tsfresh's own result
    depends on its rounding."""
    if feature not in LANGEVIN:
        return False
    fits = [_langevin_fit(np.asarray(x, dtype=float), m, r) for m, r in LANGEVIN_FITS]
    return any(fit is not None and not fit[0] <= 1e6 for fit in fits)


def _tolerance(x: object, column: str) -> tuple[float, float]:
    """Relative and absolute tolerance for a tolerant column: 1e-9, for a
    polynomial coefficient relative to the largest coefficient of its fit
    (least squares is accurate relative to the whole solution vector)."""
    name, _, rest = column.partition("__")
    if name != "friedrich_coefficients":
        return 1e-9, 1e-9
    m, r = map(int, re.search(r"m_(\d+)__r_(\d+)", rest).groups())
    fit = _langevin_fit(x, m, r)
    if fit is None:
        return 1e-9, 1e-9
    return 1e-9, 1e-9 * max([1.0] + [abs(c) for c in fit[1] if math.isfinite(c)])


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
        hypothesis.assume(not _ill_conditioned(feature, values))
        ours = _tsfx(values)
        x = np.array(values, dtype=float)
        mismatches = []
        for column, value in ours.items():
            if _feature(column) != feature or _tsfx_only(column):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = _tsfresh(x, column)
                tol = _tolerance(x, column) if tolerant else (0.0, 0.0)
            if not _same(value, expected, tolerant, tol):
                mismatches.append(f"{column}: tsfx {value!r}, tsfresh {expected!r}")
        assert not mismatches, "\n".join(mismatches)

    check()
