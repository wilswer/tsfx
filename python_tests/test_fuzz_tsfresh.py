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
no counterpart for in ``_tsfx_only``. ``number_cwt_peaks`` is compared with
scipy's peak finder run on tsfx's portable arithmetic (``_number_cwt_peaks``),
because tied coefficients make the count depend on rounding. Calculators get
what tsfresh's
``extract_features`` passes them: a numpy array, or a pandas Series where
the calculator asks for one.
"""

import math
import re
import tomllib
import warnings

import polars as pl
import pytest
from tsfx import ExtractionSettings, FeatureSetting, extract_features

hypothesis = pytest.importorskip("hypothesis")
fc = pytest.importorskip("tsfresh.feature_extraction.feature_calculators")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
signal = pytest.importorskip("scipy.signal")
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
    "benford_correlation",
    "fft_aggregated",
    "fourier_entropy",
    "cwt_coefficients",
    # statsmodels (np.correlate / BLAS dot, LAPACK SVD least squares)
    "agg_autocorrelation",
    "partial_autocorrelation",
    "ar_coefficient",
    "augmented_dickey_fuller",
}

# FFT-based features, compared with tolerances that scale with the series
# (see _same_spectral)
SPECTRAL = {
    "fft_coefficient",
    "fft_aggregated",
    "spkt_welch_density",
    "fourier_entropy",
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


def _plain_convolve(a: object, v: object, mode: str = "same") -> object:
    """``np.convolve`` with each output a plain sequential sum, as tsfx."""
    a, v = (v, a) if len(v) > len(a) else (a, v)
    n1, n2 = len(a), len(v)
    full = []
    for k in range(n1 + n2 - 1):
        r = 0.0
        for j in range(max(0, k - n1 + 1), min(k, n2 - 1) + 1):
            r += a[k - j] * v[j]
        full.append(r)
    start = 0 if mode == "full" else (n2 - 1) // 2
    return np.array(full[start : start + (n1 + n2 - 1 if mode == "full" else n1)])


def _ricker(points: int, a: float) -> object:
    """tsfresh's ``_ricker`` with the C library's ``exp`` per value, as tsfx."""
    amp = 2 / (math.sqrt(3 * a) * math.pi**0.25)
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    return np.array(
        [amp * (1 - v * v / wsq) * math.exp(-v * v / (2 * wsq)) for v in vec]
    )


def _number_cwt_peaks(x: object, n: int) -> int:
    """tsfresh's ``number_cwt_peaks`` (scipy's ``find_peaks_cwt``) evaluated
    with tsfx's portable convolution and ``exp``: tied neighbouring
    coefficients are common (any series symmetric about a point between two
    samples), and which one wins is decided by rounding."""
    import scipy.signal._wavelets as wavelets
    from scipy.signal import find_peaks_cwt

    original = wavelets.convolve
    wavelets.convolve = _plain_convolve
    try:
        return len(find_peaks_cwt(x, np.arange(1, n + 1), wavelet=_ricker))
    finally:
        wavelets.convolve = original


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
    if name in GROUP_A:
        return GROUP_A[name](x, rest)
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
    if name == "number_cwt_peaks":
        return _number_cwt_peaks(x, int(params["n"]))
    if name == "cwt_coefficients":
        param = {
            "widths": tuple(int(w) for w in params["widths"].split("_")),
            "coeff": int(params["coeff"]),
            "w": int(params["w"]),
        }
        return _combiner(fc.cwt_coefficients, x, param)
    if name == "mean_n_absolute_max":
        return fc.mean_n_absolute_max(x, int(params["n"]))
    if name == "fft_coefficient":
        param = {"attr": params["attr"], "coeff": int(params["coeff"])}
        return _combiner(fc.fft_coefficient, x, param)
    if name == "fft_aggregated":
        return _combiner(fc.fft_aggregated, x, {"aggtype": params["aggtype"]})
    if name == "spkt_welch_density":
        return _combiner(fc.spkt_welch_density, x, {"coeff": int(params["coeff"])})
    if name == "fourier_entropy":
        return fc.fourier_entropy(x, int(params["bins"]))
    fn = getattr(fc, TSFRESH_NAMES.get(name, name))
    if params:
        ((key, value),) = params.items()
        cast = int if key in ("lag", "n") else float
        return fn(_input(fn, x), cast(value))
    return fn(_input(fn, x))


def _energy_ratio_by_chunks(x: object, rest: str) -> float:
    m = re.fullmatch(r"num_segments_(\d+)__segment_focus_(\d+)", rest)
    param = {"num_segments": int(m.group(1)), "segment_focus": int(m.group(2))}
    return _combiner(fc.energy_ratio_by_chunks, x, param)


def _change_quantiles(x: object, rest: str) -> float:
    m = re.fullmatch(r"ql_(.+)__qh_(.+)__isabs_(true|false)__f_agg_(\w+)", rest)
    ql, qh, isabs, f_agg = m.groups()
    return fc.change_quantiles(x, float(ql), float(qh), isabs == "true", f_agg)


def _query_similarity_count(x: object, rest: str) -> float:
    """tsfresh's count for the i-th query of the parity config; NaN where
    stumpy raises (a series shorter than the query)."""
    with open(CONFIG, "rb") as f:
        queries = tomllib.load(f)["query_similarity_count"]["parameters"]
    param = dict(queries[int(rest.removeprefix("query_"))])
    param["query"] = np.array(param["query"])
    try:
        return _combiner(fc.query_similarity_count, x, param)
    except ValueError:
        return math.nan


# Features whose column names need their own parsing
GROUP_A = {
    "energy_ratio_by_chunks": _energy_ratio_by_chunks,
    "change_quantiles": _change_quantiles,
    "query_similarity_count": _query_similarity_count,
}


def _tsfx_only(column: str) -> bool:
    """Columns tsfresh cannot compute: agg_linear_trend's median aggregator
    is a tsfx addition (tsfresh's extraction hands it a numpy array, which
    has no ``median`` method)."""
    return column.endswith("__agg_median")


def _deviation(feature: str, x: list[float]) -> bool:
    """Documented deliberate deviations from tsfresh (see the parity tests)."""
    if feature == "sample_entropy":
        return any(math.isinf(v) for v in x)
    if feature == "fft_coefficient":
        # tsfx: NaN for every attribute; numpy keeps some parts finite
        return any(not math.isfinite(v) for v in x)
    if feature in ("sum_of_reoccurring_values", "sum_of_reoccurring_data_points"):
        # a NaN or ±inf that occurs only once is left out by tsfx
        odd = [v for v in x if not math.isfinite(v)]
        return any(
            sum(1 for w in odd if (math.isnan(v) and math.isnan(w)) or v == w) == 1
            for v in odd
        )
    if feature == "ar_coefficient":
        # tsfresh raises on NaN / ±inf (MissingDataError); tsfx gives NaN
        if not all(math.isfinite(v) for v in x):
            return True
        return not _well_posed(x, _ar_designs(x))
    if feature == "augmented_dickey_fuller":
        return not _well_posed(x, _adf_design(x)) or _adf_exact_fit(x)
    return False


# Least squares designs conditioned worse than this are not fuzzed: their
# statsmodels results depend on rounding (LAPACK's as much as tsfx's)
MAX_CONDITION = 1e6


def _well_posed(values: list[float], designs: list) -> bool:
    """Whether the regression features are well-posed on this series: no
    nonzero value so small its square underflows (|v| < 1e-150), and every
    design condition number at most MAX_CONDITION. This excludes
    rank-deficient and near-collinear designs, so also the AIC / BIC ties of
    collinear extra lags (tsfx takes the smaller lag, statsmodels decides by
    rounding)."""
    if any(0 < abs(v) < 1e-150 for v in values):
        return False
    return all(np.linalg.cond(d) <= MAX_CONDITION for d in designs)


def _ar_designs(values: list[float]) -> list:
    """AutoReg's designs [1, x_{t-1}, ..., x_{t-k}] for the parity config's
    k that can be fitted (n >= 2k + 2)."""
    x = np.array(values, dtype=float)
    n = len(x)
    return [
        np.column_stack([np.ones(n - k)] + [x[k - c : n - c] for c in range(1, k + 1)])
        for k in (1, 2, 4)
        if n >= 2 * k + 2
    ]


def _adf_design(values: list[float]) -> list:
    """The largest ADF regression design (constant, level, all lags), or none
    if statsmodels raises before fitting."""
    x = np.array(values, dtype=float)
    n = len(x)
    if n < 4 or not np.isfinite(x).all() or x.max() == x.min():
        return []
    maxlag = min(math.ceil(12 * (n / 100) ** 0.25), n // 2 - 2)
    xd = np.diff(x)
    nobs = n - 1 - maxlag
    cols = [np.ones(nobs), x[maxlag : maxlag + nobs]]
    cols += [xd[maxlag - lag : maxlag - lag + nobs] for lag in range(1, maxlag + 1)]
    return [np.column_stack(cols)]


def _adf_exact_fit(values: list[float]) -> bool:
    """Whether the largest ADF regression fits (nearly) exactly: tsfx takes
    the residual variance as 0 below a relative residual of 1e-10, where
    statsmodels divides by rounding noise; up to 1e-5 the residual's noise
    can still exceed the comparison tolerance."""
    designs = _adf_design(values)
    if not designs:
        return False
    (design,) = designs
    x = np.array(values, dtype=float)
    maxlag = design.shape[1] - 2
    xd = np.diff(x)
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


def _same_spectral(column: str, a: float, b: float, x: object) -> bool:
    """FFT-based columns: tsfx's FFT (rustfft) rounds differently from
    numpy's (pocketfft), so the absolute error scales with the whole series,
    not the value. Values that are 0 in exact arithmetic come out as rounding
    noise, so their angle is arbitrary and not compared."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return _same(a, b, tolerant=True)
    if column.startswith("fft_coefficient__attr_angle"):
        coeff = int(column.rsplit("_", 1)[1])
        magnitude = abs(np.fft.rfft(x)[coeff])
        noise = 1e-12 * np.sum(np.abs(x))
        if magnitude <= 1e4 * noise:
            return True
        diff = abs((a - b + 180.0) % 360.0 - 180.0)
        return diff <= 1e-4
    if column.startswith("fft_coefficient"):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12 * np.sum(np.abs(x)))
    if column.startswith("spkt_welch_density"):
        scale = len(x) * np.sum((x - np.mean(x)) ** 2)
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-13 * scale)
    if column.startswith("fourier_entropy") and _on_bin_edge(
        x, int(column.rsplit("_", 1)[1])
    ):
        return True
    return _same(a, b, tolerant=True)


def _on_bin_edge(x: object, bins: int) -> bool:
    """Whether a value of fourier_entropy's scaled spectrum is within rounding
    of an inner histogram bin edge, so that the FFT's rounding decides its
    bin (for example a bin with exactly half the peak power and 2 bins)."""
    _, pxx = signal.welch(x, nperseg=min(len(x), 256))
    scaled = pxx / np.max(pxx)
    if not np.all(np.isfinite(scaled)):
        return False
    edges = np.histogram_bin_edges(scaled, bins)[1:-1]
    return bool(np.any(np.abs(scaled[:, None] - edges[None, :]) <= 1e-9))


def _same_ar(x: object, column: str, a: float, b: float) -> bool:
    """ar_coefficient compared with an absolute tolerance relative to the
    largest coefficient of the same fit: least squares rounding errors scale
    with the size of the whole solution, not of each coefficient."""
    if a is None or b is None or math.isnan(float(a)) or math.isnan(float(b)):
        return _same(a, b, tolerant=True)
    k = int(column.rsplit("__k_", 1)[1])
    fit = fc.ar_coefficient(x, [{"coeff": c, "k": k} for c in range(k + 1)])
    scale = max([1.0] + [abs(v) for _, v in fit if math.isfinite(v)])
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9 * scale)


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
            if feature in SPECTRAL:
                same = _same_spectral(column, value, expected, x)
            elif feature == "ar_coefficient":
                same = _same_ar(x, column, value, expected)
            else:
                same = _same(value, expected, tolerant)
            if not same:
                mismatches.append(f"{column}: tsfx {value!r}, tsfresh {expected!r}")
        assert not mismatches, "\n".join(mismatches)

    check()
