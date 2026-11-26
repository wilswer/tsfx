"""Differential fuzzing of the feature reduction's first three steps.

``reduction_report`` (noise step off, fewer rows than ``max_samples``, so no
sampling) is compared with a numpy/scipy reference of the missing, one-value
and near-duplicate steps on the same extracted features:

- missing share: share of null, NaN and ±inf values;
- top share: share of the most frequent non-missing value (-0.0 as 0.0);
- near-duplicates: missing values filled with the median, average ranks
  (``scipy.stats.rankdata``), standardized with the population standard
  deviation (zero variance gives zeros), correlations ``Zᵀ Z / n``, and a
  greedy walk in column order that keeps a feature unless its |ρ| with a kept
  one is above the threshold; it then copies the kept feature with the
  highest |ρ| (the first kept on ties).

The shares and the decisions must match exactly, the correlations to 1e-9.
Examples where a compared |ρ| is within 1e-9 of the threshold, or where two
kept features tie for the highest |ρ| within 1e-12, are skipped: rounding can
decide those either way.

Opt-in (marker ``fuzz``, needs hypothesis, numpy and scipy):

    pytest -m fuzz -k reduction

Arbitrary feature matrices come from one-row groups, whose Minimal features
repeat the drawn values; some value columns are monotone maps of others, to
make exact and monotone copies. The tests on Efficient features of real
series are slow and run at most 20 examples.

The cap (step 5) is compared with the greedy by definition on the step-3
survivors' standardized ranks: per step, every unpicked candidate's share of
the total spread explained together with the picks (``np.linalg.lstsq``
projections), a residual variance of at most 1e-9 meaning fully explained,
and the spec's stop rules; gains within a relative 1e-9 of the best are ties,
won by the earliest candidate. Picks and ``dropped_by`` must match exactly,
gains and shares to 1e-9. Examples with a gain between a relative 0.5e-9 and
2e-9 below the best, a residual variance between 1e-11 and 1e-7, or a share
within 1e-9 of ``min_explained`` are skipped.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import pytest
from tsfx import (
    CapSetting,
    ExtractionSettings,
    FeatureSetting,
    ReductionSetting,
    extract_features,
    reduction_report,
)

hypothesis = pytest.importorskip("hypothesis")
np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")
st = hypothesis.strategies

pytestmark = pytest.mark.fuzz

MAX_MISSING = 0.05
MAX_TOP_SHARE = 0.95
MAX_CORRELATION = 0.95
SPECIAL = [math.nan, math.inf, -math.inf, 0.0, -0.0]

MONOTONE: dict[str, Callable[[float], float]] = {
    "cube": lambda x: x**3 if abs(x) < 1e100 else math.copysign(math.inf, x),
    "negate": lambda x: -x,
    "exp": lambda x: math.exp(x / 1e150),
}


@st.composite
def value_column(draw: st.DrawFn, n: int) -> list[float]:
    """Draw `n` values: arbitrary, small integers or (near-)constant."""
    finite = st.floats(-1e150, 1e150, allow_nan=False, allow_infinity=False)
    kind = draw(st.sampled_from(["any", "integers", "near_constant", "constant"]))
    if kind == "any":
        value = st.one_of(finite, st.sampled_from(SPECIAL))
        return draw(st.lists(value, min_size=n, max_size=n))
    if kind == "integers":
        value = st.one_of(st.integers(-3, 3).map(float), st.sampled_from(SPECIAL))
        return draw(st.lists(value, min_size=n, max_size=n))
    base = draw(st.floats(-1e12, 1e12, allow_nan=False, allow_infinity=False))
    if kind == "constant":
        return [base] * n
    steps = draw(st.lists(st.integers(-4, 4), min_size=n, max_size=n))
    return [base + k * math.ulp(base or 1.0) for k in steps]


@st.composite
def one_row_groups(draw: st.DrawFn) -> tuple[pl.LazyFrame, list[str]]:
    """Draw groups of one row each with 2-6 value columns."""
    n = draw(st.integers(10, 60))
    k = draw(st.integers(2, 6))
    columns: dict[str, list[float]] = {}
    for c in range(k):
        if c > 0 and draw(st.integers(0, 3)) == 0:
            source = columns[f"v{draw(st.integers(0, c - 1))}"]
            transform = MONOTONE[draw(st.sampled_from(sorted(MONOTONE)))]
            columns[f"v{c}"] = [transform(x) for x in source]
        else:
            columns[f"v{c}"] = draw(value_column(n))
    ids = [f"g{i:03}" for i in range(n)]
    return pl.DataFrame({"id": ids, **columns}).lazy(), list(columns)


@st.composite
def real_series(draw: st.DrawFn) -> tuple[pl.LazyFrame, list[str]]:
    """Draw 10-20 series of 5-30 values each."""
    value = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
    ids: list[str] = []
    values: list[float] = []
    for g in range(draw(st.integers(10, 20))):
        series = draw(st.lists(value, min_size=5, max_size=30))
        ids.extend([f"g{g:03}"] * len(series))
        values.extend(series)
    return pl.DataFrame({"id": ids, "v": values}).lazy(), ["v"]


def _top_share(column: np.ndarray) -> float:
    present = column[~np.isnan(column)] + 0.0
    if len(present) == 0:
        return 1.0
    return float(np.unique(present, return_counts=True)[1].max() / len(present))


def _standardized_ranks(column: np.ndarray) -> np.ndarray:
    column = column + 0.0
    present = column[~np.isnan(column)]
    fill = float(np.median(present)) if len(present) else 0.0
    ranks = stats.rankdata(np.where(np.isnan(column), fill, column), method="average")
    sd = ranks.std()
    return (ranks - ranks.mean()) / sd if sd > 0 else np.zeros(len(ranks))


def _reference(features: pl.DataFrame, keys: Sequence[str]) -> dict[str, list[Any]]:
    """Run steps 1-3 in numpy/scipy, `hypothesis.assume` away borderline cases."""
    df = features.sort(keys)
    names = [c for c in df.columns if c not in keys]
    x = df.select(names).cast(pl.Float64).to_numpy()
    x = np.where(np.isfinite(x), x, np.nan)
    d = len(names)
    dropped: list[str | None] = [None] * d
    missing = [float(np.isnan(x[:, j]).mean()) for j in range(d)]
    for j in range(d):
        if missing[j] > MAX_MISSING:
            dropped[j] = "missing"
    top: list[float | None] = [None] * d
    for j in range(d):
        if dropped[j] is None:
            top[j] = _top_share(x[:, j])
            if top[j] > MAX_TOP_SHARE:  # type: ignore[operator]
                dropped[j] = "one_value"
    alive = [j for j in range(d) if dropped[j] is None]
    duplicate_of: list[str | None] = [None] * d
    correlation: list[float | None] = [None] * d
    if alive:
        z = np.column_stack([_standardized_ranks(x[:, j]) for j in alive])
        corr = z.T @ z / len(z)
        kept: list[int] = []
        for a in range(len(alive)):
            above = []
            for b in kept:
                r = float(corr[a, b])
                hypothesis.assume(abs(abs(r) - MAX_CORRELATION) > 1e-9)
                if abs(r) > MAX_CORRELATION:
                    above.append((abs(r), b, r))
            if not above:
                kept.append(a)
                continue
            best = max(above, key=lambda t: t[0])
            ties = [t for t in above if best[0] - t[0] <= 1e-12]
            hypothesis.assume(len(ties) == 1)
            j = alive[a]
            dropped[j] = "duplicate"
            duplicate_of[j] = names[alive[best[1]]]
            correlation[j] = best[2]
    return {
        "feature": names,
        "dropped_by": dropped,
        "missing_share": missing,
        "top_share": top,
        "duplicate_of": duplicate_of,
        "correlation": correlation,
    }


def _check(lf: pl.LazyFrame, value_cols: list[str], tier: FeatureSetting) -> None:
    plain = ExtractionSettings(["id"], value_cols, tier)
    want = _reference(extract_features(lf, plain), ["id"])
    reduction = ReductionSetting(
        max_missing=MAX_MISSING,
        max_top_share=MAX_TOP_SHARE,
        max_correlation=MAX_CORRELATION,
        noise_sensitivity=None,
    )
    report = reduction_report(
        lf,
        ExtractionSettings(["id"], value_cols, tier, reduction=reduction),
    )
    for name in ["feature", "dropped_by", "missing_share", "top_share", "duplicate_of"]:
        assert report[name].to_list() == want[name], name
    assert report["kept"].to_list() == [d is None for d in want["dropped_by"]]
    for got, expected in zip(
        report["correlation"].to_list(),
        want["correlation"],
        strict=True,
    ):
        assert (got is None) == (expected is None)
        if expected is not None:
            assert got == pytest.approx(expected, abs=1e-9)


@hypothesis.given(one_row_groups())
def test_matches_reference_matrices(case: tuple[pl.LazyFrame, list[str]]) -> None:
    lf, value_cols = case
    _check(lf, value_cols, FeatureSetting.Minimal)


@hypothesis.settings(max_examples=min(hypothesis.settings().max_examples, 20))
@hypothesis.given(real_series())
def test_matches_reference_real_series(case: tuple[pl.LazyFrame, list[str]]) -> None:
    lf, value_cols = case
    _check(lf, value_cols, FeatureSetting.Efficient)


@st.composite
def cap_case(draw: st.DrawFn) -> tuple[pl.LazyFrame, list[str], CapSetting]:
    """Draw one-row groups of independent value columns, and a cap."""
    n = draw(st.integers(10, 40))
    k = draw(st.integers(2, 25))
    value = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
    columns = {f"v{c}": draw(st.lists(value, min_size=n, max_size=n)) for c in range(k)}
    ids = [f"g{i:03}" for i in range(n)]
    cap = CapSetting(
        max_features=draw(st.one_of(st.none(), st.integers(1, 30))),
        min_explained=draw(st.one_of(st.just(1.0), st.floats(0.05, 0.99))),
    )
    return pl.DataFrame({"id": ids, **columns}).lazy(), list(columns), cap


def _cap_reference(
    features: pl.DataFrame,
    keys: Sequence[str],
    cap: CapSetting,
) -> dict[str, list[Any]]:
    """Run steps 1-3 and the cap in numpy, `hypothesis.assume` away borderline cases."""
    want = _reference(features, keys)
    names = want["feature"]
    x = features.sort(keys).select(names).cast(pl.Float64).to_numpy()
    x = np.where(np.isfinite(x), x, np.nan)
    n = len(x)
    candidates = [j for j, drop in enumerate(want["dropped_by"]) if drop is None]
    dropped = list(want["dropped_by"])
    rank: list[int | None] = [None] * len(names)
    gain: list[float | None] = [None] * len(names)
    explained: list[float | None] = [None] * len(names)
    if not candidates:
        return {
            "dropped_by": dropped,
            "cap_rank": rank,
            "cap_gain": gain,
            "cap_explained": explained,
        }
    z = np.column_stack([_standardized_ranks(x[:, j]) for j in candidates])

    def captured(basis: list[int]) -> float:
        """Σᵢ ‖P zᵢ‖², P the projection onto the `basis` columns of z."""
        if not basis:
            return 0.0
        coef = np.linalg.lstsq(z[:, basis], z, rcond=None)[0]
        return float(((z[:, basis] @ coef) ** 2).sum())

    def residual_variance(p: int, basis: list[int]) -> float:
        r = z[:, p]
        if basis:
            r = r - z[:, basis] @ np.linalg.lstsq(z[:, basis], r, rcond=None)[0]
        v = float(r @ r) / n
        hypothesis.assume(not 1e-11 <= v <= 1e-7)
        return v

    total = float((z**2).sum())
    picked: list[int] = []
    share = 0.0
    while True:
        options = [
            (captured([*picked, p]) / total, p)
            for p in range(len(candidates))
            if p not in picked and residual_variance(p, picked) > 1e-9
        ]
        if not options:
            break
        # Gains within a relative 1e-9 of the best are ties, won by the
        # earliest candidate; skip gains that rounding could put on either
        # side of that line
        top = max(s for s, _ in options) - share
        for s, _ in options:
            hypothesis.assume(not 0.5e-9 <= 1 - (s - share) / top <= 2e-9)
        new_share, p = min(
            (o for o in options if o[0] - share >= top * (1 - 1e-9)),
            key=lambda o: o[1],
        )
        j = candidates[p]
        picked.append(p)
        rank[j] = len(picked)
        gain[j] = (new_share - share) * total / n
        explained[j] = share = new_share
        if cap.max_features == len(picked):
            break
        if cap.min_explained < 1.0:
            hypothesis.assume(abs(share - cap.min_explained) > 1e-9)
            if share >= cap.min_explained:
                break
    before = captured(picked)
    for p in range(len(candidates)):
        if p in picked:
            continue
        j = candidates[p]
        if residual_variance(p, picked) <= 1e-9:
            dropped[j], gain[j] = "explained", 0.0
        else:
            dropped[j], gain[j] = "cap", (captured([*picked, p]) - before) / n
    return {
        "dropped_by": dropped,
        "cap_rank": rank,
        "cap_gain": gain,
        "cap_explained": explained,
    }


@hypothesis.given(cap_case())
def test_cap_matches_reference(
    case: tuple[pl.LazyFrame, list[str], CapSetting],
) -> None:
    lf, value_cols, cap = case
    plain = ExtractionSettings(["id"], value_cols, FeatureSetting.Minimal)
    want = _cap_reference(extract_features(lf, plain), ["id"], cap)
    reduction = ReductionSetting(
        max_missing=MAX_MISSING,
        max_top_share=MAX_TOP_SHARE,
        max_correlation=MAX_CORRELATION,
        noise_sensitivity=None,
        cap=cap,
    )
    settings = ExtractionSettings(
        ["id"],
        value_cols,
        FeatureSetting.Minimal,
        reduction=reduction,
    )
    with warnings.catch_warnings():
        # The cap warns without the noise step, and when the rows stop it
        warnings.simplefilter("ignore", UserWarning)
        report = reduction_report(lf, settings)
    assert report["dropped_by"].to_list() == want["dropped_by"]
    assert report["cap_rank"].to_list() == want["cap_rank"]
    for name in ["cap_gain", "cap_explained"]:
        for got, expected in zip(report[name].to_list(), want[name], strict=True):
            assert (got is None) == (expected is None), name
            if expected is not None:
                assert got == pytest.approx(expected, abs=1e-9), name
