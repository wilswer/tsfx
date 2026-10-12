"""Randomised comparison of number_peaks against tsfresh (skipped when tsfresh
is not installed, e.g. in CI)."""

import math
import random

import polars as pl
import pytest
from tsfx import ExtractionSettings, FeatureSetting, extract_features

tsfresh_fc = pytest.importorskip("tsfresh.feature_extraction.feature_calculators")
np = pytest.importorskip("numpy")

NS = [1, 3, 5, 10, 50]
CONFIG = "./python_tests/data/.tsfx-config-number-peaks.toml"


def _series(rng: random.Random) -> list[float]:
    n = rng.choice([1, 2, 3, 5, 7, 11, 20, 60, 101, 150])
    kind = rng.choice(["noise", "ties", "smooth", "special"])
    if kind == "noise":
        return [rng.gauss(0, 1) for _ in range(n)]
    if kind == "ties":
        return [float(rng.randint(0, 3)) for _ in range(n)]
    if kind == "smooth":
        return [math.sin(i / rng.uniform(1, 8)) for i in range(n)]
    pool = [math.nan, math.inf, -math.inf, 0.0, 1.0, 2.0]
    return [
        rng.choice(pool) if rng.random() < 0.3 else rng.gauss(0, 1) for _ in range(n)
    ]


def test_number_peaks_matches_tsfresh_on_random_series():
    rng = random.Random(42)
    series = {f"s{i}": _series(rng) for i in range(200)}
    df = pl.DataFrame(
        {
            "id": [k for k, v in series.items() for _ in v],
            "val": [x for v in series.values() for x in v],
        },
    )
    out = extract_features(
        df.lazy(),
        ExtractionSettings(
            grouping_cols=["id"],
            value_cols=["val"],
            feature_setting=FeatureSetting.Efficient,
            config_path=CONFIG,
        ),
    )
    rows = {r["id"]: r for r in out.iter_rows(named=True)}
    mismatches = []
    for key, values in series.items():
        for n in NS:
            expected = tsfresh_fc.number_peaks(np.array(values), n)
            got = rows[key][f"val__number_peaks__n_{n}"]
            if got != expected:
                mismatches.append((key, len(values), n, got, expected))
    assert not mismatches, mismatches[:10]
