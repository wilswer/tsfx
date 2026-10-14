"""Head-to-head benchmark of TSFX against TSFresh.

Usage (from anywhere):

    python python_benchmarks/tsfresh_compare.py --tier efficient
    python python_benchmarks/tsfresh_compare.py --dataset 1brc --rows 10_000_000

The comparison is kept fair by:

- Same features: TSFresh runs exactly the feature calculators, parameters and
  output attributes that TSFX runs for the chosen tier, derived from the same
  `.tsfx-config.toml`. The script checks that both produce the same number of
  features per value column before timing anything.
- Same input: the data is read, cleaned, cast to Float64 and sorted by
  (id, time) once, outside the timed region. Both libraries then get the same
  in-memory rows in the same order, so neither pays for CSV parsing or sorting.
- Same hardware budget: TSFresh's `n_jobs` is set to the size of the Polars
  thread pool TSFX runs on. Set `POLARS_MAX_THREADS` to limit both. BLAS is
  pinned to one thread per process, as TSFresh recommends for multiprocessing.
- Repeated runs: both get an untimed warm-up on a small subset, then
  `--repeats` timed runs; the best and median are reported.
"""

# ruff: noqa: E402
import os

# TSFresh parallelises over processes; multithreaded BLAS inside each worker
# oversubscribes the CPU and slows it down. TSFX does not use BLAS.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import statistics
import time
import tomllib
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import tsfresh
from tsfx import ExtractionSettings, FeatureSetting, extract_features

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".tsfx-config.toml"
TIERS = {
    "minimal": FeatureSetting.Minimal,
    "efficient": FeatureSetting.Efficient,
    "comprehensive": FeatureSetting.Comprehensive,
}
# TSFX config names that TSFresh spells differently
RENAMED = {
    "absolute_energy": "abs_energy",
    "mean_absolute_change": "mean_abs_change",
    "percentage_of_reoccurring_values_to_all_datapoints": (
        "percentage_of_reoccurring_datapoints_to_all_datapoints"
    ),
}
# TSFX only emits these attributes of the linear-trend fits
TREND_ATTRS = ("intercept", "slope")


def active_calculators(tier: FeatureSetting) -> set[str]:
    """Feature calculators TSFX actually computes for `tier`.

    Read off TSFX's output columns on a tiny frame, so tier membership never
    has to be duplicated here.
    """
    df = pl.DataFrame({"id": [0] * 50, "x": np.random.default_rng(0).normal(size=50)})
    settings = ExtractionSettings(
        grouping_cols=["id"],
        value_cols=["x"],
        feature_setting=tier,
        config_path=str(CONFIG_PATH),
    )
    cols = set(extract_features(df.lazy(), settings).columns) - {"id"}
    active = set()
    for name in [*tomllib.loads(CONFIG_PATH.read_text()), "length"]:
        prefix = f"x__{name}"
        # Trend features are named `x__linear_trend_slope` etc.
        starts = (f"{prefix}__", *(f"{prefix}_{a}" for a in TREND_ATTRS))
        matched = {c for c in cols if c in (name, prefix) or c.startswith(starts)}
        if matched:
            active.add(name)
            cols -= matched
    # Features without a config section
    active |= {c.removeprefix("x__") for c in cols}
    return active


def tsfresh_parameters(tier: FeatureSetting) -> dict:
    """TSFresh `default_fc_parameters` matching what TSFX computes for `tier`."""
    config = tomllib.loads(CONFIG_PATH.read_text())
    fc_parameters = {}
    for name in sorted(active_calculators(tier)):
        params = config.get(name, {}).get("parameters")
        if name == "linear_trend":
            params = [{"attr": a} for a in TREND_ATTRS]
        elif name == "agg_linear_trend":
            params = [
                {"attr": a, "chunk_len": p["chunk_size"], "f_agg": p["aggregator"]}
                for p in params
                for a in TREND_ATTRS
            ]
        elif name == "mean_n_absolute_max":
            params = [{"number_of_maxima": p["n"]} for p in params]
        elif name == "cwt_coefficients":
            params = [{**p, "widths": tuple(p["widths"])} for p in params]
        fc_parameters[RENAMED.get(name, name)] = params
    return fc_parameters


def load_stocks() -> tuple[pl.DataFrame, str, list[str]]:
    """Kaggle S&P 500 daily prices: one series per ticker."""
    value_cols = ["open", "high", "low", "close", "volume"]
    df = (
        pl.read_csv(ROOT / "test_data/all_stocks_5yr.csv")
        .drop_nulls()
        .with_columns(pl.col(value_cols).cast(pl.Float64))
        .sort("Name", "date", maintain_order=True)
        .select("Name", *value_cols)
    )
    return df, "Name", value_cols


def load_1brc(rows: int | None) -> tuple[pl.DataFrame, str, list[str]]:
    """First `rows` 1BRC measurements: one series per weather station."""
    # No timestamps: each station's series is in file order, which a stable
    # sort by station keeps
    df = (
        pl.read_csv(
            ROOT / "test_data/measurements.txt",
            separator=";",
            has_header=False,
            new_columns=["location", "temperature"],
            schema_overrides={"temperature": pl.Float64},
            n_rows=rows,
        )
        .drop_nulls()
        .sort("location", maintain_order=True)
    )
    return df, "location", ["temperature"]


def time_runs(name: str, run: Callable[[], object], repeats: int) -> list[float]:
    """Wall-clock seconds of `repeats` calls to `run`."""
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
        print(f"  {name} run {i + 1}/{repeats}: {times[-1]:.2f} s")
    return times


def main() -> None:
    """Time both libraries on one dataset and tier."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["stocks", "1brc"], default="stocks")
    parser.add_argument("--tier", choices=list(TIERS), default="efficient")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000_000,
        help="1brc only: rows to read (0 for all, which is slow)",
    )
    args = parser.parse_args()

    tier = TIERS[args.tier]
    n_threads = pl.thread_pool_size()
    if args.dataset == "stocks":
        df, id_col, value_cols = load_stocks()
    else:
        df, id_col, value_cols = load_1brc(args.rows or None)
    pdf = df.to_pandas()
    fc_parameters = tsfresh_parameters(tier)
    settings = ExtractionSettings(
        grouping_cols=[id_col],
        value_cols=value_cols,
        feature_setting=tier,
        config_path=str(CONFIG_PATH),
    )

    def run_tsfx(data: pl.DataFrame = df) -> pl.DataFrame:
        return extract_features(data.lazy(), settings)

    def run_tsfresh(data: pd.DataFrame = pdf) -> pd.DataFrame:
        with warnings.catch_warnings():
            # Constant or short series trigger numpy/statsmodels warnings
            warnings.simplefilter("ignore")
            return tsfresh.extract_features(
                data,
                column_id=id_col,
                default_fc_parameters=fc_parameters,
                n_jobs=n_threads,
                disable_progressbar=True,
            )

    n_series = df[id_col].n_unique()
    print(
        f"{args.dataset}: {df.height:,} rows, {n_series:,} series x "
        f"{len(value_cols)} value columns, tier={args.tier}, "
        f"{n_threads} threads/processes"
    )

    # Warm-up (thread/process pools, lazy imports) and feature-parity check
    first_ids = df[id_col].unique(maintain_order=True)[:5]
    head = df.filter(pl.col(id_col).is_in(first_ids.implode()))
    tsfx_head, tsfresh_head = run_tsfx(head), run_tsfresh(head.to_pandas())
    for col in value_cols:
        # TSFX emits one shared `length` column, TSFresh one per value column
        n_tsfx = sum(c.startswith(f"{col}__") for c in tsfx_head.columns)
        n_tsfx += "length" in tsfx_head.columns
        n_tsfresh = sum(c.startswith(f"{col}__") for c in tsfresh_head.columns)
        if n_tsfx != n_tsfresh:
            raise SystemExit(
                f"Feature mismatch for {col!r}: TSFX {n_tsfx} vs TSFresh {n_tsfresh}"
            )
    print(f"Both compute {n_tsfresh} features per value column\n")

    print("TSFX")
    tsfx_times = time_runs("TSFX", run_tsfx, args.repeats)
    print("TSFresh")
    tsfresh_times = time_runs("TSFresh", run_tsfresh, args.repeats)

    print(f"\n{'':8}{'best':>10}{'median':>10}")
    for name, times in (("TSFX", tsfx_times), ("TSFresh", tsfresh_times)):
        print(f"{name:8}{min(times):>9.2f}s{statistics.median(times):>9.2f}s")
    best = min(tsfresh_times) / min(tsfx_times)
    median = statistics.median(tsfresh_times) / statistics.median(tsfx_times)
    print(f"Speed-up: {best:.1f}x (best), {median:.1f}x (median)")


if __name__ == "__main__":
    main()
