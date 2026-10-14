"""Benchmark TSFX, functime and TSFresh on the features all three implement.

Usage (from anywhere), one library and size per process:

    python python_benchmarks/library_compare.py run --lib tsfx --rows 10_000_000 \
        --out results.jsonl
    python python_benchmarks/library_compare.py table results.jsonl

Each `run` appends one JSON line to `--out`; `table` prints a Markdown table of
the best times. Running each library in its own process keeps TSFresh's pandas
copy and worker processes from affecting the others, and lets the others
finish if one runs out of memory.

The comparison follows `tsfresh_compare.py` (same 1BRC input, cleaning and
sorting outside the timed region, thread budget, warm-up, best of
`--repeats`), restricted to the `Efficient`-tier features functime also
implements, with the parameters from `.tsfx-config.toml` (tsfresh's defaults):

- TSFX runs a config trimmed to those sections.
- TSFresh runs exactly the calculators and parameters TSFX runs for that
  config, as in `tsfresh_compare.py`.
- functime runs the equivalent `pl.col(x).ts.*` expressions (plain Polars for
  basic statistics) in one `group_by().agg()` on the same Polars thread pool.
  Its `linear_trend` also returns the residual sum of squares, so it reports
  one value more per series.

Not compared: features functime lacks (FFT, wavelets, autoregression, Langevin
and some entropies), and `energy_ratio_by_chunks`, whose functime version
fails on Polars 1.44.
"""

# ruff: noqa: E402
import argparse
import json
import resource
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import polars as pl

# Sets single-threaded BLAS for TSFresh's worker processes on import
import tsfresh_compare as tc
from tsfx import ExtractionSettings, FeatureSetting, extract_features

TIER = FeatureSetting.Efficient
LIBS = ("tsfx", "functime", "tsfresh")
Params = list[dict]
ExprsFn = Callable[[pl.Expr, Params], list[pl.Expr]]


def scalar(fn: Callable[[pl.Expr], pl.Expr]) -> ExprsFn:
    """Build the expression of a section without parameters."""
    return lambda x, params: [fn(x)]


def per_param(fn: Callable[[pl.Expr, dict], pl.Expr]) -> ExprsFn:
    """Build one expression per parameter set."""
    return lambda x, params: [fn(x, p) for p in params]


def change_quantiles(x: pl.Expr, params: Params) -> list[pl.Expr]:
    """Aggregate the changes functime returns for each corridor once."""
    out = []
    for ql, qh, isabs in sorted({(p["ql"], p["qh"], p["isabs"]) for p in params}):
        changes = x.ts.change_quantiles(q_low=ql, q_high=qh, is_abs=isabs)
        aggs = {
            p["f_agg"]
            for p in params
            if (p["ql"], p["qh"], p["isabs"]) == (ql, qh, isabs)
        }
        if "mean" in aggs:
            out.append(changes.list.mean())
        if "var" in aggs:
            out.append(changes.list.var(ddof=0))
    return out


# TSFX config section -> functime/Polars expressions for its parameter sets
FUNCTIME: dict[str, ExprsFn] = {
    "length": scalar(lambda x: x.len()),
    "sum_values": scalar(lambda x: x.sum()),
    "mean": scalar(lambda x: x.mean()),
    "median": scalar(lambda x: x.median()),
    "minimum": scalar(lambda x: x.min()),
    "maximum": scalar(lambda x: x.max()),
    "standard_deviation": scalar(lambda x: x.std(ddof=0)),
    "variance": scalar(lambda x: x.var(ddof=0)),
    "skewness": scalar(lambda x: x.skew(bias=False)),
    "kurtosis": scalar(lambda x: x.kurtosis(bias=False)),
    "root_mean_square": scalar(lambda x: x.ts.root_mean_square()),
    "absolute_energy": scalar(lambda x: x.ts.absolute_energy()),
    "mean_absolute_change": scalar(lambda x: x.ts.mean_abs_change()),
    "linear_trend": scalar(lambda x: x.ts.linear_trend()),
    "variance_larger_than_standard_deviation": scalar(
        lambda x: x.ts.var_gt_std(ddof=0),
    ),
    "ratio_beyond_r_sigma": per_param(
        lambda x, p: x.ts.ratio_beyond_r_sigma(ratio=p["r"]),
    ),
    "large_standard_deviation": per_param(
        lambda x, p: x.ts.large_standard_deviation(ratio=p["r"]),
    ),
    "symmetry_looking": per_param(lambda x, p: x.ts.symmetry_looking(ratio=p["r"])),
    "has_duplicate_max": scalar(lambda x: x.ts.has_duplicate_max()),
    "has_duplicate_min": scalar(lambda x: x.ts.has_duplicate_min()),
    "has_duplicate": scalar(lambda x: x.ts.has_duplicate()),
    "cid_ce": per_param(lambda x, p: x.ts.cid_ce(normalize=p["normalize"])),
    "absolute_maximum": scalar(lambda x: x.ts.absolute_maximum()),
    "absolute_sum_of_changes": scalar(lambda x: x.ts.absolute_sum_of_changes()),
    "count_above_mean": scalar(lambda x: x.ts.count_above_mean()),
    "count_below_mean": scalar(lambda x: x.ts.count_below_mean()),
    "count_above": per_param(lambda x, p: x.ts.count_above(threshold=p["t"])),
    "count_below": per_param(lambda x, p: x.ts.count_below(threshold=p["t"])),
    "first_location_of_maximum": scalar(lambda x: x.ts.first_location_of_maximum()),
    "first_location_of_minimum": scalar(lambda x: x.ts.first_location_of_minimum()),
    "last_location_of_maximum": scalar(lambda x: x.ts.last_location_of_maximum()),
    "last_location_of_minimum": scalar(lambda x: x.ts.last_location_of_minimum()),
    "longest_strike_above_mean": scalar(lambda x: x.ts.longest_streak_above_mean()),
    "longest_strike_below_mean": scalar(lambda x: x.ts.longest_streak_below_mean()),
    "variation_coefficient": scalar(lambda x: x.ts.variation_coefficient()),
    "mean_change": scalar(lambda x: x.ts.mean_change()),
    "ratio_value_number_to_time_series_length": scalar(
        lambda x: x.ts.ratio_n_unique_to_length(),
    ),
    "sum_of_reoccurring_values": scalar(lambda x: x.ts.sum_reoccurring_values()),
    "sum_of_reoccurring_data_points": scalar(lambda x: x.ts.sum_reoccurring_points()),
    "percentage_of_reoccurring_values_to_all_values": scalar(
        lambda x: x.ts.percent_reoccurring_values(),
    ),
    "percentage_of_reoccurring_values_to_all_datapoints": scalar(
        lambda x: x.ts.percent_reoccurring_points(),
    ),
    "mean_n_absolute_max": per_param(
        lambda x, p: x.ts.mean_n_absolute_max(n_maxima=p["n"]),
    ),
    "autocorrelation": per_param(lambda x, p: x.ts.autocorrelation(n_lags=p["lag"])),
    "quantile": per_param(lambda x, p: x.quantile(p["q"], "linear")),
    "number_crossing_m": per_param(
        lambda x, p: x.ts.number_crossings(crossing_value=p["m"]),
    ),
    "range_count": per_param(
        lambda x, p: x.ts.range_count(lower=p["min"], upper=p["max"], closed="left"),
    ),
    "index_mass_quantile": per_param(lambda x, p: x.ts.index_mass_quantile(q=p["q"])),
    "c3": per_param(lambda x, p: x.ts.c3(n_lags=p["lag"])),
    "time_reversal_asymmetry_statistic": per_param(
        lambda x, p: x.ts.time_reversal_asymmetry_statistic(n_lags=p["lag"]),
    ),
    "number_peaks": per_param(lambda x, p: x.ts.number_peaks(support=p["n"])),
    "mean_second_derivative_central": scalar(
        lambda x: x.ts.mean_second_derivative_central(),
    ),
    "change_quantiles": change_quantiles,
    "benford_correlation": scalar(lambda x: x.ts.benford_correlation()),
    "binned_entropy": per_param(
        lambda x, p: x.ts.binned_entropy(bin_count=p["max_bins"]),
    ),
    "permutation_entropy": per_param(
        lambda x, p: x.ts.permutation_entropy(tau=p["tau"], n_dims=p["dimension"]),
    ),
}


def toml_value(v: object) -> str:
    """`v` as a TOML literal (bools, numbers and strings only)."""
    if isinstance(v, bool):
        return str(v).lower()
    return json.dumps(v) if isinstance(v, str) else repr(v)


def write_shared_config(path: Path) -> dict:
    """Write the TSFX config trimmed to the sections functime implements."""
    config = tomllib.loads(tc.CONFIG_PATH.read_text())
    shared = {name: config[name] for name in config if name in FUNCTIME}
    lines = []
    for name, section in shared.items():
        lines.append(f"[{name}]")
        if params := section.get("parameters"):
            items = (
                ", ".join(f"{k} = {toml_value(v)}" for k, v in p.items())
                for p in params
            )
            rows = [f"    {{ {item} }}," for item in items]
            lines += ["parameters = [", *rows, "]"]
        lines.append("")
    path.write_text("\n".join(lines))
    return shared


def functime_exprs(col: str, shared: dict) -> list[pl.Expr]:
    """Build the functime expressions for every feature in `shared`."""
    import functime.feature_extractors  # noqa: F401  registers the `ts` namespace

    exprs = []
    for name, section in shared.items():
        for expr in FUNCTIME[name](pl.col(col), section.get("parameters", [])):
            # functime returns a plain constant for some parameters
            # (autocorrelation at lag 0 is 1.0)
            if not isinstance(expr, pl.Expr):
                expr = pl.lit(expr)
            exprs.append(expr.alias(f"{col}__{name}__{len(exprs)}"))
    return exprs


def n_values(df: pl.DataFrame, id_col: str) -> int:
    """Feature values per series, counting list and struct entries."""
    total = 0
    for col in df.columns:
        if col == id_col:
            continue
        dtype = df.schema[col]
        if isinstance(dtype, pl.List):
            total += len(df[col][0])
        elif isinstance(dtype, pl.Struct):
            total += len(dtype.fields)
        else:
            total += 1
    return total


def run(args: argparse.Namespace) -> None:
    """Time one library at one size and append the result to `args.out`."""
    shared_path = args.out.with_suffix(".shared-config.toml")
    shared = write_shared_config(shared_path)
    # TSFresh's parameters are derived from what TSFX computes for this config
    tc.CONFIG_PATH = shared_path
    df, id_col, value_cols = tc.load_1brc(args.rows or None)
    n_threads = pl.thread_pool_size()
    first_ids = df[id_col].unique(maintain_order=True)[:50]
    head = df.filter(pl.col(id_col).is_in(first_ids.implode()))

    if args.lib == "tsfx":
        settings = ExtractionSettings(
            grouping_cols=[id_col],
            value_cols=value_cols,
            feature_setting=TIER,
            config_path=str(shared_path),
        )

        def extract(data: pl.DataFrame) -> pl.DataFrame:
            return extract_features(data.lazy(), settings)

        n_features = extract(head).width - 1
    elif args.lib == "functime":
        exprs = [e for col in value_cols for e in functime_exprs(col, shared)]

        def extract(data: pl.DataFrame) -> pl.DataFrame:
            return data.lazy().group_by(id_col).agg(exprs).collect()

        n_features = n_values(extract(head), id_col)
    else:
        import warnings

        import pandas as pd
        import tsfresh

        fc_parameters = tc.tsfresh_parameters(TIER)

        def extract(data: pd.DataFrame) -> pd.DataFrame:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return tsfresh.extract_features(
                    data,
                    column_id=id_col,
                    default_fc_parameters=fc_parameters,
                    n_jobs=n_threads,
                    disable_progressbar=True,
                )

        n_features = extract(head.to_pandas()).shape[1]

    rows, n_series = df.height, df[id_col].n_unique()
    data = df.to_pandas() if args.lib == "tsfresh" else df
    if args.lib == "tsfresh":
        del df
    print(
        f"{args.lib}: {rows:,} rows, {n_series:,} series, {n_features} features, "
        f"{n_threads} threads",
        flush=True,
    )
    times = tc.time_runs(args.lib, lambda: extract(data), args.repeats)
    result = {
        "lib": args.lib,
        "rows": rows,
        "series": n_series,
        "features": n_features,
        "threads": n_threads,
        "times": times,
        # Peak resident memory in GiB of this process and of its largest
        # child (TSFresh's worker processes)
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "peak_child_rss_gib": (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
        ),
    }
    with args.out.open("a") as f:
        f.write(json.dumps(result) + "\n")


def human(n: int) -> str:
    """Format a row count as in the README: 1M, 10M, 1B."""
    for div, unit in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
        if n >= div:
            return f"{n / div:.3g}{unit}"
    return str(n)


def seconds(t: float) -> str:
    """Format a time with about three significant digits."""
    return f"{t:.0f} s" if t >= 100 else f"{t:.3g} s"


def table(args: argparse.Namespace) -> None:
    """Markdown table of the best time per library and size."""
    results = [json.loads(line) for line in args.results.read_text().splitlines()]
    best: dict[int, dict[str, float]] = {}
    for r in results:
        best.setdefault(r["rows"], {})[r["lib"]] = min(r["times"])
    print("| rows | TSFX | functime | TSFresh | vs functime | vs TSFresh |")
    print("| ---- | ---- | -------- | ------- | ----------- | ---------- |")
    for rows in sorted(best):
        t = best[rows]
        cells = [seconds(t[lib]) if lib in t else "–" for lib in LIBS]
        speedups = [
            f"{t[lib] / t['tsfx']:.0f}x" if lib in t and "tsfx" in t else "–"
            for lib in ("functime", "tsfresh")
        ]
        print(f"| {human(rows)} | " + " | ".join(cells + speedups) + " |")


def main() -> None:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p_run = sub.add_parser("run", help="time one library at one size")
    p_run.add_argument("--lib", choices=LIBS, required=True)
    p_run.add_argument(
        "--rows",
        type=int,
        default=10_000_000,
        help="rows to read (0 for all)",
    )
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.set_defaults(func=run)
    p_table = sub.add_parser("table", help="print a Markdown table of results")
    p_table.add_argument("results", type=Path)
    p_table.set_defaults(func=table)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
