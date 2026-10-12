"""Snapshot extract_features output to check that a change keeps every value.

Usage:
    python benches/tools/snapshot.py OUT.json
    python benches/tools/snapshot.py compare BEFORE.json AFTER.json

Runs every tier with three configs on small groups that cover the edge
cases (constant, single value, NaN, all null, nulls, an Int64 column), plus
a dynamic (time-window) group-by, and writes the column names and values.
Take one snapshot before the change and one after (with the extension
rebuilt). ``cmp`` them to check column order as well, or use ``compare`` to
match columns by name: it reports missing or extra columns and every value
that is not bit-identical.
"""

import datetime
import json
import math
import random
import sys

import polars as pl
from tsfx import (
    DynamicGroupBySettings,
    ExtractionSettings,
    FeatureSetting,
    extract_features,
)

CONFIGS = {
    "default": None,
    "parity": "python_tests/data/.tsfx-config-tsfresh-parity.toml",
    "test": "python_tests/data/.tsfx-config-test.toml",
}
TIERS = ("Minimal", "Efficient", "Comprehensive")


def _frame() -> pl.DataFrame:
    rng = random.Random(0)  # noqa: S311 (test data, not security)
    nan = math.nan
    groups = {
        "rand": [round(rng.gauss(0, 1), 3) for _ in range(60)],
        "walk": [float(i % 7) + 0.5 * (i % 3) for i in range(40)],
        "const": [2.0] * 12,
        "single": [5.0],
        "withnan": [1.0, nan, 3.0, 4.0, nan, 6.0, 0.0, 5.0, 2.0],
        "allnull": [None, None, None],
        "mixed": [1.0, None, 3.0, 3.0, -2.0, None, 7.0, 7.0, 0.0, 1.0],
    }
    ids = [k for k, v in groups.items() for _ in v]
    vals = [x for v in groups.values() for x in v]
    ivals = [None if v is None or math.isnan(v) else int(v * 10) for v in vals]
    # increasing within each group, as the dynamic group-by requires
    days = [
        f"2020-{1 + j // 28:02d}-{1 + j % 28:02d}"
        for v in groups.values()
        for j in range(len(v))
    ]
    return pl.DataFrame(
        {"id": ids, "time": days, "val": vals, "ival": ivals},
        schema={"id": pl.Utf8, "time": pl.Utf8, "val": pl.Float64, "ival": pl.Int64},
    )


def _encode(x: object) -> object:
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return x.isoformat() if isinstance(x, datetime.date) else x


def _run(df: pl.DataFrame, settings: ExtractionSettings, keys: list[str]) -> dict:
    result = extract_features(df.lazy(), settings).sort(keys)
    return {
        "columns": result.columns,
        "rows": [[_encode(x) for x in row] for row in result.rows()],
    }


def snapshot(out_path: str) -> None:
    """Write the snapshot of every config and tier to ``out_path``."""
    df = _frame()
    out = {}
    for config_name, config_path in CONFIGS.items():
        for tier in TIERS:
            settings = ExtractionSettings(
                grouping_cols=["id"],
                value_cols=["val", "ival"],
                feature_setting=getattr(FeatureSetting, tier),
                config_path=config_path,
            )
            out[f"{config_name}/{tier}"] = _run(df, settings, ["id"])
    dynamic = ExtractionSettings(
        grouping_cols=["id"],
        value_cols=["val", "ival"],
        feature_setting=FeatureSetting.Efficient,
        dynamic_settings=DynamicGroupBySettings(
            time_col="time",
            every="1mo",
            period="1mo",
            offset="0d",
            datetime_format="%Y-%m-%d",
        ),
    )
    out["dynamic/Efficient"] = _run(df, dynamic, ["id", "time"])
    with open(out_path, "w") as f:
        json.dump(out, f, sort_keys=True)
    print({k: len(v["columns"]) for k, v in out.items()})


def _same(a: object, b: object) -> bool:
    return a == b and type(a) is type(b)


def compare(before_path: str, after_path: str) -> bool:
    """Match columns by name; report differences. True if none."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)
    ok = True
    for run in sorted(before.keys() | after.keys()):
        if run not in before or run not in after:
            print(f"{run}: only in {'after' if run in after else 'before'}")
            ok = False
            continue
        a, b = before[run], after[run]
        missing = [c for c in a["columns"] if c not in b["columns"]]
        extra = [c for c in b["columns"] if c not in a["columns"]]
        if missing or extra:
            print(f"{run}: missing {missing[:5]} extra {extra[:5]}")
            ok = False
        if len(a["rows"]) != len(b["rows"]):
            print(f"{run}: {len(a['rows'])} rows before, {len(b['rows'])} after")
            ok = False
            continue
        idx = {c: i for i, c in enumerate(b["columns"])}
        for j, column in enumerate(a["columns"]):
            if column not in idx:
                continue
            for r, (row_a, row_b) in enumerate(zip(a["rows"], b["rows"], strict=True)):
                old, new = row_a[j], row_b[idx[column]]
                if not _same(old, new):
                    print(f"{run}: {column} row {r}: {old!r} -> {new!r}")
                    ok = False
    print("IDENTICAL (by column name)" if ok else "DIFFERENT")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    snapshot(sys.argv[1])
