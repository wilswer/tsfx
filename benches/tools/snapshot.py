"""Snapshot extract_features output to check that a change keeps every value.

Usage: python benches/tools/snapshot.py OUT.json

Runs every tier with three configs on small groups that cover the edge
cases (constant, single value, NaN, all null, nulls, an Int64 column) and
writes the column names and values. Take one snapshot before the change and
one after (with the extension rebuilt), then ``cmp`` them: identical files
mean identical column order and bit-identical values.
"""

import json
import math
import random
import sys

import polars as pl
from tsfx import ExtractionSettings, FeatureSetting, extract_features

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
    return pl.DataFrame(
        {"id": ids, "val": vals, "ival": ivals},
        schema={"id": pl.Utf8, "val": pl.Float64, "ival": pl.Int64},
    )


def _encode(x: object) -> object:
    return "NaN" if isinstance(x, float) and math.isnan(x) else x


def main(out_path: str) -> None:
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
            result = extract_features(df.lazy(), settings).sort("id")
            out[f"{config_name}/{tier}"] = {
                "columns": result.columns,
                "rows": [[_encode(x) for x in row] for row in result.rows()],
            }
    with open(out_path, "w") as f:
        json.dump(out, f, sort_keys=True)
    print({k: len(v["columns"]) for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1])
