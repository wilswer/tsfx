from __future__ import annotations

import polars as pl

class ExtractionSettings:
    def __init__(self) -> None:
        self.grouping_col: str
        self.value_cols: list[str]
        self.dynamic_opts: DynamicGroupBySettings | None

class DynamicGroupBySettings:
    def __init__(self) -> None:
        self.time_col: str
        self.every: str
        self.period: str
        self.offset: str

def extract_features(df: pl.LazyFrame, opts: ExtractionSettings) -> pl.DataFrame: ...
