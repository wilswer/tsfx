from __future__ import annotations

from enum import Enum, auto

import polars as pl

class FeatureSetting(Enum):
    Mimimal = auto()
    Efficient = auto()
    Comprehensive = auto()

class ExtractionSettings:
    def __init__(self) -> None:
        self.grouping_col: str
        self.value_cols: list[str]
        self.feature_setting: FeatureSetting
        self.dynamic_settings: DynamicGroupBySettings | None = None

class DynamicGroupBySettings:
    def __init__(self) -> None:
        self.time_col: str
        self.every: str
        self.period: str
        self.offset: str
        self.datetime_format: str | None = None

def extract_features(df: pl.LazyFrame, opts: ExtractionSettings) -> pl.DataFrame: ...
