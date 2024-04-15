from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

class FeatureSetting(Enum):
    Minimal = auto()
    Efficient = auto()
    Comprehensive = auto()

class ExtractionSettings:
    def __init__(
        self,
        grouping_col: str,
        value_cols: Sequence[str],
        feature_setting: FeatureSetting,
        config_path: str | None = None,
        dynamic_settings: DynamicGroupBySettings | None = None,
    ) -> None:
        self.grouping_col = grouping_col
        self.value_cols = value_cols
        self.feature_setting = feature_setting
        self.dynamic_settings = dynamic_settings

class DynamicGroupBySettings:
    def __init__(
        self,
        time_col: str,
        every: str,
        period: str,
        offset: str,
        datetime_format: str,
    ) -> None:
        self.time_col = time_col
        self.every = every
        self.period = period
        self.offset = offset
        self.datetime_format = datetime_format
