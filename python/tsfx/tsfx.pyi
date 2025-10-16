from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

"""Type stubs for the tsfx public configuration types."""

class FeatureSetting(Enum):
    """Preset groups of feature extractors.

    - `Minimal`: smallest, fastest set of features.
    - `Efficient`: balanced set of commonly useful features (default choice).
    - `Comprehensive`: largest set of features for maximum coverage.
    """

    Minimal = auto()
    Efficient = auto()
    Comprehensive = auto()

class DynamicGroupBySettings:
    r"""Settings for dynamic (time-windowed) group-by extraction.

    Parameters
    ----------
    time_col:
        Name of the datetime column used for windowing.
    every:
        Window stride (e.g. \"1d\", \"3y\").
    period:
        Window length (e.g. \"3y\").
    offset:
        Offset for the windowing grid (e.g. \"0y\").
    datetime_format:
        Format string used to parse time column values when needed.

    """

    time_col: str
    every: str
    period: str
    offset: str
    datetime_format: str

    def __init__(
        self,
        time_col: str,
        every: str,
        period: str,
        offset: str,
        datetime_format: str,
    ) -> None: ...

class ExtractionSettings:
    """Configuration for a feature extraction run.

    Parameters
    ----------
    grouping_cols:
        Column names used to group rows into separate time series (e.g. an id).
    value_cols:
        Column names containing the numeric values to extract features from.
    feature_setting:
        One of the `FeatureSetting` presets controlling which feature calculators run.
    config_path:
        Optional path to `.tsfx-config.yaml` config file.
    dynamic_settings:
        Optional `DynamicGroupBySettings` for time-windowed extraction.

    """

    grouping_cols: Sequence[str]
    value_cols: Sequence[str]
    feature_setting: FeatureSetting
    config_path: str | None
    dynamic_settings: DynamicGroupBySettings | None

    def __init__(
        self,
        grouping_cols: Sequence[str],
        value_cols: Sequence[str],
        feature_setting: FeatureSetting,
        config_path: str | None = None,
        dynamic_settings: DynamicGroupBySettings | None = None,
    ) -> None: ...
