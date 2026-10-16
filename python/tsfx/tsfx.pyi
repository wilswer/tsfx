from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

import polars as pl

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

class ScalerSetting:
    """How to scale the extracted features, column-wise across groups.

    Matches scikit-learn's `StandardScaler` and `MinMaxScaler` fitted on the
    feature matrix: NaN and null are ignored when fitting and kept in the
    output, as are infinities (which scikit-learn rejects), and
    (near-)constant features map to 0 (`Standard`) or the lower end of the
    range (`MinMax`). Grouping and time columns are not scaled, and the
    scaled features are Float64.

    - `Standard()`: zero mean and unit (population) variance.
    - `MinMax(feature_range=(0, 1))`: maps each feature's [min, max] onto
      `feature_range`.
    """

    class Standard(ScalerSetting):
        def __init__(self) -> None: ...

    class MinMax(ScalerSetting):
        feature_range: tuple[float, float]

        def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None: ...

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
    datetime_format: str | None

    def __init__(
        self,
        time_col: str,
        every: str,
        period: str,
        offset: str,
        datetime_format: str | None = None,
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
        Optional path to a TOML feature config file; a path that does not
        exist raises `ValueError` on extraction. Without one,
        `.tsfx-config.toml` in the working directory is used if present, else
        the default config.
    dynamic_settings:
        Optional `DynamicGroupBySettings` for time-windowed extraction.
    scaling:
        Optional `ScalerSetting` to scale each feature across groups after
        extraction, e.g. `ScalerSetting.Standard()` or
        `ScalerSetting.MinMax(feature_range=(-1, 1))`; a variant without
        arguments may be given without calling it (`ScalerSetting.Standard`).
        Defaults to None: no scaling. An invalid `feature_range` (not finite,
        or minimum not below maximum) raises `ValueError`.

    """

    grouping_cols: Sequence[str]
    value_cols: Sequence[str]
    feature_setting: FeatureSetting
    config_path: str | None
    dynamic_settings: DynamicGroupBySettings | None
    scaling: ScalerSetting | None

    @property
    def _input_columns(self) -> list[str]:
        """Columns extraction reads: grouping, value and any time column."""
        ...

    def __init__(
        self,
        grouping_cols: Sequence[str],
        value_cols: Sequence[str],
        feature_setting: FeatureSetting,
        config_path: str | None = None,
        dynamic_settings: DynamicGroupBySettings | None = None,
        scaling: ScalerSetting | type[ScalerSetting] | None = None,
    ) -> None: ...

def extract_features_df(
    df: pl.DataFrame,
    settings: ExtractionSettings,
) -> pl.DataFrame:
    """Extract features from an in-memory DataFrame (used by the Python wrapper)."""
    ...

def extract_features(
    lf: pl.LazyFrame,
    settings: ExtractionSettings,
    streaming: bool = False,
) -> pl.DataFrame:
    """Extract time-series features from a Polars LazyFrame.

    Parameters
    ----------
    lf:
        The input data to extract features from.
    settings:
        The configuration controlling grouping and feature complexity.
    streaming:
        If True, executes the query using Polars' streaming engine
        for out-of-core processing.

    Returns
    -------
    polars.DataFrame
        A new DataFrame containing the grouped IDs and their extracted features.

    """
    ...
