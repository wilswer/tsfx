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

class ReductionSetting:
    """How to drop redundant and uninformative features after extraction.

    Runs after extraction and before scaling. Drops features that are mostly
    missing, almost always one value, near-duplicates of an earlier feature
    (Spearman correlation), or noise (no better than shuffled shadow features
    at respecting which rows are neighbors). Each step is turned off with
    `None`. The reduction is fitted across the extracted rows (groups or
    windows), so it needs several of them and cannot be fitted on a single
    series. `reduction_report` explains every decision.

    Parameters
    ----------
    max_missing:
        Drop a feature whose share of missing values (null, NaN, ±inf) is
        above this, in [0, 1). `0.0` drops every feature with a missing value.
    max_top_share:
        Drop a feature whose most frequent value covers more than this share
        of its non-missing values, in (0, 1).
    max_correlation:
        Drop a feature whose |Spearman correlation| with an earlier kept
        feature is above this, in (0, 1).
    noise_sensitivity:
        About this share of pure-noise features is dropped (0.99 lets about 1
        in 100 through), in [0.5, 0.999]. Higher is stricter and slower and,
        with few features left, drops more real features by mistake: with
        fewer than 50 features or rows reaching the noise step, a
        `UserWarning` says so. `None` turns the step off.
    noise_neighbors:
        Nearest rows per row in the noise step, at least 1.
    max_samples:
        Row cap for the correlation and noise steps (a seeded sample above
        it), larger than `noise_neighbors`.
    seed:
        Seed of the row sample and the noise step.
    cap:
        A `CapSetting` to keep only the most informative features after the
        noise step. `None` (the default): no cap.

    """

    max_missing: float | None
    max_top_share: float | None
    max_correlation: float | None
    noise_sensitivity: float | None
    noise_neighbors: int
    max_samples: int
    seed: int
    cap: CapSetting | None

    def __init__(
        self,
        max_missing: float | None = 0.05,
        max_top_share: float | None = 0.95,
        max_correlation: float | None = 0.95,
        noise_sensitivity: float | None = 0.99,
        noise_neighbors: int = 10,
        max_samples: int = 5000,
        seed: int = 0,
        cap: CapSetting | None = None,
    ) -> None: ...

class CapSetting:
    """Caps the number of feature columns kept; never changes values.

    The cap is the reduction's last step: after the noise step it keeps the
    most informative of the remaining features, by greedy column subset
    selection. Each pick is the feature that explains the most of the
    remaining features' spread (Spearman, on ranks) not yet explained by the
    ones picked before it; its gain, in features' worth, is the information it
    adds per dimension (`cap_gain` in the reduction report, with the pick
    order in `cap_rank` and the share explained so far in `cap_explained`).

    It never keeps more features than the rows used minus one (all rows, or a
    sample of `max_samples` of them): past that, the kept features explain the
    others fully, and those are dropped as `explained`. `CapSetting()` keeps
    picking until nothing is left to explain, so it caps the features at the
    number of rows; `ReductionSetting(cap=None)` turns the cap off. Run it with
    the noise step on: it cannot tell noise from a feature unrelated to all
    the others.

    Parameters
    ----------
    max_features:
        Keep at most this many features, at least 1. `None` (the default): no
        count limit.
    min_explained:
        Stop once the kept features explain this share of the spread of the
        features reaching the cap, in (0, 1]. 1.0 (the default): until
        nothing is left to explain.

    """

    max_features: int | None
    min_explained: float

    def __init__(
        self,
        max_features: int | None = None,
        min_explained: float = 1.0,
    ) -> None: ...

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
    reduction:
        Optional `ReductionSetting` to drop redundant and uninformative
        features after extraction, before scaling. Defaults to None: no
        reduction.

    """

    grouping_cols: Sequence[str]
    value_cols: Sequence[str]
    feature_setting: FeatureSetting
    config_path: str | None
    dynamic_settings: DynamicGroupBySettings | None
    scaling: ScalerSetting | None
    reduction: ReductionSetting | None

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
        reduction: ReductionSetting | None = None,
    ) -> None: ...

def extract_features_df(
    df: pl.DataFrame,
    settings: ExtractionSettings,
) -> pl.DataFrame:
    """Extract features from an in-memory DataFrame (used by the Python wrapper)."""
    ...

def reduction_report_df(
    df: pl.DataFrame,
    settings: ExtractionSettings,
) -> pl.DataFrame:
    """Return the reduction's report for an in-memory DataFrame.

    Used by `reduction_report`.
    """
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
